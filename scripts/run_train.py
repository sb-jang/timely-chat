import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from logging import Logger
from math import ceil, exp

import mlflow
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.optim import Adam8bit, GlobalOptimManager
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from timeliness.config import ArtifactConfig, ExecutionConfig, ExperimentConfig, TrainConfig
from timeliness.dataset import TimelinessDataset
from timeliness.utils.argparser import build_parser, parse_args
from timeliness.utils.logging import TQDM_FORMAT, create_logger
from timeliness.utils.utils import log_mlflow_metric, log_mlflow_param, print_config, set_seed

# Load dotenv
load_dotenv()

# Create and build parser
parser = ArgumentParser()
build_parser(parser, [ArtifactConfig, TrainConfig, ExperimentConfig, ExecutionConfig])

# Constants
LABEL_MASK_ID = 0

# Logger, initialized inside main function
logger: Logger = None


# ==================================================
# > Setup and Cleanup Functions
# ==================================================
def setup(rank: int, world_size: int, experiment_config: ExperimentConfig, execution_config: ExecutionConfig):
    """분산 학습을 위해 초기화 합니다."""
    global logger
    global log_mlflow_metric, log_mlflow_param, print_config

    # setup distributed setup
    os.environ["MASTER_ADDR"] = execution_config.ddp_master_address
    os.environ["MASTER_PORT"] = execution_config.ddp_master_port

    # Intialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    # Initlaize Logger
    log_level = logging.INFO if rank == 0 else logging.ERROR
    logger = create_logger(__name__, experiment_config.log_output_dir, log_level, rank)

    if rank == 0:
        # MLflow 관련 설정
        mlflow.set_experiment(experiment_config.experiment_name)
        mlflow_run = mlflow.start_run(run_name=experiment_config.run_name)

        logger.info(f"[+] Start tracking to MLflow with Run ID: {mlflow_run.info.run_id}")

    # 유틸리티 함수 로거들 Partial Patch
    log_mlflow_metric = partial(log_mlflow_metric, logger=logger)
    log_mlflow_param = partial(log_mlflow_param, logger=logger)
    print_config = partial(print_config, logger=logger)


def cleanup():
    """학습 종료 후 리소스를 정리합니다."""
    rank = dist.get_rank()

    if rank == 0:
        mlflow.end_run()

    dist.barrier()
    dist.destroy_process_group()


# ==================================================
# > Validation Function
# ==================================================
def validation(model: AutoModelForCausalLM, dataloader: DataLoader):
    """
    학습된 모델에 validation을 진행합니다.

    :param model: 생성 모델
    :param dataloader: dev 데이터셋 DataLoader
    """
    rank = dist.get_rank()
    dev_loss = torch.tensor([0.0], dtype=torch.float32).to(rank)
    num_non_mask = torch.tensor([0.0], dtype=torch.float32).to(rank)

    with torch.no_grad():
        for input_ids, label_ids in tqdm(dataloader, disable=rank != 0):
            input_ids = input_ids.to(rank)
            label_ids = label_ids.to(rank)
            num_non_mask += (label_ids != LABEL_MASK_ID).sum()

            lm_logits = model(input_ids=input_ids).logits

            loss = F.cross_entropy(lm_logits.transpose(1, 2), label_ids, ignore_index=LABEL_MASK_ID, reduction="sum")
            dev_loss += loss

    dist.all_reduce(dev_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_non_mask, op=dist.ReduceOp.SUM)
    dev_loss /= num_non_mask
    dev_ppl = torch.exp(dev_loss)

    return dev_loss.item(), dev_ppl.item()


# ==================================================
# > Train Function
# ==================================================
def run(
    rank: int,
    world_size: int,
    artifact_config: ArtifactConfig,
    train_config: TrainConfig,
    experiment_config: ExperimentConfig,
    execution_config: ExecutionConfig,
):
    # Setup Experiment for Distributed Training
    setup(rank, world_size, experiment_config, execution_config)
    # Show all prepared configurations
    print_config(artifact_config, train_config, experiment_config, execution_config)
    # Set seed
    set_seed(train_config.random_seed)

    # ==================================================
    # > Tokenizer Initialization
    # ==================================================
    logger.info("[+] Tokenizer Initializing...")
    tokenizer = AutoTokenizer.from_pretrained(artifact_config.pretrained_model)
    tokenizer.model_max_length = train_config.max_sequence_length

    # ==================================================
    # > Dataset Initialization
    # ==================================================
    logger.info("[+] Load train/dev dataset...")

    # Load "Train" dataset
    # TODO: Load dataset from HuggingFace Datasets
    with open("./resources/data/train.jsonl", "r") as f:
        train_dataset = [json.loads(line) for line in f]
    train_dataset = TimelinessDataset(train_dataset, tokenizer)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.train_batch_size,
        num_workers=execution_config.num_dataloader_workers,
        prefetch_factor=execution_config.dataloader_prefetch_factor,
        sampler=train_datasampler,
        pin_memory=True,
    )
    logger.info(f"[+] The size of train dataset: {len(train_dataset)}")

    # Load "Dev" dataset
    # TODO: Load dataset from HuggingFace Datasets
    with open("./resources/data/valid.jsonl", "r") as f:
        dev_dataset = [json.loads(line) for line in f]
    dev_dataset = TimelinessDataset(dev_dataset, tokenizer)
    dev_datasampler = DistributedSampler(dev_dataset, shuffle=False, drop_last=False)
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=train_config.dev_batch_size,
        num_workers=execution_config.num_dataloader_workers,
        prefetch_factor=execution_config.dataloader_prefetch_factor,
        sampler=dev_datasampler,
        pin_memory=True,
    )
    logger.info(f"[+] The size of dev datatset: {len(dev_dataset)}")

    total_train_batch_size = train_config.train_batch_size * train_config.gradient_accumulation_steps * world_size
    logger.info(f"[+] Total train batch size: {total_train_batch_size}")
    log_mlflow_param("total_train_batch_size", total_train_batch_size)

    # ==================================================
    # > Model Initialization
    # ==================================================
    # Initialize CausalLM Model
    logger.info("[+] Load model...")

    model_config = AutoConfig.from_pretrained(artifact_config.pretrained_model)

    # 추가 파라미터 Override
    # gradient checkpoint 적용 시 attention 등 결과를 output으로 반환하는 use_cache 사용 불가
    model_config.use_cache = False
    model_config.attention_dropout = train_config.attention_dropout
    model_config.hidden_dropout = train_config.hidden_dropout

    print_config(model_config)

    if artifact_config.pretrained_model_weight_path is None:
        model = AutoModelForCausalLM.from_pretrained(artifact_config.pretrained_model, config=model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(artifact_config.pretrained_model_weight_path, config=model_config)

    # Patch Embedding Layers
    if train_config.use_8bit_adam:
        manager = GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})

    model.gradient_checkpointing_enable()
    model.to(rank)

    logger.info(f"[+] Upload model parameters to rank {rank}")
    model = DistributedDataParallel(model, device_ids=[rank], gradient_as_bucket_view=True)

    # ==================================================
    # > Trainable Module's Initialization
    # ==================================================
    # Optimizer
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": train_config.weight_decay_ratio,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = ZeroRedundancyOptimizer(
        optimizer_parameters,
        optimizer_class=Adam8bit if train_config.use_8bit_adam else AdamW,
        process_group=None,
        parameters_as_bucket_view=True,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
    )

    steps_per_epoch = ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * train_config.epoch
    warmup_steps = int(train_config.warmup_step_ratio * total_steps)

    logger.info(f"[+] Total training steps: {total_steps}")
    logger.info(f"[+] Learning rate warmup steps: {warmup_steps}")
    log_mlflow_param("total_steps", total_steps)
    log_mlflow_param("warmup_steps", warmup_steps)

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # PyTorch AMP
    use_amp = train_config.use_amp
    scaler = GradScaler(enabled=use_amp)

    # 로깅에 필요한 변수 초기화
    logging_loss = torch.tensor(0.0, device=torch.device(rank))

    # ==================================================
    # > Go Training
    # ==================================================
    # Training Process
    logger.info("[+] Start Training Process")
    step, accumulated_steps = 0, 0

    for epoch in range(train_config.epoch):
        train_datasampler.set_epoch(epoch)
        model.train()

        for input_ids, label_ids in tqdm(
            train_dataloader,
            desc=f"[Epoch: {epoch}]",
            total=len(train_dataloader),
            unit_scale=1 / train_config.gradient_accumulation_steps,
            bar_format=TQDM_FORMAT,
            disable=rank != 0,
        ):
            accumulated_steps += 1

            input_ids = input_ids.to(rank)
            label_ids = label_ids.to(rank)

            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                output = model(input_ids=input_ids).logits
            loss = F.cross_entropy(output.transpose(1, 2), label_ids, ignore_index=LABEL_MASK_ID)
            loss /= train_config.gradient_accumulation_steps
            logging_loss += loss.detach()

            scaler.scale(loss).backward()

            # 아직 Gradient Accumulation 중인 경우
            # 단, 가장 마지막 epoch의 마지막 배치에 대한 업데이트는 이루어지지 않음
            if accumulated_steps < train_config.gradient_accumulation_steps:
                continue

            # Accumulation을 모두 시도하여 0으로 초기화 후 업데이트 진행
            accumulated_steps = 0
            step += 1

            # Unscale gradient first, then clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            # Update accumulated gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Clear gradient
            optimizer.zero_grad(set_to_none=True)

            # ==================================================
            # > Periodic Evaluation, Logging and CheckPointing
            # ==================================================
            # Logging the loss and Validate the model (for only rank 0)
            if step % experiment_config.steps_per_log == 0:
                dist.all_reduce(logging_loss, op=dist.ReduceOp.SUM)

                usage = torch.tensor(
                    [
                        torch.cuda.memory_allocated(rank),
                        torch.cuda.max_memory_allocated(rank),
                        torch.cuda.memory_reserved(rank),
                        torch.cuda.max_memory_reserved(rank),
                    ],
                    dtype=torch.float32,
                ).to(rank)
                dist.all_reduce(usage, op=dist.ReduceOp.SUM)
                usage = usage / (world_size * 1024 * 1024)

                mean_loss = logging_loss / (world_size * experiment_config.steps_per_log)
                mean_loss = mean_loss.item()

                ppl = exp(mean_loss)
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[+] Epoch: {epoch}, Step: {step}/{total_steps}, "
                    f"LR: {lr:.4e}, Loss: {mean_loss:2.4f}, PPL: {ppl:.4f}"
                )
                log_mlflow_metric("lr", lr, step)
                log_mlflow_metric("train/loss", mean_loss, step)
                log_mlflow_metric("train/ppl", ppl, step)

                # 평균 GPU 사용량
                usage = usage.cpu()
                log_mlflow_metric("gpu_allocated", usage[0], step)
                log_mlflow_metric("gpu_max_allocated", usage[1], step)
                log_mlflow_metric("gpu_reserved", usage[2], step)
                log_mlflow_metric("gpu_max_reserved", usage[3], step)

                logging_loss = torch.tensor(0.0, device=torch.device(rank))

            # Evaluation
            if step % experiment_config.steps_per_valid == 0:
                model.eval()
                dev_loss, dev_ppl = validation(model, dev_dataloader)

                logger.info("[+] Validation Result")
                logger.info(
                    f"\t[+] Epoch: {epoch}, Step: {step}/{total_steps}, Loss: {dev_loss:2.4f}, PPL: {dev_ppl:.4f}"
                )
                log_mlflow_metric("valid/loss", dev_loss, step)
                log_mlflow_metric("valid/ppl", dev_ppl, step)

                model.train()

            # Model checkpointing
            if step % experiment_config.steps_per_model_save == 0:
                if rank == 0:
                    model_weight_path = os.path.join(
                        experiment_config.weight_output_dir,
                        f"{experiment_config.run_name}-epoch-{epoch}-step-{step}",
                    )
                    logger.info(f"[+] Save the model weight (Epoch: {epoch}, Step: {step}/{total_steps})")
                    model.module.save_pretrained(model_weight_path)
                torch.cuda.empty_cache()
                dist.barrier()

    # ==================================================
    # > Final Evaluation
    # ==================================================
    # Validation for the final model
    model.eval()
    dev_loss, dev_ppl = validation(model, dev_dataloader)

    logger.info("[+] Final Validation Result")
    logger.info(f"\t[+] Loss: {dev_loss:2.4f}, PPL: {dev_ppl:.4f}")
    log_mlflow_metric("valid/loss", dev_loss, total_steps)
    log_mlflow_metric("valid/ppl", dev_ppl, total_steps)

    # ==================================================
    # > Save trained model
    # ==================================================
    if rank == 0:
        model_weight_path = os.path.join(experiment_config.weight_output_dir, f"{experiment_config.run_name}-final")

        # 파일이 중복될 경우, 다른 이름으로 저장
        if os.path.exists(model_weight_path):
            model_weight_path += f"-{datetime.now().strftime('%f')}"
            logger.warning(f"[-] Weight file already exists. Try to save with another name ({model_weight_path})")

        logger.info("[+] Saving the final model's parameters")
        model.module.save_pretrained(model_weight_path)
        logger.info("[+] Model is saved")

        logger.info("[+] Training Complete!")

    cleanup()


def main():
    # 실행에 필요한 설정값을 인자로부터 불러옴
    artifact_config = parse_args(parser, ArtifactConfig)
    train_config = parse_args(parser, TrainConfig)
    experiment_config = parse_args(parser, ExperimentConfig)
    execution_config = parse_args(parser, ExecutionConfig)

    # 모델 체크포인트는 서브 디렉토리에 저장
    experiment_config.weight_output_dir = os.path.join(experiment_config.weight_output_dir, experiment_config.run_name)

    # 필요한 폴더 생성
    os.makedirs(experiment_config.weight_output_dir, exist_ok=True)
    os.makedirs(experiment_config.log_output_dir, exist_ok=True)

    assert torch.cuda.is_available()

    # 로깅에 필요한 MLflow 설정이 되어 있는지 검증
    if any(
        [
            env not in os.environ
            for env in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
        ]
    ):
        raise ValueError("MLFlow tracking URI, username and password must be set.")

    # 학습 시작
    print("Spawning processes for training...")
    world_size = torch.cuda.device_count()
    mp.spawn(
        run,
        args=(world_size, artifact_config, train_config, experiment_config, execution_config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger:
            logger.error(e, exc_info=e)
        else:
            raise
        exit(-1)
