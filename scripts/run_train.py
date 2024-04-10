import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from logging import Logger
from math import ceil, exp
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_parameter_names

from timely_chat.config import ArtifactConfig, ExecutionConfig, ExperimentConfig, TrainConfig
from timely_chat.dataset import TimelyChatDataset
from timely_chat.utils.argparser import build_parser, parse_args
from timely_chat.utils.logging import TQDM_FORMAT, create_logger
from timely_chat.utils.utils import log_wandb_metric, log_wandb_param, print_config, set_seed

# Global setup
load_dotenv()

parser = ArgumentParser()
build_parser(parser, [ArtifactConfig, TrainConfig, ExperimentConfig, ExecutionConfig])
logger: Logger = None

LABEL_MASK_ID = 0

SEQ2SEQ_MODELS = [
    "microsoft/GODEL-v1_1-base-seq2seq",
    "microsoft/GODEL-v1_1-large-seq2seq",
    "allenai/cosmo-xl",
    "ToddGoldfarb/Cadet-Tiny",
]


# ==================================================
# > Setup and Cleanup Functions
# ==================================================
def setup(rank: int, world_size: int, experiment_config: ExperimentConfig, execution_config: ExecutionConfig) -> None:
    """Initialize for distributed training."""
    global logger
    global log_wandb_metric, log_wandb_param, print_config

    # distributed setup
    os.environ["MASTER_ADDR"] = execution_config.ddp_master_address
    os.environ["MASTER_PORT"] = execution_config.ddp_master_port

    # intialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    # initlaize logger
    log_level = logging.INFO if rank == 0 else logging.ERROR
    logger = create_logger(__name__, experiment_config.log_output_dir, log_level, rank)

    # wandb setup
    if rank == 0:
        wandb.init(project=experiment_config.experiment_name)
        logger.info(f"[+] Start tracking to WandB with project: {experiment_config.experiment_name}")

    # partial patch for utility functions
    log_wandb_metric = partial(log_wandb_metric, logger=logger)
    log_wandb_param = partial(log_wandb_param, logger=logger)
    print_config = partial(print_config, logger=logger)


def cleanup() -> None:
    """Clean up the resources used during distributed training."""
    rank = dist.get_rank()

    if rank == 0:
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


# ==================================================
# > Validation Function
# ==================================================
def validation(
    model,
    dataloader: DataLoader,
    model_type: str = "causal",
) -> Tuple[float, float]:
    """
    Validate the model with val dataset.

    :param model: CausalLM or Seq2SeqLM model
    :param dataloader: dataLoader for val dataset
    :param model_type: model type (causal or seq2seq)
    :return: tuple of (val_loss, val_ppl)
    """
    rank = dist.get_rank()
    val_loss = torch.tensor([0.0], dtype=torch.float32).to(rank)
    num_non_mask = torch.tensor([0.0], dtype=torch.float32).to(rank)

    with torch.no_grad():
        for input_ids, label_ids in tqdm(dataloader, disable=rank != 0):
            input_ids = input_ids.to(rank)
            label_ids = label_ids.to(rank)

            if model_type == "causal":
                num_non_mask += (label_ids != LABEL_MASK_ID).sum()
                logits = model(input_ids=input_ids).logits
                loss = F.cross_entropy(logits.transpose(1, 2), label_ids, ignore_index=LABEL_MASK_ID, reduction="sum")
            else:
                loss = model(input_ids=input_ids, labels=label_ids).loss
            val_loss += loss

    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_non_mask, op=dist.ReduceOp.SUM)
    if model_type == "causal":
        val_loss /= num_non_mask
    val_ppl = torch.exp(val_loss)

    return val_loss.item(), val_ppl.item()


# ==================================================
# > Training Function
# ==================================================
def run(
    rank: int,
    world_size: int,
    artifact_config: ArtifactConfig,
    train_config: TrainConfig,
    experiment_config: ExperimentConfig,
    execution_config: ExecutionConfig,
):
    # setup and initialization
    setup(rank, world_size, experiment_config, execution_config)
    print_config(artifact_config, train_config, experiment_config, execution_config)
    set_seed(train_config.random_seed)

    # ==================================================
    # > Tokenizer Setup
    # ==================================================
    logger.info("[+] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(artifact_config.pretrained_model)
    tokenizer.model_max_length = train_config.max_sequence_length
    # set pad token to eos token if not exists
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    LABEL_MASK_ID = tokenizer.pad_token_id

    # ==================================================
    # > Determine model_type before dataset loading
    # ==================================================
    model_type = "seq2seq" if artifact_config.pretrained_model in SEQ2SEQ_MODELS else "causal"

    # ==================================================
    # > Dataset Loading
    # ==================================================
    # load train split
    logger.info("[+] Loading train/val datasets...")
    with open(artifact_config.train_dataset_path, "r") as f:
        train_dataset = json.load(f)
    train_dataset = TimelyChatDataset(
        train_dataset, tokenizer, instantaneous_dropout=train_config.instantaneous_dropout, model_type=model_type
    )
    train_datasampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.train_batch_size,
        num_workers=execution_config.num_dataloader_workers,
        prefetch_factor=execution_config.dataloader_prefetch_factor,
        sampler=train_datasampler,
        pin_memory=True,
    )
    logger.info(f"[+] training data size: {len(train_dataset)}")

    # load validation split
    with open(artifact_config.val_dataset_path, "r") as f:
        val_dataset = json.load(f)
    val_dataset = TimelyChatDataset(val_dataset, tokenizer, model_type=model_type)
    val_datasampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=train_config.val_batch_size,
        num_workers=execution_config.num_dataloader_workers,
        prefetch_factor=execution_config.dataloader_prefetch_factor,
        sampler=val_datasampler,
        pin_memory=True,
    )
    logger.info(f"[+] validation data size: {len(val_dataset)}")

    total_train_batch_size = train_config.train_batch_size * train_config.gradient_accumulation_steps * world_size
    logger.info(f"[+] Total train batch size: {total_train_batch_size}")
    log_wandb_param("total_train_batch_size", total_train_batch_size)

    # ==================================================
    # > Model Initialization
    # ==================================================
    logger.info("[+] Loading model...")
    model_config = AutoConfig.from_pretrained(artifact_config.pretrained_model)

    # override additional parameters
    # NOTE: use_cache unavailable if gradient checkpoint is enabled
    model_config.use_cache = False
    if model_type == "seq2seq":
        model_config.dropout_rate = train_config.hidden_dropout
    else:
        model_config.attention_dropout = train_config.attention_dropout

    print_config(model_config)

    if artifact_config.pretrained_model_weight_path is None:
        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(artifact_config.pretrained_model, config=model_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(artifact_config.pretrained_model, config=model_config)
    else:
        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                artifact_config.pretrained_model_weight_path, config=model_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                artifact_config.pretrained_model_weight_path, config=model_config
            )

    # patch embedding layers
    if train_config.use_8bit_adam:
        manager = GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})

    model.gradient_checkpointing_enable()
    model.to(rank)

    logger.info(f"[+] Uploading model parameters to rank {rank}...")
    model = DistributedDataParallel(model, device_ids=[rank], gradient_as_bucket_view=True)

    # ==================================================
    # > Trainable Module Initialization
    # ==================================================
    # optimizer setup
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
    logger.info(f"[+] LR warmup steps: {warmup_steps}")
    log_wandb_param("total_steps", total_steps)
    log_wandb_param("warmup_steps", warmup_steps)

    # scheduler setup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # PyTorch AMP
    use_amp = train_config.use_amp
    scaler = GradScaler(enabled=use_amp)

    logging_loss = torch.tensor(0.0, device=torch.device(rank))

    # ==================================================
    # > Train Loop
    # ==================================================
    logger.info("[+] START TRAINING!")
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
                if model_type == "causal":
                    output = model(input_ids=input_ids).logits
                    loss = F.cross_entropy(output.transpose(1, 2), label_ids, ignore_index=LABEL_MASK_ID)
                else:
                    loss = model(input_ids=input_ids, labels=label_ids).loss
            loss /= train_config.gradient_accumulation_steps
            logging_loss += loss.detach()

            scaler.scale(loss).backward()

            # accumulating gradients
            if accumulated_steps < train_config.gradient_accumulation_steps:
                continue

            # accumulation done
            accumulated_steps = 0
            step += 1

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            # ==================================================
            # > Validation & Logging & Checkpointing
            # ==================================================
            # logging
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
                log_wandb_metric("lr", lr, step)
                log_wandb_metric("train/loss", mean_loss, step)
                log_wandb_metric("train/ppl", ppl, step)

                # average GPU usage
                usage = usage.cpu()
                log_wandb_metric("gpu_allocated", usage[0], step)
                log_wandb_metric("gpu_max_allocated", usage[1], step)
                log_wandb_metric("gpu_reserved", usage[2], step)
                log_wandb_metric("gpu_max_reserved", usage[3], step)

                logging_loss = torch.tensor(0.0, device=torch.device(rank))

            # validation
            if step % experiment_config.steps_per_valid == 0:
                model.eval()
                val_loss, val_ppl = validation(model, val_dataloader, model_type)

                logger.info("[+] Validation Results")
                logger.info(
                    f"\t[+] Epoch: {epoch}, Step: {step}/{total_steps}, Loss: {val_loss:2.4f}, PPL: {val_ppl:.4f}"
                )
                log_wandb_metric("valid/loss", val_loss, step)
                log_wandb_metric("valid/ppl", val_ppl, step)

                model.train()

            # model checkpointing
            if step % experiment_config.steps_per_model_save == 0:
                if rank == 0:
                    model_weight_path = os.path.join(
                        experiment_config.weight_output_dir,
                        experiment_config.run_name,
                        f"epoch-{epoch}-step-{step}",
                    )
                    logger.info(f"[+] Saving the model weights (Epoch: {epoch}, Step: {step}/{total_steps})...")
                    model.module.save_pretrained(model_weight_path, safe_serialization=False)
                    tokenizer.save_pretrained(model_weight_path)
                torch.cuda.empty_cache()
                dist.barrier()

    # ==================================================
    # > Validation after Training
    # ==================================================
    model.eval()
    val_loss, val_ppl = validation(model, val_dataloader, model_type)

    logger.info("[+] Final Validation Result")
    logger.info(f"\t[+] Loss: {val_loss:2.4f}, PPL: {val_ppl:.4f}")
    log_wandb_metric("valid/loss", val_loss, total_steps)
    log_wandb_metric("valid/ppl", val_ppl, total_steps)

    # ==================================================
    # > Save the trained model
    # ==================================================
    if rank == 0:
        model_weight_path = os.path.join(
            experiment_config.weight_output_dir,
            experiment_config.run_name,
            "final",
        )

        # if already exists, save with temporary name
        if os.path.exists(model_weight_path):
            model_weight_path += f"-{datetime.now().strftime('%f')}"
            logger.warning(f"[-] Weight file already exists. Try to save with another name ({model_weight_path})")

        logger.info("[+] Saving the final model's parameters...")
        model.module.save_pretrained(model_weight_path, safe_serialization=False)
        tokenizer.save_pretrained(model_weight_path)
        logger.info("[+] Model saved!")

        logger.info("[+] FINISH TRAINING!")

    cleanup()


def main():
    # load configs
    artifact_config = parse_args(parser, ArtifactConfig)
    train_config = parse_args(parser, TrainConfig)
    experiment_config = parse_args(parser, ExperimentConfig)
    execution_config = parse_args(parser, ExecutionConfig)

    experiment_config.weight_output_dir = os.path.join(experiment_config.weight_output_dir, experiment_config.run_name)

    os.makedirs(experiment_config.weight_output_dir, exist_ok=True)
    os.makedirs(experiment_config.log_output_dir, exist_ok=True)

    assert torch.cuda.is_available()

    # start training
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
