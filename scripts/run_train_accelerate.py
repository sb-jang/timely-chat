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
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
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
from timely_chat.dataset import TimelyChatDataset, GapChatDataset, AugmentedDataset
from timely_chat.utils.argparser import build_parser, parse_args
from timely_chat.utils.logging import TQDM_FORMAT, create_logger
from timely_chat.utils.accelerate_utils import log_wandb_metric, log_wandb_param, print_config, set_seed

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, DummyOptim

#os.environ['WANDB_SILENT']="True"
os.environ["OMP_NUM_THREADS"] = '8'

# Global setup
parser = ArgumentParser()
build_parser(parser, [ArtifactConfig, TrainConfig, ExperimentConfig, ExecutionConfig])
logger: Logger = None

LABEL_MASK_ID = 0
SEQ2SEQ_MODELS = [
    "microsoft/GODEL-v1_1-base-seq2seq",
    "microsoft/GODEL-v1_1-large-seq2seq",
    "allenai/cosmo-xl",
    "ToddGoldfarb/Cadet-Tiny",
    "seongbo-research/msc-3B",
]

# ==================================================
# > Setup and Cleanup Functions
# ==================================================
def setup(experiment_config: ExperimentConfig, execution_config: ExecutionConfig, accelerator) -> None:
    """Initialize for distributed training."""
    global logger
    global log_wandb_metric, log_wandb_param, print_config

    # initlaize logger
    log_level = logging.INFO
    logger = create_logger(__name__, experiment_config.log_output_dir, log_level)

    # wandb setup
    logger.info(f"[+] Start tracking to WandB with project: {experiment_config.experiment_name}")

    # partial patch for utility functions
    log_wandb_metric = partial(log_wandb_metric, logger=logger, accelerator=accelerator)
    log_wandb_param = partial(log_wandb_param, logger=logger, accelerator=accelerator)
    print_config = partial(print_config, logger=logger, accelerator=accelerator)

# ==================================================
# > Validation Function
# ==================================================
def validation(
    accelerator,
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
    device = accelerator.device
    val_loss = torch.tensor([0.0], dtype=torch.float32).to(device)
    #num_non_mask = torch.tensor([0.0], dtype=torch.float32).to(device)

    with torch.no_grad():
        for input_ids, attention_mask, label_ids in tqdm(dataloader, disable= not accelerator.is_main_process):
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            attention_mask = attention_mask.to(device)            
            loss = model(input_ids=input_ids, attention_mask = attention_mask, labels=label_ids).loss
            val_loss += loss

    #val_loss = accelerator.reduce(val_loss,reduction="sum")
    #num_non_mask = accelerator.reduce(num_non_mask, reduction="sum")
    #denominator = num_non_mask if model_type == "causal" else len(dataloader)
    val_loss /= len(dataloader)
    val_ppl = torch.exp(val_loss)
    return val_loss.item(), val_ppl.item()


# ==================================================
# > Training Function
# ==================================================
def run(
    accelerator,
    artifact_config: ArtifactConfig,
    train_config: TrainConfig,
    experiment_config: ExperimentConfig,
    execution_config: ExecutionConfig,
):
    device = accelerator.device
    # setup and initialization
    setup(experiment_config, execution_config, accelerator)
    if accelerator.is_main_process:
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
    if "gapchat" in artifact_config.train_dataset_path:
        from_hf = False
        split = None
        if "seongbo-research" in artifact_config.train_dataset_path:
            from_hf = True
            split = "train"
        train_dataset = GapChatDataset(data_source=artifact_config.train_dataset_path, 
                                       tokenizer=tokenizer, 
                                       instantaneous_dropout=train_config.instantaneous_dropout, 
                                       model_type=model_type, 
                                       from_hf=from_hf, 
                                       split=split)
    else:
        if artifact_config.data_augment:
            train_dataset = AugmentedDataset(data_path=artifact_config.train_dataset_path, tokenizer=tokenizer, instantaneous_dropout=train_config.instantaneous_dropout)
        else:
            with open(artifact_config.train_dataset_path, "r") as f:
                train_dataset = json.load(f)
            train_dataset = TimelyChatDataset(train_dataset, tokenizer, instantaneous_dropout=train_config.instantaneous_dropout, model_type=model_type, loss_response_only=train_config.loss_response_only)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.train_batch_size,
        num_workers=execution_config.num_dataloader_workers,
        prefetch_factor=execution_config.dataloader_prefetch_factor,
        pin_memory=True,
    )
    logger.info(f"[+] training data size: {len(train_dataset)}")

    if "gapchat" in artifact_config.val_dataset_path:
        from_hf = False
        split = None
        if "seongbo-research" in artifact_config.val_dataset_path:
            from_hf = True
            split = "valid"
        val_dataset = GapChatDataset(
        data_source=artifact_config.val_dataset_path, tokenizer=tokenizer, model_type=model_type,from_hf=from_hf,split=split
    )
    else:
        with open(artifact_config.val_dataset_path, "r") as f:
           val_dataset = json.load(f)
        val_dataset = TimelyChatDataset(val_dataset, tokenizer, model_type=model_type, loss_response_only=train_config.loss_response_only)

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=train_config.val_batch_size,
        num_workers=execution_config.num_dataloader_workers,
        prefetch_factor=execution_config.dataloader_prefetch_factor,
        pin_memory=True,
    )
    logger.info(f"[+] validation data size: {len(val_dataset)}")
    
    total_train_batch_size = train_config.train_batch_size * train_config.gradient_accumulation_steps
    logger.info(f"[+] Total train batch size: {total_train_batch_size}")
    log_wandb_param("total_train_batch_size", total_train_batch_size)
    # ==================================================
    # > Model Initialization
    # ==================================================
    logger.info("[+] Loading model...")
    model_config = AutoConfig.from_pretrained(artifact_config.pretrained_model) ##

    # override additional parameters
    # NOTE: use_cache unavailable if gradient checkpoint is enabled
    model_config.use_cache = False
    if model_type == "seq2seq":
        model_config.dropout_rate = train_config.hidden_dropout
    else:
        model_config.attention_dropout = train_config.attention_dropout
    if accelerator.is_main_process:
        print_config(model_config)

    if artifact_config.pretrained_model_weight_path is None:
        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(artifact_config.pretrained_model, config=model_config)
        else:
            model =  AutoModelForCausalLM.from_pretrained(artifact_config.pretrained_model, config=model_config)
    else:
        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                artifact_config.pretrained_model_weight_path, config=model_config
            )
        else:
            model =  AutoModelForCausalLM.from_pretrained(
                artifact_config.pretrained_model_weight_path, config=model_config
            )

    
    logger.info(f"[+] Uploading model parameters to Device...")
    model.gradient_checkpointing_enable()
    model.to(device)

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
    #optimizer = ZeroRedundancyOptimizer(
    #    optimizer_parameters,
    #    optimizer_class=Adam8bit if train_config.use_8bit_adam else AdamW,
    #    process_group=None,
    #    parameters_as_bucket_view=True,
    #    lr=train_config.learning_rate,
    #    betas=(train_config.adam_beta1, train_config.adam_beta2),
    #    eps=train_config.adam_eps,
    #)
    #optimizer = AdamW(optimizer_parameters, lr=train_config.learning_rate)
    optimizer = DummyOptim(model.parameters())

    steps_per_epoch = ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * train_config.epoch
    warmup_steps = int(train_config.warmup_step_ratio * total_steps)

    logger.info(f"[+] Total training steps: {total_steps}")
    logger.info(f"[+] LR warmup steps: {warmup_steps}")
    
    log_wandb_param("total_steps", total_steps)
    log_wandb_param("warmup_steps", warmup_steps)

    # scheduler setup
    #scheduler = get_cosine_schedule_with_warmup(
    #    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    #)
    logging_loss = torch.tensor(0.0, device=device)
    
    # Accelerate
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    

    # ==================================================
    # > Train Loop
    # ==================================================
    logger.info("[+] START TRAINING!")
    step, accumulated_steps = 0, 0
    for epoch in range(train_config.epoch):
        model.train()

        for step, (input_ids, attention_mask, label_ids) in enumerate(tqdm(
            train_dataloader,
            desc=f"[Epoch: {epoch}]",
            total=len(train_dataloader),
            unit_scale=1 / train_config.gradient_accumulation_steps,
            bar_format=TQDM_FORMAT,
            disable= not accelerator.is_main_process,
        )):
            with accelerator.accumulate(model):
                with accelerator.autocast(model):
                    if model_type == "causal":
                        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids).loss
                accelerator.backward(loss)
                logging_loss += loss.detach()
                if accelerator.sync_gradients: ##확인 필요
                    accelerator.clip_grad_norm_(model.parameters(),train_config.max_grad_norm)
                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # ==================================================
            # > Validation & Logging & Checkpointing
            # ==================================================
            # logging
            if step % experiment_config.steps_per_log == 0:
                if accelerator.is_main_process:
                    usage = torch.tensor(
                        [
                            torch.cuda.memory_allocated(device),
                            torch.cuda.max_memory_allocated(device),
                            torch.cuda.memory_reserved(device),
                            torch.cuda.max_memory_reserved(device),
                        ],
                        dtype=torch.float32,
                    ).to(device)
                    usage = usage / (accelerator.num_processes * 1024 * 1024)
                    mean_loss = logging_loss / (experiment_config.steps_per_log)
                    mean_loss = mean_loss.item()

                    ppl = exp(mean_loss)
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[+] Epoch: {epoch}, Step: {step}/{total_steps}, "
                        f"LR: {lr:.4e}, Loss: {mean_loss:2.4f}, PPL: {ppl:.4f}"
                    )
                    wandb.log({"lr":lr})
                    wandb.log({"train/loss":mean_loss})
                    wandb.log({"train/ppl":ppl})


                    # average GPU usage
                    usage = usage.cpu()
                    wandb.log({"gpu_allocated":usage[0]})
                    wandb.log({"gpu_max_allocated":usage[1]})
                    wandb.log({"gpu_reserved":usage[2]})
                    wandb.log({"gpu_max_reserved":usage[3]})


                    logging_loss = torch.tensor(0.0, device=device)

            # validation
            if step % experiment_config.steps_per_valid == 0:
                model.eval()
                val_loss, val_ppl = validation(accelerator, model, val_dataloader, model_type)
                if accelerator.is_main_process:
                    logger.info("[+] Validation Results")
                    logger.info(
                        f"\t[+] Epoch: {epoch}, Step: {step}/{total_steps}, Loss: {val_loss:2.4f}, PPL: {val_ppl:.4f}"
                    )

                    wandb.log({"valid/loss":val_loss})
                    wandb.log({"valid/ppl":val_ppl})

                model.train()

            # model checkpointing
            if step % experiment_config.steps_per_model_save == 0:
                model_weight_path = os.path.join(
                    experiment_config.weight_output_dir,
                    experiment_config.run_name,
                    f"epoch-{epoch}-step-{step}",
                )
                if accelerator.is_main_process:
                    logger.info(f"[+] Saving the model weights (Epoch: {epoch}, Step: {step}/{total_steps})...")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model = unwrapped_model.bfloat16()
                unwrapped_model.save_pretrained(
                    model_weight_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )

                torch.cuda.empty_cache()

    # ==================================================
    # > Validation after Training
    # ==================================================
    model.eval()
    val_loss, val_ppl = validation(accelerator, model, val_dataloader, model_type)
    
    if accelerator.is_main_process:
        logger.info("[+] Final Validation Result")
        logger.info(f"\t[+] Loss: {val_loss:2.4f}, PPL: {val_ppl:.4f}")
        wandb.log({"valid/loss" : val_loss})
        wandb.log({"valid/ppl" : val_ppl})


    # ==================================================
    # > Save the trained model
    # ==================================================

    model_weight_path = os.path.join(
        experiment_config.weight_output_dir,
        experiment_config.run_name,
        "final",
    )

        # if already exists, save with temporary name
    if os.path.exists(model_weight_path):
        model_weight_path += f"-{datetime.now().strftime('%f')}"
        logger.warning(f"[-] Weight file already exists. Try to save with another name ({model_weight_path})")
    if accelerator.is_main_process:
        logger.info("[+] Saving the final model's parameters...")
    accelerator.wait_for_everyone()
    
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.bfloat16()
    unwrapped_model.save_pretrained(
        model_weight_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        logger.info("[+] Model saved!")
        logger.info("[+] FINISH TRAINING!")




def main():
    # load configs
    artifact_config = parse_args(parser, ArtifactConfig)
    train_config = parse_args(parser, TrainConfig)
    experiment_config = parse_args(parser, ExperimentConfig)
    execution_config = parse_args(parser, ExecutionConfig)

    os.makedirs(experiment_config.weight_output_dir, exist_ok=True)
    os.makedirs(experiment_config.log_output_dir, exist_ok=True)

    assert torch.cuda.is_available()

    # start training
    print("accelerator for training...")
    world_size = torch.cuda.device_count()
    accelerator = Accelerator(mixed_precision="bf16" if train_config.use_amp else None, gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    if accelerator.is_main_process:
        wandb.init(project=experiment_config.experiment_name,    
            config={"artifact_config": artifact_config, 
            "train_config" : train_config,
            "experiment_config" : experiment_config,
            "execution_config" : execution_config,
            })
    
    run(accelerator, artifact_config, train_config, experiment_config, execution_config)

    accelerator.end_training()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger:
            logger.error(e, exc_info=e)
        else:
            raise
        exit(-1)
