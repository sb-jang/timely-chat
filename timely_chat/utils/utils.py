import random
from dataclasses import fields
from logging import Logger
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import wandb
from transformers import PretrainedConfig, T5Config


def set_seed(seed: int):
    """
    랜덤 시드를 고정합니다.

    :param seed: 랜덤 시드로 사용할 정수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_wandb_metric(key: str, value: float, step: int, logger: Logger = Logger.root):
    """wandb 서버에 메트릭을 로깅합니다."""
    rank = dist.get_rank()
    if rank == 0:
        try:
            wandb.log({key: value}, step)
        except Exception:
            logger.error(f"[-] wandb server error when logging metric ({key})")


def log_wandb_metrics(metrics: Dict[str, float], step: int, logger: Logger = Logger.root):
    """wandb 서버에 메트릭을 로깅합니다."""
    rank = dist.get_rank()
    if rank == 0:
        try:
            wandb.log(metrics, step)
        except Exception:
            logger.error("[-] wandb server error when logging metrics")


def log_wandb_param(key: str, value: Any, logger: Logger = Logger.root):
    """wandb 서버에 파라미터를 로깅합니다."""
    rank = dist.get_rank()
    if rank == 0:
        try:
            wandb.config.update({key: value})
        except Exception:
            logger.error(f"[-] wandb server error when logging param ({key})")


def print_config(*configurations, logger: Logger = Logger.root):
    """config를 출력합니다."""
    for config in configurations:
        if isinstance(config, T5Config):
            logger.info("[+] Model Config")
            logger.info(f"[+] \tvocab_size: {config.vocab_size}")
            logger.info(f"[+] \thidden_size: {config.d_model}")
            logger.info(f"[+] \tnum_hidden_layers: {config.num_layers}")
            logger.info(f"[+] \tnum_attention_heads: {config.num_heads}")
            logger.info(f"[+] \tbos_token_id: {config.decoder_start_token_id}")
            logger.info(f"[+] \teos_token_id: {config.eos_token_id}")
            logger.info(f"[+] \thidden_dropout: {config.dropout_rate}")
            logger.info(f"[+] \tclassifier_dropout: {config.classifier_dropout}")
        elif isinstance(config, PretrainedConfig):
            logger.info("[+] ModelConfig")
            logger.info(f"[+] \tvocab_size: {config.vocab_size}")
            logger.info(f"[+] \tmax_position_embeddings: {config.max_position_embeddings}")
            logger.info(f"[+] \thidden_size: {config.hidden_size}")
            logger.info(f"[+] \tnum_hidden_layers: {config.num_hidden_layers}")
            logger.info(f"[+] \tnum_attention_heads: {config.num_attention_heads}")
            logger.info(f"[+] \tbos_token_id: {config.bos_token_id}")
            logger.info(f"[+] \teos_token_id: {config.eos_token_id}")
            logger.info(f"[+] \tattention_dropout: {config.attention_dropout}")
        else:
            logger.info("[+] " + config.__class__.__name__)
            for param in fields(config):
                value = getattr(config, param.name)
                # 파라미터의 길이가 너무 긴 경우 wandb에서 받는 파라미터의 최고 길이로 잘라줍니다.
                if isinstance(value, str) and len(value) > 250:
                    clipped_value = value[:250]
                else:
                    clipped_value = value
                log_wandb_param(param.name, clipped_value)
                logger.info(f"[+] \t{param.name}: {value}")
