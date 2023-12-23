import random
from dataclasses import fields
from logging import Logger
from typing import Any, Dict, List

import mlflow
import numpy as np
import torch
import torch.distributed as dist
import ujson as json
from transformers import PretrainedConfig


def set_seed(seed: int):
    """
    랜덤 시드를 고정합니다.

    :param seed: 랜덤 시드로 사용할 정수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_mlflow_metric(key: str, value: float, step: int, logger: Logger = Logger.root):
    """MLflow 서버에 메트릭을 로깅합니다."""
    rank = dist.get_rank()
    if rank == 0:
        try:
            mlflow.log_metric(key, value, step)
        except Exception:
            logger.error(f"[-] MLFlow server error when logging metric ({key})")


def log_mlflow_metrics(metrics: Dict[str, float], step: int, logger: Logger = Logger.root):
    """MLflow 서버에 메트릭을 로깅합니다."""
    rank = dist.get_rank()
    if rank == 0:
        try:
            mlflow.log_metrics(metrics, step)
        except Exception:
            logger.error("[-] MLFlow server error when logging metrics")


def log_mlflow_param(key: str, value: Any, logger: Logger = Logger.root):
    """MLflow 서버에 파라미터를 로깅합니다."""
    rank = dist.get_rank()
    if rank == 0:
        try:
            mlflow.log_param(key, value)
        except Exception:
            logger.error(f"[-] MLFlow server error when logging param ({key})")


def print_config(*configurations, logger: Logger = Logger.root):
    """config를 출력합니다."""
    for config in configurations:
        if isinstance(config, PretrainedConfig):
            logger.info("[+] ModelConfig")
            logger.info(f"[+] \tvocab_size: {config.vocab_size}")
            logger.info(f"[+] \tmax_position_embeddings: {config.max_position_embeddings}")
            logger.info(f"[+] \thidden_size: {config.hidden_size}")
            logger.info(f"[+] \tnum_hidden_layers: {config.num_hidden_layers}")
            logger.info(f"[+] \tnum_attention_heads: {config.num_attention_heads}")
            logger.info(f"[+] \tbos_token_id: {config.bos_token_id}")
            logger.info(f"[+] \teos_token_id: {config.eos_token_id}")
            logger.info(f"[+] \thidden_dropout: {config.hidden_dropout}")
            logger.info(f"[+] \tattention_dropout: {config.attention_dropout}")
        else:
            logger.info("[+] " + config.__class__.__name__)
            for param in fields(config):
                value = getattr(config, param.name)
                # 파라미터의 길이가 너무 긴 경우 mlflow에서 받는 파라미터의 최고 길이로 잘라줍니다.
                if isinstance(value, str) and len(value) > 250:
                    clipped_value = value[:250]
                else:
                    clipped_value = value
                log_mlflow_param(param.name, clipped_value)
                logger.info(f"[+] \t{param.name}: {value}")


def load_dataset(data_file_path: str) -> List[Dict[str, Any]]:
    """
    jsonl 형식의 데이터 파일을 읽어서 dict 아이템 리스트 데이터셋으로 반환합니다.

    :param data_file_path: 데이터 파일 경로
    """
    with open(data_file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data
