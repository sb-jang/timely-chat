from dataclasses import dataclass
from typing import Optional

from timely_chat.utils.argparser import field, group


# fmt: off
@dataclass
@group("Artifact Config", "모델, 데이터셋 등 아티팩트에 관한 설정")
class ArtifactConfig:
    pretrained_model: str = field("allenai/cosmo-xl", help="사용할 사전 학습된 모델 이름/버전")
    pretrained_model_weight_path: Optional[str] = field(help="모델 weight 경로, Optional - 지정하지 않으면 사전 학습 모델의 weight가 사용됩니다.")
    train_dataset_path: str = field("/home/namomo73/timely-chat/resources/data/train_augmented.json", help="학습에 사용할 데이터셋 경로")
    val_dataset_path: str = field("/home/namomo73/timely-chat/resources/data/valid_augmented.json", help="Validation에 사용할 데이터셋 경로")
    data_augment: bool = field(False, help="sliding window 방식 채택 여부")


@dataclass
@group("Train Experiment Setting", "학습 관련 설정")
class TrainConfig:
    epoch: int = field(3, help="학습 진행할 epoch 수")
    learning_rate: float = field(1e-4, help="Learning Rate")
    train_batch_size: int = field(32, help="학습에 사용할 배치수")
    val_batch_size: int = field(32, help="Validation에 사용할 배치수")
    warmup_step_ratio: float = field(0.1, help="Warmup Ratio")
    weight_decay_ratio: float = field(0.1, help="Optimizer 내 Weight Decay Ratio")
    gradient_accumulation_steps: int = field(1, help="Gradient Accumulation할 Step 수")
    max_grad_norm: float = field(1.0, help="Back Propagation에서 Max Gradient Norm")
    random_seed: int = field(0, help="Random Seed")
    use_amp: bool = field(help="Automatic mixed precision 사용 여부")
    use_8bit_adam: bool = field(help="8Bit Adam 사용 여부")
    max_sequence_length: int = field(512, help="Max Sequence Length")
    adam_beta1: float = field(0.9, help="Adam optimizer beta1")
    adam_beta2: float = field(0.999, help="Adam optimizer beta2")
    adam_eps: float = field(1e-8, help="Adam optimizer epsilon")
    hidden_dropout: float = field(0.2, help="Residual dropout ratio")
    attention_dropout: float = field(0.2, help="Attention dropout ratio")
    instantaneous_dropout: float = field(0.0, help="Instantaneous response time dropout ratio")
    loss_response_only: bool=field(help="If True, only compute loss for delayed response")


@dataclass
@group("Experiment Config", "실험 관련 설정")
class ExperimentConfig:
    # 실험 중 로깅에 필요한 메타정보
    run_name: str = field(required=True, help="모델 weight과 학습 로그 명에 저장될 실험 이름")
    experiment_name: str = field("Timely-Chat", help="해당 실험에 대한 이름")

    # 실험 중 출력되는 파일들을 저장하는 위치
    log_output_dir: str = field("./logs/sft", help="학습 로그를 지정할 폴더")
    weight_output_dir: str = field("./checkpoints", help="모델 Weight를 저장할 폴더")

    # 실험 중 출력되는 파일들이 작성되는 주기 설정
    steps_per_log: int = field(10, help="학습 로그 출력 빈도")
    steps_per_valid: int = field(100, help="validation을 시행할 빈도")
    steps_per_model_save: int = field(500, help="모델 체크포인트를 저장할 빈도")


@dataclass
@group("Execution Config", "실행에 필요한 설정")
class ExecutionConfig:
    # 데이터 처리 관련 머신 설정
    num_dataloader_workers: int = field(4, help="Dataloader 멀티 프로세스 개수")
    dataloader_prefetch_factor: int = field(32, help="Dataloader에서 한번에 prefetch할 비율")

    # 분산 학습 관련 설정
    ddp_master_address: str = field("localhost", help="DDP에 사용될 Master Address")
    ddp_master_port: str = field("1357", help="DDP에 사용될 Master Port")