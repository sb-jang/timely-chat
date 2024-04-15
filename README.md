# Timely Chat

## Installation

```
pip install -r requirements.txt -r requirements-dev.txt
wandb login
```

## Training

더 많은 옵션은 `timely-chat/config.py` 참고
```
python -m scripts.run_train \
    --pretrained-model allenai/cosmo-xl \
    --run-name cosmo-xl \
    --train-dataset-path ./resources/data/train_augmented.json \
    --val-dataset-path ./resources/data/valid_augmented.json \
    --epoch 1 \
    --learning-rate 2e-5 \
    --train-batch-size 32 \
    --val-batch-size 32 \
    --instantaneous-dropout 0.2
```
- `instantaneous_dropout`: context 내의 발화 사이에 ` <time> 0 minutes later`를 생략할 확률. 0.2이면 20%의 발화에 안 붙게 됨.

## Inference & Evaluation

- Inference: `notebooks/inference.ipynb` 참고
- Automatic evaluation: `notebooks/automatic_evaluation.ipynb` 참고

## Notebooks

- `augment_training_set.ipynb`: delayed response뿐만 아니라 instantaneous response도 0 minutes later로 학습하도록 데이터를 2배로 불리는 스크립트
- `automatic_evaluation.ipynb`: `scripts/inference.py`를 통해 추론이 완료된 파일로 Automatic evaluation metric들을 계산하는 스크립트
- `inference.ipynb`: 모델 체크포인트를 불러와서 test set에 대해 모델 추론을 진행하여 결과 파일을 내놓는 스크립트
- `intro_analysis.ipynb`: 한국어 코퍼스를 가지고 발화 간 시간 차이의 통계를 분석하는 스크립트
- `lexical_diversity.ipynb`: 여러 대화 데이터셋에 대해 총 세션 수, 평균 턴 수, 평균 발화 길이, MTLD를 측정하는 스크립트
- `make_chatgpt_requests_file.ipynb`: `scripts/chatgpt.py`의 input이 될 requests 파일을 만드는 스크립트
- `make_timelychat_dataset.ipynb`: `scripts/chatgpt.py`의 output 파일을 가지고 학습 데이터를 만드는 스크립트
- `postprocess_atomic_2020_duration_estimation.ipynb`: ATOMIC2020 데이터를 ChatGPT로 duration pseudo-labeling한 결과 중 유효한 것만 뽑아서 narrative 파일로 만드는 스크립트
- `process_raw_tcs_data.ipynb` (Deprecated): 여러 가지 Raw Temporal Common Sense 데이터를 정제하는 스크립트