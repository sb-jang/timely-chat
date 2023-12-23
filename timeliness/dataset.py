from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from timeliness.template.date import DateTemplate
from timeliness.template.timestamp import TimestampTemplate
from timeliness.template.utterance import UtteranceTemplate


def _make_session_with_template(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    (화자, 발화 메시지, 발화 시간)으로 구성된 세션 데이터셋을 Supervised Finetuning 학습용 데이터로 변환합니다.
    미리 정의된 템플릿을 사용하여, 세 가지 정보를 적절한 형식의 텍스트로 변환합니다.

    :param instances: raw한 형태의 인스턴스
    :return: Supervised Finetuning용 인스턴스
    """
    supervised_finetuning_instances = []
    for instance in instances:
        session = instance["session"]
        time_template = TimestampTemplate.get_template()
        utterance_template = UtteranceTemplate.get_template()
        date_template = DateTemplate.get_template()

        session_with_template = []
        prev_timestamp = None
        for turn in session:
            timestamp = datetime.strptime(turn["datetime"], "%Y-%m-%d %H:%M")
            if prev_timestamp is None or timestamp.date() != prev_timestamp.date():
                session_with_template.append(date_template(timestamp.date()))
            prev_timestamp = timestamp

            session_with_template.append(
                utterance_template(turn["utterance"], turn["speaker"], timestamp, time_template)
            )

        supervised_finetuning_instances.append(session_with_template)

    return supervised_finetuning_instances


class TimelinessDataset(Dataset):
    def __init__(self, raw_instances: List[Dict[str, Any]], tokenizer: AutoTokenizer):
        """
        LM을 파인튜닝하기 위한 데이터셋

        :param raw_instances: json 형식의 대화 리스트
        :param tokenizer: 토크나이저
        """
        self.tokenizer = tokenizer

        self.supervised_finetuning_instances = _make_session_with_template(raw_instances)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]
        concatenated_session = "\n".join(instance)

        input_ids = self.tokenizer(concatenated_session, padding="max_length", truncation=True)["input_ids"]
        label_ids = input_ids

        input_ids = input_ids[:-1]
        label_ids = label_ids[1:]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)
