from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def sample_minute() -> int:
    """
    Sample a minute from 0 to 5 from the following distribution:
    0.5 for 0, 0.1 otherwise

    :return: random minute for immediate response
    """
    return np.random.choice([0, 1, 2, 3, 4, 5], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])


class TimelyChatDataset(Dataset):
    def __init__(
        self,
        raw_instances: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        turn_separator: str = " <turn> ",
        immediate_dropout: float = 0.0,
    ):
        """
        Dataset for supervised finetuning

        :param raw_instances: JSON format instances in the following format
        Example:
        [
            {
                "context": list of str,
                "time_elapsed": str,
                "delayed_response": str,
                "immediate_response": str,
            }
        ]
        :param tokenizer: Huggingface tokenizer
        :param turn_separator: Separator for each turn
        :param immediate_dropout: Probability of dropping time_elapsed sequence for immediate response
        """
        self.tokenizer = tokenizer
        self.supervised_finetuning_instances = raw_instances
        self.turn_separator = turn_separator
        self.immediate_dropout = immediate_dropout

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]

        # make dialogue history
        context = f"{self.turn_separator}{instance['context'][0]}"
        for turn in instance["context"][1:]:
            drop = np.random.rand() < self.immediate_dropout
            if not drop:
                minute = sample_minute()
                suffix = "minute" if minute in [0, 1] else "minutes"
                context += f" <sep> {minute} {suffix} later"
            context += f"{self.turn_separator}{turn}"
        # append time-conditional sequence to the end of encoder input
        context += f" <sep> {instance['time_elapsed']} later{self.turn_separator}"

        response = instance["delayed_response"]

        input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
        label_ids = self.tokenizer(response, padding="max_length", truncation=True)["input_ids"]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)
