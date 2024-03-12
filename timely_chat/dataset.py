from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TimelyChatDataset(Dataset):
    def __init__(self, raw_instances: List[Dict[str, Any]], tokenizer: AutoTokenizer, turn_separator: str = " <turn> "):
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
        """
        self.tokenizer = tokenizer
        self.supervised_finetuning_instances = raw_instances
        self.turn_separator = turn_separator

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]

        context = self.turn_separator.join(instance["context"])
        time_elapsed = instance["time_elapsed"]
        response = instance["delayed_response"]

        input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
        label_ids = self.tokenizer(response, padding="max_length", truncation=True)["input_ids"]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)
