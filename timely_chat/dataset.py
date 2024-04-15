from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TimelyChatDataset(Dataset):
    def __init__(
        self,
        raw_instances: List[Dict[str, Union[str, List[str]]]],
        tokenizer: AutoTokenizer,
        speaker_token: str = "<spk>",
        time_token: str = "<time>",
        utterance_token: str = "<utt>",
        instantaneous_dropout: float = 0.0,
        model_type: str = "causal",
    ):
        """
        Dataset for supervised finetuning

        :param raw_instances: JSON format instances in the following format
        Example:
        [
            {
                "context": list of str,
                "speaker_list": list of str,
                "time_elapsed": str,
                "target_speaker": str,
                "timely_response": str,
                "untimely_response": str,
                "narrative": str,
            }
        ]
        :param tokenizer: huggingface tokenizer
        :param speaker_token: token to denote start of speaker of current turn
        :param time_token: token to denote start of time difference between consecutive messages
        :param utterance_token: token to denote start of utterance
        :param instantaneous_dropout: dropout rate of time_elapsed sequence for immediate response
        :param model_type: causal (decoder) or seq2seq (encoder-decoder)
        """
        self.tokenizer = tokenizer
        self.supervised_finetuning_instances = raw_instances
        self.speaker_token = speaker_token
        self.time_token = time_token
        self.utterance_token = utterance_token
        self.instantaneous_dropout = instantaneous_dropout
        self.model_type = model_type

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]

        # make dialogue history
        context = f"{self.speaker_token} {instance['speaker_list'][0]}: {self.utterance_token}{instance['context'][0]}"
        for speaker, utterance in zip(instance["speaker_list"][1:], instance["context"][1:]):
            context += f" {self.speaker_token} {speaker}:"
            drop = np.random.rand() < self.instantaneous_dropout
            if not drop:
                context += f" {self.time_token} 0 minutes later"
            context += f" {self.utterance_token} {utterance}"
        # append time-conditional sequence to the end of encoder input
        context += f" {self.speaker_token} {instance['target_speaker']}: {self.time_token} {instance['time_elapsed']} later {self.utterance_token} "

        response = instance["timely_response"]

        if self.model_type == "seq2seq":
            input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
            label_ids = self.tokenizer(response, padding="max_length", truncation=True)["input_ids"]
        else:
            context += response
            input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
            label_ids = input_ids

            input_ids = input_ids[:-1]
            label_ids = label_ids[1:]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)
