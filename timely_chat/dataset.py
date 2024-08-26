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


class GapChatDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        speaker_token: str = "<spk>",
        time_token: str = "<time>",
        utterance_token: str = "<utt>",
        instantaneous_dropout: float = 0.0,
        model_type: str = "seq2seq",
    ):
        """
        Dataset for supervised finetuning
        
        :param tokenizer: huggingface tokenizer
        :param speaker_token: token to denote start of speaker of current turn
        :param time_token: token to denote start of time difference between consecutive messages
        :param utterance_token: token to denote start of utterance
        :param instantaneous_dropout: dropout rate of time_elapsed sequence for immediate response
        :param model_type: causal (decoder) or seq2seq (encoder-decoder)
        """
        self.tokenizer = tokenizer
        self.supervised_finetuning_instances = self._set_up(data_path)
        self.speaker_token = speaker_token
        self.time_token = time_token
        self.utterance_token = utterance_token
        self.instantaneous_dropout = instantaneous_dropout
        self.model_type = model_type
        self.data_path = data_path
        
    def _set_up(self, data_path:str)->list:
        data_lst = []
        text = ""
        label =""
        data_type = None
        if data_path.split('/')[-2] == 'time':
            data_type = 'time'
        elif data_path.split('/')[-2] == 'schedule':
            data_type = 'schedule'
        else:
            data_type = 'both'
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                text += label+'\n'
                text += line.split('\\n')[0].split("text:")[1]+'\n'
                
                if data_type!='schedule':
                    try:
                        progress = line.split('\\n')[1].split("Progress: ")[1].split('\t')[0]
                    except:
                        progress = ""
                
                if data_type!='time':
                    if data_type == 'schedule':
                        try:
                            schedule = line.split('\\n')[1].split("Schedule: ")[1].split('\t')[0]
                        except:
                            print(line.split('\\n'))
                    else:
                        try:
                            schedule = line.split('\\n')[2].split("Schedule: ")[1].split('\t')[0]
                        except:
                            print(line.split('\\n'))
                label = line.split('labels:')[1].split('\t')[0]
                
                if data_type == 'time':
                    data = {"text":text.strip(),
                        "Progress":progress.strip(),
                        "labels":label.strip(),
                        }
                elif data_type == 'schedule':
                    data = {"text":text.strip(),
                            "Schedule":schedule.strip(),
                            "labels":label.strip(),
                            }
                else:
                    data = {"text":text.strip(),
                            "Progress":progress.strip(),
                            "Schedule":schedule.strip(),
                            "labels":label.strip(),
                            }
                
                data_lst.append(data)
                if "episode_done" in line:
                    text = ""
                    label = ""
        return data_lst
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]
        if self.data_path.split('/')[-2] == 'time':
            context = f"Text:{instance['text']}\nProgess:{instance['Progress']}\nlabel:"
        elif self.data_path.split('/')[-2] == "schedule":
            context = f"Text:{instance['text']}\nSchedule:{instance['Schedule']}\nlabel:"
        else:
            context = f"Text:{instance['text']}\nProgess:{instance['Progress']}\nSchedule:{instance['Schedule']}\nlabel:"
        
        response = instance["labels"]

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
    
    

class GapChatDatasetSPK(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        speaker_token: str = "<spk>",
        time_token: str = "<time>",
        utterance_token: str = "<utt>",
        instantaneous_dropout: float = 0.0,
        model_type: str = "seq2seq",
    ):
        """
        Dataset for supervised finetuning
        
        :param tokenizer: huggingface tokenizer
        :param speaker_token: token to denote start of speaker of current turn
        :param time_token: token to denote start of time difference between consecutive messages
        :param utterance_token: token to denote start of utterance
        :param instantaneous_dropout: dropout rate of time_elapsed sequence for immediate response
        :param model_type: causal (decoder) or seq2seq (encoder-decoder)
        """
        self.speaker_token = speaker_token
        self.time_token = time_token
        self.utterance_token = utterance_token
        self.tokenizer = tokenizer
        self.supervised_finetuning_instances = []
        self.data_path = data_path
        self._set_up()
        self.instantaneous_dropout = instantaneous_dropout
        self.model_type = model_type
        self.data_type = None
        
    def _set_up(self):
        #data_lst = []
        text = ""
        label =""
        if self.data_path.split('/')[-1] == 'time.txt':
            self.data_type = 'time'
        elif self.data_path.split('/')[-1] == 'schedule.txt':
            self.data_type = 'schedule'
        else:
            self.data_type = 'both'
        with open(self.data_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                text += label
                if line.split('\\n')[0].split("text:")[1] == "":
                    text += ""
                else:
                    text += f"{self.speaker_token} speaker_1: {self.utterance_token} {line.split('\\n')[0].split("text:")[1]} "
                if self.data_type!='schedule':
                    try:
                        progress = line.split('\\n')[1].split("Progress: ")[1].split('\t')[0]
                    except:
                        progress = ""
                
                if self.data_type!='time':
                    if self.data_type == 'schedule':
                        try:
                            schedule = line.split('\\n')[1].split("Schedule: ")[1].split('\t')[0]
                        except:
                            print(line.split('\\n'))
                    else:
                        try:
                            schedule = line.split('\\n')[2].split("Schedule: ")[1].split('\t')[0]
                        except:
                            print(line.split('\\n'))
                label = f"{self.speaker_token} speaker_2: {self.utterance_token} {line.split('labels:')[1].split('\t')[0]} "
                
                
                if self.data_type == 'time':
                    data = {"text":text.strip(),
                        "Progress":f"{self.speaker_token} speaker_1: {progress.strip()}",
                        "labels":label.strip(),
                        }
                elif self.data_type == 'schedule':
                    data = {"text":text.strip(),
                            "Schedule":f"{self.speaker_token} speaker_1: {schedule.strip()}",
                            "labels":label.strip(),
                            }
                else:
                    data = {"text":text.strip(),
                            "Progress":f"{self.speaker_token} speaker_1: {progress.strip()}",
                            "Schedule":f"{self.speaker_token} speaker_1: {schedule.strip()}",
                            "labels":label.strip(),
                            }
                
                self.supervised_finetuning_instances.append(data)
                if "episode_done" in line:
                    text = ""
                    label = ""
        #return data_lst
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]
        if self.data_type == 'time':
            context = f"Text:{instance['text']}\nProgess:{instance['Progress']}\nlabel:{self.speaker_token} speaker_2: {self.utterance_token} "
        elif self.data_type == "schedule":
            context = f"Text:{instance['text']}\nSchedule:{instance['Schedule']}\nlabel:{self.speaker_token} speaker_2: {self.utterance_token} "
        else:
            context = f"Text:{instance['text']}\nProgess:{instance['Progress']}\nSchedule:{instance['Schedule']}\nlabel:{self.speaker_token} speaker_2: {self.utterance_token} "
        
        response = instance["labels"].split("<utt> ")[1]
        
        
        if self.model_type == "seq2seq":
            input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
            attention_mask = self.tokenizer(context, padding="max_length", truncation=True)["attention_mask"]
            label_ids = self.tokenizer(response, padding="max_length", truncation=True)["input_ids"]
        else:
            context += response
            input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
            label_ids = input_ids

            input_ids = input_ids[:-1]
            label_ids = label_ids[1:]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)
        #return (context, response)
    
    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)