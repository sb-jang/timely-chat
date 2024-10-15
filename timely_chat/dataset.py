from typing import Dict, List, Tuple, Union

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple
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
        context = f"{self.speaker_token} {instance['speaker_list'][0]}: {self.utterance_token} {instance['context'][0]}"
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
            tokenized_inputs = self.tokenizer(context, padding="max_length", truncation=True)
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            label_ids = self.tokenizer(response, padding="max_length", truncation=True)["input_ids"]
        else:
            context += response
            tokenized_inputs = self.tokenizer(context, padding="max_length", truncation=True)
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            label_ids = input_ids[1:]
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)

class AugmentedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        speaker_token: str = "<spk>",
        time_token: str = "<time>",
        utterance_token: str = "<utt>",
        instantaneous_dropout: float = 0.0,
        ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.speaker_token = speaker_token
        self.time_token = time_token
        self.utterance_token = utterance_token
        self.instantaneous_dropout = instantaneous_dropout
        self._set_up()
    
    def _set_up(self):
        timely_dataset = load_dataset("json",data_files=self.data_path,split='train')
        timelychat_augmented_dataset = timely_dataset.map(self._augment_data, remove_columns=timely_dataset.column_names)
        all_augmented_data = [item for sublist in timelychat_augmented_dataset['augmented_data'] for item in sublist]
        self.inputs = [item["input"] for item in all_augmented_data]
        self.outputs = [item["output"] for item in all_augmented_data]

    def _augment_data(self,instance):
        outputs = []
        inputs = []
        # make dialogue history
        context = f"{self.speaker_token} {instance['speaker_list'][0]}: {self.utterance_token} {instance['context'][0]}"
        n_turn = len(instance['context'])
        n_response = 5 if n_turn > 6 else n_turn-1
        for idx,(speaker, utterance) in enumerate(zip(instance["speaker_list"][1:], instance["context"][1:])):
            context += f" {self.speaker_token} {speaker}:"
            drop = np.random.rand() < self.instantaneous_dropout
            if not drop:
                context += f" {self.time_token} 0 minutes later"
            if idx >= n_turn-n_response:
                suffix = f" {self.utterance_token} "
                response = utterance
                outputs.append({"input":context + suffix,"output":response})
            context += f" {self.utterance_token} {utterance}"
        # append time-conditional sequence to the end of encoder input
        context += f" {self.speaker_token} {instance['target_speaker']}: {self.time_token} {instance['time_elapsed']} later {self.utterance_token} "
        response = instance["timely_response"]
        outputs.append({"input":context,"output":response})
        return {"augmented_data" : outputs}
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        tokenized_inputs = self.tokenizer(input_text, padding="max_length", truncation=True)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        label_ids = self.tokenizer(output_text, padding="max_length", truncation=True)["input_ids"]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


class GapChatDataset(Dataset):
    def __init__(
        self,
        data_source: str,
        tokenizer: AutoTokenizer,
        split: str=None,
        file_type: str="txt",
        from_hf: bool=False,
        speaker_token: str = "<spk>",
        time_token: str = "<time>",
        utterance_token: str = "<utt>",
        instantaneous_dropout: float = 0.0,
        model_type: str = "seq2seq",
        time_unaware: bool = False,
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
        self.instantaneous_dropout = instantaneous_dropout
        self.model_type = model_type
        self.supervised_finetuning_instances = None
        self.time_unaware = time_unaware
        if from_hf:
            self.supervised_finetuning_instances = load_dataset(data_source,split=split)
            self.data_type = data_source[25:]
        else:
            if file_type == 'txt':
                self.data_path = data_source
                self.supervised_finetuning_instances = self._set_up()
            else:
                raise ValueError(f"Unsupported file type:{file_type}")

        
    def _set_up(self) -> List[Dict[str, str]]:
        data_lst = []
        text = ""
        label = ""
        previous_session = ""
        even_previous_session = ""
        odd_previous_session = ""
        prefix = ""
        first_turn = True
        episode_cnt = 0
        # set data type
        filename = os.path.basename(self.data_path)
        self.data_type = filename.split(".")[0]
        if self.time_unaware:
            self.data_type = "time_unaware"
        with open(self.data_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                text += label
                # extract text in the line
                gap = line.split('\\n')[-1].split(" Gap:")[1].split('\t')[0]
                content = line.split('\\n')[0].split("text:")[1]
                prefix = ""
                label_prefix = ""
                if self.data_type == "time_unaware":
                    label_prefix = f"{self.time_token} 0 minutes later "
                    prefix = f"{self.time_token} 0 minutes later "
                    if first_turn:
                        prefix = "" if gap == "No Gap" else f"{self.time_token} {gap} later "
                        if content == "":
                            label_prefix = "" if gap == "No Gap" else f"{self.time_token} {gap} later "
                        drop = np.random.rand() < self.instantaneous_dropout
                        if drop:
                            prefix = ""
                        first_turn = False
                if content != "":
                    text += f"{self.speaker_token} speaker_1: {prefix}{self.utterance_token} {content} "
                # extract the label in the line
                label = f"{self.speaker_token} speaker_2: {label_prefix}{self.utterance_token} {line.split('labels:')[1].split('\t')[0]} "
                            
                # extract progress in the line
                if self.data_type =='time' or self.data_type == 'both':
                    progress = line.split('\\n')[1].split(" Progress:")[1].split('\t')[0]
                # extract schedule in the line
                if self.data_type == "schedule":
                    schedule = line.split('\\n')[1].split(" Schedule:")[1].split('\t')[0]
                elif self.data_type == "both":
                    schedule = line.split('\\n')[2].split(" Schedule:")[1].split('\t')[0]

                # data formatting following data type
                if self.data_type == 'time':
                    data = {"text":text.strip(),
                        "Progress":f"{progress.strip()}",
                        "labels":label.strip(),
                        }
                elif self.data_type == 'schedule':
                    data = {"text":text.strip(),
                            "Schedule":f"{schedule.strip()}",
                            "labels":label.strip(),
                            }
                elif self.data_type == 'both':
                    data = {"text":text.strip(),
                            "Progress":f"{progress.strip()}",
                            "Schedule":f"{schedule.strip()}",
                            "labels":label.strip(),
                            }
                elif self.data_type == 'time_unaware':
                    if episode_cnt%2 == 0:
                        previous_session = even_previous_session
                    else:
                        previous_session = odd_previous_session
                    data = {"text": text.strip(),
                            "previous_session" : previous_session.strip(),
                            "gap" : gap,
                            "labels" : label.strip()
                            }
                data_lst.append(data)
                
                if "episode_done" in line:
                    if self.data_type == "time_unaware":
                        previous_session = ""
                        first_turn = True
                        if "final_session" not in line:
                            text += label
                            utterance_lst = text.strip().split(f"{self.speaker_token} ")[1:]
                            for utterance in utterance_lst[-5:]:
                                previous_session += f"{self.speaker_token} {utterance}"
                            previous_session += " "
                        if episode_cnt%2 == 0:
                            even_previous_session = previous_session
                        else:
                            odd_previous_session = previous_session
                    text = ""
                    label = ""
                    episode_cnt += 1
        return data_lst
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instance = self.supervised_finetuning_instances[index]
        if self.data_type == "time_unaware":
            suffix = instance["labels"].split("<utt> ")[0]
            context = f"Text:{instance['previous_session']}{instance['text']}\nLabel:{suffix}{self.utterance_token} "
        elif (self.data_type == 'time') or (self.data_type == 'progress'):
            context = f"Text:{instance['text']}\nProgress:{instance['Progress']}\nLabel:{self.speaker_token} speaker_2: {self.utterance_token} "
        elif self.data_type == "schedule":
            context = f"Text:{instance['text']}\nSchedule:{instance['Schedule']}\nLabel:{self.speaker_token} speaker_2: {self.utterance_token} "
        elif self.data_type == "both":
            context = f"Text:{instance['text']}\nProgress:{instance['Progress']}\nSchedule:{instance['Schedule']}\nLabel:{self.speaker_token} speaker_2: {self.utterance_token} "
        
        response = instance["labels"].split("<utt> ")[1]
        
        if self.model_type == "seq2seq":
            input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
            attention_mask = self.tokenizer(context, padding="max_length", truncation=True)["attention_mask"]
            label_ids = self.tokenizer(response, padding="max_length", truncation=True)["input_ids"]
        else:
            context += response
            input_ids = self.tokenizer(context, padding="max_length", truncation=True)["input_ids"]
            attention_mask = self.tokenizer(context, padding="max_length", truncation=True)["attention_mask"]
            label_ids = label_ids[1:]
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)
        #return (context, response)
    
    def __len__(self) -> int:
        return len(self.supervised_finetuning_instances)