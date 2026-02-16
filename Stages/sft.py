from typing import Optional, Dict
from dataclasses import dataclass, field
import json
import os

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from transformers.training_args import TrainingArguments

from copy import deepcopy

from trl import setup_chat_format

def load_jsonl(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return [json.loads(l) for l in f]
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    torch_dtype: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    dataset_name: str = field(default=None)
    dataset_train_split: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(default="wandb")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        data_split,
        tokenizer,
        model_max_length
    ):
        super(SupervisedDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        self.data = load_dataset(data_path, split=data_split)
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        content = example['messages']
        question_template = self.tokenizer.apply_chat_template(
            [content[0]], tokenize=True, add_generation_prompt=True, use_system_prompt=False
        )
        qa_template = self.tokenizer.apply_chat_template(
            [content[0], content[1]],
            tokenize=True
        )
        stop_token_id = self.tokenizer.encode("<|im_end|>")[-1]
        input_ids = qa_template + [stop_token_id]
        labels = [self.ignore_index] * len(question_template) + qa_template[len(question_template):] + [stop_token_id]
        assert len(input_ids) == len(labels)
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        labels = torch.tensor(labels, dtype=torch.int)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
    )
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = SupervisedDataset(
        data_args.dataset_name, data_args.dataset_train_split, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)
if __name__ == "__main__":
    train()