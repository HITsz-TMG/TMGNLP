# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import (
    ModelConfig,
    SFTConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from trl.commands.cli_utils import SFTScriptArguments

from tqdm import tqdm
import os
from typing import Dict

from trainer_utils import MCSoftTrainer, LlamaForMCS, QwenForMCS

import random
from scipy.stats import norm
import numpy as np

import json
def load_jsonl(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return [json.loads(l) for l in f]

def calculate_gaussian(mu, num_labels):
    probs = []
    num_labels -= 1
    sigma = np.sqrt(mu*(1-mu)/num_labels)
    if sigma == 0:
        for i in range(num_labels+1):
            if i != int(mu*num_labels):
                probs.append(0.0)
            else:
                probs.append(1.0)
        return probs
    if mu == 0 or mu == 1:
        probs = [0] * (num_labels+1)
        probs[int(mu*num_labels)] = 1
    else:
        for i in range(num_labels+1):
            lower_bound = (2 * i - 1) / (2 * num_labels)
            upper_bound = (2 * i + 1) / (2 * num_labels)
            prob = norm.cdf(upper_bound, loc=mu, scale=sigma) - norm.cdf(lower_bound, loc=mu, scale=sigma)
            probs.append(prob)
    total_prob = sum(probs)
    probs = [prob / total_prob for prob in probs]
    return probs

def calculate_td(vs, vsi, scale, num_labels):
    probs = []
    num_labels -= 1
    sigma = abs(vs-vsi)/scale
    if sigma == 0:
        for i in range(num_labels + 1):
            if i != int(vs*num_labels):
                probs.append(0.0)
            else:
                probs.append(1.0)
        return probs
    if vs == 0 or vs == 1:
        probs = [0] * (num_labels+1)
        probs[int(vs*num_labels)] = 1
    else:
        for i in range(num_labels+1):
            lower_bound = (2 * i - 1) / (2 * num_labels)
            upper_bound = (2 * i + 1) / (2 * num_labels)
            prob = norm.cdf(upper_bound, loc=vs, scale=sigma) - norm.cdf(lower_bound, loc=vs, scale=sigma)
            probs.append(prob)
    total_prob = sum(probs)
    probs = [prob / total_prob for prob in probs]
    return probs

def calculate_binary(mu, num_labels):
    return [mu, 1-mu]

import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


DIST_MAP = {
    "gaussian": calculate_gaussian,
    "td": calculate_td,
    "binary": calculate_binary
}

@dataclass
class ScriptArguments:
    loss_type: str = field(default='bce')
    distribution_type: str = field(default="gaussian")
    num_labels: int = field(default=9)
    scale: int = field(default=1)
    coefficient: float = field(default=1.0)

class MCSoftDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        num_labels=8,
        sample_size=-1,
        custom_args = None
    ):
        super(MCSoftDataset, self).__init__()
        
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        self.num_labels = num_labels
        self.distribution_func = DIST_MAP[custom_args.distribution_type]
        self.scale = custom_args.scale
        self.coefficient = custom_args.coefficient
        
        self.raw_data = self.preprocess_function(load_jsonl(data_path))
        self.data = []
        for item in self.raw_data:
            data_item = self.preprocessing(item)
            if data_item is not None:
                self.data.append(data_item)
        if sample_size != -1:
            self.data = random.sample(self.data, sample_size)
        item = random.sample(self.data, 1)[0]
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue
            labels.append(id_)
        print("label:", labels)

    def __len__(self):
        return len(self.data)
    
    def preprocess_function(self, raw_data):
        data = []
        for example in tqdm(raw_data):
            steps = example['answer'].rstrip(self.tokenizer.eos_token).split('\n')
            message = [
                {"role": "user", "content": example['problem']},
            ]
            content = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            data.append({
                "content": content,
                "steps": steps,
                "score": example['score'],
                "splitter": "<request>"
            })
        return data

    def preprocessing(self, example):
        if len(example['score']) == 0:
            return None
        input_ids = self.tokenizer.encode(example['content'])[1:]
        labels = [self.ignore_index] * (len(input_ids))
        distributions = [[0.1]*self.num_labels]*(len(input_ids))
        assert len(example['steps']) == len(example['score'])
        vs = [0] * len(input_ids)
        vsi = [0] * len(input_ids)
        for idx in range(len(example['steps'])):
            step = example['steps'][idx]
            new_input_tensor = self.tokenizer.encode(step + example['splitter'])[1:]
            distributions += [[0.1]*self.num_labels]*(len(new_input_tensor)-1)
            vs += [0] * (len(new_input_tensor)-1)
            vs.append(example['score'][idx])
            vsi += [0] * (len(new_input_tensor)-1)
            if idx < len(example['steps']) -1:
                vsi.append(example['score'][idx+1])
            else:
                vsi.append(example['score'][idx])
            if self.num_labels > 1:
                mu = example['score'][idx]
                if self.distribution_func == calculate_td:
                    distributions.append(
                        self.distribution_func(
                            vs=example['score'][idx],
                            vsi=example['score'][idx+1] if idx < len(example['steps'])-1 else example['score'][idx],
                            scale=self.scale,
                            num_labels=self.num_labels, 
                        )
                    )
                elif self.distribution_func == calculate_gaussian and idx < len(example['steps'])-1:
                    distributions.append(
                        self.distribution_func(
                            mu= self.coefficient * example['score'][idx] + (1-self.coefficient) * example['score'][idx+1], 
                            num_labels=self.num_labels, 
                        )
                    )
                else:
                    distributions.append(
                        self.distribution_func(
                            mu=example['score'][idx], 
                            num_labels=self.num_labels, 
                        )
                    )
            else:
                mu = example['score'][idx]
                distributions.append([0.1]*self.num_labels)
            new_labels = [self.ignore_index] * (len(new_input_tensor)-1) + [mu]
            input_ids += new_input_tensor
            labels += new_labels

        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        labels = torch.tensor(labels, dtype=torch.float32)

        distributions = distributions[:self.model_max_length]
        distributions += [[0.1]*self.num_labels]*(self.model_max_length-len(distributions))
        distributions = torch.tensor(distributions, dtype=torch.float32)

        vs = vs[:self.model_max_length]
        vs += [0] * (self.model_max_length-len(vs))
        vs = torch.tensor(vs, dtype=torch.float32)

        vsi = vsi[:self.model_max_length]
        vsi += [0] * (self.model_max_length-len(vsi))
        vsi = torch.tensor(vsi, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "distributions": distributions,
            "vs": vs,
            "vsi": vsi
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.data[idx]

if __name__ == "__main__":
    setup_seed(42)
    parser = HfArgumentParser((ScriptArguments, SFTScriptArguments, SFTConfig, ModelConfig))
    custom_args, script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        attn_implementation=model_config.attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if "qwen" in model_config.model_name_or_path.lower():
        print("using QwenForMCS")
        model = QwenForMCS.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, num_labels=custom_args.num_labels, **model_kwargs
        )
    else:
        print("using LlamaForMCS")
        model = LlamaForMCS.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, num_labels=custom_args.num_labels, **model_kwargs
        )
    special_tokens_dict = {"additional_special_tokens": ["<request>", "<score>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Align padding tokens between tokenizer and model
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    ##############
    # Load dataset
    ##############
    train_path = os.path.join(script_args.dataset_name, script_args.dataset_train_split+'.jsonl')
    test_path = os.path.join(script_args.dataset_name, script_args.dataset_test_split+'.jsonl')
    train_dataset = MCSoftDataset(data_path=train_path, tokenizer=tokenizer, num_labels=model.num_labels, model_max_length=training_args.max_seq_length, custom_args=custom_args)
    eval_dataset = MCSoftDataset(data_path=test_path, tokenizer=tokenizer, num_labels=model.num_labels, model_max_length=training_args.max_seq_length, custom_args=custom_args)

    ##########
    # Training
    ##########
    trainer = MCSoftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        loss_type=custom_args.loss_type,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
