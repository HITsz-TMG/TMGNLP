from dataclasses import dataclass
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    TrainerCallback,
)
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction

from transformers.models.llama import (
    LlamaPreTrainedModel, 
    LlamaModel
)
from transformers.models.qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache

from scipy.stats import norm


@dataclass
class ClassifierDataCollatorWithPadding:
    r"""
    PRM DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_processed = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids" not in feature
                or "attention_mask" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids`, `attention_mask`"
                )

            features_processed.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature['labels']
                }
            )
        batch = self.tokenizer.pad(
            features_processed,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch['labels'],
            "return_loss": True,
        }
        return batch

def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.array([i[0] for i in predictions])
    predictions = np.array(predictions > 0.5, dtype=float)
    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}


def calculate_td(vs, vsi, scale, num_labels):
    vs = vs.to(torch.float32)
    vsi = vsi.to(torch.float32)
    probs = []
    sigma = abs(vs-vsi)/scale
    if sigma == 0:
        for i in range(num_labels + 1):
            if i != int(vs*num_labels):
                probs.append(0.0)
            else:
                probs.append(1.0)
        return torch.tensor(probs)
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
    return torch.tensor(probs, dtype=torch.bfloat16)

def variance_cauculation(mu, k):
    return mu*(1-mu)/k

def variance_next(distribution, k):
    variance, expectation = 0, 0
    bins_vector = torch.linspace(0, 1, steps=len(distribution))
    vs = distribution.clone().detach().requires_grad_(False)@bins_vector
    for idx, value in enumerate(distribution):
        variance += (distribution[idx]**2) * ((value-vs)**2)
        if distribution[idx] > 0:
            expectation += 1/distribution[idx] * variance_cauculation(value, k)
    return 1/k * expectation + variance

def gaussian(vs, vsi, distribution, num_labels):
    for weight in [0.9, 0.99]:
        variance_current = variance_cauculation(vs, num_labels)
        variance_next_state = variance_next(distribution, num_labels)
        updated_variance = weight**2 * variance_current + (1-weight)**2 * variance_next_state
        if updated_variance < variance_current:
            return calculate_td(weight*vs+(1-weight)*vsi, vsi, 2, num_labels)
    return calculate_td(vs, vsi, 1, num_labels)

def distribution_mapping(vs, vsi, logits, distribution):
    num_labels = distribution.shape[-1] - 1
    for i in range(vs.shape[0]):
        if vs[i] == vsi[i]:
            distribution[i] = torch.zeros_like(distribution[0]).to(distribution.device)
            distribution[i, int(num_labels*vs[i])] = 1
        else:
            logit = logits[i].clone().detach().requires_grad_(False)
            distribution[i] = gaussian(
                vs[i].to('cpu'), 
                vsi[i].to('cpu'), 
                nn.functional.softmax(logit, dim=-1).to('cpu'), 
                num_labels
            ).to(distribution.device)
    return distribution


class DataCollatorForCompletionOnlyLMWoTemplate(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]] = None,
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            # Make pytorch loss function ignore all tokens up through the end of the response key
            batch["labels"][i, :-2] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch


class MCSoftTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        loss_type: str = "cross_entropy",
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.score_token_id = tokenizer.encode("<score>")[-1]
        self.loss_type = loss_type
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits.float()
        shift_labels = labels[..., 1:].contiguous()
        if self.loss_type == 'mse':
            if logits.shape[-1] != 1:
                shift_logits = logits[..., :-1, self.score_token_id].contiguous()
            else:
                shift_logits = logits[..., :-1, :].squeeze(-1).contiguous()
            shift_logits = torch.sigmoid(shift_logits) + 1e-16
            shift_logits = shift_logits[shift_labels!=-100]
            shift_labels = shift_labels[shift_labels!=-100]
            loss = (shift_logits-shift_labels)**2
            loss = loss.mean()
            if return_outputs:
                return (loss, {"logits": shift_logits})
            return loss
        elif self.loss_type == "weighted_cross_entropy":
            shift_labels = labels.contiguous()
            shift_logits = logits.squeeze(-1).contiguous()
            shift_logits = shift_logits[shift_labels!=-100]
            shift_distributions = inputs['distributions'].contiguous()
            shift_distributions = shift_distributions[shift_labels!=-100]
            loss_func = nn.KLDivLoss(reduction="batchmean")
            loss = loss_func(nn.LogSoftmax(dim=-1)(shift_logits), shift_distributions)
            if return_outputs:
                return (loss, {"logits": shift_logits})
            else:
                return loss
        elif self.loss_type == 'dv1':
            shift_labels = labels.contiguous()
            shift_logits = logits.squeeze(-1).contiguous()
            shift_logits = shift_logits[shift_labels!=-100]
            shift_distributions = inputs['distributions'].contiguous()
            shift_distributions = shift_distributions[shift_labels!=-100]
            shift_vs = inputs['vs'].contiguous()
            shift_vs = shift_vs[shift_labels!=-100]
            shift_vsi = inputs['vsi'].contiguous()
            shift_vsi = shift_vsi[shift_labels!=-100]

            shift_distributions = distribution_mapping(shift_vs, shift_vsi, shift_logits, shift_distributions)

            loss_func = nn.KLDivLoss(reduction="batchmean")
            loss = loss_func(nn.LogSoftmax(dim=-1)(shift_logits), shift_distributions)
            if return_outputs:
                return (loss, {"logits": shift_logits})
            else:
                return loss
        else:
            raise NotImplementedError

class LlamaForMCS(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class QwenForMCS(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    