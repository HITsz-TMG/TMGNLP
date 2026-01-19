from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils.data_utils import load_jsonl
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from utils.data_utils import save_jsonl

import sys
sys.path.append("..")
from metrics import grade_answer_math
from trainer_utils import LlamaForMCS, QwenForMCS

@dataclass
class ScriptArguments:
    model_path: str = field(default='/XYFS01/hitsz_bthu_ldf_1/szt/reasoning/workspace/outputs/verifier/prm_qwen_classifier_td_value1.0')
    log_path: str = field(default="/XYFS01/hitsz_bthu_ldf_1/szt/reasoning/workspace/best_of_n/gaokao2023.jsonl")
    data_path: str = field(default="/XYFS01/hitsz_bthu_ldf_1/szt/reasoning/workspace/outputs/best_of_n/gaokao.jsonl")

@torch.no_grad()
def get_score(model, tokenizer, best_of, item, mode):
    question = item['problem']
    question = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )
    ground_truth = item['gt_answer']
    answers = []
    for answer in item['answers'][:best_of]:
        answer = answer.replace("\n", "<request>") + "<request>"
        answers.append(answer)
    ph_token_ids = tokenizer.encode("<request>", return_tensors="pt")[0, -1].to(model.device)
    outcome_scores = []
    inputs = tokenizer(
        [question + answer for answer in answers],
        max_length=1536,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # for key, _ in inputs.items():
    #     inputs[key] = inputs[key].to(model.device)
    # candidate_positions = torch.eq(inputs["input_ids"], ph_token_ids)
    # logits = model(**inputs).logits
    # if mode == "regression":
    #     candidate_tokens = [tokenizer.encode(f"<score>")[-1]]
    #     logits = logits[:, :, candidate_tokens]
    #     scores = torch.sigmoid(logits)[..., 0]
    #     step_score = scores[0][candidate_positions[0]].tolist()
    #     outcome_scores.append(step_score[-1])
    # elif mode == "categorical":
    #     scores = torch.nn.functional.softmax(logits, dim=-1)
    #     outcome_positions = (candidate_positions.cumsum(dim=1) * candidate_positions).argmax(dim=-1)
    #     batch_indices = torch.arange(outcome_positions.shape[0], device=outcome_positions.device)
    #     outcome_scores = scores[batch_indices, outcome_positions].tolist()
    # else:
    #     raise NotImplementedError

    for answer in answers:
        inputs = tokenizer(question+answer, return_tensors='pt')
        for key, _ in inputs.items():
            inputs[key] = inputs[key].to(model.device)
        candidate_positions = torch.eq(inputs["input_ids"], ph_token_ids)
        logits = model(**inputs).logits
        if mode == "regression":
            candidate_tokens = [tokenizer.encode(f"<score>")[-1]]
            logits = logits[:, :, candidate_tokens]
            scores = torch.sigmoid(logits)[..., 0]
            step_score = scores[0][candidate_positions[0]].tolist()
            outcome_scores.append(step_score[-1])
        elif mode == "categorical":
            scores = torch.nn.functional.softmax(logits, dim=-1)
            step_score = scores[0][candidate_positions[0]].tolist()
            outcome_scores.append(step_score[-1])
        else:
            raise NotImplementedError
        
    if mode == "regression":
        _, index = torch.max(torch.tensor(outcome_scores), dim=-1)
        final_answer = item['answers'][index].split('\n')[-1] + "<|im_end|>"
    elif mode == "categorical":
        bins_vector = torch.linspace(0, 1, steps=len(outcome_scores[0]))
        scores = torch.tensor(outcome_scores)@bins_vector
        _, idx = scores.max(dim=-1)
        final_answer = item['answers'][idx].split('\n')[-1] + "<|im_end|>"
    else:
        raise NotImplementedError

    return grade_answer_math(final_answer, ground_truth)

def get_raw_score(best_of, item):
    raw_answers = item['answers'][:best_of]
    answers = []
    for answer in raw_answers:
        answers.append(answer.split('\n')[-1] + "<|im_end|>")
    ground_truth = item['gt_answer']
    for answer in answers:
        if grade_answer_math(answer, ground_truth) == True:
            return True
    return False


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments,))
    (custom_args,) = parser.parse_args_into_dataclasses()

    model_path = custom_args.model_path
    data_path = custom_args.data_path
    best_of = [2,4,8,16,32,64,128]
    if "classifier" in model_path:
        if "qwen" in model_path:
            model = QwenForMCS.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        else:
            model = LlamaForMCS.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        mode = "categorical"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        mode = "regression"
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    data = load_jsonl(data_path)
    for i in best_of:
        results = []
        for item in tqdm(data):
            ret = get_score(model=model,
                            tokenizer=tokenizer,
                            best_of=i,
                            item=item,
                            mode=mode)
            results.append(ret)
        print("mode_path: {}".format(model_path))
        print("mode: {}".format(mode))
        print("best_of: {}".format(i))
        print("acc: {}".format(sum(results)/len(results)))

        log_item = {
            "verifier": model_path,
            "mode": mode,
            "best_of": i,
            "acc": sum(results)/len(results)
        }
        save_jsonl([log_item], path=custom_args.log_path)

