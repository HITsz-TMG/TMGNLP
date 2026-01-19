from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from utils.data_utils import save_jsonl
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field

import sys
sys.path.append("..")
from metrics import grade_answer_math
from trainer_utils import LlamaForMCS, QwenForMCS

@dataclass
class ScriptArguments:
    verifier_path: str = field(default='')
    log_path: str = field(default="")


def preprocess_function(examples):
    new_examples = {
        "problem": [],
        "gt_answer": []
    }
    for problem, answer in zip(examples['problem'], examples['answer']):
        content = problem
        message = [
            {"role": "user", "content": content}
        ]
        new_examples["problem"].append(message)
        new_examples['gt_answer'].append(answer)
    return new_examples

def get_datasets(data_path, split, sanity_check: bool = False):
    print("load dataset {}".format(data_path))
    dataset = load_dataset(data_path, split=split)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 2)))
    dataset = dataset.map(
        preprocess_function,
        batched=True, 
        num_proc=4,
    )
    return dataset

@torch.no_grad()
def get_score(model, tokenizer, question, candidates, mode):
    question = tokenizer.apply_chat_template(
        question,
        tokenize=False,
        add_generation_prompt=True
    )
    answers = [i.replace("\n", "<request>") + "<request>" for i in candidates]
    ph_token_ids = tokenizer.encode("<request>", return_tensors="pt")[0, -1].to(model.device)
    outcome_scores = []
    for answer in answers:
        inputs = tokenizer(question+answer, return_tensors='pt')
        for key, _ in inputs.items():
            inputs[key] = inputs[key].to(model.device)
        candidate_positions = torch.eq(inputs["input_ids"], ph_token_ids)
        logits = model(**inputs).logits
        if mode == "regression":
            # candidate_tokens = tokenizer.encode(f"<score>")[1:]
            # logits = logits[:, :, candidate_tokens]
            scores = torch.sigmoid(logits)[:, :, 0]
            step_score = scores[0][candidate_positions[0]].tolist()
            outcome_scores.append(step_score[-1])
        elif mode == "categorical":
            scores = torch.nn.functional.softmax(logits, dim=-1)
            step_score = scores[0][candidate_positions[0]].tolist()
            outcome_scores.append(step_score[-1])
        else:
            raise NotImplementedError
    if mode == "regression":
        scores = torch.tensor(outcome_scores)
    elif mode == "categorical":
        scores = torch.tensor(outcome_scores)@torch.arange(0, 9, dtype=torch.float)
    else:
        raise NotImplementedError
    return scores

def get_rank(scores, num_beams):
    sorted_indices = torch.argsort(scores, descending=True)
    # sorted_answers = [answers[i] for i in sorted_indices]
    return sorted_indices[:num_beams]

def get_answers(model, tokenizer, question, prefixes, beam_size):
    sampling_params = SamplingParams(
        n=beam_size,
        max_tokens=1024,
        stop_token_ids=[tokenizer.encode('\n')[-1], tokenizer.eos_token_id],
        include_stop_str_in_output=True,
        detokenize=False
    )
    question = tokenizer.apply_chat_template(
        question, 
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = [question+prefix for prefix in prefixes]
    generation_outputs = model.generate(inputs, sampling_params, use_tqdm=False)
    outputs = []
    for prefix, beam_outputs in zip(prefixes, generation_outputs):
        for output in beam_outputs.outputs:
            output_text = tokenizer.decode(output.token_ids, skip_special_tokens=False)
            outputs.append(prefix + output_text)
    return outputs

def post_process(outputs, tokenizer):
    eos_token = tokenizer.eos_token
    answers = []
    for answer in outputs:
        if eos_token in answer or len(answer) > 2048:
            answers.append({
                "candidate": answer.replace(eos_token, ""),
                "is_finished": True
            })
        else:
            answers.append({
                "candidate": answer.rstrip('\n'),
                "is_finished": False
            })
    return answers

def search_loop(generator, verifier, generator_tokenizer, verifier_tokenizer, question, beam_size, num_beams, mode):
    prefixes = [""]
    finished_answers = []
    finished = False
    steps = 0
    while not finished and steps < 50:
        outputs = get_answers(generator, generator_tokenizer, question, prefixes, beam_size)
        answers = post_process(outputs, generator_tokenizer)
        candidates = [i['candidate'] for i in answers]
        scores = get_score(verifier, verifier_tokenizer, question, candidates, mode)
        ranks = get_rank(scores, num_beams)
        prefixes = []
        for idx in ranks:
            if answers[idx]['is_finished']:
                finished_answers.append({
                    "candidate": answers[idx]['candidate'] + "<|im_end|>",
                    "score": scores[idx]
                })
            else:
                prefixes.append(answers[idx]['candidate'] + '\n')
        if len(prefixes) == 0:
            finished = True
        steps += 1
    scores = [i['score'] for i in finished_answers]
    ranks = get_rank(torch.tensor(scores), 1)
    return finished_answers[ranks[0]]['candidate'] if len(finished_answers) > 0 else prefixes[0] + "<|im_end|>"

if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments,))
    (custom_args,) = parser.parse_args_into_dataclasses()

    generator_path = "/home/szt/workspace/reasoning/outputs/generator/llemma-7b"
    verifier_path = custom_args.verifier_path
    data_path = "/home/szt/workspace/reasoning/datasets/math500"
    split = "test"
    if "classifier" in verifier_path:
        if "qwen" in verifier_path:
            base_model = QwenForMCS
        else:
            base_model = LlamaForMCS
        verifier = base_model.from_pretrained(
            verifier_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0"
        )
        mode = "categorical"
    else:
        if "qwen" in verifier_path:
            base_model = QwenForMCS
        else:
            base_model = LlamaForMCS
        verifier = base_model.from_pretrained(
            verifier_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0"
        )
        mode = "regression"
    
    geneartor = LLM(model=generator_path, tokenizer=generator_path, max_model_len=2048)
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)
    verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_path)
    data = get_datasets(data_path, split=split, sanity_check=False)

    for i in [2,4,8]:
        beam_size = num_beams = i
        results = []
        for item in tqdm(data):
            answer = search_loop(
                generator=geneartor,
                verifier=verifier,
                generator_tokenizer=generator_tokenizer,
                verifier_tokenizer=verifier_tokenizer,
                question=item['problem'],
                mode=mode,
                beam_size=beam_size,
                num_beams=num_beams
            )
            answer = answer.split('\n')[-1]
            result = grade_answer_math(answer, item['gt_answer'])
            print(result)
            results.append(result)

        print("verifier_path: {}".format(verifier_path))
        print("mode: {}".format(mode))
        print("beam_size: {}\tnum_beams".format(beam_size, num_beams))
        print("acc: {}".format(sum(results)/len(results)))

        log_item = {
            "verifier": verifier_path,
            "mode": mode,
            "beam_size": beam_size,
            "acc": sum(results)/len(results)
        }
        save_jsonl([log_item], path=custom_args.log_path)
