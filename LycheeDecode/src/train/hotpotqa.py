import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM




# HotpotQA dataset processing
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
    return total_qas, total_docs


def generate_input_output(index, num_docs):
    global QAS, DOCS
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])

    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs,
                                                             max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS

    random.Random(4).shuffle(all_docs)
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)])

    formatted_output = {
        "data_source": "hotpotqa",
        "prompt": [{
            "role": "user",
            "content": curr_q,
        }],
        "context": context,
        "ability": "memory",
        "reward_model": {
            "style": "rule",
            "ground_truth": curr_a
        },
        "extra_info": {
            'index': index,
            "question": curr_q,
            "num_docs": num_docs,
        }
    }
    return formatted_output


class HotpotQADistractorDataset(Dataset):
    def __init__(self, model_name_or_path, context_length_min, context_length_max, num_samples=None, context_lengths_num_intervals=20, file_path='datasets/hotpotqa.json', filtered_ids_path: str = 'datasets/hotpotqa_filtered_ids.json',):
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")

        self.file_path = file_path
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.context_length_min = context_length_min
        self.context_length_max = context_length_max
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.filtered_ids_path = Path(filtered_ids_path)

        self.DOCUMENT_PROMPT = "Document {i}:\n{document}"
        self.PROMPT = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given document. Only give me the answer and do not output any other words.\nQuestion: {question}\nAnswer:"

        self.qas, self.docs, self.doc_dict, self.doc_token_counts = self._load_and_process_data()

        valid_indices = self._filter()
        self.qas = [self.qas[i] for i in valid_indices]
        print(f"Filtered down to {len(self.qas)} questions that likely require context.")

        if num_samples and num_samples < len(self.qas):
            random.shuffle(self.qas)
            self.qas = self.qas[:num_samples]

        self.context_length_intervals = torch.linspace(
            context_length_min,
            context_length_max,
            context_lengths_num_intervals,
            dtype=torch.int,
        )


    def _load_and_process_data(self):
        print(f"Loading and processing data from {self.file_path}...")
        with open(self.file_path) as f:
            data = json.load(f)

        total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
        total_docs = sorted(list(set(total_docs)))
        total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

        print("Pre-calculating token counts for all documents...")
        doc_token_counts = [
            self._get_token_nums(self.DOCUMENT_PROMPT.format(i=1, document=doc))
            for doc in tqdm(total_docs, desc="Tokenizing docs")
        ]

        total_qas = []
        for d in data:
            total_qas.append({
                'query': d['question'],
                'outputs': [d['answer']],
                'context_indices': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
            })

        print(f"Data loaded. Found {len(total_qas)} questions and {len(total_docs)} unique documents.")
        return total_qas, total_docs, total_docs_dict, doc_token_counts

    def __len__(self):
        return len(self.qas)

    def _filter(self):
        """
        Filters out questions that the model can answer without any context.
        Uses a cache file to avoid re-running this expensive operation.
        """
        if self.filtered_ids_path.exists():
            print(f"Loading filtered question indices from cache: {self.filtered_ids_path}")
            with open(self.filtered_ids_path, 'r') as f:
                return json.load(f)

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).cuda()

        print("Filtering questions to find those that require context. This may take a while...")

        valid_indices = []
        device = model.device

        # A simple prompt to ask the question without context
        no_context_prompt_template = "Question: {question}\nAnswer:"

        for i, qa_item in enumerate(tqdm(self.qas, desc="Filtering Questions")):
            question = qa_item['query']
            true_answer = qa_item['outputs'][0]

            prompt = no_context_prompt_template.format(question=question)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,  # Keep it short for faster inference
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            if true_answer.lower() not in generated_text.lower():
                valid_indices.append(i)

        print(f"Saving {len(valid_indices)} valid question indices to {self.filtered_ids_path}")
        with open(self.filtered_ids_path, 'w') as f:
            json.dump(valid_indices, f)

        return valid_indices

    def _get_token_nums(self, context):
        return len(self.tokenizer.encode(context))

    def _create_context(self, gold_doc_indices, target_max_tokens):
        selected_docs_content = [self.docs[i] for i in gold_doc_indices]
        distractor_indices = [i for i in range(len(self.docs)) if i not in gold_doc_indices]
        random.shuffle(distractor_indices)

        current_tokens = sum(self.doc_token_counts[i] for i in gold_doc_indices)

        if current_tokens > target_max_tokens:
            return selected_docs_content, current_tokens

        for doc_idx in distractor_indices:
            doc_tokens = self.doc_token_counts[doc_idx]

            if current_tokens + doc_tokens <= target_max_tokens:
                current_tokens += doc_tokens
                selected_docs_content.append(self.docs[doc_idx])

            if current_tokens >= target_max_tokens:
                break

        return selected_docs_content, current_tokens

    def _construct_input(self, question, answer, context):
        context_tokens = self.PROMPT.format(context=context, question=question)

        input_ids = self.tokenizer(context_tokens, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(answer, add_special_tokens=False,return_tensors="pt").input_ids[0]
        return dict(input_ids= input_ids, labels= labels)

    def __getitem__(self, index):
        target_max_tokens = self.context_length_intervals[index%self.context_lengths_num_intervals]
        qa_item = self.qas[index]

        selected_docs, token_count = self._create_context(qa_item['context_indices'],target_max_tokens)

        if token_count < self.context_length_min:
            return self.__getitem__(random.randint(0, len(self) - 1))

        random.Random(index).shuffle(selected_docs)
        context_str = '\n\n'.join(
            [self.DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(selected_docs)])


        return self._construct_input(qa_item['query'], qa_item['outputs'], context_str)

