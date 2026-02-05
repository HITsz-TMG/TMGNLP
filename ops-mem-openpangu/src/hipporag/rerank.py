import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
import os
import requests
from .prompts.filter_default_prompt import best_dspy_prompt
from .utils.config_utils import BaseConfig
from .utils.logging_utils import get_logger

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class DSPyFilter:
    def __init__(self, hipporag):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    def parse_filter(self, response):
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                # Handle empty value
                if not value or value.strip() == '':
                    print(f"Warning: fact_after_filter field is empty, returning empty list")
                    parsed = []
                    continue
                
                try:
                    # Try to fix incomplete JSON by attempting to close it
                    value_to_parse = value.strip()
                    
                    # Try JSON parsing first
                    try:
                        parsed_value = json.loads(value_to_parse)
                    except json.JSONDecodeError as json_err:
                        # Try to fix incomplete JSON
                        # Check if it starts with {"fact": but is incomplete
                        if value_to_parse.startswith('{"fact"') or value_to_parse.startswith("{'fact'"):
                            # Try to close the JSON structure
                            # Count open brackets and quotes
                            open_braces = value_to_parse.count('{') - value_to_parse.count('}')
                            open_brackets = value_to_parse.count('[') - value_to_parse.count(']')
                            
                            # Try to complete the JSON
                            fixed_value = value_to_parse
                            # Close brackets first
                            fixed_value += ']' * open_brackets
                            # Close braces
                            fixed_value += '}' * open_braces
                            
                            try:
                                parsed_value = json.loads(fixed_value)
                            except:
                                # If fixing didn't work, try ast.literal_eval
                                try:
                                    parsed_value = ast.literal_eval(value_to_parse)
                                except (ValueError, SyntaxError):
                                    # Last resort: try to extract what we can
                                    print(f"Warning: Could not parse fact_after_filter, attempting partial extraction. Original error: {json_err}")
                                    parsed = []
                                    continue
                        else:
                            # Try ast.literal_eval for Python dict format
                            try:
                                parsed_value = ast.literal_eval(value_to_parse)
                            except (ValueError, SyntaxError):
                                print(f"Warning: Could not parse fact_after_filter as JSON or Python literal. Value: {value_to_parse[:100]}...")
                                parsed = []
                                continue
                    
                    # Validate with Pydantic
                    try:
                        parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                    except Exception as validation_err:
                        print(f"Warning: fact_after_filter validation failed: {validation_err}. Value: {value_to_parse[:100]}...")
                        parsed = []
                        
                except Exception as e:
                    print(f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value[:200]}\n```")
                    parsed = []

        return parsed

    def llm_call(self, question, fact_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})
        # call openai

        self.default_gen_kwargs['max_completion_tokens'] = 512

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        # 注释掉LLM filter部分，直接返回top-k结果
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:
            # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print('exception', e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print('result_indices exception', e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}
        # # 直接返回原始的top-k结果，不进行LLM过滤
        # if len_after_rerank is None:
        #     len_after_rerank = len(candidate_items)
        
        # # 直接返回前len_after_rerank个结果
        # sorted_candidate_indices = candidate_indices[:len_after_rerank]
        # sorted_candidate_items = candidate_items[:len_after_rerank]
        # return sorted_candidate_indices, sorted_candidate_items, {'confidence': None}


class Qwen3Reranker:
    """
    Qwen3-Reranker-4B模型的重排序器实现
    """
    
    def __init__(self, global_config: BaseConfig):
        """
        初始化Qwen3Reranker
        
        Args:
            global_config: 全局配置对象
        """
        self.global_config = global_config
        self.rerank_llm_base_url = global_config.rerank_llm_base_url
        self.rerank_model_name = global_config.rerank_model_name
        
        if self.rerank_llm_base_url is None:
            raise ValueError("rerank_llm_base_url must be provided for Qwen3Reranker")
        
        # 轻量日志记录（避免引入额外机制）
        try:
            self._logger = get_logger(__name__)
            self._logger.info(f"Initializing Qwen3Reranker with model: {self.rerank_model_name}")
        except Exception:
            self._logger = None
    
    def _format_instruction(self, instruction: str, query: str, doc: str) -> List[Dict]:
        """
        构建OpenAI兼容messages
        """
        return [
            {
                "role": "system", 
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user", 
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]

    def _call_rerank_api(self, query: str, facts: List[Tuple]) -> List[Dict]:
        # 将facts转换为字符串格式
        fact_texts = []
        for fact in facts:
            if len(fact) == 3:
                fact_text = f"{fact[0]} {fact[1]} {fact[2]}"
            else:
                fact_text = str(fact)
            fact_texts.append(fact_text)
        
        task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        try:
            base = (self.rerank_llm_base_url or "").rstrip("/")
            if base.endswith("/v1"):
                endpoint = f"{base}/chat/completions"
            else:
                endpoint = f"{base}/v1/chat/completions"

            headers = {"Content-Type": "application/json"}
            api_key = os.environ.get("RERANK_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            results = []
            for i, fact_text in enumerate(fact_texts):
                single_messages = self._format_instruction(task_instruction, query, fact_text)

                payload_primary = {
                    "model": self.rerank_model_name,
                    "messages": single_messages,
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False,
                    "logprobs": 20
                }

                payload_fallback = {
                    "model": self.rerank_model_name,
                    "messages": single_messages,
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False
                }

                def _post_and_score(payload_obj):
                    resp = requests.post(
                        endpoint,
                        json=payload_obj,
                        headers=headers,
                        timeout=30
                    )
                    return resp

                response = _post_and_score(payload_primary)
                if response.status_code != 200:
                    response = _post_and_score(payload_fallback)

                if response.status_code != 200:
                    results.append({"index": i, "text": fact_text, "score": 0.5})
                    continue

                try:
                    result = response.json()
                except Exception:
                    results.append({"index": i, "text": fact_text, "score": 0.5})
                    continue

                if not isinstance(result, dict) or "choices" not in result or len(result["choices"]) == 0:
                    results.append({"index": i, "text": fact_text, "score": 0.5})
                    continue

                score = 0.5
                choice0 = result["choices"][0]
                logprobs_obj = choice0.get("logprobs") if isinstance(choice0, dict) else None
                if isinstance(logprobs_obj, dict):
                    content_logprobs = logprobs_obj.get("content")
                    if isinstance(content_logprobs, list) and len(content_logprobs) > 0:
                        last_token_alternatives = content_logprobs[-1]
                        yes_lp = None
                        no_lp = None
                        for alt in last_token_alternatives:
                            if isinstance(alt, dict):
                                token_str = alt.get("token")
                                lp = alt.get("logprob")
                            else:
                                continue
                            if token_str is None or lp is None:
                                continue
                            ts = str(token_str).strip().lower()
                            if ts == "yes":
                                yes_lp = lp
                            elif ts == "no":
                                no_lp = lp
                        if yes_lp is not None or no_lp is not None:
                            import math
                            if yes_lp is None:
                                yes_lp = -10.0
                            if no_lp is None:
                                no_lp = -10.0
                            yes_p = math.exp(yes_lp)
                            no_p = math.exp(no_lp)
                            denom = yes_p + no_p
                            score = (yes_p / denom) if denom > 0 else 0.5
                if score == 0.5:
                    message_obj = choice0.get("message", {}) if isinstance(choice0, dict) else {}
                    content = message_obj.get("content", "") if isinstance(message_obj, dict) else str(message_obj)
                    content = content.strip().lower()
                    if content == "yes":
                        score = 0.9
                    elif content == "no":
                        score = 0.1

                results.append({"index": i, "text": fact_text, "score": float(score)})

            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except requests.exceptions.RequestException:
            return [{"index": i, "text": fact_texts[i], "score": 0.5} for i in range(len(fact_texts))]
        except Exception:
            return [{"index": i, "text": fact_texts[i], "score": 0.5} for i in range(len(fact_texts))]
    
    def rerank(self, 
               query: str, 
               candidate_items: List[Tuple], 
               candidate_indices: List[int], 
               len_after_rerank: int = None) -> Tuple[List[int], List[Tuple], Dict]:
        if len_after_rerank is None:
            len_after_rerank = len(candidate_items)
        if not candidate_items:
            return [], [], {'confidence': None}

        rerank_results = self._call_rerank_api(query, candidate_items)
        if not rerank_results:
            top_k = max(1, (len(candidate_items) + 1)// 2)
            return candidate_indices[:top_k], candidate_items[:top_k], {'confidence': None}

        try:
            if isinstance(rerank_results[0], dict):
                if "index" in rerank_results[0]:
                    sorted_results = sorted(rerank_results, key=lambda x: x.get("score", 0), reverse=True)
                    reranked_indices = [result["index"] for result in sorted_results]
                    confidence_scores = [result.get("score", 0) for result in sorted_results]
                else:
                    return candidate_indices[:len_after_rerank], candidate_items[:len_after_rerank], {'confidence': None}
            else:
                if isinstance(rerank_results[0], (int, float)):
                    sorted_indices = sorted(range(len(rerank_results)), key=lambda i: rerank_results[i], reverse=True)
                    reranked_indices = sorted_indices
                    confidence_scores = [rerank_results[i] for i in sorted_indices]
                else:
                    return candidate_indices[:len_after_rerank], candidate_items[:len_after_rerank], {'confidence': None}

            valid_indices = [idx for idx in reranked_indices if 0 <= idx < len(candidate_items)]
            if not valid_indices:
                return candidate_indices[:len_after_rerank], candidate_items[:len_after_rerank], {'confidence': None}

            top_k = max(1, (len(valid_indices) + 1)// 2)
            sorted_candidate_indices = [candidate_indices[idx] for idx in valid_indices[:top_k]]
            sorted_candidate_items = [candidate_items[idx] for idx in valid_indices[:top_k]]
            avg_confidence = sum(confidence_scores[:len(valid_indices)]) / len(valid_indices) if valid_indices else None
            return sorted_candidate_indices, sorted_candidate_items, {'confidence': avg_confidence}
        except Exception:
            top_k = max(1, (len(candidate_items) + 1)// 2)
            return candidate_indices[:top_k], candidate_items[:top_k], {'confidence': None}
    
    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)