from typing import List, Dict, Tuple, Optional, Callable
import json
import numpy as np
import re

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..llm import _get_llm_class, BaseLLM
from ..utils.llm_utils import TextChatMessage, fix_broken_generated_json

logger = get_logger(__name__)


class QALLMJudge(BaseMetric):
    metric_name: str = "qa_llm_judge"

    def __init__(self, global_config: Optional[BaseConfig] = None, llm_model: Optional[BaseLLM] = None):
        """
        Initialize the LLM as a Judge evaluator.

        Args:
            global_config: The global configuration object. If None, a default BaseConfig will be used.
            llm_model: Optional pre-initialized LLM model. If None, will be created from global_config.
        """
        super().__init__(global_config)
        
        if llm_model is not None:
            self.llm_model = llm_model
        else:
            # Create LLM instance from config
            self.llm_model = _get_llm_class(self.global_config)
        
        logger.info(f"Initialized {self.__class__.__name__} with LLM: {self.llm_model.__class__.__name__}")

    def _build_judge_prompt(self, question: str, gold_answer: str, generated_answer: str) -> str:
        """
        Build the LLM as a Judge prompt according to the MemO paper.

        Args:
            question: The question being evaluated.
            gold_answer: The ground truth answer.
            generated_answer: The generated answer to be evaluated.

        Returns:
            The complete prompt string.
        """
        prompt = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example: Question: Do you remember what I got the last time I went to Hawaii? Gold answer: A shell necklace The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question: 

Question: {question} 

Gold answer: {gold_answer} 

Generated answer: {generated_answer} 

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""

        return prompt.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )

    def _parse_judge_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the label (CORRECT or WRONG).

        Args:
            response: The raw response from the LLM.

        Returns:
            "CORRECT" or "WRONG", or "WRONG" if parsing fails.
        """
        # First, try to find and extract JSON from the response
        # Look for JSON pattern that may span multiple lines
        json_pattern = r'\{[^{}]*(?:"label"\s*:\s*["\']?(CORRECT|WRONG)["\']?)[^{}]*\}'
        match = re.search(json_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            try:
                # Try to parse the matched JSON
                json_str = match.group(0)
                # Try to fix broken JSON if needed
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = fix_broken_generated_json(json_str)
                    result = json.loads(json_str)
                
                label = result.get("label", "").upper()
                if label in ["CORRECT", "WRONG"]:
                    return label
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Failed to parse JSON from response: {str(e)}")
        
        # Fallback: look for JSON block with multiline support
        json_block_pattern = r'```(?:json)?\s*(\{[^{}]*"label"[^{}]*\})\s*```'
        match = re.search(json_block_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                result = json.loads(json_str)
                label = result.get("label", "").upper()
                if label in ["CORRECT", "WRONG"]:
                    return label
            except (json.JSONDecodeError, Exception):
                pass
        
        # Fallback: look for CORRECT or WRONG in the response (case-insensitive)
        response_upper = response.upper()
        if "CORRECT" in response_upper:
            # Make sure WRONG doesn't appear before CORRECT
            correct_pos = response_upper.find("CORRECT")
            wrong_pos = response_upper.find("WRONG")
            if wrong_pos == -1 or correct_pos < wrong_pos:
                return "CORRECT"
        
        if "WRONG" in response_upper:
            return "WRONG"
        
        # Default to WRONG if we can't parse
        logger.warning(f"Could not parse judge response: {response[:200]}... Defaulting to WRONG.")
        return "WRONG"

    def calculate_metric_scores(
        self, 
        gold_answers: List[List[str]], 
        predicted_answers: List[str], 
        questions: Optional[List[str]] = None,
        aggregation_fn: Callable = np.max
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate the LLM as a Judge score.

        Args:
            gold_answers: List of lists containing ground truth answers.
            predicted_answers: List of predicted answers.
            questions: Optional list of questions. If not provided, will use placeholder.
            aggregation_fn: Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged LLM Judge score.
                - A list of dictionaries with LLM Judge scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."
        
        if questions is None:
            questions = [f"Question {i+1}" for i in range(len(gold_answers))]
        else:
            assert len(questions) == len(gold_answers), \
                "Length of questions should match length of gold answers."

        example_eval_results = []
        total_correct = 0

        for idx, (gold_list, predicted, question) in enumerate(zip(gold_answers, predicted_answers, questions)):
            # Handle empty gold answer lists
            if not gold_list or len(gold_list) == 0:
                aggregated_score = 0.0
                example_eval_results.append({"LLMJudge": aggregated_score})
                continue

            # For multiple gold answers, evaluate against each and take the max
            judge_scores = []
            
            for gold_answer in gold_list:
                # Build the prompt
                prompt_text = self._build_judge_prompt(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=predicted
                )
                
                # Create the message for LLM
                messages: List[TextChatMessage] = [
                    {"role": "user", "content": prompt_text}
                ]
                
                try:
                    # Call LLM
                    # Handle different return formats: some LLMs return 2 values (response, metadata)
                    # while others return 3 values (response, metadata, cache_hit)
                    infer_result = self.llm_model.infer(messages)
                    
                    # Try to unpack 3 values first, fallback to 2 if that fails
                    try:
                        llm_responses, metadata, cache_hit = infer_result
                    except ValueError:
                        # If unpacking 3 values fails, try 2 values
                        llm_responses, metadata = infer_result
                    
                    # Extract the response content
                    # Handle both string and list formats
                    if isinstance(llm_responses, str):
                        response_content = llm_responses
                    elif isinstance(llm_responses, list) and len(llm_responses) > 0:
                        if isinstance(llm_responses[0], dict):
                            response_content = llm_responses[0].get("content", "")
                        else:
                            response_content = str(llm_responses[0])
                    else:
                        response_content = ""
                    
                    # Parse the response
                    label = self._parse_judge_response(response_content)
                    
                    # Convert to score: CORRECT = 1.0, WRONG = 0.0
                    score = 1.0 if label == "CORRECT" else 0.0
                    judge_scores.append(score)
                    
                    logger.debug(
                        f"Question {idx+1}: Label={label}, Score={score}, "
                        f"Gold={gold_answer[:50]}..., Predicted={predicted[:50]}..."
                    )
                    
                except Exception as e:
                    logger.error(f"Error evaluating question {idx+1} with LLM Judge: {str(e)}")
                    judge_scores.append(0.0)
            
            # Aggregate scores across multiple gold answers
            aggregated_score = aggregation_fn(judge_scores) if judge_scores else 0.0
            example_eval_results.append({"LLMJudge": aggregated_score})
            total_correct += aggregated_score

        avg_score = total_correct / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"LLMJudge": avg_score}

        logger.info(f"LLM Judge evaluation completed. Average score: {avg_score:.4f}")

        return pooled_eval_results, example_eval_results

