from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer

logger = get_logger(__name__)

# Reference: MRQA official eval
class QAExactMatch(BaseMetric):
    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # 处理空答案列表的情况（不可回答的问题）
            if not gold_list or len(gold_list) == 0:
                aggregated_em = 0.0
            else:
                em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
                aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

class QAF1Score(BaseMetric):
    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # 处理空答案列表的情况（不可回答的问题）
            if not gold_list or len(gold_list) == 0:
                aggregated_f1 = 0.0
            else:
                f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
                aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results

class QABLEU1(BaseMetric):
    metric_name: str = "qa_bleu_1"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the BLEU-1 score (unigram precision with brevity penalty).

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged BLEU-1 score.
                - A list of dictionaries with BLEU-1 scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_bleu1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()

            if len(predicted_tokens) == 0:
                return 0.0

            gold_counts = Counter(gold_tokens)
            predicted_counts = Counter(predicted_tokens)
            overlap = sum((predicted_counts & gold_counts).values())

            precision = overlap / len(predicted_tokens)

            ref_len = len(gold_tokens)
            cand_len = len(predicted_tokens)
            if cand_len == 0:
                bp = 0.0
            elif cand_len > ref_len:
                bp = 1.0
            else:
                bp = np.exp(1 - ref_len / cand_len)

            return bp * precision

        example_eval_results = []
        total_bleu1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            if not gold_list or len(gold_list) == 0:
                aggregated_bleu1 = 0.0
            else:
                bleu1_scores = [compute_bleu1(gold, predicted) for gold in gold_list]
                aggregated_bleu1 = aggregation_fn(bleu1_scores)
            example_eval_results.append({"BLEU-1": aggregated_bleu1})
            total_bleu1 += aggregated_bleu1

        avg_bleu1 = total_bleu1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"BLEU-1": avg_bleu1}

        return pooled_eval_results, example_eval_results