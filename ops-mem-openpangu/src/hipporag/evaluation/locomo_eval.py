from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseMetric
from ..utils.locomo_eval_utils import f1_multi_answer_locomo, f1_score_locomo


class LocomoQAEvaluator(BaseMetric):
    """
    LoCoMo category-aware evaluator.

    IMPORTANT:
    - This evaluator is additive: call it in addition to existing QAExactMatch/QAF1Score/QABLEU1/LLMJudge,
      to avoid affecting other dataset pipelines.
    """

    metric_name: str = "locomo_qa"

    def calculate_metric_scores(
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],
        qa_items: Optional[List[Dict[str, Any]]] = None,
        aggregation_fn: Callable = np.max,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        # Keep signature compatible with other evaluators; qa_items is optional.
        assert len(gold_answers) == len(predicted_answers), "gold_answers/predicted_answers length mismatch"
        if qa_items is not None:
            assert len(qa_items) == len(predicted_answers), "qa_items/predicted_answers length mismatch"

        if qa_items is None:
            # If caller didn't provide category metadata, we cannot do category-aware scoring.
            return {"LoCoMoScore": 0.0}, [{"LoCoMoScore": 0.0} for _ in predicted_answers]

        example_results: List[Dict[str, float]] = []
        per_cat: Dict[int, List[float]] = defaultdict(list)

        for gold_list, pred, qa in zip(gold_answers, predicted_answers, qa_items):
            category = int(qa.get("category", 0) or 0)
            
            # Align with HippoRAG: for category 3, only process the first gold answer
            # locomo/task_eval/evaluation.py: for category 3, use text before ';'
            if gold_list is None:
                gold_list_norm = []
            elif isinstance(gold_list, set):
                gold_list_norm = list(gold_list)
            else:
                gold_list_norm = list(gold_list)
            
            # LoCoMo 数据里通常每题只有一个 gold，但这里仍保留多答案兼容
            # For category 3, align with HippoRAG: only use the first gold answer
            if category == 3:
                gold_main = (gold_list_norm[0] if gold_list_norm else "") or ""
                if isinstance(gold_main, str):
                    gold_main = gold_main.split(";")[0].strip()
                gold_list_norm = [gold_main] if gold_main else []
            else:
                # For other categories, process all gold answers (original behavior)
                gold_list_norm_processed = []
                for g in gold_list_norm:
                    gs = str(g)
                    gold_list_norm_processed.append(gs)
                gold_list_norm = gold_list_norm_processed

            score = self._score_one(category=category, predicted=str(pred), gold_list=gold_list_norm, qa_item=qa)
            example_results.append({"LoCoMoScore": float(score), "LoCoMoCategory": float(category)})
            per_cat[category].append(float(score))

        overall = float(np.mean([r["LoCoMoScore"] for r in example_results])) if example_results else 0.0
        pooled: Dict[str, float] = {"LoCoMoScore": overall}
        for cat, scores in per_cat.items():
            pooled[f"LoCoMoScore_Cat{cat}"] = float(np.mean(scores)) if scores else 0.0

        return pooled, example_results

    def _score_one(self, category: int, predicted: str, gold_list: List[str], qa_item: Dict[str, Any]) -> float:
        # Category mapping mirrors locomo/task_eval/evaluation.py
        if category in (2, 3, 4):
            # single-hop / temporal / open-domain
            if not gold_list:
                return 0.0
            return float(max(f1_score_locomo(predicted, g) for g in gold_list))

        if category == 1:
            # multi-hop
            if not gold_list:
                return 0.0
            return float(max(f1_multi_answer_locomo(predicted, g) for g in gold_list))

        if category == 5:
            # Match locomo/task_eval/evaluation.py for category 5:
            # only check whether output selects/expresses the unanswerable option.
            low = predicted.lower()
            return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0

        # unknown category: fall back to standard single-hop f1
        if not gold_list:
            return 0.0
        return float(max(f1_score_locomo(predicted, g) for g in gold_list))

