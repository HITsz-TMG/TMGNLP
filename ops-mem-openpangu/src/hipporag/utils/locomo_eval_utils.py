import string
from collections import Counter

from nltk.stem import PorterStemmer


_ps = PorterStemmer()


def _normalize_answer_locomo(s: str) -> str:
    # Mirror locomo/task_eval/evaluation.py normalization closely (remove commas, articles + and, punct, lowercase)
    if s is None:
        s = ""
    s = str(s).replace(",", "")

    def remove_articles(text: str) -> str:
        # locomo used (a|an|the|and)
        import regex
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_locomo(prediction: str, ground_truth: str) -> float:
    """
    locomo/task_eval/evaluation.py f1_score() with stemming + token overlap F1.
    """
    pred_tokens = [_ps.stem(w) for w in _normalize_answer_locomo(prediction).split()]
    gt_tokens = [_ps.stem(w) for w in _normalize_answer_locomo(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = 1.0 * num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def f1_multi_answer_locomo(prediction: str, ground_truth: str) -> float:
    """
    locomo/task_eval/evaluation.py f1() for multi-hop:
    split by ',' and compute mean(max f1 for each gt sub-answer).
    """
    predictions = [p.strip() for p in str(prediction).split(",") if p.strip()]
    ground_truths = [g.strip() for g in str(ground_truth).split(",") if g.strip()]
    if not predictions or not ground_truths:
        return 0.0
    return sum(max(f1_score_locomo(p, g) for p in predictions) for g in ground_truths) / len(ground_truths)

