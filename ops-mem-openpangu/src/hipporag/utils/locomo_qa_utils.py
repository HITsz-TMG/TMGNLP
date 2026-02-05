import random
import re
from typing import Any, Dict, Optional, Tuple


NO_INFO_OPTION = "Not mentioned in the conversation"
DEFAULT_MC_TEMPLATE = " (a) {} (b) {}. Select the correct answer by writing (a) or (b)."


def _clean_llm_choice_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Take first non-empty line (LoCoMo hf_llm_utils behavior)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    s = lines[0] if lines else s
    return s


def preprocess_question(qa_item: Dict[str, Any], random_seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """
    LoCoMo-specific question preprocessing.

    - category == 2: append temporal instruction
    - category == 5: convert to a 2-choice multiple-choice question using NO_INFO_OPTION and qa_item['adversarial_answer']

    Returns:
        processed_question, locomo_metadata
    """
    category = qa_item.get("category", None)
    question = str(qa_item.get("question", "")).strip()

    metadata: Dict[str, Any] = {
        "category": category,
        "original_question": question,
    }

    if category == 2:
        processed = question + " Use DATE of CONVERSATION to answer with an approximate date."
        return processed, metadata

    if category == 5:
        # For HippoRAG_Episodic's LoCoMo datasets, category-5 adversarial option is stored in `adversarial_answer`.
        # NOTE: `answer` is the gold label (often a list like ["Not mentioned in the conversation"]) and should NOT
        # be used as the adversarial option, otherwise it can create degenerate/duplicate choices.
        adversarial = qa_item.get("adversarial_answer", None)
        if adversarial is None:
            # Fallback: still ask as an unanswerable question; evaluation will rely on NOT_MENTIONED
            processed = question + DEFAULT_MC_TEMPLATE.format(NO_INFO_OPTION, NO_INFO_OPTION)
            metadata["choice_map"] = {"a": NO_INFO_OPTION, "b": NO_INFO_OPTION}
            return processed, metadata

        rng = random.Random(random_seed) if random_seed is not None else random
        # 50/50 randomize option order
        if rng.random() < 0.5:
            a_opt, b_opt = NO_INFO_OPTION, str(adversarial)
        else:
            a_opt, b_opt = str(adversarial), NO_INFO_OPTION

        processed = question + DEFAULT_MC_TEMPLATE.format(a_opt, b_opt)
        metadata["choice_map"] = {"a": a_opt, "b": b_opt}
        return processed, metadata

    # Default: no preprocessing
    return question, metadata


def postprocess_answer(raw_answer: str, locomo_metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    LoCoMo-specific answer postprocessing.

    For category 5, map model output to option (a)/(b) -> actual answer string.
    Otherwise, strip common prefixes/markers and return.
    """
    if locomo_metadata is None:
        locomo_metadata = {}

    category = locomo_metadata.get("category", None)
    s = _clean_llm_choice_text(raw_answer)

    # Generic cleanup (do not overdo; keep compatibility with existing evaluation)
    s_norm = re.sub(r"^\s*answer\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    if category != 5:
        # Remove stray choice tokens if present
        s_norm = (
            s_norm.replace("(a)", "")
            .replace("(b)", "")
            .replace("a)", "")
            .replace("b)", "")
            .strip()
        )
        return s_norm

    choice_map = locomo_metadata.get("choice_map", None) or {}
    low = s_norm.lower()

    # Heuristic choice detection
    picked = None
    if "(a)" in low or re.search(r"\b[aA]\b", s_norm):
        picked = "a"
    if "(b)" in low or re.search(r"\b[bB]\b", s_norm):
        # If both detected, prefer explicit (a)/(b) tokens; else last wins.
        picked = "b" if "(b)" in low else (picked or "b")

    # Handle outputs like "a", "b"
    if picked is None:
        if low.strip() in {"a", "(a)"}:
            picked = "a"
        elif low.strip() in {"b", "(b)"}:
            picked = "b"

    if picked in choice_map:
        return str(choice_map[picked]).strip()

    # Fallback: if model directly outputs NO_INFO semantics
    if "no information available" in low or "not mentioned" in low or "no information" in low:
        return NO_INFO_OPTION

    return s_norm

