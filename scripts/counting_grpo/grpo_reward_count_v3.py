#!/usr/bin/env python3
"""Count-only GRPO reward for Stage 3.2.1."""

import re
from typing import Any, Iterable, List, Optional


TOTAL_COUNT_PATTERNS = (
    re.compile(r'"total_count"\s*:\s*(-?\d+)'),
    re.compile(r"'total_count'\s*:\s*(-?\d+)"),
    re.compile(r"\btotal_count\b\s*[:=]\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"\btotal\s+count\b\s*[:=]\s*(-?\d+)", re.IGNORECASE),
)


def extract_total_count(prediction: Any) -> Optional[int]:
    text = _extract_text(prediction)
    for pattern in TOTAL_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def score_prediction(prediction: Any, gt_count: int) -> float:
    pred_count = extract_total_count(prediction)
    if pred_count is None:
        return -1.0

    gt = max(1, int(gt_count))
    error = abs(gt - int(pred_count))
    accuracy_reward = max(0.0, 1.0 - (error / gt))
    penalty = -0.2 if (int(pred_count) == 20 and gt > 50) else 0.0
    return float(accuracy_reward + penalty)


def _extract_text(completion_item: Any) -> str:
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, list) and completion_item:
        first = completion_item[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    if isinstance(completion_item, dict) and "content" in completion_item:
        return str(completion_item["content"])
    return str(completion_item)


def _gt_counts_from_kwargs(length: int, kwargs: dict) -> List[int]:
    raw = kwargs.get("gt_count", kwargs.get("ground_truth_count", []))
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        values = [raw]
    out = []
    for idx in range(length):
        value = values[idx] if idx < len(values) else 0
        out.append(int(value or 0))
    return out


def reward_function(*args, **kwargs):
    """TRL-style list reward plus direct score compatibility.

    Direct use:
        reward_function(prediction, gt_count)

    Trainer use:
        reward_function(prompts=[...], completions=[...], gt_count=[...])
    """
    if "completions" in kwargs:
        completions = kwargs["completions"]
        gt_counts = _gt_counts_from_kwargs(len(completions), kwargs)
        return [score_prediction(c, gt) for c, gt in zip(completions, gt_counts)]

    if len(args) >= 2 and isinstance(args[1], (int, float)):
        return score_prediction(args[0], int(args[1]))

    if len(args) >= 2 and isinstance(args[1], list):
        completions = args[1]
        gt_counts = _gt_counts_from_kwargs(len(completions), kwargs)
        return [score_prediction(c, gt) for c, gt in zip(completions, gt_counts)]

    raise TypeError("reward_function expects either (prediction, gt_count) or completions=[...]")
