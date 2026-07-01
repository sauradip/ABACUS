#!/usr/bin/env python3
"""Consensus GRPO reward for dual-head counting alignment.

Designed for Stage-2 RL refinement where token and regression heads must agree.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_JSON_RE = re.compile(r'\{\s*"total_count"\s*:\s*(-?\d+)\s*\}')


def _to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def extract_token_count(completion: Any) -> Optional[int]:
    text = _to_text(completion)
    m = _JSON_RE.search(text.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_valid_json_count(completion: Any) -> bool:
    text = _to_text(completion)
    m = _JSON_RE.search(text.strip())
    if not m:
        return False
    try:
        payload = json.loads(m.group(0))
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("total_count"), int)


def _quadratic_relative(pred: float, gt: float) -> float:
    gt_safe = max(float(gt), 1.0)
    rel = (float(pred) - float(gt)) / gt_safe
    return 1.0 - (rel * rel)


def _component_row(gt_count: float, token_count: Optional[int], reg_count: Optional[float]) -> Dict[str, float]:
    gt = max(float(gt_count), 1.0)

    r_form = 1.0 if token_count is not None else 0.0
    r_tok = _quadratic_relative(token_count, gt) if token_count is not None else 0.0
    r_reg = _quadratic_relative(reg_count, gt) if reg_count is not None else 0.0

    p_con = 0.0
    if token_count is not None and reg_count is not None:
        if abs(float(token_count) - float(reg_count)) > 0.1 * gt:
            p_con = -0.5

    total = r_form + r_tok + r_reg + p_con
    return {
        "r_form": float(r_form),
        "r_tok": float(r_tok),
        "r_reg": float(r_reg),
        "p_con": float(p_con),
        "total": float(total),
    }


def consensus_quadratic_v1_components(
    completions: List[Any],
    gt_count: List[Any],
    reg_pred_count: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    reg_values = reg_pred_count or [None] * len(completions)
    components: List[Dict[str, float]] = []
    for i, completion in enumerate(completions):
        gt = float(gt_count[i])
        tok = extract_token_count(completion)
        reg = reg_values[i] if i < len(reg_values) else None
        if reg is not None:
            reg = max(float(reg), 0.0)
        components.append(_component_row(gt, tok, reg))
    return components


def consensus_quadratic_v1(
    prompts: List[Any],
    completions: List[Any],
    gt_count: List[Any],
    reg_pred_count: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[float]:
    comps = consensus_quadratic_v1_components(
        completions=completions,
        gt_count=gt_count,
        reg_pred_count=reg_pred_count,
    )
    return [row["total"] for row in comps]
