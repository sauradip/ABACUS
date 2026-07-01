#!/usr/bin/env python3
"""Stage 3.2 GRPO reward (v32).

Reward components:
- R_format: +1.0 perfect JSON, +0.8 recovered JSON, -1.0 fatal parse failure
- R_schema: -1.0 when gt>30 and model emits long clusters list
- R_math: +1.0 when sum(clusters.count) == total_count (full schema)
- R_acc:  1 - |GT - Pred| / GT
"""

import json
import re
from typing import Any, Dict, List, Tuple


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


def _parse_clusters_total(clusters: Any) -> int:
    if not isinstance(clusters, list):
        return 0
    total = 0
    for c in clusters:
        if isinstance(c, dict):
            total += int(c.get("count", 0))
        elif isinstance(c, (int, float)):
            total += int(c)
    return total


def _safe_json_loads(text: str) -> Tuple[Dict[str, Any], str]:
    """Return (obj, status) where status is perfect|recovered|fatal."""
    raw = text.strip()

    # Direct parse.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "perfect"
    except Exception:
        pass

    # If response contains extra prose around JSON, try slicing to first/last brace.
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        candidate = raw[start:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, "recovered"
        except Exception:
            pass

    # Missing-brace syndrome from Stage 1.6b: starts with "clusters".
    stripped = raw.lstrip()
    if stripped.startswith('"clusters"') or stripped.startswith("'clusters'") or stripped.startswith("clusters"):
        recovered = stripped
        # normalize bare clusters key case: clusters: ... -> "clusters": ...
        if recovered.startswith("clusters"):
            recovered = re.sub(r"^clusters\s*:", '"clusters":', recovered, count=1)
        recovered = "{" + recovered
        if not recovered.rstrip().endswith("}"):
            recovered = recovered + "}"
        try:
            obj = json.loads(recovered)
            if isinstance(obj, dict):
                return obj, "recovered"
        except Exception:
            pass

    return {}, "fatal"


def _pred_count(obj: Dict[str, Any]) -> int:
    tc = obj.get("total_count", None)
    if isinstance(tc, (int, float)):
        return int(tc)
    clusters = obj.get("clusters", None)
    if clusters is not None:
        return _parse_clusters_total(clusters)
    return 0


def _format_reward(status: str) -> float:
    if status == "perfect":
        return 1.0
    if status == "recovered":
        return 0.8
    return -1.0


def _schema_reward(gt_count: int, response_schema: str, obj: Dict[str, Any]) -> float:
    if gt_count <= 30:
        return 0.0

    # Explicit schema mismatch
    if response_schema == "summary" and "clusters" in obj:
        clusters = obj.get("clusters")
        if isinstance(clusters, list) and len(clusters) > 0:
            return -1.0

    # Even without response_schema metadata, penalize long clusters for high density.
    clusters = obj.get("clusters")
    if isinstance(clusters, list) and len(clusters) > 8:
        return -1.0

    return 0.0


def _math_reward(obj: Dict[str, Any]) -> float:
    clusters = obj.get("clusters")
    total_count = obj.get("total_count")
    if not isinstance(clusters, list):
        return 0.0
    if not isinstance(total_count, (int, float)):
        return 0.0
    return 1.0 if _parse_clusters_total(clusters) == int(total_count) else 0.0


def _accuracy_reward(gt_count: int, pred_count: int) -> float:
    gt = max(int(gt_count), 1)
    return 1.0 - (abs(gt - int(pred_count)) / gt)


def compute_reward_components(
    completion_text: str,
    gt_count: int,
    response_schema: str,
) -> Dict[str, float]:
    parsed, status = _safe_json_loads(completion_text)

    r_format = _format_reward(status)
    if status == "fatal":
        # Fatal parse still gets a numerical gradient from 0 prediction.
        pred_count = 0
        r_acc = _accuracy_reward(gt_count, pred_count)
        return {
            "R_format": r_format,
            "R_schema": 0.0,
            "R_math": 0.0,
            "R_acc": r_acc,
            "total": r_format + r_acc,
        }

    pred_count = _pred_count(parsed)
    r_schema = _schema_reward(gt_count, response_schema, parsed)
    r_math = _math_reward(parsed)
    r_acc = _accuracy_reward(gt_count, pred_count)
    total = r_format + r_schema + r_math + r_acc

    return {
        "R_format": r_format,
        "R_schema": r_schema,
        "R_math": r_math,
        "R_acc": r_acc,
        "total": total,
    }


def reward_function(prompts: List[str], completions: List[Any], **kwargs) -> List[float]:
    del prompts
    gt_counts = kwargs.get("gt_count", kwargs.get("ground_truth_count", []))
    schemas = kwargs.get("response_schema", [""] * len(completions))

    out = []
    for i, completion in enumerate(completions):
        text = _extract_text(completion)
        gt = int(gt_counts[i]) if i < len(gt_counts) else 0
        schema = str(schemas[i]) if i < len(schemas) else ""
        parts = compute_reward_components(text, gt, schema)
        out.append(float(parts["total"]))
    return out
