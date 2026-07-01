#!/usr/bin/env python3
"""Audit count-only predictions with the Stage 3.2.1 parser."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.grpo_reward_count_v3 import extract_total_count


def _get_field(row: Dict[str, Any], names: List[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _prediction_text(row: Dict[str, Any], field: str) -> Any:
    if field:
        return row.get(field, "")
    return _get_field(row, ["prediction", "completion", "response", "generated_text", "text"]) or ""


def _bucket(gt: int) -> str:
    if gt < 30:
        return "low_lt30"
    if gt > 100:
        return "high_gt100"
    return "mid_30_100"


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
        stripped = raw.lstrip()
        if not stripped:
            return []
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None

        if isinstance(payload, dict):
            if "rows" in payload:
                rows = payload["rows"]
                if not isinstance(rows, list):
                    raise ValueError(f"JSON payload in {path} does not contain a rows list")
                return rows
            return [payload]
        if isinstance(payload, list):
            return payload

        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def audit(path: str, prediction_field: str = "") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = load_rows(path)
    buckets: Dict[str, List[float]] = {"low_lt30": [], "mid_30_100": [], "high_gt100": []}
    missing = 0
    examples: List[Dict[str, Any]] = []

    for row in rows:
        gt = int(_get_field(row, ["gt_count", "ground_truth_count", "total_count"]) or 0)
        pred: Optional[int] = extract_total_count(_prediction_text(row, prediction_field))
        if pred is None:
            missing += 1
            err = float(max(gt, 1))
        else:
            err = float(abs(gt - pred))
        buckets[_bucket(gt)].append(err)
        if len(examples) < 20:
            examples.append({"id": row.get("id"), "gt_count": gt, "pred_count": pred, "abs_error": err})

    summary: Dict[str, Any] = {"rows": len(rows), "missing_predictions": missing}
    for name, values in buckets.items():
        if values:
            summary[name] = {"n": len(values), "mae": sum(values) / len(values)}
        else:
            summary[name] = {"n": 0, "mae": None}
    all_errors = [err for values in buckets.values() for err in values]
    summary["overall"] = {"n": len(all_errors), "mae": sum(all_errors) / len(all_errors) if all_errors else None}
    return summary, examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--prediction_field", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, examples = audit(args.predictions_jsonl, args.prediction_field)
    print(json.dumps({"summary": summary, "examples": examples}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
