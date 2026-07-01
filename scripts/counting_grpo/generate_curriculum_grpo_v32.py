#!/usr/bin/env python3
"""Generate Stage 3.2 GRPO curriculum dataset.

Input source (default): outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl
Output: outputs/scaffold_rex_5k_pca/curriculum_grpo_v32.jsonl

Schema:
- gt_count <= 30  -> response_schema="full"
- gt_count > 30   -> response_schema="summary"
"""

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = "outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl"
FALLBACK_SOURCE = "outputs/scaffold_rex_5k_pca/all.jsonl"
DEFAULT_OUT = "outputs/scaffold_rex_5k_pca/curriculum_grpo_v32.jsonl"


def _gt_count(row: dict) -> int:
    return int(row.get("gt_count", row.get("ground_truth_count", 0)))


def _get_prompt(row: dict) -> str:
    convs = row.get("conversations", [])
    if convs:
        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            if role == "human":
                return str(turn.get("value", turn.get("content", "")))
    return str(row.get("problem", ""))


def _cluster_total(clusters: list) -> int:
    return sum(int(c.get("count", 0)) for c in clusters if isinstance(c, dict))


def _quadrant_summary(clusters: list) -> str:
    # 6x6 anchors -> TL/TR/BL/BR split by x<=3 and y<=3
    quads = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for c in clusters:
        if not isinstance(c, dict):
            continue
        anchor = c.get("anchor", [0, 0])
        if not isinstance(anchor, list) or len(anchor) != 2:
            continue
        x, y = int(anchor[0]), int(anchor[1])
        q = ("T" if x <= 3 else "B") + ("L" if y <= 3 else "R")
        quads[q] += int(c.get("count", 0))

    peak = max(quads.items(), key=lambda kv: kv[1])[0]
    return f"High density observed across all quadrants. Concentration peak in {peak}."


def _full_thought(clusters: list, category: str) -> str:
    subtotal = 0
    lines = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        anchor = c.get("anchor", [0, 0])
        if not isinstance(anchor, list) or len(anchor) != 2:
            continue
        x, y = int(anchor[0]), int(anchor[1])
        n = int(c.get("count", 0))
        subtotal += n
        lines.append(f"Anchor ({x},{y}) contains {n} {category}. Subtotal: {subtotal}.")
    if not lines:
        return "No valid anchors parsed."
    return " ".join(lines)


def build_row(row: dict, threshold: int) -> dict:
    gt = _gt_count(row)
    clusters = row.get("clusters", [])
    category = str(row.get("category", "objects"))

    out = dict(row)
    out["gt_count"] = gt
    out["ground_truth_count"] = gt

    if gt <= threshold:
        out["response_schema"] = "full"
        target_obj = {
            "thought": _full_thought(clusters, category),
            "clusters": clusters,
            "total_count": gt,
        }
    else:
        out["response_schema"] = "summary"
        target_obj = {
            "thought": _quadrant_summary(clusters),
            "total_count": gt,
        }

    out["target_response"] = json.dumps(target_obj, ensure_ascii=False)

    prompt = _get_prompt(row)
    out["conversations"] = [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": out["target_response"]},
    ]
    out["stage"] = "stage32_grpo_v32"

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=int, default=30)
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        fallback = Path(FALLBACK_SOURCE)
        if not fallback.exists():
            raise FileNotFoundError(f"Neither {src} nor {fallback} exists")
        print(f"[warn] Source not found: {src}; using fallback: {fallback}")
        src = fallback

    rows = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    transformed = [build_row(r, args.threshold) for r in rows]

    # Integrity pass: for full schema rows, ensure cluster sum can recover total_count.
    mismatch = 0
    for r in transformed:
        if r.get("response_schema") != "full":
            continue
        clusters = r.get("clusters", [])
        if _cluster_total(clusters) != int(r["gt_count"]):
            mismatch += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in transformed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    full_n = sum(1 for r in transformed if r.get("response_schema") == "full")
    summary_n = len(transformed) - full_n
    print(f"Source rows: {len(rows)}")
    print(f"Output rows: {len(transformed)}")
    print(f"Full schema rows (<=30): {full_n}")
    print(f"Summary schema rows (>30): {summary_n}")
    print(f"Cluster/GT mismatches (full): {mismatch}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
