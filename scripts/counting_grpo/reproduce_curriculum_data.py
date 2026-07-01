#!/usr/bin/env python3
"""Build the reproducible Stage 1.6b/3.2.1 pivot curricula.

Outputs:
- sft_low_30_scaffold.jsonl: low-count scaffold SFT rows.
- grpo_full_count_only.jsonl: full-count count-only GRPO rows.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SOURCE_JSONL = "outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl"
SFT_OUT = "outputs/scaffold_rex_5k_pca/sft_low_30_scaffold.jsonl"
GRPO_OUT = "outputs/scaffold_rex_5k_pca/grpo_full_count_only.jsonl"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _resolve_image_path(value: Any, source_dir: Path) -> str:
    if not value:
        return ""
    raw = str(value)
    if os.path.isabs(raw):
        return raw
    return str((source_dir / raw).resolve())


def _instruction(row: Dict[str, Any]) -> str:
    return str(row.get("instruction") or row.get("problem") or "")


def _count_only_instruction(row: Dict[str, Any]) -> str:
    prompt = _instruction(row)
    if "Count the " in prompt:
        prefix, count_clause = prompt.rsplit("Count the ", 1)
        category = count_clause.split(" and ", 1)[0].rstrip(".")
        return (
            prefix.strip()
            + "\nReturn strict count-only JSON with exactly one key: total_count.\n"
            + f"Count the {category}."
        )
    return (
        prompt.strip()
        + "\nReturn strict count-only JSON with exactly one key: total_count."
    )


def _scaffold_response(anchors: List[Dict[str, Any]], gt_count: int) -> str:
    subtotal = 0
    tally_parts = []
    for anchor in anchors:
        count = int(anchor.get("count", 0))
        subtotal += count
        tally_parts.append(
            f"Anchor {anchor.get('anchor')} has {count}. Subtotal: {subtotal}."
        )
    payload = {"clusters": anchors, "total_count": gt_count}
    return f"Thought: {' '.join(tally_parts)} Answer: {json.dumps(payload, separators=(',', ':'))}"


def build_data(source_jsonl: str, sft_out: str, grpo_out: str) -> Tuple[int, int, int]:
    source_path = Path(source_jsonl).resolve()
    source_dir = source_path.parent
    data = _load_jsonl(source_path)

    sft_rows: List[Dict[str, Any]] = []
    grpo_rows: List[Dict[str, Any]] = []

    for row in data:
        gt = int(row.get("gt_count", row.get("ground_truth_count", 0)))
        anchors = row.get("anchors") or row.get("clusters") or []
        if not isinstance(anchors, list):
            anchors = []

        image = _resolve_image_path(row.get("image"), source_dir)
        pca_image = _resolve_image_path(row.get("pca_image") or row.get("image"), source_dir)

        if gt <= 30:
            instruction = _instruction(row)
            response = _scaffold_response(anchors, gt)
            sft_rows.append(
                {
                    "id": row.get("id"),
                    "split": row.get("split"),
                    "category": row.get("category"),
                    "image": image,
                    "pca_image": pca_image,
                    "instruction": instruction,
                    "response": response,
                    "ground_truth_count": gt,
                    "gt_count": gt,
                    "clusters": anchors,
                    "conversations": [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": response},
                    ],
                    "stage": "stage16b_low30_scaffold_sft",
                }
            )

        instruction = _count_only_instruction(row)
        target_response = json.dumps({"total_count": gt}, separators=(",", ":"))
        grpo_rows.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "category": row.get("category"),
                "image": image,
                "pca_image": pca_image,
                "instruction": instruction,
                "problem": instruction,
                "gt_count": gt,
                "ground_truth_count": gt,
                "target_response": target_response,
                "response_schema": "count_only",
                "conversations": [
                    {"from": "human", "value": instruction},
                ],
                "stage": "stage321_count_only_grpo",
            }
        )

    sft_count = _write_jsonl(Path(sft_out), sft_rows)
    grpo_count = _write_jsonl(Path(grpo_out), grpo_rows)
    return len(data), sft_count, grpo_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", default=SOURCE_JSONL)
    parser.add_argument("--sft_out", default=SFT_OUT)
    parser.add_argument("--grpo_out", default=GRPO_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, sft_count, grpo_count = build_data(
        source_jsonl=args.source_jsonl,
        sft_out=args.sft_out,
        grpo_out=args.grpo_out,
    )
    print(f"source_rows={total}")
    print(f"sft_low30_rows={sft_count} -> {args.sft_out}")
    print(f"grpo_full_rows={grpo_count} -> {args.grpo_out}")


if __name__ == "__main__":
    main()
