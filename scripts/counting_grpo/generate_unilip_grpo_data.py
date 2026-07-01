#!/usr/bin/env python3
"""Build the HF-native multi-image GRPO JSONL for UniLIP count-only RL."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm


PROMPT_JSONL = "outputs/scaffold_prompt_fsc147_train_adaptive/train_scaffold_input_only.jsonl"
MASTER_GT_JSONL = "outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl"
OUTPUT_JSONL = "outputs/unilip_grpo_adaptive/hf_native_multi_image_grpo_train_dense.jsonl"
FSC147_ANNOTATIONS = "/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json"


def normalize_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return Path(raw).stem or raw


def parse_gt_count(row: Dict[str, Any]) -> Optional[int]:
    for key in ("gt_count", "ground_truth_count", "total_count"):
        value = row.get(key)
        if value is not None:
            return int(value)

    for key in ("target_response", "solution", "response"):
        value = row.get(key)
        if not value:
            continue
        if isinstance(value, dict) and value.get("total_count") is not None:
            return int(value["total_count"])
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("total_count") is not None:
                return int(payload["total_count"])
    return None


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            yield row


def load_gt_mapping(master_gt_jsonl: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not master_gt_jsonl.exists():
        return mapping

    for row in iter_jsonl(master_gt_jsonl):
        gt_count = parse_gt_count(row)
        if gt_count is None:
            continue
        for key in ("id", "question_id", "image", "source_image", "pca_image"):
            value = row.get(key)
            if not value:
                continue
            raw = str(value)
            mapping[raw] = gt_count
            norm = normalize_id(raw)
            if norm:
                mapping[norm] = gt_count
    return mapping


def load_fsc147_annotation_mapping(annotation_json: Path) -> Dict[str, int]:
    if not annotation_json.exists():
        raise FileNotFoundError(f"Missing FSC147 annotations: {annotation_json}")
    payload = json.loads(annotation_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in FSC147 annotation file: {annotation_json}")

    mapping: Dict[str, int] = {}
    for image_name, item in payload.items():
        if not isinstance(item, dict):
            continue
        count = len(item.get("points", []))
        mapping[str(image_name)] = int(count)
        mapping[normalize_id(image_name)] = int(count)
    return mapping


def history_text(row: Dict[str, Any]) -> str:
    history = row.get("history") or []
    if not history:
        return ""
    content = history[0].get("content", [])
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


def validate_row(row: Dict[str, Any]) -> None:
    if sorted(row.keys()) != ["gt_count", "id", "prompt"]:
        raise ValueError(f"GRPO row must have exactly id, prompt, gt_count keys: {row.keys()}")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
        raise ValueError(f"Row {row.get('id')} prompt must contain only one user message")
    content = prompt[0].get("content", [])
    images = [item for item in content if isinstance(item, dict) and item.get("type") == "image"]
    texts = [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if len(images) != 2 or len(texts) != 1:
        raise ValueError(f"Row {row.get('id')} must have exactly two images and one text item")
    if "assistant" in json.dumps(prompt).lower() or "gt_count" in json.dumps(prompt):
        raise ValueError(f"Row {row.get('id')} prompt contains supervision leakage")
    for image in images:
        path = Path(str(image.get("url", "")))
        if not path.exists():
            raise FileNotFoundError(f"Missing image for row {row.get('id')}: {path}")


def build_dataset(
    prompt_jsonl: Path,
    master_gt_jsonl: Path,
    fsc147_annotations: Optional[Path],
    output_jsonl: Path,
    limit: int = 0,
    min_gt: int = 0,
) -> int:
    gt_mapping = load_gt_mapping(master_gt_jsonl)
    if fsc147_annotations is not None:
        gt_mapping.update(load_fsc147_annotation_mapping(fsc147_annotations))
    if not gt_mapping:
        raise RuntimeError(
            f"No GT counts loaded from {master_gt_jsonl}"
            + (f" or {fsc147_annotations}" if fsc147_annotations is not None else "")
        )

    input_data = list(iter_jsonl(prompt_jsonl))
    if limit > 0:
        input_data = input_data[:limit]

    grpo_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in tqdm(input_data, desc=f"Building GRPO Dataset (min_gt={min_gt})"):
        q_id = normalize_id(row["question_id"])
        gt_count = gt_mapping.get(q_id)
        if gt_count is None:
            missing.append(q_id)
            continue

        if int(gt_count) < int(min_gt):
            continue

        image_paths = row.get("image_paths")
        if not isinstance(image_paths, list) or len(image_paths) != 2:
            raise ValueError(f"Row {q_id} must contain exactly two image_paths")
        original_img, scaffold_img = image_paths
        question = str(row["question"]).strip()
        system_prompt = history_text(row)

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(original_img)},
                    {"type": "image", "url": str(scaffold_img)},
                    {"type": "text", "text": f"{system_prompt}\n\n{question}".strip()},
                ],
            }
        ]

        out_row = {"id": q_id, "prompt": prompt_messages, "gt_count": int(gt_count)}
        validate_row(out_row)
        grpo_rows.append(out_row)

    if missing:
        raise KeyError(f"Missing GT for {len(missing)} prompt rows, first few: {missing[:20]}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in grpo_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Success! Generated {len(grpo_rows)} GRPO rows at {output_jsonl}")
    return len(grpo_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_jsonl", default=PROMPT_JSONL)
    parser.add_argument("--master_gt_jsonl", default=MASTER_GT_JSONL)
    parser.add_argument("--fsc147_annotations", default=FSC147_ANNOTATIONS)
    parser.add_argument("--output_jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--min_gt",
        type=int,
        default=50,
        help="Minimum Ground Truth count required to include the row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        prompt_jsonl=Path(args.prompt_jsonl),
        master_gt_jsonl=Path(args.master_gt_jsonl),
        fsc147_annotations=Path(args.fsc147_annotations) if args.fsc147_annotations else None,
        output_jsonl=Path(args.output_jsonl),
        limit=args.limit,
        min_gt=args.min_gt,
    )


if __name__ == "__main__":
    main()
