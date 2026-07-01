#!/usr/bin/env python3
"""Build count-only eval JSONL files from the original FSC147 dataset."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_ROOT = "/projects/u6fb/myprojects/FSC147_hf"
DEFAULT_OUT_DIR = "outputs/fsc147_stage321_eval"


def load_classes(path: Path) -> Dict[str, str]:
    classes: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            image_name, category = line.split("\t", 1)
            classes[image_name] = category
    return classes


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build_rows(root: Path, split: str) -> List[Dict[str, Any]]:
    split_path = root / "Train_Test_Val_FSC_147.json"
    ann_path = root / "annotation_FSC147_384.json"
    class_path = root / "ImageClasses_FSC147.txt"
    image_dir = root / "images_384_VarV2"

    with split_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    with ann_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    classes = load_classes(class_path)

    if split not in splits:
        raise KeyError(f"Split '{split}' not found in {split_path}. Available: {sorted(splits)}")

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for image_name in splits[split]:
        ann = annotations.get(image_name)
        category = classes.get(image_name, "objects")
        image_path = image_dir / image_name
        if ann is None:
            missing.append(f"{image_name}: missing annotation")
            continue
        if not image_path.exists():
            missing.append(f"{image_name}: missing image file {image_path}")
            continue

        gt_count = len(ann.get("points", []))
        instruction = (
            "<image>\n"
            "Return strict count-only JSON with exactly one key: total_count.\n"
            f"Count the {category}."
        )
        rows.append(
            {
                "id": image_name,
                "split": split,
                "category": category,
                "image": str(image_path),
                "instruction": instruction,
                "problem": instruction,
                "gt_count": gt_count,
                "ground_truth_count": gt_count,
                "target_response": json.dumps({"total_count": gt_count}, separators=(",", ":")),
                "response_schema": "count_only",
                "source": "original_fsc147",
                "conversations": [{"from": "human", "value": instruction}],
            }
        )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"FSC147 split '{split}' has {len(missing)} missing item(s):\n{preview}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc147_root", default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.fsc147_root).resolve()
    out_dir = Path(args.output_dir)

    for split in args.splits:
        rows = build_rows(root, split)
        out_path = out_dir / f"fsc147_{split}_count_only.jsonl"
        count = write_jsonl(out_path, rows)
        counts = [int(row["gt_count"]) for row in rows]
        print(
            json.dumps(
                {
                    "split": split,
                    "rows": count,
                    "output": str(out_path),
                    "gt_min": min(counts) if counts else None,
                    "gt_max": max(counts) if counts else None,
                    "gt_mean": sum(counts) / len(counts) if counts else None,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
