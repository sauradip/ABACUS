#!/usr/bin/env python3
"""Prepare FSC-147 dataset splits for LoRA SFT in LLaVA-style conversation format.

Reproduces the data preparation step for the Variant B counter described in
ADAPTIVE_TILING_FULL_SPEC.md §A.  Produces one JSON file per split (train /
val / test) in the LLaVA conversations format consumed by train_lora_understanding.py.

Output JSON schema (list of dicts, saved to <split>/<split>_counting.json):
    [
      {
        "image": "/abs/path/FSC147/images_384_VarV2/1234.jpg",
        "conversations": [
          {"from": "system", "value": "You are a helpful counting assistant. Answer with only a number."},
          {"from": "human",  "value": "<image>\\nHow many sea shells are present in this image? Answer with only a number."},
          {"from": "gpt",    "value": "136"}
        ]
      },
      ...
    ]

Usage:
    python3 scripts/experiment_lora_counting_sft/prepare_fsc147_splits.py \\
        --fsc147_root   /data/amondal/FSC147_hf \\
        --images_subdir images_384_VarV2 \\
        --output_root   outputs/experiment_lora_counting_sft
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


SYSTEM_PROMPT = "You are a helpful counting assistant. Answer with only a number."
USER_TMPL     = "<image>\nHow many {category} are present in this image? Answer with only a number."


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Category map
# ---------------------------------------------------------------------------

def load_category_map(txt_path: Path) -> Dict[str, str]:
    """Parse ImageClasses_FSC147.txt → {filename: category_string}."""
    cat: Dict[str, str] = {}
    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                cat[parts[0].strip()] = parts[1].strip()
            else:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    cat[parts[0].strip()] = parts[1].strip()
    return cat


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_split_entries(
    split_filenames: List[str],
    annotations: Dict[str, Any],
    category_map: Dict[str, str],
    images_dir: Path,
) -> List[Dict[str, Any]]:
    """Build LLaVA-style conversation entries for a single split.

    Each entry:
        {
            "image": "<abs_path>",
            "conversations": [
                {"from": "system", "value": "<SYSTEM_PROMPT>"},
                {"from": "human",  "value": "<image>\\nHow many ... ?"},
                {"from": "gpt",    "value": "<count>"}
            ]
        }
    """
    entries: List[Dict[str, Any]] = []
    missing_ann = missing_img = missing_cat = 0

    for fname in split_filenames:
        stem = Path(fname).stem
        ann  = (
            annotations.get(fname)
            or annotations.get(stem)
            or annotations.get(stem + ".jpg")
        )
        if ann is None:
            missing_ann += 1
            continue

        img_path = images_dir / fname
        if not img_path.exists():
            alt = images_dir / (stem + ".jpg")
            if alt.exists():
                img_path = alt
            else:
                missing_img += 1
                continue

        count    = len(ann.get("points", []))
        category = (
            category_map.get(fname)
            or category_map.get(stem + ".jpg")
            or "objects"
        )
        if fname not in category_map and stem + ".jpg" not in category_map:
            missing_cat += 1

        entries.append({
            "image": str(img_path.resolve()),
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human",  "value": USER_TMPL.format(category=category)},
                {"from": "gpt",    "value": str(count)},
            ],
        })

    print(
        f"  built {len(entries):5d} entries  |  "
        f"missing_ann={missing_ann}  missing_img={missing_img}  missing_cat={missing_cat}"
    )
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fsc147_root",
        default="/data/amondal/FSC147_hf",
        help="Root directory containing annotation_FSC147_384.json, "
             "Train_Test_Val_FSC_147.json, ImageClasses_FSC147.txt, "
             "and the images subfolder.",
    )
    p.add_argument(
        "--images_subdir",
        default="images_384_VarV2",
        help="Subdirectory under fsc147_root that holds the JPEG images.",
    )
    p.add_argument(
        "--output_root",
        default="outputs/experiment_lora_counting_sft",
        help="Root output directory (relative to repo root, or absolute).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test", "test_coco", "val_coco"],
        help="Which splits to generate.  Default: train val test",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root   = Path(__file__).resolve().parents[2]
    fsc_root    = Path(args.fsc147_root).resolve()
    images_dir  = fsc_root / args.images_subdir
    output_root = (
        Path(args.output_root)
        if Path(args.output_root).is_absolute()
        else repo_root / args.output_root
    )

    assert fsc_root.is_dir(),   f"FSC-147 root not found: {fsc_root}"
    assert images_dir.is_dir(), f"Images dir not found: {images_dir}"

    ann_path     = fsc_root / "annotation_FSC147_384.json"
    splits_path  = fsc_root / "Train_Test_Val_FSC_147.json"
    classes_path = fsc_root / "ImageClasses_FSC147.txt"
    for required in (ann_path, splits_path, classes_path):
        assert required.exists(), f"Required file not found: {required}"

    print(f"Loading annotations from {ann_path} …")
    annotations  = load_json(ann_path)
    print(f"  {len(annotations):,} annotated images")

    splits_data  = load_json(splits_path)
    category_map = load_category_map(classes_path)
    print(f"  {len(category_map):,} category entries")

    summary: List[str] = []
    for split in args.splits:
        filenames = splits_data.get(split)
        if filenames is None:
            print(f"[WARN] split '{split}' not found in splits JSON, skipping.")
            continue

        print(f"\n[{split.upper()}] {len(filenames)} images …")
        entries = build_split_entries(filenames, annotations, category_map, images_dir)

        out_path = output_root / split / f"{split}_counting.json"
        write_json(out_path, entries)
        print(f"  → {out_path}  ({len(entries)} entries)")
        summary.append(f"  {split:10s}: {len(entries):5d} entries  →  {out_path}")

    print("\n=== Summary ===")
    for line in summary:
        print(line)

    # Print one sample from train for verification
    train_json = output_root / "train" / "train_counting.json"
    if train_json.exists():
        data = load_json(train_json)
        print(f"\nSample entry (train[0]):\n{json.dumps(data[0], indent=4)}")

    # Count stats per split
    for split in args.splits:
        json_path = output_root / split / f"{split}_counting.json"
        if not json_path.exists():
            continue
        data   = load_json(json_path)
        counts = [int(e["conversations"][2]["value"]) for e in data]
        print(
            f"\n[{split}] count stats: "
            f"min={min(counts)}  max={max(counts)}  "
            f"mean={statistics.mean(counts):.1f}  median={statistics.median(counts)}"
        )


if __name__ == "__main__":
    main()
