#!/usr/bin/env python3
"""Step 1: build the FSC-147 CRCO ranking JSON.

For each FSC-147 train category, partition images by GT count into K=4
quartiles and emit N ranking samples per category by drawing one image per
quartile, shuffling the display order, and emitting the ascending-count
ranking string as the supervision target.

Output schema (list of dicts, JSON):
    {
      "type": "ranking",
      "category": "<category name>",
      "image": ["<abs/path/to/img1.jpg>", ...4...],
      "counts": [c1, c2, c3, c4],
      "conversations": [
          {"from": "system", "value": "You are a helpful counting assistant."},
          {"from": "human",  "value": "<image>\\n<image>\\n<image>\\n<image>\\nGiven four images, ..."},
          {"from": "gpt",    "value": "Image 4 < Image 2 < Image 1 < Image 3"}
      ]
    }

The conversations field is the same shape consumed by the existing
``CountingSFTDataset`` so the dataloader can dispatch ranking and counting
samples through the same tokenisation path (one ``<image>`` placeholder per
image, expanded to the InternVL ``<IMG_CONTEXT>`` block).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

K = 4  # quartiles per category
SYSTEM_PROMPT = "You are a helpful counting assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc_root", type=str, default="../FSC147_hf",
                        help="Directory containing Train_Test_Val_FSC_147.json, "
                             "annotation_FSC147_384.json, ImageClasses_FSC147.txt, "
                             "images_384_VarV2/")
    parser.add_argument("--images_subdir", type=str, default="images_384_VarV2")
    parser.add_argument("--n_per_category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=str,
                        default="crco/data/fsc147_crco_ranking.json")
    parser.add_argument("--strict_existence", action="store_true", default=True,
                        help="Skip rows whose image files do not exist on disk.")
    return parser.parse_args()


def load_categories(path: Path) -> Dict[str, str]:
    """Parse ImageClasses_FSC147.txt — '<image_id>.jpg<TAB><category>' per line."""
    cats: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # Tab- or whitespace-separated; only first split matters.
            parts = line.split("\t")
            if len(parts) != 2:
                # Some distributions use multiple spaces.
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            img_id, cat = parts[0].strip(), parts[1].strip()
            cats[img_id] = cat
    return cats


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    fsc_root = Path(args.fsc_root).resolve()
    images_dir = fsc_root / args.images_subdir
    splits_path = fsc_root / "Train_Test_Val_FSC_147.json"
    ann_path = fsc_root / "annotation_FSC147_384.json"
    cats_path = fsc_root / "ImageClasses_FSC147.txt"
    for required in (splits_path, ann_path, cats_path, images_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required FSC-147 input: {required}")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train_ids: List[str] = splits["train"]
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = load_categories(cats_path)

    gt_counts: Dict[str, int] = {}
    missing_ann: List[str] = []
    for img_id in train_ids:
        rec = ann.get(img_id)
        if rec is None or "points" not in rec:
            missing_ann.append(img_id)
            continue
        gt_counts[img_id] = len(rec["points"])

    images_per_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_cat = 0
    missing_file = 0
    for img_id, count in gt_counts.items():
        cat = categories.get(img_id)
        if cat is None:
            missing_cat += 1
            continue
        img_path = images_dir / img_id
        if args.strict_existence and not img_path.exists():
            missing_file += 1
            continue
        images_per_cat[cat].append({
            "image_id": img_id,
            "path": str(img_path),
            "count": count,
        })

    skipped_lt_k = 0
    skipped_const = 0
    skipped_empty_bin = 0
    samples: List[Dict[str, Any]] = []
    cat_count_ranges: List[int] = []

    for cat, imgs in sorted(images_per_cat.items()):
        if len(imgs) < K:
            skipped_lt_k += 1
            continue
        counts = [im["count"] for im in imgs]
        c_min, c_max = min(counts), max(counts)
        if c_max == c_min:
            skipped_const += 1
            continue
        step = (c_max - c_min) / K
        bins: List[List[Dict[str, Any]]] = [[] for _ in range(K)]
        for im in imgs:
            idx = min(int((im["count"] - c_min) / step), K - 1)
            bins[idx].append(im)
        if any(len(b) == 0 for b in bins):
            skipped_empty_bin += 1
            continue
        cat_count_ranges.append(c_max - c_min)

        for _ in range(args.n_per_category):
            selected = [rng.choice(b) for b in bins]
            order = list(range(K))
            rng.shuffle(order)
            shuffled = [selected[i] for i in order]

            sorted_indices = sorted(range(K), key=lambda i: shuffled[i]["count"])
            ranking_str = " < ".join(f"Image {i + 1}" for i in sorted_indices)

            instruction = (
                f"Given four images, rank them in ascending order based on "
                f"their counts of {cat}."
            )
            human_text = (
                "<image>\n<image>\n<image>\n<image>\n" + instruction
            )

            samples.append({
                "type": "ranking",
                "category": cat,
                "image": [im["path"] for im in shuffled],
                "counts": [im["count"] for im in shuffled],
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": ranking_str},
                ],
            })

    rng.shuffle(samples)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    avg_range = (sum(cat_count_ranges) / len(cat_count_ranges)) if cat_count_ranges else 0.0
    print(f"[CRCO] Wrote {len(samples)} ranking samples to {out_path}")
    print(f"[CRCO] Categories used:               {len(cat_count_ranges)}")
    print(f"[CRCO] Categories skipped (<{K} images):  {skipped_lt_k}")
    print(f"[CRCO] Categories skipped (constant):  {skipped_const}")
    print(f"[CRCO] Categories skipped (empty bin): {skipped_empty_bin}")
    print(f"[CRCO] Train images missing category:  {missing_cat}")
    print(f"[CRCO] Train images missing on disk:   {missing_file}")
    print(f"[CRCO] Train images missing annotation:{len(missing_ann)}")
    print(f"[CRCO] Avg per-category count range:   {avg_range:.1f}")

    # Sanity-check 5 random samples: response order matches counts ascending.
    if samples:
        for sample in rng.sample(samples, k=min(5, len(samples))):
            counts = sample["counts"]
            tokens = [tok.strip() for tok in sample["conversations"][2]["value"].split("<")]
            order = [int(tok.split()[-1]) - 1 for tok in tokens if tok.startswith("Image")]
            ascending_counts = [counts[i] for i in order]
            assert ascending_counts == sorted(ascending_counts), (
                f"Order check failed for sample with counts={counts}, order={order}"
            )
        print("[CRCO] Spot-check on 5 random samples: PASS")


if __name__ == "__main__":
    main()
