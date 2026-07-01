#!/usr/bin/env python3
"""
Build pairwise overlap GRPO training data.

For each image, generates 4 pairwise records (one per shared edge):
  TL↔TR  vertical cut,   top half
  TL↔BL  horizontal cut, left half
  TR↔BR  horizontal cut, right half
  BL↔BR  vertical cut,   bottom half

GT overlap = number of objects within MARGIN of the cut line on the correct side.

Uses existing quadrant crops from boundary_crops/{bmix,sht_a,sht_b}/.
Does NOT re-open or re-crop images.

Output: outputs/experiment_ctap_aware_pipeline/pairwise_overlap_grpo_train.jsonl

Usage:
    python3 bound_reward/build_pairwise_overlap_grpo_data.py [--max_samples N] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BALANCED_MIX = (
    REPO / "outputs/experiment_lora_counting_sft/balanced_mix_v3s"
    / "balanced_mix_train_with_centers.json"
)
SHT_A_JSON = REPO / "data/attn_regularizer_dataset/attn_regularizer_shanghaitech_a.json"
SHT_B_JSON = REPO / "data/attn_regularizer_dataset/attn_regularizer_shanghaitech_b.json"

CROP_DIR = REPO / "data/ctap_aware_pipeline/boundary_crops"
OUT_DIR   = REPO / "outputs/experiment_ctap_aware_pipeline"

MARGIN = 0.04   # normalized coords — objects within 4% of cut line

# (tag_a, tag_b, axis, side_for_membership, label_a, label_b, orientation)
EDGE_PAIRS = [
    # tag_a, tag_b, cut_axis, membership_axis, cut_side, label_a, label_b
    ("tl", "tr", "x", "y", "lt",  "Top-Left",    "Top-Right"),
    ("tl", "bl", "y", "x", "lt",  "Top-Left",    "Bottom-Left"),
    ("tr", "br", "y", "x", "gte", "Top-Right",   "Bottom-Right"),
    ("bl", "br", "x", "y", "gte", "Bottom-Left", "Bottom-Right"),
]
# cut_axis="x" means the cut is at x=0.5 (vertical line), check |cx-0.5|<MARGIN
# membership_axis="y" means side selection uses y coord
# cut_side="lt" means cy<0.5 (top half), "gte" means cy>=0.5 (bottom half)


def _count_overlap(
    centers: list,
    cut_axis: str,
    membership_axis: str,
    cut_side: str,
) -> int:
    """Count objects in the margin zone of one shared edge."""
    n = 0
    for pt in centers:
        cx, cy = pt[0], pt[1]
        cut_val   = cx if cut_axis == "x" else cy
        member_val = cy if membership_axis == "y" else cx

        if abs(cut_val - 0.5) >= MARGIN:
            continue
        if cut_side == "lt" and member_val >= 0.5:
            continue
        if cut_side == "gte" and member_val < 0.5:
            continue
        n += 1
    return n


def _prompt_text(label_a: str, label_b: str) -> str:
    return (
        f"Image 1 is the {label_a} quadrant crop. "
        f"Image 2 is the {label_b} quadrant crop. "
        "These two crops share one edge. "
        "Count the objects that appear visually cut/truncated at that shared edge "
        "(objects straddling the boundary between the two crops). "
        'Output only valid JSON: {"overlap": N}'
    )


def _make_pairs(
    uid: str,
    centers: list,
    crop_dir: Path,
) -> list[dict]:
    """Generate up to 4 pairwise records for one image."""
    records = []
    for tag_a, tag_b, cut_ax, mem_ax, side, lbl_a, lbl_b in EDGE_PAIRS:
        path_a = crop_dir / f"{uid}_{tag_a}.jpg"
        path_b = crop_dir / f"{uid}_{tag_b}.jpg"
        if not path_a.exists() or not path_b.exists():
            continue

        gt = _count_overlap(centers, cut_ax, mem_ax, side)
        pair_id = f"{uid}_{tag_a}_{tag_b}"

        records.append({
            "id":       pair_id,
            "gt_count": gt,
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image", "url": str(path_a)},
                    {"type": "image", "url": str(path_b)},
                    {"type": "text",  "text": _prompt_text(lbl_a, lbl_b)},
                ],
            }],
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples",    type=int,   default=0,    help="Cap total records (0=all)")
    ap.add_argument("--neg_keep_rate",  type=float, default=0.15, help="Fraction of zero-overlap pairs to keep (bmix only)")
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pairwise_overlap_grpo_train.jsonl"
    rng = random.Random(args.seed)

    all_records: list[dict] = []
    n_zero_dropped = 0

    # ── balanced_mix_v3s (bmix) ──────────────────────────────────────────────
    print("Loading balanced_mix …")
    bm_rows = json.loads(BALANCED_MIX.read_text())
    bmix_dir = CROP_DIR / "bmix"
    n_before = 0
    for i, row in enumerate(bm_rows):
        uid     = f"bmix_{i:06d}"
        centers = row.get("object_centers", [])
        if not centers:
            continue
        pairs = _make_pairs(uid, centers, bmix_dir)
        for p in pairs:
            if p["gt_count"] == 0 and rng.random() > args.neg_keep_rate:
                n_zero_dropped += 1
                continue
            all_records.append(p)
    print(f"  balanced_mix : {len(bm_rows):,} images → {len(all_records)-n_before:,} pairs  "
          f"(zero-overlap dropped: {n_zero_dropped})")

    def _load_sht(json_path: Path, prefix: str, crop_subdir: str) -> int:
        rows = json.loads(json_path.read_text())
        sht_dir = CROP_DIR / crop_subdir
        n_added = 0
        for i, row in enumerate(rows):
            uid     = f"{prefix}_{i:05d}"
            centers = row.get("object_centers", [])
            if not centers:
                continue
            pairs = _make_pairs(uid, centers, sht_dir)
            # Keep all SHT pairs (dense crowd → overlap almost always > 0)
            all_records.extend(pairs)
            n_added += len(pairs)
        return n_added

    # ── SHT-A ─────────────────────────────────────────────────────────────────
    print("Loading SHT-A …")
    n_shta = _load_sht(SHT_A_JSON, "shta", "sht_a")
    print(f"  SHT-A  : {n_shta:,} pairs added")

    # ── SHT-B ─────────────────────────────────────────────────────────────────
    print("Loading SHT-B …")
    n_shtb = _load_sht(SHT_B_JSON, "shtb", "sht_b")
    print(f"  SHT-B  : {n_shtb:,} pairs added")

    rng.shuffle(all_records)
    if args.max_samples > 0:
        all_records = all_records[: args.max_samples]

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    gt_vals = [r["gt_count"] for r in all_records]
    n_pos = sum(1 for g in gt_vals if g > 0)
    n_zero = sum(1 for g in gt_vals if g == 0)
    print(f"\nTotal pairs    : {len(all_records):,}")
    print(f"  overlap > 0  : {n_pos:,}")
    print(f"  overlap == 0 : {n_zero:,}")
    if gt_vals:
        print(f"  mean overlap : {sum(gt_vals)/len(gt_vals):.2f}  "
              f"max: {max(gt_vals)}")
    print(f"Output         : {out_path}")


if __name__ == "__main__":
    main()
