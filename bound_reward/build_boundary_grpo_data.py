#!/usr/bin/env python3
"""
Build boundary-aware GRPO training data.

Each output record sends 4 quadrant crops (TL, TR, BL, BR) as separate images
to the model, matching exactly how CTAP partitions images at inference.

Crop layout (in normalized [0,1] coords):
  TL: cx<0.5, cy<0.5   TR: cx>=0.5, cy<0.5
  BL: cx<0.5, cy>=0.5  BR: cx>=0.5, cy>=0.5

Output: outputs/experiment_ctap_aware_pipeline/boundary_grpo_train.jsonl

Usage:
    python3 bound_reward/build_boundary_grpo_data.py [--max_samples N] [--sht_only]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bound_reward"))

from prompt_template import CountingPromptBuilder  # noqa: E402

BALANCED_MIX = (
    REPO / "outputs/experiment_lora_counting_sft/balanced_mix_v3s"
    / "balanced_mix_train_with_centers.json"
)
SHT_A_JSON = REPO / "data/attn_regularizer_dataset/attn_regularizer_shanghaitech_a.json"
SHT_B_JSON = REPO / "data/attn_regularizer_dataset/attn_regularizer_shanghaitech_b.json"

CROP_DIR = REPO / "data/ctap_aware_pipeline/boundary_crops"
OUT_DIR   = REPO / "outputs/experiment_ctap_aware_pipeline"

_BUILDER = CountingPromptBuilder()
_CAT_RE  = re.compile(r"How many ([\w\s]+?) (?:are|is)\b", re.IGNORECASE)


def _extract_category(conversations: list) -> str:
    for turn in conversations:
        if turn.get("from") == "human":
            m = _CAT_RE.search(turn.get("value", ""))
            if m:
                return m.group(1).strip().lower()
    return "objects"


def _prompt_text(category: str) -> str:
    pkg = _BUILDER.build(category=category)
    return pkg["system"] + "\n\n" + pkg["user"]


def _quadrant_counts(centers: list[list[float]]) -> dict[str, int]:
    """Assign each object center to exactly one quadrant."""
    counts = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
    for cx, cy in centers:
        row = "bottom" if cy >= 0.5 else "top"
        col = "right"  if cx >= 0.5 else "left"
        counts[f"{row}_{col}"] += 1
    return counts


def _save_crops(pil: Image.Image, save_dir: Path, uid: str) -> list[str]:
    """
    Crop image into 4 quadrants, save as JPEG, return list of absolute paths
    in reading order: [TL, TR, BL, BR].
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    W, H = pil.size
    mx, my = W // 2, H // 2
    boxes = [
        (0,  0,  mx, my),   # top_left
        (mx, 0,  W,  my),   # top_right
        (0,  my, mx, H),    # bottom_left
        (mx, my, W,  H),    # bottom_right
    ]
    tags = ["tl", "tr", "bl", "br"]
    paths = []
    for box, tag in zip(boxes, tags):
        crop = pil.crop(box).convert("RGB")
        p = save_dir / f"{uid}_{tag}.jpg"
        crop.save(str(p), format="JPEG", quality=90)
        paths.append(str(p))
    return paths


def _make_record(
    uid: str,
    gt_count: int,
    image_path: str,
    category: str,
    save_dir: Path,
) -> dict | None:
    try:
        pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [skip] cannot open {image_path}: {e}")
        return None

    crop_paths = _save_crops(pil, save_dir, uid)
    text = _prompt_text(category)

    return {
        "id": uid,
        "gt_count": gt_count,
        "prompt": [{
            "role": "user",
            "content": [
                {"type": "image", "url": crop_paths[0]},  # TL
                {"type": "image", "url": crop_paths[1]},  # TR
                {"type": "image", "url": crop_paths[2]},  # BL
                {"type": "image", "url": crop_paths[3]},  # BR
                {"type": "text",  "text": text},
            ],
        }],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=0, help="Cap total records (0=all)")
    ap.add_argument("--sht_only", action="store_true", help="Skip balanced_mix, use only SHT")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "boundary_grpo_train.jsonl"
    random.seed(args.seed)

    records: list[dict] = []
    n_skip = 0

    # ── balanced_mix_v3s ──────────────────────────────────────────────────────
    if not args.sht_only:
        print("Loading balanced_mix …")
        bm_rows = json.loads(BALANCED_MIX.read_text())
        save_dir = CROP_DIR / "bmix"
        for i, row in enumerate(bm_rows):
            gt_str = ""
            for turn in row.get("conversations", []):
                if turn.get("from") == "gpt":
                    gt_str = turn.get("value", "0")
                    break
            try:
                gt_count = int(str(gt_str).strip())
            except ValueError:
                n_skip += 1
                continue

            category = _extract_category(row.get("conversations", []))
            uid = f"bmix_{i:06d}"
            rec = _make_record(uid, gt_count, row["image"], category, save_dir)
            if rec is None:
                n_skip += 1
            else:
                records.append(rec)

        print(f"  balanced_mix : {len(bm_rows):,} input → {len(records):,} valid  (skip={n_skip})")

    # ── SHT-A ─────────────────────────────────────────────────────────────────
    print("Loading SHT-A …")
    sht_a_rows = json.loads(SHT_A_JSON.read_text())
    save_dir = CROP_DIR / "sht_a"
    n_before = len(records)
    for i, row in enumerate(sht_a_rows):
        uid = f"shta_{i:05d}"
        rec = _make_record(uid, int(row["count"]), row["image"],
                           row.get("category", "people").lower(), save_dir)
        if rec is None:
            n_skip += 1
        else:
            records.append(rec)
    print(f"  SHT-A        : {len(sht_a_rows):,} input → {len(records)-n_before:,} valid")

    # ── SHT-B ─────────────────────────────────────────────────────────────────
    print("Loading SHT-B …")
    sht_b_rows = json.loads(SHT_B_JSON.read_text())
    save_dir = CROP_DIR / "sht_b"
    n_before = len(records)
    for i, row in enumerate(sht_b_rows):
        uid = f"shtb_{i:05d}"
        rec = _make_record(uid, int(row["count"]), row["image"],
                           row.get("category", "people").lower(), save_dir)
        if rec is None:
            n_skip += 1
        else:
            records.append(rec)
    print(f"  SHT-B        : {len(sht_b_rows):,} input → {len(records)-n_before:,} valid")

    random.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nTotal records  : {len(records):,}  (skipped {n_skip})")
    print(f"Crops saved to : {CROP_DIR}")
    print(f"Output         : {out_path}")


if __name__ == "__main__":
    main()
