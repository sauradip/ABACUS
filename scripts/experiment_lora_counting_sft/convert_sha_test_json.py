#!/usr/bin/env python3
"""Convert ShanghaiTech test data (.mat ground truth) to LLaVA-format JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import scipy.io as sio


def create_sha_test_json(part: str, output_json: str) -> None:
    """Create LLaVA-format JSON for ShanghaiTech part_A or part_B test split.

    Reads:
      /data/amondal/ShanghaiTech/part_{part}/test_data/images/*.jpg
      /data/amondal/ShanghaiTech/part_{part}/test_data/ground-truth/GT_IMG_*.mat

    Writes LLaVA-format JSON with image paths and ground truth counts.
    """
    part = part.upper()  # A or B
    if part not in ("A", "B"):
        raise ValueError(f"part must be 'A' or 'B', got {part}")

    base = Path(f"/data/amondal/ShanghaiTech/part_{part}/test_data")
    images_dir = base / "images"
    gt_dir = base / "ground-truth"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    rows = []

    for img_path in sorted(images_dir.glob("IMG_*.jpg")):
        img_num = img_path.stem.replace("IMG_", "")
        gt_path = gt_dir / f"GT_IMG_{img_num}.mat"

        if not gt_path.exists():
            print(f"[WARN] Missing ground truth for {img_path.name}")
            continue

        try:
            mat = sio.loadmat(str(gt_path))
            # ShanghaiTech .mat files have image_info with structured dtype
            # Structure: [('location', 'O'), ('number', 'O')]
            if "image_info" in mat:
                info = mat["image_info"][0, 0]
                if hasattr(info, "dtype") and "number" in info.dtype.names:
                    count = info["number"][0, 0]
                    if hasattr(count, "item"):
                        count = int(count.item())
                    else:
                        count = int(count)
                else:
                    # Fallback: count annotation points
                    points = info[0]
                    if hasattr(points, "shape"):
                        count = len(points)
                    else:
                        print(f"[WARN] Unexpected structure in {gt_path}")
                        continue
            else:
                print(f"[WARN] No image_info key in {gt_path}")
                continue
        except Exception as e:
            print(f"[WARN] Error reading {gt_path}: {e}")
            continue

        rows.append({
            "image": str(img_path),
            "conversations": [
                {
                    "from": "system",
                    "value": "You are a helpful counting assistant. Answer with only a number.",
                },
                {
                    "from": "human",
                    "value": "How many people are present in this image? Answer with only a number.",
                },
                {
                    "from": "gpt",
                    "value": str(count),
                },
            ],
        })

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"Created {output_json}: {len(rows)} samples")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert ShanghaiTech test data to LLaVA JSON")
    ap.add_argument("--part", required=True, choices=["A", "B"], help="ShanghaiTech part")
    ap.add_argument("--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    create_sha_test_json(args.part, args.output)
