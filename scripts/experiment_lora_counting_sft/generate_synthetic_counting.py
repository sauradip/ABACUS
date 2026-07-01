#!/usr/bin/env python3
"""
Generate synthetic counting images: colored circles on a white canvas.

Reference:
  "Counting Circuits" (arXiv 2603.18523, March 2026) — fine-tuning on simple
  synthetic dot images with a uniform count distribution recalibrates a VLM's
  numerical output mapping without disturbing learned visual features.

Spec:
  - 448x448 PNG, white background.
  - All circles in a single image share the same radius (5..15 px) so the
    model cannot use size variation as a counting shortcut.
  - Each circle gets a random RGB colour in [30, 230] per channel.
  - Circle centres are non-overlapping (min_dist >= 2*radius).
  - Counts are sampled uniformly inside 7 buckets, 1,200 images per bucket
    (8,400 images total).
  - For very dense buckets (501..800), if placement saturates we shrink the
    radius / relax the min-distance, and finally accept whatever was placed
    (its actual_count becomes the GT).

Outputs:
  /data/amondal/UniCountData/synthetic_dots/images/synthetic_NNNNN.png
  /data/amondal/UniCount/data/synthetic_dots_train.json
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ─── Repro ─────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ─────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("/data/amondal/UniCountData/synthetic_dots/images")
OUTPUT_JSON = Path("/data/amondal/UniCount/data/synthetic_dots_train.json")
REFERENCE_JSON = Path(
    "/data/amondal/UniCount/outputs/experiment_lora_counting_sft/train/train_counting.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ─── Spec ──────────────────────────────────────────────────────────────────
CANVAS_SIZE = 448
BUCKETS = [
    (1, 10),
    (11, 30),
    (31, 70),
    (71, 150),
    (151, 300),
    (301, 500),
    (501, 800),
]
N_PER_BUCKET = 1200

# ─── Placement ─────────────────────────────────────────────────────────────

def _try_place(n_dots: int, radius: int, min_dist_factor: float,
               canvas: int = CANVAS_SIZE,
               max_attempts: int = 1000):
    """Sample n_dots non-overlapping centres at the given radius.

    Returns the list of (x, y) actually placed. Uses a uniform grid hash
    to make overlap checking O(1) on average so high counts are tractable.
    """
    min_dist = min_dist_factor * radius
    min_dist_sq = min_dist * min_dist
    cell = max(1, int(min_dist))           # one cell ~ one disc diameter
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}

    positions: list[tuple[int, int]] = []
    lo, hi = radius, canvas - radius - 1
    if hi <= lo:
        return positions

    for _ in range(n_dots):
        placed = False
        for _att in range(max_attempts):
            x = random.randint(lo, hi)
            y = random.randint(lo, hi)
            cx, cy = x // cell, y // cell
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = grid.get((cx + dx, cy + dy))
                    if not bucket:
                        continue
                    for px, py in bucket:
                        ddx = x - px
                        ddy = y - py
                        if ddx * ddx + ddy * ddy < min_dist_sq:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                positions.append((x, y))
                grid.setdefault((cx, cy), []).append((x, y))
                placed = True
                break
        if not placed:
            break
    return positions


def generate_dot_image(target_count: int, canvas_size: int = CANVAS_SIZE,
                       high_density: bool = False):
    """Generate a white-background image with up-to target_count circles.

    Returns (PIL.Image, actual_count). actual_count may be < target_count
    when placement saturates (very dense buckets).
    """
    # Pick radius. For dense buckets, prefer smaller circles.
    if high_density:
        radius = random.randint(3, 6)
        min_dist_factor = 1.5
    else:
        radius = random.randint(5, 15)
        min_dist_factor = 2.0

    positions = _try_place(target_count, radius, min_dist_factor, canvas_size)

    # If placement saturated, progressively shrink radius / relax spacing.
    while len(positions) < target_count and (radius > 3 or min_dist_factor > 1.2):
        if min_dist_factor > 1.5:
            min_dist_factor = max(1.5, min_dist_factor - 0.25)
        elif radius > 3:
            radius -= 1
        else:
            min_dist_factor = max(1.2, min_dist_factor - 0.15)
        positions = _try_place(target_count, radius, min_dist_factor, canvas_size)

    # Render
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x, y in positions:
        color = (
            random.randint(30, 230),
            random.randint(30, 230),
            random.randint(30, 230),
        )
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    return img, len(positions)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ref_data = json.load(open(REFERENCE_JSON))
    ref_sample = ref_data[0]

    system_prompt = None
    ref_human = None
    for conv in ref_sample["conversations"]:
        if conv["from"] == "system":
            system_prompt = conv["value"]
        elif conv["from"] == "human":
            ref_human = conv["value"]
    print(f"Reference system: {system_prompt}")
    print(f"Reference human : {ref_human[:80]!r}")

    # Use the same '<image>\n' placeholder format as reference.
    HUMAN_TEMPLATE = "<image>\nHow many circles are present in this image? Answer with only a number."

    samples: list[dict] = []
    counts: list[int] = []
    total = 0
    t0 = time.time()

    for bi, (lo, hi) in enumerate(BUCKETS):
        high_density = (lo >= 301)
        bucket_actual: list[int] = []
        bucket_target: list[int] = []
        for _ in range(N_PER_BUCKET):
            target = random.randint(lo, hi)
            img, actual = generate_dot_image(target, CANVAS_SIZE, high_density=high_density)
            if actual < 1:
                continue
            fname = f"synthetic_{total:05d}.png"
            fpath = OUTPUT_DIR / fname
            img.save(fpath)
            samples.append({
                "image": str(fpath),
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human",  "value": HUMAN_TEMPLATE},
                    {"from": "gpt",    "value": str(actual)},
                ],
            })
            counts.append(actual)
            bucket_actual.append(actual)
            bucket_target.append(target)
            total += 1
        bt = np.asarray(bucket_target)
        ba = np.asarray(bucket_actual)
        sat = float((ba < bt).mean()) if len(ba) else 0.0
        elapsed = time.time() - t0
        print(f"  bucket [{lo:>4},{hi:>4}]: n={len(bucket_actual):>5} "
              f"target_mean={bt.mean():.0f} actual_mean={ba.mean():.0f} "
              f"saturated_frac={sat:.2%}  elapsed={elapsed:.0f}s")

    random.shuffle(samples)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(samples, f)

    c = np.asarray(counts)
    print(f"\nGenerated {len(samples)} synthetic samples in {time.time() - t0:.0f}s")
    print(f"Count range: [{c.min()}, {c.max()}]  mean={c.mean():.1f}  median={np.median(c):.0f}")
    print("Per-bucket histogram of ACTUAL counts:")
    for lo, hi in BUCKETS:
        m = (c >= lo) & (c <= hi)
        print(f"  [{lo:>4},{hi:>4}]: {m.sum():>5} images, mean={c[m].mean() if m.any() else 0:.1f}")

    # Sanity spot-check
    for s in random.sample(samples, 3):
        assert os.path.exists(s["image"]), f"Missing {s['image']}"
        im = Image.open(s["image"])
        assert im.size == (CANVAS_SIZE, CANVAS_SIZE), im.size
        print(f"  spot-check {s['image']} -> GT={s['conversations'][-1]['value']}, size={im.size}")


if __name__ == "__main__":
    main()
