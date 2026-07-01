#!/usr/bin/env python3
"""
Visualize 10 random samples from balanced_mix_train_with_centers.json
with extracted object_centers overlaid on images.
"""

import json
import random
import os
from PIL import Image, ImageDraw
from collections import defaultdict

# Load augmented data
print("Loading augmented dataset...")
with open("outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json") as f:
    augmented = json.load(f)

# Group by source
by_source = defaultdict(list)
for record in augmented:
    if 'object_centers' in record:
        src = record.get('source', 'unknown')
        by_source[src].append(record)

print(f"Records by source: {[(k, len(v)) for k, v in sorted(by_source.items())]}")

# Visualize samples per source
output_dir = "outputs/experiment_lora_counting_sft/eval/extraction_samples_final"
os.makedirs(output_dir, exist_ok=True)

print(f"\nGenerating 10 samples per source...")
print("=" * 70)

for source in sorted(by_source.keys()):
    records = by_source[source]
    samples = random.sample(records, min(10, len(records)))

    print(f"\n{source}: {len(samples)} samples")

    for idx, record in enumerate(samples):
        img_path = record['image']

        if not os.path.exists(img_path):
            print(f"  [{idx+1:2d}] SKIP: Image not found: {img_path}")
            continue

        try:
            # Load and draw
            img = Image.open(img_path).convert('RGB')
            W_img, H_img = img.size
            draw = ImageDraw.Draw(img)

            centers = record['object_centers']
            count = len(centers)

            # Draw points
            for cx_norm, cy_norm in centers:
                px = cx_norm * (W_img - 1)
                py = cy_norm * (H_img - 1)
                r = 5
                draw.ellipse([px-r, py-r, px+r, py+r], fill='red', outline='yellow')

            # Add text overlay
            draw.text((10, 10), f"{source} | Count: {count} | Size: {W_img}×{H_img}", fill='white')

            # Save
            fname = f"{source.replace('_', '-').lower()}-{idx+1:02d}.jpg"
            out_path = os.path.join(output_dir, fname)
            img.save(out_path)
            print(f"  [{idx+1:2d}] {fname} ({count} objects)")

        except Exception as e:
            print(f"  [{idx+1:2d}] ERROR: {e}")

print(f"\n{'=' * 70}")
print(f"Visualizations saved to: {output_dir}")
print(f"View with: ls -lh {output_dir}")
