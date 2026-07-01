#!/usr/bin/env python3
"""
Visualize extracted object_centers with overlay on images.
Sample 10 images from each matched source and save visualization grid.
"""

import json
import random
import os
from PIL import Image, ImageDraw
import numpy as np

# Load augmented data
with open("outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json") as f:
    augmented = json.load(f)

# Group by source
by_source = {}
for record in augmented:
    if 'object_centers' in record:
        src = record.get('source', 'unknown')
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(record)

print(f"Records by source: {[(k, len(v)) for k, v in by_source.items()]}")

# Visualize 10 samples per source
output_dir = "outputs/experiment_lora_counting_sft/eval/extraction_visualizations"
os.makedirs(output_dir, exist_ok=True)

for source, records in sorted(by_source.items()):
    print(f"\n{source}: Processing {min(10, len(records))} samples...")

    samples = random.sample(records, min(10, len(records)))

    for idx, record in enumerate(samples):
        img_path = record['image']

        if not os.path.exists(img_path):
            print(f"  Skip {idx+1}: Image not found: {img_path}")
            continue

        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            W_img, H_img = img.size

            # Draw centers
            draw = ImageDraw.Draw(img)
            centers = record['object_centers']

            for cx_norm, cy_norm in centers:
                # Convert normalized to pixel coords
                px = cx_norm * (W_img - 1)
                py = cy_norm * (H_img - 1)

                # Draw circle
                r = 5
                draw.ellipse([px-r, py-r, px+r, py+r], fill='red', outline='yellow')

            # Add text
            count = len(centers)
            draw.text((10, 10), f"{source} | Count: {count}", fill='white')

            # Save
            fname = f"{source.replace('_', '-').lower()}-{idx+1:02d}.jpg"
            out_path = os.path.join(output_dir, fname)
            img.save(out_path)
            print(f"  {idx+1}: Saved {fname} ({count} objects)")

        except Exception as e:
            print(f"  {idx+1}: Error - {e}")

print(f"\nVisualizations saved to {output_dir}")
print(f"View with: ls -lh {output_dir}")
