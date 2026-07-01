#!/usr/bin/env python3
"""
Extract object_centers for all 49.8K records in balanced_mix_v3s.

Sources:
  1. FSC-147 (30.7K): Extract from annotation 'points' field
  2. UCount Part 1 (8.9K): Extract bbox centers from consolidated labels
  3. UCount Part 2 (34.1K): Extract bbox centers from metadata
  4. UCount Part 3 / SKU110k (7.4K): Extract bbox centers from labels
  5. ShanghaiTech (3.5K): Extract from ground-truth point annotations

Output: balanced_mix_v3s_with_centers.json (49.8K records)
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.io as sio

# =====================================================================
# 1. FSC-147 Extraction
# =====================================================================

def extract_fsc147_centers():
    """Extract object_centers from FSC-147 annotations."""
    fsc_annot_file = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"

    with open(fsc_annot_file) as f:
        fsc_annot = json.load(f)

    fsc_centers = {}
    for img_id, annot in fsc_annot.items():
        H, W = annot['H'], annot['W']
        points = annot['points']  # Already in image coordinates [x, y]

        # Normalize to [0, 1]
        normalized_centers = []
        for x, y in points:
            norm_x = x / W
            norm_y = y / H
            normalized_centers.append([norm_x, norm_y])

        # Map to image path format used in balanced_mix
        img_path = f"/data/amondal/FSC147_hf/images_384_VarV2/{img_id}"
        fsc_centers[img_path] = {
            'object_centers': normalized_centers,
            'H': H,
            'W': W,
            'source': 'FSC147'
        }

    print(f"FSC-147: Extracted {len(fsc_centers)} records with object_centers")
    return fsc_centers


# =====================================================================
# 2. UCount Part 1 Extraction
# =====================================================================

def extract_ucount_part1_centers():
    """Extract from UCount Part 1 bounding boxes."""
    part1_label_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part1/snapshots/*/dense_objects_labels.json"
    )

    if not part1_label_files:
        return {}

    with open(part1_label_files[0]) as f:
        labels = json.load(f)

    part1_centers = {}
    for item in labels:
        img_id = item['image_id']
        H = item['image_dimensions']['height']
        W = item['image_dimensions']['width']

        # Extract bbox centers
        centers = []
        for obj in item['objects']:
            bbox = obj['bbox']  # [left, top, right, bottom]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # Normalize to [0, 1]
            norm_cx = cx / W
            norm_cy = cy / H
            centers.append([norm_cx, norm_cy])

        # Use basename for matching (path-agnostic)
        basename = os.path.basename(img_id)
        part1_centers[basename] = {
            'object_centers': centers,
            'H': H,
            'W': W,
            'source': 'UCount_Part1'
        }

    print(f"UCount Part 1: Extracted {len(part1_centers)} records with object_centers")
    return part1_centers


# =====================================================================
# 3. UCount Part 2 Extraction
# =====================================================================

def extract_ucount_part2_centers():
    """Extract from UCount Part 2 JSONL metadata."""
    part2_meta_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part2/snapshots/*/metadata_updated.jsonl"
    )

    if not part2_meta_files:
        return {}

    part2_centers = {}
    with open(part2_meta_files[0]) as f:
        for line_no, line in enumerate(f):
            item = json.loads(line)

            file_name = item['file_name']
            H = item['image_dimensions']['height']
            W = item['image_dimensions']['width']

            # Extract bbox centers
            centers = []
            for obj in item['objects']:
                bbox = obj['bbox']  # [left, top, right, bottom]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                # Normalize
                norm_cx = cx / W
                norm_cy = cy / H
                centers.append([norm_cx, norm_cy])

            # Use basename for matching (path-agnostic)
            basename = os.path.basename(file_name)
            part2_centers[basename] = {
                'object_centers': centers,
                'H': H,
                'W': W,
                'source': 'UCount_Part2'
            }

    print(f"UCount Part 2: Extracted {len(part2_centers)} records with object_centers")
    return part2_centers


# =====================================================================
# 4. UCount Part 3 (SKU110k) Extraction
# =====================================================================

def extract_ucount_part3_centers():
    """Extract from UCount Part 3 / SKU110k."""
    part3_file = "/data/amondal/UniCountData/ucount_part3/sku110k_labels_fixed.json"

    with open(part3_file) as f:
        labels = json.load(f)

    part3_centers = {}
    for item in labels:
        img_id = item['image_id']
        H = item['image_dimensions']['height']
        W = item['image_dimensions']['width']

        # Extract bbox centers
        centers = []
        for obj in item['objects']:
            bbox = obj['bbox']  # [left, top, right, bottom]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # Normalize
            norm_cx = cx / W
            norm_cy = cy / H
            centers.append([norm_cx, norm_cy])

        # Use basename for matching (path-agnostic)
        basename = os.path.basename(img_id)
        part3_centers[basename] = {
            'object_centers': centers,
            'H': H,
            'W': W,
            'source': 'UCount_Part3_SKU110k'
        }

    print(f"UCount Part 3 (SKU110k): Extracted {len(part3_centers)} records with object_centers")
    return part3_centers


# =====================================================================
# 5. ShanghaiTech Extraction
# =====================================================================

def extract_shanghaitech_centers():
    """Extract from ShanghaiTech ground-truth annotations."""
    sha_centers = {}

    # Process both Part A and Part B
    for part in ['part_A', 'part_B']:
        root = f"/data/amondal/ShanghaiTech/{part}"
        gt_dir = os.path.join(root, "train_data", "ground-truth")
        img_dir = os.path.join(root, "train_data", "images")

        for mat_file in sorted(glob.glob(os.path.join(gt_dir, "GT_IMG_*.mat"))):
            img_name = os.path.basename(mat_file).replace("GT_IMG_", "").replace(".mat", ".jpg")
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            try:
                mat = sio.loadmat(mat_file)
                points = mat['image_info'][0, 0][0, 0][0]  # Shape: (n, 2)

                # Load image to get dimensions
                from PIL import Image
                img = Image.open(img_path)
                H, W = img.size[1], img.size[0]

                # Normalize points to [0, 1]
                centers = []
                for x, y in points:
                    norm_x = float(x) / W
                    norm_y = float(y) / H
                    centers.append([norm_x, norm_y])

                # Store with full absolute path
                sha_centers[img_path] = {
                    'object_centers': centers,
                    'H': H,
                    'W': W,
                    'source': f'ShanghaiTech_{part}'
                }
            except Exception as e:
                print(f"  [skip] {mat_file}: {e}")
                continue

    print(f"ShanghaiTech: Extracted {len(sha_centers)} records with object_centers")
    return sha_centers


# =====================================================================
# 6. Combine & Create Output Dataset
# =====================================================================

def create_balanced_mix_with_centers():
    """Load balanced_mix_v3s and augment with object_centers."""

    # Extract all sources
    print("\nExtracting object_centers from all sources...")
    print("=" * 70)

    fsc_centers = extract_fsc147_centers()
    part1_centers = extract_ucount_part1_centers()
    part2_centers = extract_ucount_part2_centers()
    part3_centers = extract_ucount_part3_centers()
    sha_centers = extract_shanghaitech_centers()

    # Combine
    all_centers = {}
    all_centers.update(fsc_centers)
    all_centers.update(part1_centers)
    all_centers.update(part2_centers)
    all_centers.update(part3_centers)
    all_centers.update(sha_centers)

    print(f"\nTotal extracted: {len(all_centers)} unique image paths")

    # Load balanced_mix and augment
    print("\nLoading balanced_mix_v3s...")
    with open("/data/amondal/UniCount/outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json") as f:
        balanced = json.load(f)

    print(f"Augmenting {len(balanced)} records with object_centers...")

    augmented = []
    missing_centers = []

    for record in balanced:
        img_path = record['image']
        basename = os.path.basename(img_path)

        if basename in all_centers:
            # Add object_centers and dimensions
            record['object_centers'] = all_centers[basename]['object_centers']
            record['H'] = all_centers[basename]['H']
            record['W'] = all_centers[basename]['W']
            record['annotation_source'] = all_centers[basename]['source']
            augmented.append(record)
        else:
            # Keep record without centers (will use CE loss only)
            augmented.append(record)
            missing_centers.append(img_path)

    print(f"Records with object_centers: {len(augmented) - len(missing_centers)}")
    print(f"Records without centers: {len(missing_centers)}")

    if missing_centers:
        print(f"\nMissing annotation examples:")
        for path in missing_centers[:5]:
            print(f"  {path}")

    # Save output
    output_file = "/data/amondal/UniCount/outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(augmented, f)

    print(f"Done! Saved {len(augmented)} records")

    # Print statistics
    with_centers = sum(1 for r in augmented if 'object_centers' in r)
    print(f"\n" + "=" * 70)
    print(f"FINAL STATISTICS")
    print(f"=" * 70)
    print(f"Total records: {len(augmented)}")
    print(f"Records with object_centers: {with_centers}")
    print(f"Coverage: {100*with_centers/len(augmented):.1f}%")

    # Breakdown by source
    sources = defaultdict(int)
    for r in augmented:
        if 'annotation_source' in r:
            sources[r['annotation_source']] += 1

    print(f"\nBy source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")


if __name__ == '__main__':
    create_balanced_mix_with_centers()
