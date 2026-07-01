#!/usr/bin/env python3
"""
Smart object_centers extraction matching balanced_mix_v3s paths exactly.
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.io as sio

def extract_fsc147_centers():
    """Extract from FSC-147."""
    fsc_annot_file = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"

    with open(fsc_annot_file) as f:
        fsc_annot = json.load(f)

    fsc_centers = {}
    for img_id, annot in fsc_annot.items():
        H, W = annot['H'], annot['W']
        points = annot['points']

        normalized_centers = []
        for x, y in points:
            norm_x = x / W
            norm_y = y / H
            normalized_centers.append([norm_x, norm_y])

        # Store with full path as it appears in balanced_mix
        img_path = f"/data/amondal/FSC147_hf/images_384_VarV2/{img_id}"
        fsc_centers[img_path] = {
            'object_centers': normalized_centers,
            'H': H,
            'W': W,
            'source': 'FSC147'
        }

    print(f"FSC-147: Extracted {len(fsc_centers)} records")
    return fsc_centers


def extract_ucount_consolidated():
    """Extract from UCount Parts 1, 2, 3 with path-aware matching."""
    part1_label_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part1/snapshots/*/dense_objects_labels.json"
    )
    part2_meta_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part2/snapshots/*/metadata_updated.jsonl"
    )
    part3_file = "/data/amondal/UniCountData/ucount_part3/sku110k_labels_fixed.json"

    ucount_centers = {}

    # Part 1
    if part1_label_files:
        with open(part1_label_files[0]) as f:
            labels = json.load(f)
        for item in labels:
            img_id = item['image_id']
            H = item['image_dimensions']['height']
            W = item['image_dimensions']['width']
            centers = []
            for obj in item['objects']:
                bbox = obj['bbox']
                cx = (bbox[0] + bbox[2]) / 2.0 / W
                cy = (bbox[1] + bbox[3]) / 2.0 / H
                centers.append([cx, cy])

            # Try multiple path patterns
            for pattern in [
                f"/data/amondal/UniCountData/ucount_consolidated/images/part1/{img_id}",
                f"/data/amondal/UniCountData/ucount_consolidated/{img_id}",
            ]:
                ucount_centers[pattern] = {
                    'object_centers': centers,
                    'H': H,
                    'W': W,
                    'source': 'UCount_Part1'
                }

    # Part 2
    if part2_meta_files:
        with open(part2_meta_files[0]) as f:
            for line in f:
                item = json.loads(line)
                file_name = item['file_name']
                H = item['image_dimensions']['height']
                W = item['image_dimensions']['width']
                centers = []
                for obj in item['objects']:
                    bbox = obj['bbox']
                    cx = (bbox[0] + bbox[2]) / 2.0 / W
                    cy = (bbox[1] + bbox[3]) / 2.0 / H
                    centers.append([cx, cy])

                for pattern in [
                    f"/data/amondal/UniCountData/ucount_consolidated/images/part2/{file_name}",
                    f"/data/amondal/UniCountData/ucount_consolidated/{file_name}",
                ]:
                    ucount_centers[pattern] = {
                        'object_centers': centers,
                        'H': H,
                        'W': W,
                        'source': 'UCount_Part2'
                    }

    # Part 3
    if os.path.exists(part3_file):
        with open(part3_file) as f:
            labels = json.load(f)
        for item in labels:
            img_id = item['image_id']
            H = item['image_dimensions']['height']
            W = item['image_dimensions']['width']
            centers = []
            for obj in item['objects']:
                bbox = obj['bbox']
                cx = (bbox[0] + bbox[2]) / 2.0 / W
                cy = (bbox[1] + bbox[3]) / 2.0 / H
                centers.append([cx, cy])

            ucount_centers[f"/data/amondal/UniCountData/ucount_consolidated/{img_id}"] = {
                'object_centers': centers,
                'H': H,
                'W': W,
                'source': 'UCount_Part3'
            }

    print(f"UCount consolidated: Extracted {len(ucount_centers)} records")
    return ucount_centers


def extract_shanghaitech_centers():
    """Extract from ShanghaiTech."""
    sha_centers = {}

    for part in ['part_A', 'part_B']:
        root = f"/data/amondal/ShanghaiTech/{part}"
        gt_dir = os.path.join(root, "train_data", "ground-truth")
        img_dir = os.path.join(root, "train_data", "images")

        for mat_file in sorted(glob.glob(os.path.join(gt_dir, "GT_IMG_*.mat"))):
            img_name = os.path.basename(mat_file).replace("GT_IMG_", "").replace(".mat", ".jpg")
            img_path_local = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path_local):
                continue

            try:
                mat = sio.loadmat(mat_file)
                points = mat['image_info'][0, 0][0, 0][0]

                from PIL import Image
                img = Image.open(img_path_local)
                H, W = img.size[1], img.size[0]

                centers = []
                for x, y in points:
                    norm_x = float(x) / W
                    norm_y = float(y) / H
                    centers.append([norm_x, norm_y])

                # Store with path as it appears in balanced_mix
                img_path = f"/data/amondal/ShanghaiTech/{part}/train_data/images/{img_name}"
                sha_centers[img_path] = {
                    'object_centers': centers,
                    'H': H,
                    'W': W,
                    'source': f'ShanghaiTech_{part}'
                }
            except Exception as e:
                print(f"  [skip] {mat_file}: {e}")
                continue

    print(f"ShanghaiTech: Extracted {len(sha_centers)} records")
    return sha_centers


def main():
    print("Extracting object_centers...")
    print("=" * 70)

    fsc_centers = extract_fsc147_centers()
    ucount_centers = extract_ucount_consolidated()
    sha_centers = extract_shanghaitech_centers()

    all_centers = {}
    all_centers.update(fsc_centers)
    all_centers.update(ucount_centers)
    all_centers.update(sha_centers)

    print(f"\nTotal unique paths extracted: {len(all_centers)}")

    # Load balanced_mix
    print("\nLoading balanced_mix_v3s...")
    with open("outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json") as f:
        balanced = json.load(f)

    print(f"Augmenting {len(balanced)} records...")

    augmented = []
    missing = []

    for record in balanced:
        img_path = record['image']

        if img_path in all_centers:
            record['object_centers'] = all_centers[img_path]['object_centers']
            record['H'] = all_centers[img_path]['H']
            record['W'] = all_centers[img_path]['W']
            record['annotation_source'] = all_centers[img_path]['source']
            augmented.append(record)
        else:
            augmented.append(record)
            missing.append(img_path)

    with_centers = len([r for r in augmented if 'object_centers' in r])
    print(f"\nRecords with object_centers: {with_centers}/{len(augmented)}")
    print(f"Coverage: {100*with_centers/len(augmented):.1f}%")

    # Save
    output_file = "outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(augmented, f)

    print(f"\nSaved to {output_file}")

    # Stats by source
    by_source = defaultdict(int)
    for r in augmented:
        if 'annotation_source' in r:
            by_source[r['annotation_source']] += 1

    print(f"\nBy source:")
    for src, cnt in sorted(by_source.items()):
        print(f"  {src}: {cnt}")

    # Show missing examples
    if missing:
        print(f"\nMissing centers (first 5):")
        for path in missing[:5]:
            print(f"  {path}")


if __name__ == '__main__':
    main()
