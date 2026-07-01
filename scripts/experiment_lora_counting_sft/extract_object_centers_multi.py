#!/usr/bin/env python3
"""
Smart object_centers extraction with multi-strategy path matching.
Handles FSC-147, UCount (all parts), and ShanghaiTech.
"""

import json
import glob
import os
from collections import defaultdict

import numpy as np
import scipy.io as sio


def extract_fsc147_centers():
    """FSC-147: Full path matching."""
    fsc_annot_file = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"

    with open(fsc_annot_file) as f:
        fsc_annot = json.load(f)

    fsc_by_path = {}
    for img_id, annot in fsc_annot.items():
        H, W = annot['H'], annot['W']
        points = annot['points']

        normalized_centers = []
        for x, y in points:
            norm_x = x / W
            norm_y = y / H
            normalized_centers.append([norm_x, norm_y])

        img_path = f"/data/amondal/FSC147_hf/images_384_VarV2/{img_id}"
        fsc_by_path[img_path] = {
            'object_centers': normalized_centers,
            'H': H,
            'W': W,
            'source': 'FSC147'
        }

    print(f"FSC-147: Extracted {len(fsc_by_path)} full-path records")
    return fsc_by_path


def extract_ucount_part1():
    """UCount Part 1: Basename matching (images/set*.jpg -> part1/)"""
    part1_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part1/snapshots/*/dense_objects_labels.json"
    )

    if not part1_files:
        return {}

    with open(part1_files[0]) as f:
        labels = json.load(f)

    part1_by_basename = {}
    for item in labels:
        img_id = item['image_id']  # Format: "images/set1_cluster_157.jpg"
        basename = os.path.basename(img_id)  # "set1_cluster_157.jpg"
        H = item['image_dimensions']['height']
        W = item['image_dimensions']['width']

        centers = []
        for obj in item['objects']:
            bbox = obj['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0 / W
            cy = (bbox[1] + bbox[3]) / 2.0 / H
            centers.append([cx, cy])

        part1_by_basename[basename] = {
            'object_centers': centers,
            'H': H,
            'W': W,
            'source': 'UCount_Part1'
        }

    print(f"UCount Part 1: Extracted {len(part1_by_basename)} basename records")
    return part1_by_basename


def extract_ucount_part2():
    """UCount Part 2: Full path matching."""
    part2_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part2/snapshots/*/metadata_updated.jsonl"
    )

    if not part2_files:
        return {}

    part2_by_path = {}
    with open(part2_files[0]) as f:
        for line in f:
            item = json.loads(line)
            file_name = item['file_name']  # e.g., "pen/6156194.jpeg"
            H = item['image_dimensions']['height']
            W = item['image_dimensions']['width']

            centers = []
            for obj in item['objects']:
                bbox = obj['bbox']
                cx = (bbox[0] + bbox[2]) / 2.0 / W
                cy = (bbox[1] + bbox[3]) / 2.0 / H
                centers.append([cx, cy])

            img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/part2/{file_name}"
            part2_by_path[img_path] = {
                'object_centers': centers,
                'H': H,
                'W': W,
                'source': 'UCount_Part2'
            }

    print(f"UCount Part 2: Extracted {len(part2_by_path)} full-path records")
    return part2_by_path


def extract_ucount_part3():
    """UCount Part 3: Basename matching."""
    part3_file = "/data/amondal/UniCountData/ucount_part3/sku110k_labels_fixed.json"

    if not os.path.exists(part3_file):
        return {}

    with open(part3_file) as f:
        labels = json.load(f)

    part3_by_basename = {}
    for item in labels:
        img_id = item['image_id']
        basename = os.path.basename(img_id)
        H = item['image_dimensions']['height']
        W = item['image_dimensions']['width']

        centers = []
        for obj in item['objects']:
            bbox = obj['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0 / W
            cy = (bbox[1] + bbox[3]) / 2.0 / H
            centers.append([cx, cy])

        part3_by_basename[basename] = {
            'object_centers': centers,
            'H': H,
            'W': W,
            'source': 'UCount_Part3'
        }

    print(f"UCount Part 3: Extracted {len(part3_by_basename)} basename records")
    return part3_by_basename


def extract_shanghaitech():
    """ShanghaiTech: Full path matching."""
    sha_by_path = {}

    for part in ['part_A', 'part_B']:
        root = f"/data/amondal/ShanghaiTech/{part}"
        gt_dir = os.path.join(root, "train_data", "ground-truth")
        img_dir = os.path.join(root, "train_data", "images")

        if not os.path.exists(gt_dir):
            continue

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

                img_path = f"/data/amondal/ShanghaiTech/{part}/train_data/images/{img_name}"
                sha_by_path[img_path] = {
                    'object_centers': centers,
                    'H': H,
                    'W': W,
                    'source': f'ShanghaiTech_{part}'
                }
            except Exception as e:
                print(f"  [skip] {mat_file}: {e}")

    print(f"ShanghaiTech: Extracted {len(sha_by_path)} full-path records")
    return sha_by_path


def main():
    print("Extracting object_centers with multi-strategy matching...")
    print("=" * 70)

    # Extract from all sources
    fsc_by_path = extract_fsc147_centers()
    part1_by_basename = extract_ucount_part1()
    part2_by_path = extract_ucount_part2()
    part3_by_basename = extract_ucount_part3()
    sha_by_path = extract_shanghaitech()

    # Load balanced_mix
    print("\nLoading balanced_mix_v3s...")
    with open("outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json") as f:
        balanced = json.load(f)

    print(f"Matching {len(balanced)} records...")

    augmented = []
    stats = {'matched': 0, 'fsc': 0, 'part1': 0, 'part2': 0, 'part3': 0, 'sha': 0, 'missing': 0}

    for record in balanced:
        img_path = record['image']
        basename = os.path.basename(img_path)

        matched = False

        # Try 1: Full path match (FSC, Part2, SHA)
        if img_path in fsc_by_path:
            record.update(fsc_by_path[img_path])
            stats['matched'] += 1
            stats['fsc'] += 1
            matched = True
        elif img_path in part2_by_path:
            record.update(part2_by_path[img_path])
            stats['matched'] += 1
            stats['part2'] += 1
            matched = True
        elif img_path in sha_by_path:
            record.update(sha_by_path[img_path])
            stats['matched'] += 1
            stats['sha'] += 1
            matched = True

        # Try 2: Basename match (Part1, Part3) - only for ucount_consolidated paths
        if not matched and 'ucount_consolidated' in img_path:
            if 'part1' in img_path and basename in part1_by_basename:
                record.update(part1_by_basename[basename])
                record['annotation_source'] = 'UCount_Part1'
                stats['matched'] += 1
                stats['part1'] += 1
                matched = True
            elif 'part3' in img_path and basename in part3_by_basename:
                record.update(part3_by_basename[basename])
                record['annotation_source'] = 'UCount_Part3'
                stats['matched'] += 1
                stats['part3'] += 1
                matched = True

        if not matched:
            stats['missing'] += 1

        augmented.append(record)

    print(f"\nMatching results:")
    print(f"  Total: {len(augmented)}")
    print(f"  Matched: {stats['matched']} ({100*stats['matched']/len(augmented):.1f}%)")
    print(f"    - FSC-147: {stats['fsc']}")
    print(f"    - UCount Part1: {stats['part1']}")
    print(f"    - UCount Part2: {stats['part2']}")
    print(f"    - UCount Part3: {stats['part3']}")
    print(f"    - ShanghaiTech: {stats['sha']}")
    print(f"  Missing: {stats['missing']}")

    # Save
    output_file = "outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(augmented, f)

    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
