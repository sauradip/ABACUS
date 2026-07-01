#!/usr/bin/env python3
"""
Fixed extraction using ACTUAL image dimensions.
Test on small sample first, then full extraction.
"""

import json
import glob
import os
from PIL import Image
from collections import defaultdict

import numpy as np
import scipy.io as sio


def extract_fsc147_sample(max_records=5):
    """FSC-147: Use actual image dimensions."""
    fsc_annot_file = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"

    with open(fsc_annot_file) as f:
        fsc_annot = json.load(f)

    fsc_by_path = {}
    processed = 0

    for img_id, annot in fsc_annot.items():
        if processed >= max_records:
            break

        img_path = f"/data/amondal/FSC147_hf/images_384_VarV2/{img_id}"

        if not os.path.exists(img_path):
            continue

        try:
            # Get ACTUAL image dimensions
            img = Image.open(img_path)
            W_actual, H_actual = img.size

            points = annot['points']  # Pixel coordinates in original annotation space

            # Normalize using ACTUAL image dimensions
            normalized_centers = []
            for x, y in points:
                norm_x = x / W_actual
                norm_y = y / H_actual
                normalized_centers.append([norm_x, norm_y])

            fsc_by_path[img_path] = {
                'object_centers': normalized_centers,
                'H': H_actual,
                'W': W_actual,
                'source': 'FSC147',
                'num_objects': len(points)
            }
            processed += 1
        except Exception as e:
            print(f"  [skip FSC] {img_id}: {e}")

    print(f"FSC-147 SAMPLE: Extracted {len(fsc_by_path)} records")
    return fsc_by_path


def extract_ucount_part1_sample(max_records=5):
    """UCount Part 1: Use actual image dimensions."""
    part1_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part1/snapshots/*/dense_objects_labels.json"
    )

    if not part1_files:
        return {}

    with open(part1_files[0]) as f:
        labels = json.load(f)

    part1_by_basename = {}
    processed = 0

    for item in labels:
        if processed >= max_records:
            break

        img_id = item['image_id']
        basename = os.path.basename(img_id)

        # Try to find actual image to get dimensions
        img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/part1/{basename}"

        if not os.path.exists(img_path):
            continue

        try:
            # Get ACTUAL image dimensions
            img = Image.open(img_path)
            W_actual, H_actual = img.size

            centers = []
            for obj in item['objects']:
                bbox = obj['bbox']  # [left, top, right, bottom]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                # Normalize using ACTUAL image dimensions
                norm_cx = cx / W_actual
                norm_cy = cy / H_actual
                centers.append([norm_cx, norm_cy])

            part1_by_basename[basename] = {
                'object_centers': centers,
                'H': H_actual,
                'W': W_actual,
                'source': 'UCount_Part1',
                'num_objects': len(centers)
            }
            processed += 1
        except Exception as e:
            print(f"  [skip Part1] {basename}: {e}")

    print(f"UCount Part 1 SAMPLE: Extracted {len(part1_by_basename)} records")
    return part1_by_basename


def extract_ucount_part2_sample(max_records=5):
    """UCount Part 2: Use actual image dimensions."""
    part2_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part2/snapshots/*/metadata_updated.jsonl"
    )

    if not part2_files:
        return {}

    part2_by_path = {}
    processed = 0

    with open(part2_files[0]) as f:
        for line in f:
            if processed >= max_records:
                break

            item = json.loads(line)
            file_name = item['file_name']
            img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/part2/{file_name}"

            if not os.path.exists(img_path):
                continue

            try:
                # Get ACTUAL image dimensions
                img = Image.open(img_path)
                W_actual, H_actual = img.size

                centers = []
                for obj in item['objects']:
                    bbox = obj['bbox']
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    norm_cx = cx / W_actual
                    norm_cy = cy / H_actual
                    centers.append([norm_cx, norm_cy])

                part2_by_path[img_path] = {
                    'object_centers': centers,
                    'H': H_actual,
                    'W': W_actual,
                    'source': 'UCount_Part2',
                    'num_objects': len(centers)
                }
                processed += 1
            except Exception as e:
                print(f"  [skip Part2] {file_name}: {e}")

    print(f"UCount Part 2 SAMPLE: Extracted {len(part2_by_path)} records")
    return part2_by_path


def extract_shanghaitech_sample(max_records=5):
    """ShanghaiTech: Use actual image dimensions."""
    sha_by_path = {}
    processed = 0

    for part in ['part_A', 'part_B']:
        root = f"/data/amondal/ShanghaiTech/{part}"
        gt_dir = os.path.join(root, "train_data", "ground-truth")
        img_dir = os.path.join(root, "train_data", "images")

        if not os.path.exists(gt_dir):
            continue

        for mat_file in sorted(glob.glob(os.path.join(gt_dir, "GT_IMG_*.mat"))):
            if processed >= max_records:
                break

            img_name = os.path.basename(mat_file).replace("GT_IMG_", "").replace(".mat", ".jpg")
            img_path_local = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path_local):
                continue

            try:
                # Get ACTUAL image dimensions
                img = Image.open(img_path_local)
                W_actual, H_actual = img.size

                mat = sio.loadmat(mat_file)
                points = mat['image_info'][0, 0][0, 0][0]

                centers = []
                for x, y in points:
                    norm_x = float(x) / W_actual
                    norm_y = float(y) / H_actual
                    centers.append([norm_x, norm_y])

                img_path = f"/data/amondal/ShanghaiTech/{part}/train_data/images/{img_name}"
                sha_by_path[img_path] = {
                    'object_centers': centers,
                    'H': H_actual,
                    'W': W_actual,
                    'source': f'ShanghaiTech_{part}',
                    'num_objects': len(centers)
                }
                processed += 1
            except Exception as e:
                print(f"  [skip SHA] {mat_file}: {e}")

    print(f"ShanghaiTech SAMPLE: Extracted {len(sha_by_path)} records")
    return sha_by_path


def visualize_samples(all_records):
    """Generate visualizations for sample records."""
    from PIL import ImageDraw

    output_dir = "outputs/experiment_lora_counting_sft/eval/extraction_samples_fixed"
    os.makedirs(output_dir, exist_ok=True)

    by_source = defaultdict(list)
    for path, data in all_records.items():
        src = data['source']
        by_source[src].append((path, data))

    for source in sorted(by_source.keys()):
        records = by_source[source]
        print(f"\nVisualizing {source}: {len(records)} samples")

        for idx, (img_path, data) in enumerate(records):
            if not os.path.exists(img_path):
                print(f"  {idx+1}: Image not found: {img_path}")
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(img)
                centers = data['object_centers']

                for cx_norm, cy_norm in centers:
                    W_img, H_img = img.size
                    px = cx_norm * (W_img - 1)
                    py = cy_norm * (H_img - 1)
                    r = 5
                    draw.ellipse([px-r, py-r, px+r, py+r], fill='red', outline='yellow')

                # Add text
                draw.text((10, 10), f"{source} | Count: {len(centers)} | {data['W']}x{data['H']}", fill='white')

                fname = f"{source.replace('_', '-').lower()}-sample-{idx+1:02d}.jpg"
                out_path = os.path.join(output_dir, fname)
                img.save(out_path)
                print(f"  {idx+1}: {fname} - {len(centers)} objects, actual size {data['W']}x{data['H']}")

            except Exception as e:
                print(f"  {idx+1}: Error - {e}")

    print(f"\nSamples saved to {output_dir}")
    return output_dir


if __name__ == '__main__':
    print("Testing fixed extraction with actual image dimensions...")
    print("=" * 70)

    all_records = {}

    fsc = extract_fsc147_sample(max_records=5)
    all_records.update(fsc)

    part1 = extract_ucount_part1_sample(max_records=5)
    all_records.update(part1)

    part2 = extract_ucount_part2_sample(max_records=5)
    all_records.update(part2)

    sha = extract_shanghaitech_sample(max_records=5)
    all_records.update(sha)

    print(f"\nTotal sample records: {len(all_records)}")

    if all_records:
        print("\nGenerating visualizations...")
        visualize_samples(all_records)
    else:
        print("No records extracted!")
