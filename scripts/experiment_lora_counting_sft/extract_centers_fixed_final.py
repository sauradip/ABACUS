#!/usr/bin/env python3
"""
Full extraction using ACTUAL image dimensions for all sources.
"""

import json
import glob
import os
from PIL import Image
from collections import defaultdict

import numpy as np
import scipy.io as sio


def extract_fsc147_full():
    """FSC-147: Full extraction with actual image dimensions."""
    fsc_annot_file = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"

    with open(fsc_annot_file) as f:
        fsc_annot = json.load(f)

    fsc_by_path = {}
    skipped = 0

    for img_id, annot in fsc_annot.items():
        img_path = f"/data/amondal/FSC147_hf/images_384_VarV2/{img_id}"

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            img = Image.open(img_path)
            W_actual, H_actual = img.size
            points = annot['points']

            normalized_centers = []
            for x, y in points:
                norm_x = x / W_actual
                norm_y = y / H_actual
                normalized_centers.append([norm_x, norm_y])

            fsc_by_path[img_path] = {
                'object_centers': normalized_centers,
                'H': H_actual,
                'W': W_actual,
                'source': 'FSC147'
            }
        except Exception as e:
            skipped += 1

    print(f"FSC-147: Extracted {len(fsc_by_path)} records (skipped {skipped})")
    return fsc_by_path


def extract_ucount_part1_full():
    """UCount Part 1: Full extraction with actual image dimensions."""
    part1_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part1/snapshots/*/dense_objects_labels.json"
    )

    if not part1_files:
        return {}

    with open(part1_files[0]) as f:
        labels = json.load(f)

    part1_by_basename = {}
    skipped = 0

    for item in labels:
        img_id = item['image_id']
        basename = os.path.basename(img_id)
        img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/part1/{basename}"

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
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

            part1_by_basename[basename] = {
                'object_centers': centers,
                'H': H_actual,
                'W': W_actual,
                'source': 'UCount_Part1'
            }
        except Exception as e:
            skipped += 1

    print(f"UCount Part 1: Extracted {len(part1_by_basename)} records (skipped {skipped})")
    return part1_by_basename


def extract_ucount_part2_full():
    """UCount Part 2: Full extraction with actual image dimensions."""
    part2_files = glob.glob(
        "/data/amondal/UniCountData/datasets--sauradip--ucount_part2/snapshots/*/metadata_updated.jsonl"
    )

    if not part2_files:
        return {}

    part2_by_path = {}
    skipped = 0

    with open(part2_files[0]) as f:
        for line in f:
            item = json.loads(line)
            file_name = item['file_name']
            img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/part2/{file_name}"

            if not os.path.exists(img_path):
                skipped += 1
                continue

            try:
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
                    'source': 'UCount_Part2'
                }
            except Exception as e:
                skipped += 1

    print(f"UCount Part 2: Extracted {len(part2_by_path)} records (skipped {skipped})")
    return part2_by_path


def extract_ucount_part3_full():
    """UCount Part 3: Full extraction with actual image dimensions."""
    part3_file = "/data/amondal/UniCountData/ucount_part3/sku110k_labels_fixed.json"

    if not os.path.exists(part3_file):
        return {}

    with open(part3_file) as f:
        labels = json.load(f)

    part3_by_basename = {}
    skipped = 0

    for item in labels:
        img_id = item['image_id']
        basename = os.path.basename(img_id)
        img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/part3/{basename}"

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
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

            part3_by_basename[basename] = {
                'object_centers': centers,
                'H': H_actual,
                'W': W_actual,
                'source': 'UCount_Part3'
            }
        except Exception as e:
            skipped += 1

    print(f"UCount Part 3: Extracted {len(part3_by_basename)} records (skipped {skipped})")
    return part3_by_basename


def extract_shanghaitech_full():
    """ShanghaiTech: Full extraction with actual image dimensions."""
    sha_by_path = {}
    skipped = 0

    for part in ['part_A', 'part_B']:
        root = f"/data/amondal/ShanghaiTech/{part}"
        gt_dir = os.path.join(root, "train_data", "ground-truth")
        img_dir = os.path.join(root, "train_data", "images")

        if not os.path.exists(gt_dir):
            continue

        for mat_file in sorted(glob.glob(os.path.join(gt_dir, "GT_IMG_*.mat"))):
            img_name = os.path.basename(mat_file).replace("GT_IMG_", "IMG_").replace(".mat", ".jpg")
            img_path_local = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path_local):
                skipped += 1
                continue

            try:
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
                    'source': f'ShanghaiTech_{part}'
                }
            except Exception as e:
                skipped += 1

    print(f"ShanghaiTech: Extracted {len(sha_by_path)} records (skipped {skipped})")
    return sha_by_path


def extract_ucount_crowd_full():
    """UCount Crowd: Extraction from CrowdHuman annotations."""
    crowd_annot_file = "/data/amondal/UniCountData/saura5/crowdhuman_labels.json"

    if not os.path.exists(crowd_annot_file):
        print(f"UCount Crowd: File not found: {crowd_annot_file}")
        return {}

    with open(crowd_annot_file) as f:
        crowd_annot = json.load(f)

    # Create filename → annotation mapping
    crowd_by_filename = {}
    for annot in crowd_annot:
        filename = os.path.basename(annot['image_id'])  # Extract just filename
        crowd_by_filename[filename] = annot

    crowd_by_path = {}
    skipped = 0

    # We'll match by filename when we process balanced_mix
    # For now just load and count
    print(f"UCount Crowd: Loaded {len(crowd_by_filename)} annotations (will match by filename)")

    return crowd_by_filename


def match_with_balanced_mix():
    """Load balanced_mix and match with extracted centers."""
    print("\nLoading balanced_mix_v3s...")
    with open("outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train.json") as f:
        balanced = json.load(f)

    print(f"Matching {len(balanced)} records...")

    # Extract all sources
    fsc = extract_fsc147_full()
    part1 = extract_ucount_part1_full()
    part2 = extract_ucount_part2_full()
    part3 = extract_ucount_part3_full()
    sha = extract_shanghaitech_full()
    crowd = extract_ucount_crowd_full()

    augmented = []
    stats = {'fsc': 0, 'part1': 0, 'part2': 0, 'part3': 0, 'sha': 0, 'crowd': 0, 'missing': 0}

    for record in balanced:
        img_path = record['image']
        basename = os.path.basename(img_path)

        matched = False

        # FSC: Full path match
        if img_path in fsc:
            record.update(fsc[img_path])
            stats['fsc'] += 1
            matched = True
        # Part 2: Full path match
        elif img_path in part2:
            record.update(part2[img_path])
            stats['part2'] += 1
            matched = True
        # SHA: Full path match
        elif img_path in sha:
            record.update(sha[img_path])
            stats['sha'] += 1
            matched = True
        # Part 1: Basename match
        elif 'part1' in img_path and basename in part1:
            record.update(part1[basename])
            record['source'] = 'UCount_Part1'
            stats['part1'] += 1
            matched = True
        # Part 3: Basename match
        elif 'part3' in img_path and basename in part3:
            record.update(part3[basename])
            record['source'] = 'UCount_Part3'
            stats['part3'] += 1
            matched = True
        # UCount Crowd: Filename match
        elif 'ucount_crowd' in img_path and basename in crowd:
            try:
                img = Image.open(img_path)
                W_actual, H_actual = img.size

                annot = crowd[basename]
                centers = []
                for obj in annot['objects']:
                    bbox = obj['bbox']
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    norm_cx = cx / W_actual
                    norm_cy = cy / H_actual
                    centers.append([norm_cx, norm_cy])

                record['object_centers'] = centers
                record['H'] = H_actual
                record['W'] = W_actual
                record['source'] = 'UCount_Crowd'
                stats['crowd'] += 1
                matched = True
            except Exception as e:
                pass

        if not matched:
            stats['missing'] += 1

        augmented.append(record)

    # Save
    output_file = "outputs/experiment_lora_counting_sft/balanced_mix_v3s/balanced_mix_train_with_centers.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(augmented, f)

    # Stats
    total_matched = sum(v for k, v in stats.items() if k != 'missing')
    print(f"\n" + "=" * 70)
    print(f"EXTRACTION COMPLETE")
    print(f"=" * 70)
    print(f"Total records: {len(augmented)}")
    print(f"With object_centers: {total_matched}")
    print(f"Coverage: {100*total_matched/len(augmented):.1f}%")
    print(f"\nBy source:")
    print(f"  FSC-147: {stats['fsc']}")
    print(f"  UCount Part1: {stats['part1']}")
    print(f"  UCount Part2: {stats['part2']}")
    print(f"  UCount Part3: {stats['part3']}")
    print(f"  ShanghaiTech: {stats['sha']}")
    print(f"  UCount Crowd: {stats['crowd']}")
    print(f"  Missing: {stats['missing']}")
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    match_with_balanced_mix()
