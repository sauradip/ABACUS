"""
Build a comprehensive dataset with spatial annotations for attention regularizer training.

Sources:
  - FSC-147: point annotations
  - UniCount (consolidated): bounding box annotations
  - ShanghaiTech A/B: point annotations (.mat)

Output format:
  {
    "image": "<abs_path>",
    "image_id": "<id>",
    "H": 384,
    "W": 384,
    "category": "peppers",
    "count": 13,
    "object_centers": [[0.5, 0.3], [0.2, 0.8], ...],  # normalized [0,1]
    "annotation_source": "fsc147|ucount|shanghaitech_a|shanghaitech_b",
    "data_split": "train|val|test"
  }
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.io as sio
from PIL import Image


OUTPUT_DIR = Path("/data/amondal/UniCount/data/attn_regularizer_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Entry Builder (with LLaVA conversations)
# ============================================================================
def make_entry(image_path, image_id, H, W, category, count, object_centers, annotation_source, data_split):
    """Generate entry with metadata and LLaVA conversation format."""
    return {
        "image": image_path,
        "image_id": image_id,
        "H": H,
        "W": W,
        "category": category,
        "count": count,
        "object_centers": object_centers,
        "annotation_source": annotation_source,
        "data_split": data_split,
        "conversations": [
            {
                "from": "system",
                "value": "You are an AI assistant specialized in object counting in images."
            },
            {
                "from": "human",
                "value": f"<image>\nCount and detect all the {category} in the image. Answer with only a number."
            },
            {
                "from": "gpt",
                "value": str(count)
            }
        ]
    }


# ============================================================================
# FSC-147
# ============================================================================
def build_fsc147():
    """Extract from FSC-147 with point annotations and fine-grained categories."""
    entries = []
    annot_file = "/data/amondal/FSC147_hf/annotation_FSC147_384.json"
    split_file = "/data/amondal/FSC147_hf/Train_Test_Val_FSC_147.json"

    with open(annot_file) as f:
        annotations = json.load(f)

    with open(split_file) as f:
        splits = json.load(f)

    # Build split lookup
    split_map = {}
    for split_name in ["train", "test", "val"]:
        if split_name in splits:
            for img_id in splits[split_name]:
                split_map[img_id] = split_name

    # Load category mapping from FSC-147 metadata
    class_file = "/data/amondal/FSC147_hf/ImageClasses_FSC147.txt"
    img_to_class = {}
    if os.path.exists(class_file):
        with open(class_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_id_part = parts[0]
                        category = " ".join(parts[1:])
                        img_to_class[img_id_part] = category

    for img_id, annot in annotations.items():
        if "points" not in annot or len(annot["points"]) == 0:
            continue

        # Get fine-grained category
        category = img_to_class.get(img_id, img_id.split("_")[0] if "_" in img_id else "object")

        points = annot["points"]
        H, W = annot["H"], annot["W"]

        # Normalize to [0, 1]
        centers = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                cx, cy = float(pt[0]) / W, float(pt[1]) / H
                centers.append([cx, cy])

        if not centers:
            continue

        # Use local path, not NFS path from annotation file
        # image_id may already include .jpg
        img_filename = img_id if img_id.endswith(('.jpg', '.jpeg', '.png')) else f"{img_id}.jpg"
        img_path = f"/data/amondal/FSC147_hf/images_384_VarV2/{img_filename}"

        entries.append(make_entry(
            image_path=img_path,
            image_id=img_id,
            H=H,
            W=W,
            category=category,
            count=len(centers),
            object_centers=centers,
            annotation_source="fsc147",
            data_split=split_map.get(img_id, "train"),
        ))

    print(f"[FSC-147] Loaded {len(entries)} images with point annotations")
    return entries


# ============================================================================
# UniCount (from source annotations)
# ============================================================================
def build_ucount():
    """Extract from UniCount with bbox annotations."""
    entries = []

    # Find dense_objects_labels.json
    labels_paths = glob.glob("/data/amondal/UniCountData/datasets--sauradip--ucount_part1/snapshots/*/dense_objects_labels.json")

    if not labels_paths:
        print("[UniCount] No labels found")
        return entries

    labels_file = labels_paths[0]

    with open(labels_file) as f:
        labels = json.load(f)

    for entry in labels:
        image_id = entry.get("image_id", "")
        if not image_id:
            continue

        dims = entry.get("image_dimensions", {})
        H, W = dims.get("height", 384), dims.get("width", 384)

        objects = entry.get("objects", [])
        if not objects:
            continue

        # Group by category
        by_category = defaultdict(list)
        for obj in objects:
            category = obj.get("category_name", "object")
            bbox = obj.get("bbox", [])
            if len(bbox) >= 4:
                # bbox is [x_min, y_min, x_max, y_max]
                cx = (bbox[0] + bbox[2]) / 2 / W
                cy = (bbox[1] + bbox[3]) / 2 / H
                by_category[category].append([cx, cy])

        # Create one entry per category per image
        for category, centers in by_category.items():
            if not centers:
                continue

            img_rel_path = image_id
            img_path = f"/data/amondal/UniCountData/ucount_consolidated/images/{img_rel_path}"

            # Verify image exists
            if not os.path.exists(img_path):
                # Try alternative paths
                alt_paths = [
                    f"/data/amondal/UniCountData/ucount_consolidated/images/part1/{os.path.basename(img_rel_path)}",
                    f"/data/amondal/UniCountData/ucount_consolidated/images/part2/{img_rel_path}",
                ]
                found = False
                for alt in alt_paths:
                    if os.path.exists(alt):
                        img_path = alt
                        found = True
                        break
                if not found:
                    continue

            entries.append(make_entry(
                image_path=img_path,
                image_id=image_id,
                H=H,
                W=W,
                category=category,
                count=len(centers),
                object_centers=centers,
                annotation_source="ucount",
                data_split="train",
            ))

    print(f"[UniCount] Loaded {len(entries)} image-category pairs with bbox annotations")
    return entries


# ============================================================================
# ShanghaiTech A & B (point annotations in .mat)
# ============================================================================
def build_shanghaitech(root, part_name):
    """Extract from ShanghaiTech .mat files."""
    entries = []
    gt_dir = os.path.join(root, "train_data", "ground-truth")
    img_dir = os.path.join(root, "train_data", "images")

    if not os.path.exists(gt_dir):
        print(f"[ShanghaiTech {part_name}] ground-truth dir not found")
        return entries

    for mat_path in sorted(glob.glob(os.path.join(gt_dir, "GT_IMG_*.mat"))):
        try:
            mat = sio.loadmat(mat_path)
            # ShanghaiTech stores points as [x, y] in 'image_info'
            points = mat["image_info"][0][0][0][0][0]  # Shape: (n_objects, 2)

            if len(points) == 0:
                continue

            # Get image dimensions
            img_id = os.path.basename(mat_path).replace("GT_", "").replace(".mat", "")
            img_path = os.path.join(img_dir, f"{img_id}.jpg")

            if not os.path.exists(img_path):
                continue

            # Load image to get dimensions
            with Image.open(img_path) as img:
                W, H = img.size

            # Normalize points to [0, 1]
            centers = []
            for pt in points:
                if len(pt) >= 2:
                    cx = float(pt[0]) / W
                    cy = float(pt[1]) / H
                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    centers.append([cx, cy])

            if not centers:
                continue

            entries.append(make_entry(
                image_path=os.path.abspath(img_path),
                image_id=img_id,
                H=H,
                W=W,
                category="people",
                count=len(centers),
                object_centers=centers,
                annotation_source=f"shanghaitech_{part_name}",
                data_split="train",
            ))

        except Exception as e:
            print(f"  WARN: Failed to parse {mat_path}: {e}")
            continue

    print(f"[ShanghaiTech {part_name}] Loaded {len(entries)} images with point annotations")
    return entries


# ============================================================================
# Main
# ============================================================================
def main():
    print("Building attention regularizer dataset...\n")

    all_entries = []

    # FSC-147
    fsc_entries = build_fsc147()
    all_entries.extend(fsc_entries)

    # UniCount
    ucount_entries = build_ucount()
    all_entries.extend(ucount_entries)

    # ShanghaiTech
    sha_entries = build_shanghaitech("/data/amondal/ShanghaiTech/part_A", "a")
    shb_entries = build_shanghaitech("/data/amondal/ShanghaiTech/part_B", "b")
    all_entries.extend(sha_entries)
    all_entries.extend(shb_entries)

    print(f"\n[TOTAL] {len(all_entries)} entries across all sources\n")

    # Statistics
    by_source = defaultdict(list)
    by_split = defaultdict(list)
    for entry in all_entries:
        by_source[entry["annotation_source"]].append(entry)
        by_split[entry["data_split"]].append(entry)

    print("Distribution by source:")
    for source in sorted(by_source.keys()):
        count = len(by_source[source])
        total_objects = sum(e["count"] for e in by_source[source])
        print(f"  {source:20s}: {count:6d} images, {total_objects:8d} total objects")

    print("\nDistribution by split:")
    for split in sorted(by_split.keys()):
        count = len(by_split[split])
        print(f"  {split:10s}: {count:6d} images")

    # Write consolidated
    out_file = OUTPUT_DIR / "attn_regularizer_train.json"
    with open(out_file, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nWrote {out_file}")
    print(f"Total size: {os.path.getsize(out_file) / 1e6:.1f} MB")

    # Write by source for easier ablations
    for source, entries in by_source.items():
        out_file = OUTPUT_DIR / f"attn_regularizer_{source}.json"
        with open(out_file, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Wrote {out_file} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
