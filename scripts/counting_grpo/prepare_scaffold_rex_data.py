#!/usr/bin/env python3
"""
Build FSC147 SCAFFOLD-Rex JSONL data with 6x6 visual anchors and clustered JSON targets.

Outputs:
- Overlaid images with a 6x6 anchor grid (binary black/white dots)
- train/val/test/all JSONL records with strict JSON assistant targets
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw


GRID_GUIDELINE = (
    "The image is overlaid with a 6x6 dot matrix. Dots are labeled (x,y). "
    "Within columns, x increases top-to-bottom. Within rows, y increases left-to-right. "
    "Identify objects near their nearest coordinates and return strict JSON only."
)

PROMPT_TEMPLATE = (
    "<image>\n"
    "{guideline}\n"
    "Count the {category} and group instances by nearest anchor."
)


def resolve_fsc_root(root_arg: str) -> str:
    required = [
        "images_384_VarV2",
        "annotation_FSC147_384.json",
        "ImageClasses_FSC147.txt",
        "Train_Test_Val_FSC_147.json",
    ]

    def has_required(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, name)) for name in required)

    if has_required(root_arg):
        return root_arg

    nested = os.path.join(root_arg, "FSC147_384_V2")
    if has_required(nested):
        return nested

    raise FileNotFoundError(
        f"Could not resolve FSC root from '{root_arg}'. Expected required files in root or root/FSC147_384_V2"
    )


def load_image_classes(class_path: str) -> Dict[str, str]:
    classes: Dict[str, str] = {}
    with open(class_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                classes[parts[0].strip()] = parts[1].strip().lower()
    return classes


def grid_anchor_pixels(width: int, height: int, grid_size: int) -> Dict[Tuple[int, int], Tuple[float, float]]:
    anchors: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for row in range(1, grid_size + 1):
        py = (row / float(grid_size + 1)) * float(height)
        for col in range(1, grid_size + 1):
            px = (col / float(grid_size + 1)) * float(width)
            anchors[(row, col)] = (px, py)
    return anchors


def local_luminance(gray_img: Image.Image, x_center: int, y_center: int, radius: int = 6) -> float:
    x1 = max(0, x_center - radius)
    y1 = max(0, y_center - radius)
    x2 = min(gray_img.width, x_center + radius + 1)
    y2 = min(gray_img.height, y_center + radius + 1)
    patch = gray_img.crop((x1, y1, x2, y2))
    hist = patch.histogram()
    total = sum(hist)
    if total <= 0:
        return 255.0
    weighted = sum(i * c for i, c in enumerate(hist))
    return float(weighted) / float(total)


def draw_scaffold_overlay(image: Image.Image, grid_size: int = 6, dot_radius: int = 4) -> Tuple[Image.Image, Dict[Tuple[int, int], Tuple[float, float]]]:
    overlaid = image.copy().convert("RGB")
    gray = overlaid.convert("L")
    draw = ImageDraw.Draw(overlaid)
    anchors = grid_anchor_pixels(overlaid.width, overlaid.height, grid_size)

    for (_, _), (px, py) in anchors.items():
        xi = int(round(px))
        yi = int(round(py))
        lum = local_luminance(gray, xi, yi)
        color = (255, 255, 255) if lum < 128.0 else (0, 0, 0)
        draw.ellipse(
            [xi - dot_radius, yi - dot_radius, xi + dot_radius, yi + dot_radius],
            fill=color,
            outline=(128, 128, 128),
            width=1,
        )

    return overlaid, anchors


def nearest_anchor(point_xy: Iterable[float], anchors: Dict[Tuple[int, int], Tuple[float, float]]) -> Tuple[int, int]:
    px, py = float(point_xy[0]), float(point_xy[1])
    best_anchor = (1, 1)
    best_dist = float("inf")
    for anchor, (ax, ay) in anchors.items():
        dist = (px - ax) * (px - ax) + (py - ay) * (py - ay)
        if dist < best_dist:
            best_dist = dist
            best_anchor = anchor
    return best_anchor


def cluster_points(points: List[List[float]], anchors: Dict[Tuple[int, int], Tuple[float, float]], width: int, height: int, bbox_size: int) -> List[dict]:
    grouped: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for point in points:
        anchor = nearest_anchor(point, anchors)
        grouped[anchor].append((float(point[0]), float(point[1])))

    half = max(1, int(round(bbox_size / 2.0)))
    clusters: List[dict] = []
    for anchor in sorted(grouped.keys()):
        pts = grouped[anchor]
        x1 = min(p[0] - half for p in pts)
        y1 = min(p[1] - half for p in pts)
        x2 = max(p[0] + half for p in pts)
        y2 = max(p[1] + half for p in pts)
        region_bbox = [
            int(max(0, min(width - 1, round(x1)))),
            int(max(0, min(height - 1, round(y1)))),
            int(max(0, min(width - 1, round(x2)))),
            int(max(0, min(height - 1, round(y2)))),
        ]
        clusters.append(
            {
                "anchor": [int(anchor[0]), int(anchor[1])],
                "count": int(len(pts)),
                "region_bbox": region_bbox,
            }
        )
    return clusters


def build_json_target(clusters: List[dict]) -> str:
    total_count = int(sum(cluster["count"] for cluster in clusters))
    anchor_tokens = [f"({c['anchor'][0]},{c['anchor'][1]})" for c in clusters]
    anchors_summary = "Objects identified near coordinates " + ", ".join(anchor_tokens) + "."
    payload = {
        "total_count": total_count,
        "anchors_summary": anchors_summary,
        "clusters": clusters,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def write_jsonl(records: List[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc_root", required=True, help="Path to FSC147 root or its parent directory")
    parser.add_argument("--output_dir", required=True, help="Directory where jsonl files are written")
    parser.add_argument(
        "--overlay_dir",
        default="",
        help="Optional directory for scaffolded images; defaults to <output_dir>/images_scaffold_rex",
    )
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--dot_radius", type=int, default=4)
    parser.add_argument("--bbox_size", type=int, default=20)
    parser.add_argument("--max_samples_per_split", type=int, default=0,
                        help="Cap per split. 0 = no cap.")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated list of splits to process, e.g. 'train,val'.",
    )
    parser.add_argument(
        "--total_cap",
        type=int,
        default=0,
        help=(
            "Cap the combined all.jsonl at this many records, filling from splits in order. "
            "0 = no cap. Use 5000 for the Stage 1.5 calibration run."
        ),
    )
    parser.add_argument(
        "--smoke_n",
        type=int,
        default=0,
        help="If >0, generate only this many records total and print OCR-visibility stats.",
    )
    args = parser.parse_args()

    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    fsc_root = resolve_fsc_root(args.fsc_root)
    ann_path = os.path.join(fsc_root, "annotation_FSC147_384.json")
    cls_path = os.path.join(fsc_root, "ImageClasses_FSC147.txt")
    splits_path = os.path.join(fsc_root, "Train_Test_Val_FSC_147.json")
    image_dir = os.path.join(fsc_root, "images_384_VarV2")

    with open(ann_path, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    with open(splits_path, "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    image_classes = load_image_classes(cls_path)

    overlay_dir = args.overlay_dir or os.path.join(args.output_dir, "images_scaffold_rex")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    smoke_mode = args.smoke_n > 0
    effective_total_cap = args.smoke_n if smoke_mode else (args.total_cap if args.total_cap > 0 else 0)

    all_records: List[dict] = []
    white_dots = 0
    black_dots = 0
    total_dots = 0

    for split_name in requested_splits:
        if split_name not in splits:
            print(f"WARNING: split '{split_name}' not found in dataset; skipping.")
            continue

        split_records: List[dict] = []
        skipped = 0

        for image_name in splits[split_name]:
            if effective_total_cap > 0 and len(all_records) + len(split_records) >= effective_total_cap:
                break

            image_path = os.path.join(image_dir, image_name)
            ann = annotations.get(image_name)
            if ann is None or not os.path.exists(image_path):
                skipped += 1
                continue

            width = int(ann["W"])
            height = int(ann["H"])
            points = ann.get("points", [])
            category = image_classes.get(image_name, "objects")

            image = Image.open(image_path).convert("RGB")
            overlaid, anchors = draw_scaffold_overlay(
                image,
                grid_size=args.grid_size,
                dot_radius=args.dot_radius,
            )

            if smoke_mode:
                gray = image.convert("L")
                for (_, _), (px, py) in anchors.items():
                    lum = local_luminance(gray, int(round(px)), int(round(py)))
                    total_dots += 1
                    if lum < 128.0:
                        white_dots += 1
                    else:
                        black_dots += 1

            clusters = cluster_points(
                points=points,
                anchors=anchors,
                width=width,
                height=height,
                bbox_size=args.bbox_size,
            )
            target_json = build_json_target(clusters)

            overlay_path = os.path.abspath(os.path.join(overlay_dir, image_name))
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            overlaid.save(overlay_path)

            prompt = PROMPT_TEMPLATE.format(guideline=GRID_GUIDELINE, category=category)
            record = {
                "id": image_name,
                "image": overlay_path,
                "source_image": os.path.abspath(image_path),
                "split": split_name,
                "category": category,
                "problem": prompt,
                "solution": target_json,
                "ground_truth_count": int(len(points)),
                "clusters": clusters,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": target_json},
                ],
            }
            split_records.append(record)

            if args.max_samples_per_split > 0 and len(split_records) >= args.max_samples_per_split:
                break

        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(split_records, out_path)
        all_records.extend(split_records)
        print(f"{split_name}: wrote {len(split_records)} records to {out_path} ({skipped} skipped)")

        if effective_total_cap > 0 and len(all_records) >= effective_total_cap:
            print(f"Total cap of {effective_total_cap} reached after processing split '{split_name}'.")
            break

    all_path = os.path.join(args.output_dir, "all.jsonl")
    write_jsonl(all_records, all_path)
    print(f"all: wrote {len(all_records)} records to {all_path}")

    if smoke_mode and total_dots > 0:
        print("\n=== OCR Visibility Smoke Check ===")
        print(f"Total dots sampled      : {total_dots}")
        print(f"White dots (dark BG)    : {white_dots} ({100.0 * white_dots / total_dots:.1f}%)")
        print(f"Black dots (light BG)   : {black_dots} ({100.0 * black_dots / total_dots:.1f}%)")
        contrast_ok = (white_dots / total_dots >= 0.05) and (black_dots / total_dots >= 0.05)
        print(f"Contrast diversity check: {'PASS' if contrast_ok else 'FAIL — review dot_radius or background diversity'}")
        print(f"Overlay images saved to : {overlay_dir}")
        print("Manually inspect 2-3 images in that directory to confirm dot visibility.")


if __name__ == "__main__":
    main()
