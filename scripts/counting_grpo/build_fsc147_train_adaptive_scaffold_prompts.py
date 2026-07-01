#!/usr/bin/env python3
"""Build prompt-only adaptive scaffold JSONL for an FSC147 split.

This writes model inputs only: original/scaffolded image paths plus the
scaffold-mode counting prompt. It does not write ground-truth counts, FSC point
annotations, scaffold coordinates, boxes, clusters, or assistant targets.
Counts are used only inside the builder to choose the adaptive scaffold density.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image

from dryrun_adaptive_scaffold import (
    DEFAULT_ROOT,
    choose_foreground_mask,
    foreground_scores_from_pca,
    load_classes,
)
from dryrun_low30_scaffold_prompt_jsonl import (
    PATCH_SIZE,
    build_question,
    draw_scaffold_from_shape,
    get_textual_guidelines,
    grid_shape_from_image,
    image_to_dino_tensor,
    resize_keep_aspect,
    tensor_to_uint8_image,
    validate_input_only_row,
    write_jsonl,
)


DEFAULT_OUT_DIR = "outputs/scaffold_prompt_fsc147_train_adaptive"
DEFAULT_JSONL = "outputs/scaffold_prompt_fsc147_train_adaptive/train_scaffold_input_only.jsonl"


def scaffold_short_side_grid(gt_count: int) -> int:
    """User-specified scaffold short-side grid policy."""
    if gt_count < 30:
        return 9
    if gt_count <= 100:
        return 7
    return 6


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "objects"


def load_split_names(split_json: Path, split: str) -> List[str]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {split_json}; available={sorted(data)}")
    return [str(name) for name in data[split]]


def load_selected_rows(root: Path, split: str, limit: int) -> List[Dict[str, Any]]:
    split_names = load_split_names(root / "Train_Test_Val_FSC_147.json", split)
    annotations = json.loads((root / "annotation_FSC147_384.json").read_text(encoding="utf-8"))
    classes = load_classes(root / "ImageClasses_FSC147.txt")

    rows: List[Dict[str, Any]] = []
    for image_name in split_names:
        if image_name not in annotations:
            raise KeyError(f"Missing FSC147 annotation for {image_name}")
        image_path = root / "images_384_VarV2" / image_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        rows.append(
            {
                "question_id": Path(image_name).stem,
                "image_name": image_name,
                "source_image": str(image_path),
                "category": classes.get(image_name, "objects"),
                "_gt_count": len(annotations[image_name].get("points", [])),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def processed_size(image_path: str, short_side: int) -> Tuple[int, int]:
    image = Image.open(image_path).convert("RGB")
    processed = resize_keep_aspect(image, short_side)
    return processed.size


def build_prompt_rows(
    selected: List[Dict[str, Any]],
    image_out_dir: Path,
    img_size: int,
    mask_threshold: float,
    pca_mode: str,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[processed_size(row["source_image"], img_size)].append(row)

    print(f"Loading DINOv2 on {device}", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()

    output_rows: List[Dict[str, Any]] = []
    total_done = 0
    for size, group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        width, height = size
        if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
            raise ValueError(f"Processed image size must be divisible by {PATCH_SIZE}: {(width, height)}")
        patch_h = height // PATCH_SIZE
        patch_w = width // PATCH_SIZE

        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            processed_images: List[Image.Image] = []
            tensors: List[torch.Tensor] = []
            for meta in chunk:
                image = Image.open(meta["source_image"]).convert("RGB")
                processed = resize_keep_aspect(image, img_size)
                processed_images.append(processed)
                tensors.append(image_to_dino_tensor(processed))

            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                features = model.forward_features(batch)
            tokens = features["x_norm_patchtokens"].detach().cpu().numpy()
            scores = foreground_scores_from_pca(tokens, len(chunk), patch_h, patch_w, pca_mode)

            for idx, meta in enumerate(chunk):
                gt_count = int(meta["_gt_count"])
                image_rgb = tensor_to_uint8_image(tensors[idx])
                mask, _ = choose_foreground_mask(
                    scores[idx],
                    [],
                    gt_count,
                    max(height, width),
                    mask_threshold,
                )
                short_side_grid = scaffold_short_side_grid(gt_count)
                grid_h, grid_w = grid_shape_from_image(
                    height,
                    width,
                    short_side_grid=short_side_grid,
                )
                scaffold_img, _ = draw_scaffold_from_shape(
                    image_rgb,
                    mask,
                    grid_h,
                    grid_w,
                    sparse_factor=2,
                )

                stem = f"{meta['question_id']}_{safe_slug(meta['category'])}"
                original_path = image_out_dir / f"{stem}.jpg"
                scaffold_path = image_out_dir / f"{stem}_dots.jpg"
                processed_images[idx].save(original_path, quality=95)
                scaffold_img.save(scaffold_path, quality=95)

                out_row = {
                    "question_id": meta["question_id"],
                    "image_paths": [str(original_path), str(scaffold_path)],
                    "question": build_question(meta["category"]),
                    "history": get_textual_guidelines(grid_h, grid_w),
                }
                validate_input_only_row(out_row)
                output_rows.append(out_row)

            total_done += len(chunk)
            if total_done == len(selected) or total_done % 100 == 0:
                print(f"processed={total_done}/{len(selected)}", flush=True)

    split_order = {row["question_id"]: idx for idx, row in enumerate(selected)}
    output_rows.sort(key=lambda row: split_order[row["question_id"]])
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc147_root", default=DEFAULT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap for dry runs")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--mask_threshold", type=float, default=0.6)
    parser.add_argument("--pca_mode", choices=["per_image", "global"], default="per_image")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.img_size % PATCH_SIZE != 0:
        raise ValueError(f"--img_size must be divisible by {PATCH_SIZE}")

    root = Path(args.fsc147_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    image_out_dir = out_dir / "images"
    out_jsonl = Path(args.output_jsonl).resolve()
    image_out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected_rows(root, args.split, args.limit)
    print(f"selected_rows={len(selected)} split={args.split}", flush=True)
    if not selected:
        raise RuntimeError("No FSC147 rows selected")

    rows = build_prompt_rows(
        selected=selected,
        image_out_dir=image_out_dir,
        img_size=args.img_size,
        mask_threshold=args.mask_threshold,
        pca_mode=args.pca_mode,
        device=torch.device(args.device),
        batch_size=max(1, args.batch_size),
    )
    written = write_jsonl(out_jsonl, rows)
    print(f"wrote_rows={written} -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
