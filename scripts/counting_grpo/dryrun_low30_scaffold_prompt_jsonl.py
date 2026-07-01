#!/usr/bin/env python3
"""Build input-only scaffold-mode JSONL for the earlier low-count FSC147 samples.

This is a dry run for prompt/data shape only. It writes scaffolded images plus a
JSONL file containing the scaffold-mode counting prompt and the two image paths
used by the reference scaffold guide: original image first, scaffolded image
second. It intentionally does not write ground-truth counts, FSC point
annotations, assistant responses, target responses, object clusters, or
scaffold coordinate metadata.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from dryrun_adaptive_scaffold import (
    DEFAULT_ROOT,
    choose_foreground_mask,
    contrast_rgba,
    draw_labeled_dot,
    foreground_scores_from_pca,
    load_font,
    load_classes,
)


DEFAULT_OUT_DIR = "outputs/scaffold_prompt_low30_dryrun"
DEFAULT_JSONL = "outputs/scaffold_prompt_low30_dryrun/low30_scaffold_input_only.jsonl"
DEFAULT_SAMPLES = [
    "7.jpg",    # peppers, low-count example used in the visual dry run
    "190.jpg",  # seagulls, low-count example used in the visual dry run
    "2.jpg",    # sea shells, low-count example used in the visual dry run
]

# We only need a low-density regime proxy to select the sparse scaffold geometry.
# The exact FSC147 count is deliberately not loaded or written by this dry run.
LOW_DENSITY_REGIME_PROXY = 29
PATCH_SIZE = 14


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def get_textual_guidelines(h: int, w: int) -> List[Dict[str, Any]]:
    text = (
        f"I will provide you with two images of the same scene. The first image is the original scene. "
        f"The second image is overlaid with a dot matrix of a shape of {h} * {w} to help with counting, "
        "and each visible dot is labeled with two-dimensional coordinates (x,y).\n"
        " 1. You should identify the number of objects in the question and link them with their nearest coordinate.\n"
        " 2. Use the coordinates to estimate the objects counts in the question and avoid over-counting. Within each column, "
        "the x-coordinate increases from top to bottom, and within each row, the y-coordinate increases from left to right.\n"
        " 3. Search and count region by region with the help of the dots.\n"
        " 4. Finally, conclude with strict JSON only in the format {\"total_count\": <integer>}. Let's think step by step."
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    ]


def build_question(category: str) -> str:
    return (
        f"Count the {category} in the scene. "
        "Return strict JSON only with exactly one key: total_count."
    )


def resize_keep_aspect(image: Image.Image, short_side: int, patch_size: int = PATCH_SIZE) -> Image.Image:
    width, height = image.size
    scale = float(short_side) / float(min(width, height))
    new_w = max(patch_size, int(round(width * scale / patch_size)) * patch_size)
    new_h = max(patch_size, int(round(height * scale / patch_size)) * patch_size)
    return image.resize((new_w, new_h), Image.BICUBIC)


def image_to_dino_tensor(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    return transform(image)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    arr = ((tensor.cpu().numpy() * 0.5 + 0.5) * 255.0)
    arr = np.clip(arr, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return arr


def grid_shape_from_image(image_h: int, image_w: int, short_side_grid: int) -> Tuple[int, int]:
    if image_h <= image_w:
        grid_h = short_side_grid
        grid_w = max(short_side_grid, int(round(short_side_grid * image_w / image_h)))
    else:
        grid_w = short_side_grid
        grid_h = max(short_side_grid, int(round(short_side_grid * image_h / image_w)))
    return grid_h, grid_w


def draw_scaffold_from_shape(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    sparse_factor: int = 2,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image, "RGBA")
    image_w, image_h = image.size
    mask_big = cv2.resize(mask.astype(np.uint8), (image_w, image_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    font = load_font(max(14, min(22, int(min(image_w / max(grid_w, 1), image_h / max(grid_h, 1)) / 2.2))))
    radius = max(2, int(min(image_w / max(grid_w, 1), image_h / max(grid_h, 1)) / 22.0))

    cell_w = float(image_w) / float(grid_w + 1)
    cell_h = float(image_h) / float(grid_h + 1)
    scaffold_points: List[Dict[str, Any]] = []
    for row in range(1, grid_h + 1):
        for col in range(1, grid_w + 1):
            x = int(col * cell_w)
            y = int(row * cell_h)
            is_foreground = mask_big[min(image_h - 1, y), min(image_w - 1, x)]
            is_sparse_background = (
                not is_foreground
                and row % sparse_factor == 0
                and col % sparse_factor == 0
            )
            if not is_foreground and not is_sparse_background:
                continue

            draw_labeled_dot(
                draw,
                x,
                y,
                f"({row},{col})",
                font,
                radius,
                image_w,
                image_h,
                contrast_rgba(image_rgb, x, y),
            )
            scaffold_points.append(
                {
                    "coord": [row, col],
                    "pixel": [x, y],
                }
            )
    return image, scaffold_points


def validate_input_only_row(row: Dict[str, Any]) -> None:
    forbidden = {
        "answer",
        "annotations",
        "clusters",
        "ground_truth",
        "ground_truth_count",
        "gt_count",
        "points",
        "processed_original_image",
        "response",
        "scaffold_points",
        "source_image",
        "solution",
        "target_response",
        "total_count",
    }
    present = sorted(forbidden.intersection(row.keys()))
    if present:
        raise ValueError(f"Output row contains forbidden supervision fields: {present}")
    image_paths = row.get("image_paths")
    if not isinstance(image_paths, list) or len(image_paths) != 2:
        raise ValueError("Output row must contain image_paths as [original, scaffolded]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc147_root", default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--mask_threshold", type=float, default=0.6)
    parser.add_argument("--pca_mode", choices=["per_image", "global"], default="per_image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.fsc147_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    image_out_dir = out_dir / "images"
    out_jsonl = Path(args.output_jsonl).resolve()
    image_out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(root / "ImageClasses_FSC147.txt")

    sample_meta: List[Dict[str, Any]] = []
    for image_name in args.samples:
        source_image = root / "images_384_VarV2" / image_name
        if not source_image.exists():
            raise FileNotFoundError(source_image)
        image = Image.open(source_image).convert("RGB")
        processed_image = resize_keep_aspect(image, args.img_size)
        tensor = image_to_dino_tensor(processed_image)
        sample_meta.append(
            {
                "id": image_name,
                "category": classes.get(image_name, "objects"),
                "source_image": str(source_image),
                "processed_image": processed_image,
                "tensor": tensor,
            }
        )

    if args.img_size % 14 != 0:
        raise ValueError("--img_size must be divisible by 14")

    print(f"Loading DINOv2 on {args.device}", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(torch.device(args.device)).eval()

    rows: List[Dict[str, Any]] = []
    for meta in sample_meta:
        tensor = meta["tensor"]
        image_h = int(tensor.shape[1])
        image_w = int(tensor.shape[2])
        if image_h % PATCH_SIZE != 0 or image_w % PATCH_SIZE != 0:
            raise ValueError(f"Processed image size must be divisible by {PATCH_SIZE}: {(image_w, image_h)}")

        with torch.no_grad():
            features = model.forward_features(tensor.unsqueeze(0).to(torch.device(args.device)))
        tokens = features["x_norm_patchtokens"].detach().cpu().numpy()
        score = foreground_scores_from_pca(tokens, 1, image_h // PATCH_SIZE, image_w // PATCH_SIZE, args.pca_mode)[0]

        image_rgb = tensor_to_uint8_image(tensor)
        # Point-free foreground selection: no FSC point annotations are loaded.
        mask, _ = choose_foreground_mask(
            score,
            [],
            LOW_DENSITY_REGIME_PROXY,
            max(image_h, image_w),
            args.mask_threshold,
        )
        grid_h, grid_w = grid_shape_from_image(image_h, image_w, short_side_grid=5)
        sparse_factor = 2
        scaffold_img, _scaffold_points = draw_scaffold_from_shape(
            image_rgb,
            mask,
            grid_h,
            grid_w,
            sparse_factor=sparse_factor,
        )

        stem = f"{Path(meta['id']).stem}_{meta['category'].replace(' ', '_')}"
        original_path = image_out_dir / f"{stem}.jpg"
        scaffold_path = image_out_dir / f"{stem}_dots.jpg"
        meta["processed_image"].save(original_path, quality=95)
        scaffold_img.save(scaffold_path, quality=95)

        question = build_question(meta["category"])
        history = get_textual_guidelines(grid_h, grid_w)
        row = {
            "question_id": Path(meta["id"]).stem,
            "image_paths": [str(original_path), str(scaffold_path)],
            "question": question,
            "history": history,
        }
        validate_input_only_row(row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "question_id": row["question_id"],
                    "grid": [grid_h, grid_w],
                    "image_paths": row["image_paths"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    row_count = write_jsonl(out_jsonl, rows)
    print(f"wrote_rows={row_count} -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
