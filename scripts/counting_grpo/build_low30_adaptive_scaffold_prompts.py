#!/usr/bin/env python3
"""Build prompt-only adaptive scaffold JSONL for low-count FSC147 rows.

The output is intentionally input-only: it contains the original/scaffolded
image paths and the scaffold-mode counting prompt, but no ground-truth counts,
FSC point annotations, scaffold coordinates, boxes, clusters, or assistant
targets. A later SFT builder can merge this prompt file with a trusted GT
source to create supervised HF messages.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from dryrun_adaptive_scaffold import (
    DEFAULT_ROOT,
    choose_foreground_mask,
    foreground_scores_from_pca,
    load_classes,
)
from dryrun_low30_scaffold_prompt_jsonl import (
    LOW_DENSITY_REGIME_PROXY,
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


DEFAULT_MASTER_JSONL = "outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl"
DEFAULT_OUT_DIR = "outputs/scaffold_prompt_low30_adaptive"
DEFAULT_JSONL = "outputs/scaffold_prompt_low30_adaptive/low30_scaffold_input_only.jsonl"


def normalize_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return Path(Path(raw).name).stem


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "objects"


def parse_count(row: Dict[str, Any]) -> int:
    for key in ("gt_count", "ground_truth_count", "total_count"):
        if row.get(key) is not None:
            return int(row[key])
    solution = row.get("solution") or row.get("response") or row.get("target_response")
    if isinstance(solution, dict) and solution.get("total_count") is not None:
        return int(solution["total_count"])
    if isinstance(solution, str):
        try:
            parsed = json.loads(solution)
            if isinstance(parsed, dict) and parsed.get("total_count") is not None:
                return int(parsed["total_count"])
        except Exception:
            pass
        match = re.search(r'"total_count"\s*:\s*(-?\d+)', solution)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse count for row {row.get('id')}")


def load_low_count_rows(
    master_jsonl: Path,
    fsc147_root: Path,
    max_count: int,
    inclusive: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    classes = load_classes(fsc147_root / "ImageClasses_FSC147.txt")
    rows: List[Dict[str, Any]] = []
    seen = set()
    with master_jsonl.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            source = json.loads(raw)
            gt_count = parse_count(source)
            keep = gt_count <= max_count if inclusive else gt_count < max_count
            if not keep:
                continue

            qid = normalize_id(source.get("id") or source.get("question_id") or source.get("source_image"))
            if not qid or qid in seen:
                continue
            seen.add(qid)

            source_image = source.get("source_image") or str(fsc147_root / "images_384_VarV2" / f"{qid}.jpg")
            image_path = Path(source_image)
            if not image_path.is_absolute():
                image_path = (fsc147_root / image_path).resolve()
            category = str(source.get("category") or classes.get(f"{qid}.jpg", "objects"))
            rows.append(
                {
                    "question_id": qid,
                    "source_image": str(image_path),
                    "category": category,
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
        image_path = Path(row["source_image"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        grouped[processed_size(str(image_path), img_size)].append(row)

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
                image_rgb = tensor_to_uint8_image(tensors[idx])
                mask, _ = choose_foreground_mask(
                    scores[idx],
                    [],
                    LOW_DENSITY_REGIME_PROXY,
                    max(height, width),
                    mask_threshold,
                )
                grid_h, grid_w = grid_shape_from_image(height, width, short_side_grid=5)
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
            if total_done == len(selected) or total_done % 50 == 0:
                print(f"processed={total_done}/{len(selected)}", flush=True)

    output_rows.sort(key=lambda row: int(row["question_id"]) if str(row["question_id"]).isdigit() else row["question_id"])
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc147_root", default=DEFAULT_ROOT)
    parser.add_argument("--master_jsonl", default=DEFAULT_MASTER_JSONL)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--max_count", type=int, default=30)
    parser.add_argument("--inclusive", type=int, default=0, help="Use <= max_count instead of < max_count")
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap for dry runs")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--mask_threshold", type=float, default=0.6)
    parser.add_argument("--pca_mode", choices=["per_image", "global"], default="per_image")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.img_size % PATCH_SIZE != 0:
        raise ValueError(f"--img_size must be divisible by {PATCH_SIZE}")

    root = Path(args.fsc147_root).resolve()
    master_jsonl = Path(args.master_jsonl).resolve()
    out_dir = Path(args.output_dir).resolve()
    image_out_dir = out_dir / "images"
    out_jsonl = Path(args.output_jsonl).resolve()
    image_out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_low_count_rows(
        master_jsonl=master_jsonl,
        fsc147_root=root,
        max_count=args.max_count,
        inclusive=bool(args.inclusive),
        limit=args.limit,
    )
    comparator = "<=" if args.inclusive else "<"
    print(f"selected_rows={len(selected)} count{comparator}{args.max_count}", flush=True)
    if not selected:
        raise RuntimeError("No low-count rows selected")

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
