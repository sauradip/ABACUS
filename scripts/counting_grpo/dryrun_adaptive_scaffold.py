#!/usr/bin/env python3
"""Dry-run adaptive scaffold visualizations on FSC147 images.

This does not create training data. It writes visualizations so we can inspect
whether a DINO/PCA foreground mask plus density-adaptive, OCR-readable scaffold
spacing is sensible.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


DEFAULT_ROOT = "/projects/u6fb/myprojects/FSC147_hf"
DEFAULT_OUT = "outputs/adaptive_scaffold_dryrun"
DEFAULT_SAMPLES = [
    "7.jpg",    # low, train, peppers, 13
    "190.jpg",  # low, val, seagulls, 13
    "2.jpg",    # low, test, sea shells, 8
    "42.jpg",   # mid, train, tomatoes, 68
    "194.jpg",  # mid, val, peaches, 82
    "295.jpg",  # mid, test, strawberries, 64
    "31.jpg",   # high, train, tomatoes, 126
    "215.jpg",  # high, val, grapes, 259
    "5.jpg",    # high, test, hot air balloons, 113
]


def load_classes(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                image_name, category = line.rstrip("\n").split("\t", 1)
                out[image_name] = category
    return out


def count_regime(gt_count: int) -> str:
    if gt_count < 30:
        return "low_lt30"
    if gt_count <= 100:
        return "mid_30_100"
    return "high_gt100"


def adaptive_grid_size(gt_count: int) -> int:
    if gt_count < 30:
        return 5
    if gt_count <= 100:
        return 7
    return 9


def sparse_background_factor() -> int:
    return 2


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(img_size + int(img_size * 0.01) * 10),
            T.CenterCrop(img_size),
            T.Normalize([0.5], [0.5]),
        ]
    )


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    arr = ((tensor.cpu().numpy() * 0.5 + 0.5) * 255.0)
    arr = np.clip(arr, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return arr


def foreground_scores_from_pca(
    tokens: np.ndarray,
    img_count: int,
    patch_h: int,
    patch_w: int,
    mode: str,
) -> np.ndarray:
    if mode == "global":
        flat = tokens.reshape(img_count * patch_h * patch_w, -1)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(flat)
        scores = minmax_scale(scores.reshape(-1))
        return scores.reshape(img_count, patch_h, patch_w)

    if mode != "per_image":
        raise ValueError("--pca_mode must be 'per_image' or 'global'")

    score_maps = []
    for idx in range(img_count):
        flat = tokens[idx].reshape(patch_h * patch_w, -1)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(flat)
        scores = minmax_scale(scores.reshape(-1))
        score_maps.append(scores.reshape(patch_h, patch_w))
    return np.stack(score_maps, axis=0)


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8.astype(bool)


def transform_points(
    points: List[List[float]],
    original_size: Tuple[int, int],
    resize_short: int,
    crop_size: int,
) -> List[Tuple[float, float]]:
    width, height = original_size
    scale = float(resize_short) / float(min(width, height))
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    left = max(0.0, (resized_w - crop_size) / 2.0)
    top = max(0.0, (resized_h - crop_size) / 2.0)

    out = []
    for point in points:
        if len(point) < 2:
            continue
        x = float(point[0]) * scale - left
        y = float(point[1]) * scale - top
        if 0 <= x < crop_size and 0 <= y < crop_size:
            out.append((x, y))
    return out


def point_coverage(mask: np.ndarray, points_xy: List[Tuple[float, float]], img_size: int) -> float:
    if not points_xy:
        return 0.0
    mask_big = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST).astype(bool)
    hits = 0
    for x, y in points_xy:
        xi = min(img_size - 1, max(0, int(round(x))))
        yi = min(img_size - 1, max(0, int(round(y))))
        if mask_big[yi, xi]:
            hits += 1
    return float(hits) / float(len(points_xy))


def target_mask_ratio(gt_count: int) -> float:
    if gt_count < 30:
        return 0.25
    if gt_count <= 100:
        return 0.50
    return 0.70


def choose_foreground_mask(
    score_map: np.ndarray,
    points_xy: List[Tuple[float, float]],
    gt_count: int,
    img_size: int,
    threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    low_threshold = 1.0 - threshold
    quantile_hi = float(np.quantile(score_map, 0.65))
    quantile_lo = float(np.quantile(score_map, 0.35))
    candidates = [
        ("pca_high", score_map >= threshold),
        ("pca_low", score_map <= low_threshold),
        ("top35", score_map >= quantile_hi),
        ("bottom35", score_map <= quantile_lo),
    ]

    target_ratio = target_mask_ratio(gt_count)
    ranked = []
    for name, raw_mask in candidates:
        ratio = float(raw_mask.mean())
        coverage = point_coverage(raw_mask, points_xy, img_size)
        score = coverage - 0.30 * abs(ratio - target_ratio)
        if ratio < 0.01:
            score -= 0.50
        if ratio > 0.98:
            score -= 0.20
        ranked.append((score, name, raw_mask, ratio, coverage))

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, name, raw_mask, raw_ratio, raw_coverage = ranked[0]
    mask = raw_mask.astype(bool)
    smoothing = "none"

    info = {
        "mask_selector": name,
        "mask_score": float(ranked[0][0]),
        "raw_foreground_patch_ratio": raw_ratio,
        "raw_point_coverage": raw_coverage,
        "foreground_patch_ratio": float(mask.mean()),
        "point_coverage": point_coverage(mask, points_xy, img_size),
        "smoothing": smoothing,
    }
    return mask, info


def pca_mask_overlay(image_rgb: np.ndarray, score_map: np.ndarray, mask: np.ndarray) -> Image.Image:
    h, w = image_rgb.shape[:2]
    score_big = cv2.resize(score_map.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_big = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    score_big = np.clip(score_big, 0.0, 1.0)
    score_big[~mask_big] = 0.0
    heat = cv2.applyColorMap((score_big * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.50 * image_rgb + 0.50 * heat).astype(np.uint8)
    return Image.fromarray(overlay)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def contrast_rgba(image_rgb: np.ndarray, x: int, y: int) -> Tuple[int, int, int, int]:
    pixel = image_rgb[min(image_rgb.shape[0] - 1, y), min(image_rgb.shape[1] - 1, x)]
    if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) >= 255 * 3 / 2:
        return (0, 0, 0, 255)
    return (255, 255, 255, 255)


def draw_labeled_dot(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    font: ImageFont.ImageFont,
    radius: int,
    image_w: int,
    image_h: int,
    color: Tuple[int, int, int, int],
) -> None:
    inverse = (255 - color[0], 255 - color[1], 255 - color[2], 145)
    draw.ellipse(
        [(x - radius - 1, y - radius - 1), (x + radius + 1, y + radius + 1)],
        fill=inverse,
    )
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)

    pad = 2
    text_w, text_h = text_size(draw, label, font)
    # Place coordinate text at the bottom-right of each point by default.
    label_x = x + radius + 4
    label_y = y + radius + 4
    if label_x + text_w + 2 * pad >= image_w:
        label_x = x - radius - 4 - text_w - 2 * pad
    if label_y + text_h + 2 * pad >= image_h:
        label_y = y - radius - 4 - text_h - 2 * pad
    label_x = max(0, min(image_w - text_w - 2 * pad, label_x))
    label_y = max(0, min(image_h - text_h - 2 * pad, label_y))

    draw.text((label_x + pad, label_y + pad), label, fill=color, font=font)


def draw_adaptive_scaffold(image_rgb: np.ndarray, mask: np.ndarray, gt_count: int) -> Tuple[Image.Image, int, int, int, int, List[Dict[str, Any]]]:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    grid_n = adaptive_grid_size(gt_count)
    sparse_factor = sparse_background_factor()
    mask_big = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    font = load_font(max(14, min(22, int(w / (grid_n * 2.6)))))
    radius = max(2, int(w / (grid_n * 130)))

    cell_w = float(w) / float(grid_n + 1)
    cell_h = float(h) / float(grid_n + 1)

    fg_dot_count = 0
    bg_dot_count = 0
    scaffold_points: List[Dict[str, Any]] = []
    for row in range(1, grid_n + 1):
        for col in range(1, grid_n + 1):
            x = int(col * cell_w)
            y = int(row * cell_h)
            is_foreground = mask_big[min(h - 1, y), min(w - 1, x)]
            is_sparse_background = (
                not is_foreground
                and row % sparse_factor == 0
                and col % sparse_factor == 0
            )
            if not is_foreground and not is_sparse_background:
                continue

            if is_foreground:
                fg_dot_count += 1
                point_kind = "foreground"
            else:
                bg_dot_count += 1
                point_kind = "background"

            scaffold_points.append(
                {
                    "coord": [row, col],
                    "pixel": [x, y],
                    "kind": point_kind,
                }
            )

            draw_labeled_dot(
                draw,
                x,
                y,
                f"({row},{col})",
                font,
                radius,
                w,
                h,
                contrast_rgba(image_rgb, x, y),
            )
    return image, grid_n, fg_dot_count, sparse_factor, bg_dot_count, scaffold_points


def make_contact_sheet(records: List[Dict[str, Any]], out_path: Path, thumb: int = 448) -> None:
    title_h = 48
    cols = 3
    rows = len(records)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + title_h)), (245, 245, 245))
    font = load_font(14)
    header_font = load_font(16)
    col_titles = ["Original", "DINO PCA foreground map", "Adaptive scaffold"]
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(col_titles):
        draw.text((c * thumb + 8, 6), title, fill=(0, 0, 0), font=header_font)

    for r, rec in enumerate(records):
        y0 = r * (thumb + title_h) + title_h
        label = (
            f"{rec['regime']} | {rec['id']} | {rec['category']} | "
            f"GT={rec['gt_count']} | FG={rec['grid_n']}x{rec['grid_n']}/{rec['fg_dot_count']} | "
            f"sparse={rec['background_sparse_factor']}/{rec['background_dot_count']}"
        )
        draw.text((8, y0 - 24), label, fill=(0, 0, 0), font=font)
        for c, key in enumerate(["original_path", "mask_path", "scaffold_path"]):
            img = Image.open(rec[key]).convert("RGB")
            img.thumbnail((thumb, thumb))
            x = c * thumb + (thumb - img.width) // 2
            y = y0 + (thumb - img.height) // 2
            sheet.paste(img, (x, y))
    sheet.save(out_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc147_root", default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
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
    out_dir.mkdir(parents=True, exist_ok=True)

    with (root / "annotation_FSC147_384.json").open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    classes = load_classes(root / "ImageClasses_FSC147.txt")

    transform = build_transform(args.img_size)
    resize_short = args.img_size + int(args.img_size * 0.01) * 10
    tensors: List[torch.Tensor] = []
    metadata: List[Dict[str, Any]] = []
    for image_name in args.samples:
        image_path = root / "images_384_VarV2" / image_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        tensors.append(tensor)
        raw_points = annotations[image_name].get("points", [])
        gt_count = len(raw_points)
        points_xy = transform_points(raw_points, image.size, resize_short, args.img_size)
        metadata.append(
            {
                "id": image_name,
                "category": classes.get(image_name, "objects"),
                "gt_count": gt_count,
                "regime": count_regime(gt_count),
                "source_image": str(image_path),
                "_points_xy": points_xy,
                "points_in_crop": len(points_xy),
            }
        )

    if args.img_size % 14 != 0:
        raise ValueError("--img_size must be divisible by 14")
    patch_h = patch_w = args.img_size // 14

    print(f"Loading DINOv2 on {args.device}", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(torch.device(args.device)).eval()
    batch = torch.stack(tensors).to(torch.device(args.device))
    with torch.no_grad():
        features = model.forward_features(batch)
    tokens = features["x_norm_patchtokens"].detach().cpu().numpy()
    scores = foreground_scores_from_pca(tokens, len(tensors), patch_h, patch_w, args.pca_mode)

    records: List[Dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        image_rgb = tensor_to_uint8_image(tensors[idx])
        mask, mask_info = choose_foreground_mask(
            scores[idx],
            meta["_points_xy"],
            meta["gt_count"],
            args.img_size,
            args.mask_threshold,
        )
        original = Image.fromarray(image_rgb)
        mask_img = pca_mask_overlay(image_rgb, scores[idx], mask)
        scaffold_img, grid_n, fg_dot_count, sparse_factor, bg_dot_count, scaffold_points = draw_adaptive_scaffold(
            image_rgb,
            mask,
            meta["gt_count"],
        )

        stem = f"{meta['regime']}_{Path(meta['id']).stem}_{meta['category'].replace(' ', '_')}"
        original_path = out_dir / f"{stem}_original.jpg"
        mask_path = out_dir / f"{stem}_dino_mask.jpg"
        scaffold_path = out_dir / f"{stem}_adaptive_scaffold.jpg"
        original.save(original_path, quality=95)
        mask_img.save(mask_path, quality=95)
        scaffold_img.save(scaffold_path, quality=95)

        rec = {
            **{key: value for key, value in meta.items() if not key.startswith("_")},
            **mask_info,
            "pca_mode": args.pca_mode,
            "grid_n": grid_n,
            "fg_dot_count": fg_dot_count,
            "background_sparse_factor": sparse_factor,
            "background_dot_count": bg_dot_count,
            "scaffold_points": scaffold_points,
            "dot_count": fg_dot_count + bg_dot_count,
            "original_path": str(original_path),
            "mask_path": str(mask_path),
            "scaffold_path": str(scaffold_path),
        }
        records.append(rec)
        print(json.dumps(rec, sort_keys=True), flush=True)

    contact_path = out_dir / "contact_sheet.jpg"
    make_contact_sheet(records, contact_path)
    regime_contact_sheets: Dict[str, str] = {}
    for regime in ["low_lt30", "mid_30_100", "high_gt100"]:
        regime_records = [rec for rec in records if rec["regime"] == regime]
        if not regime_records:
            continue
        regime_contact_path = out_dir / f"contact_sheet_{regime}.jpg"
        make_contact_sheet(regime_records, regime_contact_path)
        regime_contact_sheets[regime] = str(regime_contact_path)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "records": records,
                "contact_sheet": str(contact_path),
                "regime_contact_sheets": regime_contact_sheets,
            },
            handle,
            indent=2,
        )
    print(f"contact_sheet={contact_path}", flush=True)
    for regime, path in regime_contact_sheets.items():
        print(f"{regime}_contact_sheet={path}", flush=True)


if __name__ == "__main__":
    main()
