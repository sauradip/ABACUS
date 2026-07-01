#!/usr/bin/env python3
"""Build Stage 1.5 PCA/composite dataset from an existing scaffold JSONL.

Outputs:
- <output_dir>/all.jsonl (same rows as input plus `pca_image` field)
- <output_dir>/<pca_subdir>/*.jpg semantic composites

Design goals:
- Fail-fast on missing input images when strict mode is enabled
- Resumable generation (skip existing composites by default)
- JSONL schema compatible with Stage 1.5 preflight + trainer
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


DEFAULT_PCA_SUBDIR = "images_pca"


@dataclass
class Item:
    idx: int
    row: Dict[str, Any]
    src_path: Path
    out_rel: str
    out_abs: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--pca_field", default="pca_image")
    parser.add_argument("--pca_subdir", default=DEFAULT_PCA_SUBDIR)
    parser.add_argument("--model_name", default="dinov2_vits14")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--blend_alpha", type=float, default=0.55)
    parser.add_argument("--mask_threshold", type=float, default=0.60)
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--strict", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def rank0_print(*args: Any) -> None:
    print(*args, flush=True)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list in {path}")
            return data
        rows: List[Dict[str, Any]] = []
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        return rows


def sanitize_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe or "item"


def choose_id(row: Dict[str, Any], idx: int) -> str:
    for key in ("id", "image_id", "uid"):
        value = row.get(key)
        if value:
            return str(value)
    return f"row_{idx:06d}.jpg"


def build_items(
    rows: Sequence[Dict[str, Any]],
    input_jsonl: Path,
    image_field: str,
    output_dir: Path,
    pca_subdir: str,
) -> Tuple[List[Item], List[str]]:
    data_dir = input_jsonl.parent
    out_img_dir = output_dir / pca_subdir
    out_img_dir.mkdir(parents=True, exist_ok=True)

    items: List[Item] = []
    missing: List[str] = []

    for idx, row in enumerate(rows):
        src_raw = row.get(image_field)
        if not src_raw:
            missing.append(f"row={idx} missing image field '{image_field}'")
            continue

        src_path = Path(str(src_raw))
        if not src_path.is_absolute():
            src_path = (data_dir / src_path).resolve()

        if not src_path.exists():
            missing.append(f"row={idx} missing image file {src_path}")
            continue

        row_id = choose_id(row, idx)
        stem = sanitize_name(Path(row_id).stem)
        out_name = f"{stem}.jpg"
        out_rel = f"{pca_subdir}/{out_name}"
        out_abs = out_img_dir / out_name

        items.append(Item(idx=idx, row=row, src_path=src_path, out_rel=out_rel, out_abs=out_abs))

    return items, missing


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(img_size + int(img_size * 0.01) * 10),
            T.CenterCrop(img_size),
            T.Normalize([0.5], [0.5]),
        ]
    )


def render_semantic_composite(
    image_rgb: np.ndarray,
    patch_tokens: np.ndarray,
    patch_h: int,
    patch_w: int,
    mask_threshold: float,
    blend_alpha: float,
) -> np.ndarray:
    token_count = patch_h * patch_w
    if patch_tokens.shape[0] != token_count:
        raise ValueError(f"Token count mismatch: got={patch_tokens.shape[0]} expected={token_count}")

    fg_pca = PCA(n_components=1)
    fg_vals = fg_pca.fit_transform(patch_tokens)
    fg_vals = minmax_scale(fg_vals.reshape(-1))
    mask = fg_vals > mask_threshold

    if int(mask.sum()) < 3:
        mask = np.ones_like(mask, dtype=bool)

    pca3 = PCA(n_components=3)
    pca_features = pca3.fit_transform(patch_tokens[mask])
    pca_features = minmax_scale(pca_features)

    pca_grid = np.zeros((token_count, 3), dtype=np.float32)
    pca_grid[mask] = pca_features
    pca_grid = pca_grid.reshape(patch_h, patch_w, 3)

    h, w = image_rgb.shape[:2]
    pca_up = cv2.resize(pca_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    pca_uint8 = np.clip(pca_up * 255.0, 0, 255).astype(np.uint8)

    # Keep scaffold cues while injecting semantic channels.
    comp = cv2.addWeighted(image_rgb, blend_alpha, pca_uint8, 1.0 - blend_alpha, 0)
    return comp


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.img_size % 14 != 0:
        raise ValueError("img_size must be divisible by 14")

    rows = load_json_or_jsonl(input_jsonl)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    items, missing = build_items(
        rows=rows,
        input_jsonl=input_jsonl,
        image_field=args.image_field,
        output_dir=output_dir,
        pca_subdir=args.pca_subdir,
    )

    rank0_print("=== Stage15 PCA Dataset Builder ===")
    rank0_print(f"input_jsonl   : {input_jsonl}")
    rank0_print(f"output_dir    : {output_dir}")
    rank0_print(f"rows_loaded    : {len(rows)}")
    rank0_print(f"rows_resolved  : {len(items)}")
    rank0_print(f"missing_rows   : {len(missing)}")

    if missing:
        for msg in missing[:20]:
            rank0_print("MISSING:", msg)
        if bool(args.strict):
            rank0_print("FAIL: missing input images in strict mode")
            return 1

    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        rank0_print("WARN: CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"

    device = torch.device(requested_device)
    rank0_print(f"device        : {device}")
    rank0_print(f"loading model : {args.model_name}")

    model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    model = model.to(device)
    model.eval()

    transform = build_transform(args.img_size)
    patch_h = args.img_size // 14
    patch_w = args.img_size // 14

    to_generate: List[Item] = []
    skipped = 0
    for item in items:
        if bool(args.skip_existing) and item.out_abs.exists():
            skipped += 1
            continue
        to_generate.append(item)

    rank0_print(f"existing_skipped: {skipped}")
    rank0_print(f"to_generate     : {len(to_generate)}")

    generated = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(chunked(to_generate, max(1, args.batch_size)), start=1):
            batch_tensors: List[torch.Tensor] = []
            batch_images_rgb: List[np.ndarray] = []

            for item in batch:
                with Image.open(item.src_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_rgb = np.array(img)
                    batch_images_rgb.append(img_rgb)
                    batch_tensors.append(transform(img))

            x = torch.stack(batch_tensors, dim=0).to(device)
            features = model.forward_features(x)
            tokens = features["x_norm_patchtokens"].detach().cpu().numpy()

            for i, item in enumerate(batch):
                comp = render_semantic_composite(
                    image_rgb=batch_images_rgb[i],
                    patch_tokens=tokens[i],
                    patch_h=patch_h,
                    patch_w=patch_w,
                    mask_threshold=float(args.mask_threshold),
                    blend_alpha=float(args.blend_alpha),
                )
                item.out_abs.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(comp).save(item.out_abs, format="JPEG", quality=95)
                generated += 1

            if batch_idx % 10 == 0:
                rank0_print(f"progress: batch={batch_idx} generated={generated}/{len(to_generate)}")

    # Build output rows preserving order of input rows.
    out_rows = [dict(row) for row in rows]
    for item in items:
        out_rows[item.idx][args.pca_field] = item.out_rel

    output_jsonl = output_dir / "all.jsonl"
    write_jsonl(output_jsonl, out_rows)

    rank0_print(f"generated_images: {generated}")
    rank0_print(f"written_jsonl   : {output_jsonl}")
    rank0_print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
