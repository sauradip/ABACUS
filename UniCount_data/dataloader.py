"""
dataloader.py
-------------
Reads WebDataset tar shards from the ani/data directory and provides a
torch-compatible DataLoader for the Rex-Omni pipeline.

Expected WebDataset shard structure per sample:
    <key>.jpg   – JPEG image bytes
    <key>.txt   – caption / description text

Usage:
    from dataloader import build_dataset, get_dataloader
    ds = build_dataset()
    for sample in get_dataloader(batch_size=1, num_workers=4):
        image   = sample["image"]    # PIL.Image
        caption = sample["caption"]  # str
"""

import glob
import io
import json
import os
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional

from PIL import Image
from datasets import Image as HFImage
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR   = os.environ.get("UNICOUNT_DATA_DIR", "/localhome/snag/project/huawei/ani/data")
CACHE_DIR  = os.environ.get("UNICOUNT_HF_CACHE", "/localhome/snag/.cache/huggingface")
NUM_PROC   = 128
IMAGE_SUFFIXES = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}
TEXT_SUFFIXES = {"txt", "text"}
JSON_SUFFIXES = {"json"}


def build_dataset(data_dir: str = DATA_DIR):
    """
    Discovers WebDataset shards in *data_dir* and returns a HuggingFace Dataset
    in streaming-compatible WebDataset format.

    *data_dir* may be either:
    - a directory containing one or more ``.tar`` shards, or
    - a path to a single ``.tar`` shard.

    Returns
    -------
    datasets.Dataset  (split="train")
    """
    source_path = Path(data_dir)
    if source_path.is_file():
        if source_path.suffix != ".tar":
            raise ValueError(f"Expected a .tar shard or directory, got '{data_dir}'")
        data_files = [str(source_path)]
    else:
        data_files = sorted(glob.glob(f"{data_dir}/*.tar"))

    if not data_files:
        raise FileNotFoundError(
            f"No .tar shards found in '{data_dir}'. "
            "Check that the path is correct and the shards are present."
        )

    dataset = load_dataset(
        "webdataset",
        data_files=data_files,
        cache_dir=CACHE_DIR,
        split="train",
        num_proc=NUM_PROC,
    )
    image_columns = [
        column_name
        for column_name, feature in dataset.features.items()
        if isinstance(feature, HFImage)
    ]
    for image_column in image_columns:
        dataset = dataset.cast_column(image_column, HFImage(decode=False))
    return dataset


# ──────────────────────────────────────────────
# Sample decoding helpers
# ──────────────────────────────────────────────

def _column_suffix(column_name: str) -> str:
    """Return the final extension-like suffix for a WebDataset column name."""
    return column_name.rsplit(".", 1)[-1].lower()


def _iter_sample_values(sample: dict, suffixes: set[str]):
    """Yield sample values whose column suffix matches one of *suffixes*."""
    for key, value in sample.items():
        if str(key).startswith("__") or value is None:
            continue
        suffix = _column_suffix(str(key))
        if suffix in suffixes:
            yield key, suffix, value


def _decode_text_value(value) -> str:
    """Decode a text-ish sample value to a string when possible."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        bytes_value = value.get("bytes")
        path_value = value.get("path")
        if bytes_value is not None:
            return _decode_text_value(bytes_value)
        if path_value:
            try:
                with open(path_value, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            except Exception:
                return ""
    return ""


def _caption_from_json(value) -> str:
    """Extract a natural-language caption from a JSON sidecar when present."""
    raw = _decode_text_value(value).strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except Exception:
        return ""

    if isinstance(payload, dict):
        for key in ("caption", "text", "description", "blip_caption", "alt_text"):
            text = payload.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _decode_caption(sample: dict) -> str:
    """Extract and decode the text caption from a sample."""
    for _, _, value in _iter_sample_values(sample, TEXT_SUFFIXES):
        caption = _decode_text_value(value).strip()
        if caption:
            return caption

    for _, _, value in _iter_sample_values(sample, JSON_SUFFIXES):
        caption = _caption_from_json(value)
        if caption:
            return caption

    return ""


def _decode_sample_key(sample: dict) -> str:
    """Return the original WebDataset sample key when present."""
    sample_key = sample.get("__key__", "")
    if isinstance(sample_key, bytes):
        sample_key = sample_key.decode("utf-8", errors="replace")
    return sample_key.strip()


def _normalize_caption_lookup_key(value: str) -> str:
    """Normalise a caption-map lookup key to a stable POSIX-like form."""
    key = str(value or "").strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def _load_external_caption_map(path: Optional[str]) -> dict[str, str]:
    """Load a JSON mapping of image names to external captions."""
    if not path:
        return {}

    map_path = Path(path).expanduser().resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"External caption map not found: {map_path}")

    with map_path.open() as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in external caption map: {map_path}")

    normalized = {}
    for raw_key, raw_caption in payload.items():
        caption = str(raw_caption).strip()
        if not caption:
            continue

        key = _normalize_caption_lookup_key(str(raw_key))
        if not key:
            continue

        normalized[key] = caption
        basename = PurePosixPath(key).name
        if basename and basename not in normalized:
            normalized[basename] = caption

    return normalized


def _external_caption_candidates(sample_key: str, image_ext: Optional[str]) -> list[str]:
    """Return candidate lookup keys for a sample in the external caption map."""
    ext = (image_ext or "jpg").lower()
    if ext == "jpeg":
        ext = "jpg"

    normalized_key = _normalize_caption_lookup_key(sample_key)
    seen = set()
    candidates = []

    for candidate in (normalized_key, PurePosixPath(normalized_key).name):
        if not candidate:
            continue
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

        _, existing_ext = os.path.splitext(candidate)
        if ext and not existing_ext:
            with_ext = f"{candidate}.{ext}"
            if with_ext not in seen:
                candidates.append(with_ext)
                seen.add(with_ext)

    return candidates


def _lookup_external_caption(
    caption_map: dict[str, str],
    sample_key: str,
    image_ext: Optional[str],
) -> str:
    """Look up an external caption for the given sample key."""
    if not caption_map or not sample_key:
        return ""

    for candidate in _external_caption_candidates(sample_key, image_ext):
        caption = caption_map.get(candidate)
        if caption:
            return caption
    return ""


def _image_extension(sample: dict) -> Optional[str]:
    """Return the canonical image extension for the decoded sample."""
    for _, suffix, _ in _iter_sample_values(sample, IMAGE_SUFFIXES):
        return "jpg" if suffix == "jpeg" else suffix
    return None


def _decode_image(sample: dict) -> Optional[Image.Image]:
    """Extract and decode the JPEG image from a sample."""
    image_value = None
    for _, _, value in _iter_sample_values(sample, IMAGE_SUFFIXES):
        image_value = value
        break
    if image_value is None:
        return None
    try:
        if isinstance(image_value, bytes):
            return Image.open(io.BytesIO(image_value)).convert("RGB")
        elif isinstance(image_value, dict):
            bytes_value = image_value.get("bytes")
            path_value = image_value.get("path")
            if bytes_value is not None:
                return Image.open(io.BytesIO(bytes_value)).convert("RGB")
            if path_value:
                return Image.open(path_value).convert("RGB")
        elif isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# Iterable wrapper (for DataLoader compatibility)
# ──────────────────────────────────────────────

class RexOmniDataset(IterableDataset):
    """
    Thin IterableDataset wrapper around the HuggingFace WebDataset.
    Decodes each sample into (PIL.Image, str) pairs and silently skips
    corrupt or malformed entries.

    IMPORTANT: The HF dataset is built once in __init__ (main process) so that
    load_dataset's internal multiprocessing pool is never launched from inside
    a torch DataLoader daemon worker, which would raise:
        AssertionError: daemonic processes are not allowed to have children

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .tar shards.
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        process_caption_fn=None,
        external_caption_map: Optional[str] = None,
    ):
        super().__init__()
        # Build (and cache) the HF dataset HERE in the main process.
        self._hf_dataset = build_dataset(data_dir)
        self.process_caption_fn = process_caption_fn
        self.external_caption_map = _load_external_caption_map(external_caption_map)
        if external_caption_map:
            resolved_map_path = Path(external_caption_map).expanduser().resolve()
            print(
                f"[caption_map] Loaded {len(self.external_caption_map)} normalised keys "
                f"from '{resolved_map_path}'"
            )

    def __iter__(self) -> Iterator[dict]:
        for raw in self._hf_dataset:
            sample_key = _decode_sample_key(raw)
            image_ext = _image_extension(raw)
            caption = _decode_caption(raw)
            if not caption and self.external_caption_map:
                caption = _lookup_external_caption(
                    self.external_caption_map,
                    sample_key,
                    image_ext,
                )
            
            categories = None
            if self.process_caption_fn is not None:
                categories = self.process_caption_fn(caption)
                if categories is None:
                    # Explicit None still means the sample should be dropped.
                    continue
                    
            image = _decode_image(raw)
            if image is None:
                continue

            out = {"image": image, "caption": caption}
            if sample_key:
                out["sample_key"] = sample_key
            if image_ext:
                out["image_ext"] = image_ext
            if categories is not None:
                out["categories"] = categories
            yield out


# ──────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────

def get_dataloader(
    batch_size: int = 1,
    num_workers: int = 0,
    data_dir: str = DATA_DIR,
    process_caption_fn=None,
    external_caption_map: Optional[str] = None,
) -> DataLoader:
    """
    Build and return a torch DataLoader over the WebDataset shards.

    Parameters
    ----------
    batch_size  : images per batch (default 1 for Rex-Omni single-image API)
    num_workers : parallel data workers.
                  Default is 0 (single-process) because:
                  (a) Rex-Omni is GPU-bound – CPU prefetch workers add no benefit.
                  (b) HF datasets uses its own multiprocessing pool internally;
                      launching it from daemon workers raises AssertionError.
    data_dir    : path to .tar shard directory
    process_caption_fn : optional callable taking a caption string. If filtering
                         is enabled and returns falsy, sample is skipped before image decodes.
    external_caption_map : optional JSON file mapping image names to captions.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields dicts with keys: "image", "caption", and optionally "categories".
    """
    dataset = RexOmniDataset(
        data_dir=data_dir,
        process_caption_fn=process_caption_fn,
        external_caption_map=external_caption_map,
    )

    def collate_fn(samples):
        batch = {
            "image":   [s["image"]   for s in samples],
            "caption": [s["caption"] for s in samples],
        }
        if "sample_key" in samples[0]:
            batch["sample_key"] = [s["sample_key"] for s in samples]
        if "image_ext" in samples[0]:
            batch["image_ext"] = [s["image_ext"] for s in samples]
        if "categories" in samples[0]:
            batch["categories"] = [s["categories"] for s in samples]
        return batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Building dataset …")
    ds = build_dataset()
    print(f"Dataset: {ds}")

    print("\nFirst sample keys:", list(next(iter(ds)).keys()))

    loader = get_dataloader(batch_size=1, num_workers=0)
    sample = next(iter(loader))
    print("Image:", sample["image"][0])
    print("Caption:", sample["caption"][0][:200])
