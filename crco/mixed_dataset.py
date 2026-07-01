"""Mixed counting + ranking dataset for CRCO training.

The existing single-image counting trainer (``train_lora_counting_sft.py``)
operates on rows of the form::

    {
      "image": "<abs/path>",
      "conversations": [system, human(<image>...), gpt(count)],
    }

CRCO ranking samples are a 4-image variant of the same shape::

    {
      "type": "ranking",
      "image": ["<abs/path1>", ..., "<abs/path4>"],
      "conversations": [system, human(<image>×4 + instruction), gpt(ranking_str)],
    }

This module provides:

* ``MixedCountingRankingDataset`` — yields a probabilistic mix of counting
  and ranking samples; each item carries an ``is_ranking`` flag.
* ``MixedSFTDataCollator`` — handles variable per-sample image counts by
  concatenating all ``pixel_values`` along the batch axis (the InternVL
  splice path keys off ``<IMG_CONTEXT>`` token positions, not on per-sample
  image counts), and propagates the ``is_ranking`` flag.
"""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from scripts.experiment_lora_counting_sft.train_lora_counting_sft import (
    IGNORE_INDEX,
    preprocess_internvl,
    preprocess_multimodal,
)


def _rank0(*args: Any) -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)


def _load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # pragma: no cover
        _rank0(f"[CRCO][WARN] Cannot open {path}: {exc}")
        return Image.new("RGB", (448, 448), (255, 255, 255))


class MixedCountingRankingDataset(Dataset):
    """Probabilistically yields counting (single image) and ranking (4 image) rows."""

    def __init__(
        self,
        counting_path: str,
        ranking_path: str,
        tokenizer: Any,
        data_args: Any,
        p_rank: float = 0.4,
        seed: int = 1234,
        verify_ranking_files: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= p_rank <= 1.0:
            raise ValueError(f"p_rank must be in [0, 1]; got {p_rank}")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.p_rank = p_rank
        self.rng = random.Random(seed)

        _rank0(f"[CRCO][Data] Loading counting JSON: {counting_path}")
        with open(counting_path, encoding="utf-8") as fh:
            self.counting: List[Dict[str, Any]] = json.load(fh)
        _rank0(f"[CRCO][Data] Loading ranking JSON:  {ranking_path}")
        with open(ranking_path, encoding="utf-8") as fh:
            self.ranking: List[Dict[str, Any]] = json.load(fh)

        if verify_ranking_files:
            missing = 0
            for row in self.ranking:
                for path in row["image"]:
                    if not os.path.exists(path):
                        missing += 1
            if missing:
                raise FileNotFoundError(
                    f"[CRCO] {missing} ranking image paths missing on disk; "
                    "rebuild ranking JSON with --strict_existence."
                )

        _rank0(
            f"[CRCO][Data] counting={len(self.counting)} ranking={len(self.ranking)} "
            f"p_rank={p_rank}"
        )

    def __len__(self) -> int:
        # Define epoch length as counting + ranking (both seen, on average).
        return len(self.counting) + len(self.ranking)

    # ------------------------------------------------------------------ utils
    def _process_images(self, paths: List[str]) -> torch.Tensor:
        imgs = [_load_image(p) for p in paths]
        pv = self.data_args.image_processor.preprocess(
            imgs, return_tensors="pt"
        )["pixel_values"]
        # Always return shape (N, 3, H, W); single-image rows give (1, 3, H, W).
        if pv.ndim == 3:
            pv = pv.unsqueeze(0)
        return pv

    def _tokenise(self, conversations: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        sources = preprocess_multimodal(copy.deepcopy([conversations]), self.data_args)
        prep = preprocess_internvl(sources, self.tokenizer, has_image=True)
        return {"input_ids": prep["input_ids"][0], "labels": prep["labels"][0]}

    # --------------------------------------------------------------- __getitem__
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Use both idx (for shuffling determinism inside Trainer) and rng for
        # type selection — sampler-shuffle still ensures full coverage.
        is_ranking = self.rng.random() < self.p_rank
        if is_ranking and self.ranking:
            row = self.ranking[self.rng.randrange(len(self.ranking))]
            paths = list(row["image"])
        else:
            row = self.counting[idx % len(self.counting)]
            single = row.get("image")
            if isinstance(single, list):
                paths = list(single)
            else:
                paths = [single]
            is_ranking = False

        pixel_values = self._process_images(paths)
        toks = self._tokenise(row["conversations"])
        return {
            "input_ids": toks["input_ids"],
            "labels": toks["labels"],
            "pixel_values": pixel_values,            # (N_images, 3, H, W)
            "is_ranking": bool(is_ranking),
        }


@dataclass
class MixedSFTDataCollator:
    """Pads input_ids/labels and concatenates pixel_values across samples."""

    tokenizer: Any

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = self.tokenizer.model_max_length
        input_ids = [inst["input_ids"][:max_len] for inst in instances]
        labels = [inst["labels"][:max_len] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        pv_list = [inst["pixel_values"] for inst in instances if inst.get("pixel_values") is not None]
        # Each entry is (N_i, 3, H, W); cat → (sum N_i, 3, H, W).
        pixel_values = torch.cat(pv_list, dim=0) if pv_list else None
        # Per-sample image counts so the trainer can split per-sample forwards.
        image_counts = torch.tensor(
            [(inst["pixel_values"].shape[0] if inst.get("pixel_values") is not None else 0)
             for inst in instances],
            dtype=torch.long,
        )
        is_ranking = torch.tensor(
            [bool(inst.get("is_ranking", False)) for inst in instances],
            dtype=torch.bool,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_counts": image_counts,
            "is_ranking": is_ranking,
        }
