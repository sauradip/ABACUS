#!/usr/bin/env python3
"""LoRA SFT trainer — simple single-image counting on FSC-147, UniLIP-3B.

Reproduces the Variant B counter from ADAPTIVE_TILING_FULL_SPEC.md §A.

Key spec parameters (§A.3 / §A.7):
  - LoRA r=32, alpha=64, dropout=0.05
  - task_type = FEATURE_EXTRACTION  (not CAUSAL_LM — see spec §A.3 note)
  - modules_to_save = ["lm_head"]   (Variant-B knob; applied to LLM submodule)
  - target_modules: q/k/v/o/gate/up/down on Qwen2 LLM only
  - All non-LLM weights frozen (ViT, llm_connector, DiT, VAE)
  - Trainable params: ~270M (LoRA ~37M + lm_head ~233M)
  - Gradient checkpointing ON (required when lm_head is trainable on <4 GPUs)
  - lr=4e-5, cosine, warmup_ratio=0.03, epochs=10, fp16 or bf16, eff_batch=16

Data format consumed (LLaVA conversations JSON, see prepare_fsc147_splits.py):
    [
      {
        "image": "/abs/path/1234.jpg",
        "conversations": [
          {"from": "system", "value": "You are a helpful counting assistant. Answer with only a number."},
          {"from": "human",  "value": "<image>\\nHow many apples are present ..."},
          {"from": "gpt",    "value": "47"}
        ]
      }
    ]

Usage:
    accelerate launch --num_processes=8 --mixed_precision=bf16 \\
        scripts/experiment_lora_counting_sft/train_lora_counting_sft.py \\
        --model_name_or_path /data/amondal/model_cache/UniLIP-3B \\
        --mllm_hf_path /data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2 \\
        --data_path outputs/experiment_lora_counting_sft/train/train_counting.json \\
        --output_dir /data/amondal/unicount_runs/lora_counting_sft_variantB
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# UniLIP model class (same import path as the 1B script)
sys.path.insert(0, str(REPO_ROOT))
from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)

IGNORE_INDEX     = -100
IMG_START_TOKEN  = "<img>"
IMG_END_TOKEN    = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

DEFAULT_BASE_MODEL = "/data/amondal/model_cache/UniLIP-3B"
DEFAULT_MLLM_HF    = (
    "/data/amondal/UniCount/.hf_cache/hub/"
    "models--OpenGVLab--InternVL3-2B-hf/snapshots/"
    "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
)
DEFAULT_TRAIN_JSON = "outputs/experiment_lora_counting_sft/train/train_counting.json"

# Image-context token id in UniLIP-3B (config.image_token_id == 151667).
# 256 of these tokens occupy the image embedding slots in the LLM input.
IMG_CONTEXT_TOKEN_ID = 151667


def rank0_print(*args) -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)


# ---------------------------------------------------------------------------
# Attention-focus regularizer helpers (experiment #32)
# ---------------------------------------------------------------------------

def points_to_prior(points, img_w: int, img_h: int,
                    grid: int = 16, sigma: float = 1.0) -> torch.Tensor:
    """Build a normalized (grid, grid) Gaussian point-prior from FSC-147 points.

    `points` is a list of [x, y] in the SAME pixel space as (img_w, img_h)
    (i.e. the on-disk 384-shorter-side resized image).

    Returns a flat tensor of shape (grid*grid,) summing to 1.
    """
    import numpy as _np
    from scipy.ndimage import gaussian_filter as _gf
    grid_map = _np.zeros((grid, grid), dtype=_np.float32)
    if not points or img_w <= 0 or img_h <= 0:
        # Uniform fallback (no annotation): training step will skip focus loss.
        grid_map[:] = 1.0 / (grid * grid)
        return torch.from_numpy(grid_map.flatten())
    for px, py in points:
        gx = min(int(px / img_w * grid), grid - 1)
        gy = min(int(py / img_h * grid), grid - 1)
        if 0 <= gx < grid and 0 <= gy < grid:
            grid_map[gy, gx] += 1.0
    grid_map = _gf(grid_map, sigma=sigma, mode="constant", cval=0.0)
    s = grid_map.sum()
    if s <= 0:
        grid_map[:] = 1.0 / (grid * grid)
    else:
        grid_map /= s
    return torch.from_numpy(grid_map.flatten())


def compute_attention_focus_loss(
    attentions,                # tuple of (B, n_heads, T, T) tensors, len = num_layers
    prior_flat: torch.Tensor,  # (B, grid*grid)
    image_token_mask: torch.Tensor,  # (B, T) bool — image-token positions
    label_query_mask: torch.Tensor,  # (B, T) bool — assistant tokens used as query
    target_layers,             # list[int]
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL( prior || attention(query → image_tokens) ), averaged over target layers.

    For each batch sample: take the mean-over-heads attention from every
    label_query token to every image token, average over query positions,
    renormalize over image positions, then KL with the spatial prior.

    NOTE: Direction is KL(prior || attn) (mode-covering): the model's attention
    must put mass everywhere the prior has mass, which concentrates attention on
    objects. The reverse direction KL(attn || prior) is mode-seeking and allows
    attention to collapse onto a single object while still scoring low loss.
    """
    if attentions is None:
        return torch.zeros((), device=prior_flat.device, dtype=torch.float32)

    B, T = image_token_mask.shape
    losses = []
    for ell in target_layers:
        if ell < 0 or ell >= len(attentions):
            continue
        a = attentions[ell]                    # (B, H, T, T)
        a = a.mean(dim=1)                      # (B, T, T) mean over heads
        per_sample = []
        for b in range(B):
            img_idx = image_token_mask[b].nonzero(as_tuple=False).squeeze(-1)
            q_idx   = label_query_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if img_idx.numel() == 0 or q_idx.numel() == 0:
                continue
            sub = a[b][q_idx][:, img_idx]      # (Q, N_img)
            attn_dist = sub.mean(dim=0)        # (N_img,)
            attn_dist = attn_dist / (attn_dist.sum() + eps)
            p = prior_flat[b].to(attn_dist.dtype)
            # KL(prior || attn) -- mode-covering (forces attn to span all prior support)
            kl = (p * ((p + eps).log() - (attn_dist + eps).log())).sum()
            per_sample.append(kl)
        if per_sample:
            losses.append(torch.stack(per_sample).mean())
    if not losses:
        return torch.zeros((), device=prior_flat.device, dtype=torch.float32)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=DEFAULT_BASE_MODEL)
    mllm_hf_path: str = field(default=DEFAULT_MLLM_HF,
                               metadata={"help": "Path to InternVL3-2B-hf snapshot for processor"})
    lora_rank:    int   = field(default=32)
    lora_alpha:   int   = field(default=64)
    lora_dropout: float = field(default=0.05)
    init_adapter_from: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a previously-trained LoRA adapter dir "
                          "(containing adapter_model.safetensors). When set, weights "
                          "are loaded into the freshly-built peft model AFTER "
                          "get_peft_model() — gives warm START with FRESH optimizer + "
                          "LR schedule (unlike resume_from_checkpoint)."}
    )


@dataclass
class DataArguments:
    data_path:          str  = field(default=DEFAULT_TRAIN_JSON)
    image_aspect_ratio: str  = field(default="pad")
    is_multimodal:      bool = field(default=True)
    image_processor:    Optional[object] = field(default=None)
    # ── Bucket-balanced sampling (Phase 1, count-bucket equalisation) ──
    bucket_balanced:    bool = field(
        default=False,
        metadata={"help": "Enable BucketBalancedSampler that draws roughly "
                          "n_per_bucket samples from each GT-count bucket per epoch."}
    )
    n_per_bucket:       int  = field(
        default=2000,
        metadata={"help": "Cap on samples drawn per count bucket per epoch (used iff "
                          "--bucket_balanced True). Buckets with fewer items contribute all of them."}
    )
    bucket_seed:        int  = field(
        default=42,
        metadata={"help": "Seed for BucketBalancedSampler reshuffling."}
    )
    # ── Attention-focus regularizer (experiment #32, arXiv 2603.18523) ──
    attention_regularizer: bool = field(
        default=False,
        metadata={"help": "If True, adds KL(attention || Gaussian-point-prior) loss "
                          "on selected LLM layers. Forces eager attention."}
    )
    lambda_focus:    float = field(default=1.0)
    target_layers:   str   = field(
        default="2,18,19,20,21,22",
        metadata={"help": "Comma-separated LLM layer indices to apply focus loss to. "
                          "Defaults match the paper's Qwen2.5-VL-7B layers (28-layer model)."}
    )
    attn_grid:       int   = field(default=16, metadata={"help": "Patch grid side (16 → 256 image tokens for UniLIP-3B)."})
    attn_sigma:      float = field(default=1.0, metadata={"help": "Gaussian sigma in patch units for the spatial prior."})
    fsc147_annotation_json: str = field(
        default="/data/amondal/FSC147_hf/annotation_FSC147_384.json",
        metadata={"help": "FSC-147 point annotation JSON (used iff --attention_regularizer)."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir:       Optional[str] = field(default=None)
    model_max_length: int          = field(default=512)
    remove_unused_columns: bool    = field(default=False)


# ---------------------------------------------------------------------------
# Tokenisation (identical logic to train_lora_understanding.py)
# ---------------------------------------------------------------------------

def preprocess_multimodal(sources, data_args):
    """Replace <image> with UniLIP image context tokens in the human turn."""
    und_placeholder = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}"
    for source in sources:
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace("<image>", und_placeholder).strip()
    return sources


def preprocess_internvl(sources, tokenizer, has_image: bool = False):
    """Tokenise conversations, masking prompt tokens with IGNORE_INDEX.

    Supports both {"role"/"content"} and {"from"/"value"} dicts.
    """
    roles = {"human": "user", "gpt": "assistant", "system": "system"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n'}}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\\n' }}"
        "{% elif content['type'] == 'text' %}{{ content['text'] }}"
        "{% endif %}{% endfor %}{% endif %}"
        "{{'<|im_end|>\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    input_ids_all, targets_all = [], []

    for source in sources:
        # Ensure system turn is present
        first_role = source[0].get("from", source[0].get("role"))
        if first_role != "system":
            source = [
                {"from": "system",
                 "value": "You are a helpful counting assistant. Answer with only a number."}
            ] + list(source)

        input_id: List[int] = []
        target:   List[int] = []

        for conv in source:
            role    = conv.get("role") or conv.get("from")
            content = conv.get("content") or conv.get("value")
            role    = roles.get(role, role)

            encode_id = tokenizer.apply_chat_template(
                [{"role": role, "content": content}],
                return_dict=False,
            )
            input_id += encode_id

            if role in ("user", "system"):
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target)
        input_ids_all.append(input_id)
        targets_all.append(target)

    input_ids = torch.tensor(input_ids_all, dtype=torch.long)
    targets   = torch.tensor(targets_all,   dtype=torch.long)
    return dict(input_ids=input_ids, labels=targets)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CountingSFTDataset(Dataset):
    """LLaVA-style single-image counting SFT dataset.

    Expects a JSON list where each entry has:
        "image": absolute path
        "conversations": [system, human(<image>\\n...), gpt(count)]
    """

    def __init__(self, data_path: str, tokenizer, data_args: DataArguments) -> None:
        super().__init__()
        self.tokenizer  = tokenizer
        self.data_args  = data_args

        rank0_print(f"[Data] Loading {data_path} …")
        with open(data_path, encoding="utf-8") as fh:
            self.data = json.load(fh)
        rank0_print(f"[Data] {len(self.data):,} entries loaded.")

        # Optional FSC-147 point annotations for attention regularizer.
        self._fsc_points = None
        if getattr(data_args, "attention_regularizer", False):
            ann_path = data_args.fsc147_annotation_json
            rank0_print(f"[Data] Loading FSC-147 point annotations from {ann_path}")
            with open(ann_path, encoding="utf-8") as fh:
                ann = json.load(fh)
            self._fsc_points = {k: v.get("points", []) for k, v in ann.items()}
            rank0_print(f"[Data] {len(self._fsc_points):,} annotated images for focus prior.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item      = self.data[idx]
        has_image = bool(item.get("image"))

        attention_prior = None
        img_w_disk = img_h_disk = 0
        if has_image:
            try:
                img = Image.open(item["image"]).convert("RGB")
                img_w_disk, img_h_disk = img.size
            except Exception as exc:
                rank0_print(f"[WARN] Cannot open {item['image']}: {exc}")
                img = Image.new("RGB", (448, 448), (255, 255, 255))
                img_w_disk, img_h_disk = 448, 448

            pixel_values = self.data_args.image_processor.preprocess(
                [img], return_tensors="pt"
            )["pixel_values"][0]
            conv_sources = preprocess_multimodal(
                copy.deepcopy([item["conversations"]]), self.data_args
            )
        else:
            pixel_values = None
            conv_sources = copy.deepcopy([item["conversations"]])

        if self._fsc_points is not None and has_image:
            bn = os.path.basename(item["image"])
            pts = self._fsc_points.get(bn, [])
            attention_prior = points_to_prior(
                pts, img_w_disk, img_h_disk,
                grid=self.data_args.attn_grid,
                sigma=self.data_args.attn_sigma,
            )

        prep = preprocess_internvl(conv_sources, self.tokenizer, has_image=has_image)
        return dict(
            input_ids       = prep["input_ids"][0],
            labels          = prep["labels"][0],
            pixel_values    = pixel_values,
            attention_prior = attention_prior,
        )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        max_len    = self.tokenizer.model_max_length
        input_ids  = [inst["input_ids"][:max_len] for inst in instances]
        labels     = [inst["labels"][:max_len]     for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch = dict(
            input_ids      = input_ids,
            labels         = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
        )
        pixel_list = [inst["pixel_values"] for inst in instances
                      if inst["pixel_values"] is not None]
        batch["pixel_values"] = torch.stack(pixel_list) if pixel_list else None
        prior_list = [inst.get("attention_prior") for inst in instances]
        if any(p is not None for p in prior_list):
            # Replace any None with a uniform prior so stacking succeeds; the
            # focus loss will treat all-zero label_query rows as no-op.
            grid_flat = next(p for p in prior_list if p is not None).numel()
            uniform = torch.full((grid_flat,), 1.0 / grid_flat)
            prior_list = [p if p is not None else uniform for p in prior_list]
            batch["attention_prior"] = torch.stack(prior_list)
        return batch


# ---------------------------------------------------------------------------
# Bucket-balanced sampler (Phase 1: equalise gradient contribution per count regime)
# ---------------------------------------------------------------------------

_GT_COUNT_RE = re.compile(r"\d+")
_DEFAULT_COUNT_BUCKETS = [
    (0, 5),
    (6, 20),
    (21, 50),
    (51, 100),
    (101, 200),
    (201, 10 ** 9),
]


def _extract_gt_count(item: Dict) -> Optional[int]:
    """Return the GT count from the last (gpt) turn, or None if unparseable."""
    try:
        val = item["conversations"][-1]["value"]
        m = _GT_COUNT_RE.search(val)
        return int(m.group(0)) if m else None
    except Exception:
        return None


class BucketBalancedSampler(torch.utils.data.Sampler):
    """Per-epoch sampler that draws ~n_per_bucket items from each GT-count bucket.

    Distributed-safe: every rank computes the same shuffled global index list using
    `seed + epoch`, then takes the rank-strided slice. Trainer calls
    `sampler.set_epoch(epoch)` between epochs (HF Trainer detects `set_epoch`).
    """

    def __init__(
        self,
        dataset: "CountingSFTDataset",
        buckets=_DEFAULT_COUNT_BUCKETS,
        n_per_bucket: int = 2000,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.dataset      = dataset
        self.buckets      = list(buckets)
        self.n_per_bucket = int(n_per_bucket)
        self.num_replicas = max(1, int(num_replicas))
        self.rank         = int(rank)
        self.seed         = int(seed)
        self.epoch        = 0

        # Build bucket → list[index] map (raw dataset access; no tokenisation).
        self.bucket_indices: Dict[tuple, List[int]] = {b: [] for b in self.buckets}
        unparseable = 0
        for i, item in enumerate(dataset.data):
            c = _extract_gt_count(item)
            if c is None:
                unparseable += 1
                continue
            for (lo, hi) in self.buckets:
                if lo <= c <= hi:
                    self.bucket_indices[(lo, hi)].append(i)
                    break

        self._per_bucket_used = {
            b: min(self.n_per_bucket, len(idxs))
            for b, idxs in self.bucket_indices.items()
        }
        self._global_len = sum(self._per_bucket_used.values())
        # Drop tail to keep equal shards across ranks.
        self._per_rank_len = self._global_len // self.num_replicas

        rank0_print(
            "[BucketBalancedSampler] Initialised.\n"
            f"  total dataset size : {len(dataset)}\n"
            f"  unparseable counts : {unparseable}\n"
            f"  buckets            : {self.buckets}\n"
            f"  n_per_bucket cap   : {self.n_per_bucket}\n"
            f"  per-bucket avail   : {{ {', '.join(f'{b}: {len(self.bucket_indices[b])}' for b in self.buckets)} }}\n"
            f"  per-bucket used    : {{ {', '.join(f'{b}: {self._per_bucket_used[b]}' for b in self.buckets)} }}\n"
            f"  global samples/epoch: {self._global_len}\n"
            f"  num_replicas / rank: {self.num_replicas} / {self.rank}\n"
            f"  per-rank length    : {self._per_rank_len}\n"
            f"  seed               : {self.seed}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._per_rank_len

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)
        global_indices: List[int] = []
        for bucket in self.buckets:
            idxs = self.bucket_indices[bucket]
            n    = self._per_bucket_used[bucket]
            if n <= 0:
                continue
            global_indices.extend(rng.sample(idxs, n))
        rng.shuffle(global_indices)
        # Trim to a multiple of num_replicas so every rank gets equal length.
        usable = self._per_rank_len * self.num_replicas
        global_indices = global_indices[:usable]
        # Rank stride.
        return iter(global_indices[self.rank::self.num_replicas])


# ---------------------------------------------------------------------------
# Trainer (same forward as train_lora_understanding.py)
# ---------------------------------------------------------------------------

class CountingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids      = inputs["input_ids"]
        labels         = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values   = inputs.get("pixel_values")
        attention_prior = inputs.get("attention_prior")  # (B, grid*grid) or None

        mm = model.module if hasattr(model, "module") else model

        use_focus = (
            attention_prior is not None
            and getattr(self, "data_args", None) is not None
            and getattr(self.data_args, "attention_regularizer", False)
        )

        # Use model's native forward.  Passing und_image= triggers
        # prepare_inputs_labels_for_multimodal inside forward, which:
        #   1. encodes pixel_values via the frozen vision tower
        #   2. splices the resulting embeddings into the <IMG_CONTEXT> token positions
        # We do NOT pass labels here so the model skips the generative image-loss
        # branch (llm_connector / DiT) entirely.
        outputs = mm(
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            und_image        = pixel_values,   # None → text-only path; fine
            output_attentions = True if use_focus else None,
        )
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        loss = ce_loss
        if use_focus:
            # NB: prepare_inputs_labels_for_multimodal does NOT change input_ids
            # in the simple counting path; the 256 IMG_CONTEXT tokens remain in
            # place and the LLM sees a sequence the same length as input_ids.
            image_token_mask = (input_ids == IMG_CONTEXT_TOKEN_ID)        # (B, T)
            label_query_mask = (labels != IGNORE_INDEX) & (labels != self.tokenizer.pad_token_id) \
                if hasattr(self, "tokenizer") and self.tokenizer is not None \
                else (labels != IGNORE_INDEX)
            target_layers = [int(x) for x in str(self.data_args.target_layers).split(",") if x.strip() != ""]
            attentions = getattr(outputs, "attentions", None)
            focus_loss = compute_attention_focus_loss(
                attentions      = attentions,
                prior_flat      = attention_prior.to(input_ids.device),
                image_token_mask = image_token_mask,
                label_query_mask = label_query_mask,
                target_layers   = target_layers,
            )
            loss = ce_loss + float(self.data_args.lambda_focus) * focus_loss
            if int(os.environ.get("LOCAL_RANK", 0)) == 0 and (self.state.global_step % max(1, self.args.logging_steps) == 0):
                print(f"[focus] step={self.state.global_step} ce={ce_loss.item():.4f} "
                      f"focus={float(focus_loss):.4f} lambda={self.data_args.lambda_focus}")
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Sampler override (Phase 1: bucket-balanced count sampling)
    # ------------------------------------------------------------------
    def _get_train_sampler(self, *args, **kwargs):
        data_args = getattr(self, "data_args", None)
        if data_args is None or not getattr(data_args, "bucket_balanced", False):
            return super()._get_train_sampler(*args, **kwargs)

        num_replicas = max(1, int(getattr(self.args, "world_size", 1)))
        rank         = int(getattr(self.args, "process_index", 0))
        return BucketBalancedSampler(
            dataset      = self.train_dataset,
            n_per_bucket = data_args.n_per_bucket,
            num_replicas = num_replicas,
            rank         = rank,
            seed         = data_args.bucket_seed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_base_weights(model_dir: str) -> str:
    """Return a stable weight file to checksum — first shard or single file."""
    d = Path(model_dir)
    # Prefer the first numbered shard (sharded models)
    shards = sorted(d.glob("model-*.safetensors"))
    if shards:
        return str(shards[0])
    # Fall back to monolithic file
    mono = d / "model.safetensors"
    if mono.exists():
        return str(mono)
    raise FileNotFoundError(
        f"No model.safetensors or model-*.safetensors found in {model_dir}"
    )


def md5_prefix(path: str, nbytes: int = 1024 * 1024) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read(nbytes)).hexdigest()


def smart_tokenizer_resize(special_tokens: dict, tokenizer, model) -> None:
    n = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if n > 0:
        emb = model.get_input_embeddings().weight.data
        emb[-n:] = emb[:-n].mean(dim=0, keepdim=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train() -> None:
    apply_transformers_compat_shims()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # ── Pre-flight integrity check ─────────────────────────────────────────
    base_weights = find_base_weights(model_args.model_name_or_path)
    preflight_md5 = md5_prefix(base_weights)
    rank0_print(f"=== PRE-FLIGHT MD5 (1MB prefix of {Path(base_weights).name}): {preflight_md5} ===")

    # ── Load base model ────────────────────────────────────────────────────
    rank0_print(f"[Model] Loading UniLIP-3B from {model_args.model_name_or_path}")
    model_cls = load_unilip_class()
    # Use flash_attention_2 if available, else fall back to sdpa (PyTorch built-in).
    # When the attention regularizer is on, force eager attention so attention
    # probabilities are materialized (sdpa/flash fuse them in-kernel).
    if data_args.attention_regularizer:
        _attn_impl = "eager"
        rank0_print("[Model] attention_regularizer=True → forcing attn_implementation=eager")
    else:
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
        except ImportError:
            _attn_impl = "sdpa"
    rank0_print(f"[Model] attn_implementation={_attn_impl}")
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=_attn_impl,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── Freeze everything first ────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False

    # ── Variant B LoRA (spec §A.3): apply to LLM submodule, save lm_head ──
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        r            = model_args.lora_rank,
        lora_alpha   = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # modules_to_save: saved as full weights alongside the adapter.
        # "lm_head" is relative to model.get_model().language_model — the
        # submodule on which get_peft_model is called (spec §A.3 note).
        modules_to_save = ["lm_head"],
        bias            = "none",
        # FEATURE_EXTRACTION, not CAUSAL_LM — UniLIP's forward bypasses
        # CausalLMOutputWithPast; CAUSAL_LM would inject a head wrapper (spec §A.3)
        task_type       = TaskType.FEATURE_EXTRACTION,
    )

    llm = model.get_model().language_model
    peft_llm = get_peft_model(llm, lora_cfg)
    model.get_model().language_model = peft_llm

    # ── Monkey-patch LLM forward to force output_attentions=True when the ──
    # attention regularizer is enabled.  UniLIP's outer forward() accepts
    # `output_attentions` but does NOT pass it through to its inner LLM call
    # (see UniLIP/unilip/model/language_model/unilip_internvl.py:148).  By
    # patching the LLM directly we keep UniLIP source untouched.
    if data_args.attention_regularizer:
        _orig_llm_forward = peft_llm.forward
        def _llm_forward_force_attn(*a, **kw):
            kw["output_attentions"] = True
            return _orig_llm_forward(*a, **kw)
        peft_llm.forward = _llm_forward_force_attn
        rank0_print("[FocusReg] Patched LLM forward to force output_attentions=True.")

    # ── Optional warm-start: load adapter weights from a prior run ────────
    # This is DIFFERENT from HF Trainer's resume_from_checkpoint:
    #   - resume_from_checkpoint = optimizer + scheduler + step counter
    #   - init_adapter_from      = ONLY weights, fresh optimizer/schedule
    if model_args.init_adapter_from:
        from safetensors.torch import load_file as _safe_load
        from peft.utils.save_and_load import set_peft_model_state_dict
        sd_path = os.path.join(model_args.init_adapter_from, "adapter_model.safetensors")
        if not os.path.exists(sd_path):
            raise FileNotFoundError(f"init_adapter_from: {sd_path} not found")
        sd = _safe_load(sd_path)
        load_result = set_peft_model_state_dict(peft_llm, sd)
        n_missing = len(getattr(load_result, "missing_keys", []) or [])
        n_unexpected = len(getattr(load_result, "unexpected_keys", []) or [])
        rank0_print(
            f"[Warm-start] Loaded {len(sd)} tensors from {sd_path}  "
            f"(missing={n_missing}, unexpected={n_unexpected})"
        )

    # Also make the top-level lm_head (model.lm_head) trainable so it is
    # saved with trainer checkpoints (mirrors train_lora_understanding.py §A.7)
    for p in model.lm_head.parameters():
        p.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        f"[LoRA] Total={total/1e6:.1f}M  Trainable={trainable/1e6:.1f}M  "
        f"({100*trainable/total:.2f}%)"
    )
    if trainable < 50_000_000:
        rank0_print(
            "[LoRA] WARNING: <50M trainable. If ~37M, lm_head was NOT captured "
            "(Variant A). Expected ~270M for Variant B. Check modules_to_save."
        )
    peft_llm.print_trainable_parameters()

    # ── Gradient checkpointing ────────────────────────────────────────────
    # Required when lm_head is trainable (spec §A.7)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    rank0_print(f"[Proc] Loading processor from {model_args.mllm_hf_path}")
    tokenizer = AutoProcessor.from_pretrained(
        model_args.mllm_hf_path, trust_remote_code=True
    ).tokenizer
    tokenizer.model_max_length = training_args.model_max_length

    if tokenizer.pad_token is None:
        smart_tokenizer_resize(
            {"pad_token": "<pad>",
             "additional_special_tokens": ["[IMG]", "[/IMG]", "<image>"]},
            tokenizer, model,
        )
    elif "<image>" not in tokenizer.get_added_vocab():
        smart_tokenizer_resize(
            {"additional_special_tokens": ["[IMG]", "[/IMG]", "<image>"]},
            tokenizer, model,
        )

    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.mllm_hf_path, trust_remote_code=True
    ).image_processor

    # ── Dataset & collator ────────────────────────────────────────────────
    train_dataset = CountingSFTDataset(
        data_path  = data_args.data_path,
        tokenizer  = tokenizer,
        data_args  = data_args,
    )
    collator = SFTDataCollator(tokenizer=tokenizer)

    # ── Trainer ───────────────────────────────────────────────────────────
    # transformers ≥4.47 renamed 'tokenizer' → 'processing_class' in Trainer.__init__.
    import inspect as _inspect
    _trainer_kwargs = dict(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        data_collator = collator,
    )
    _trainer_sig = _inspect.signature(transformers.Trainer.__init__).parameters
    if "processing_class" in _trainer_sig:
        _trainer_kwargs["processing_class"] = tokenizer
    else:
        _trainer_kwargs["tokenizer"] = tokenizer

    trainer = CountingTrainer(**_trainer_kwargs)
    # Expose data_args + tokenizer so the focus-reg compute_loss can read them.
    trainer.data_args = data_args
    trainer.tokenizer = tokenizer
    if data_args.bucket_balanced:
        rank0_print(
            f"[Sampler] Bucket-balanced sampling ENABLED "
            f"(n_per_bucket={data_args.n_per_bucket}, seed={data_args.bucket_seed})"
        )
    else:
        rank0_print("[Sampler] Standard (random / distributed) sampler.")

    ckpt_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_ckpt  = any((c / "trainer_state.json").exists() for c in ckpt_dirs)
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_state()

    # ── Save adapter (+ merged model when not using ZeRO-3) ───────────────
    #
    # ZeRO-3 shards ALL params across GPUs.  merge_and_unload() requires full
    # param tensors on a single device and therefore cannot run while ZeRO-3
    # is active.  Instead we:
    #   1. Let all ranks participate in the adapter save — DS gathers params
    #      to rank 0 automatically when
    #      `stage3_gather_16bit_weights_on_model_save: true` is set in the DS
    #      config (which our ds_zero3.json does).
    #   2. Skip the in-training merge.  Run scripts/merge_lora_adapter.py
    #      offline after training to produce the merged checkpoint.
    #
    # When running without DeepSpeed (or with ZeRO-0/1/2) the original
    # merge-during-training path is preserved.
    # ─────────────────────────────────────────────────────────────────────

    _ds_engine = getattr(trainer, "deepspeed", None)
    _zero_stage = (
        _ds_engine.zero_optimization_stage()
        if (_ds_engine is not None and hasattr(_ds_engine, "zero_optimization_stage"))
        else 0
    )
    is_zero3   = (_zero_stage == 3)

    adapter_dir = os.path.join(training_args.output_dir, "adapter")
    if is_main:
        os.makedirs(adapter_dir, exist_ok=True)

    peft_model_ref = model.get_model().language_model

    if is_zero3:
        # All ranks must participate in the ZeRO-3 param-gather; rank 0 writes.
        rank0_print("\n=== [ZeRO-3] Saving LoRA adapter (all-rank gather) ===")
        peft_model_ref.save_pretrained(adapter_dir)
        rank0_print(f"  LoRA adapter → {adapter_dir}")
        rank0_print(
            "  Merge skipped (ZeRO-3 active).  "
            "Run `python scripts/experiment_lora_counting_sft/merge_lora_adapter.py "
            f"--adapter_dir {adapter_dir} --base_model {model_args.model_name_or_path}` "
            "after training to produce the merged checkpoint."
        )
    else:
        if is_main:
            rank0_print("\n=== Saving LoRA adapter and merged model ===")

            merged_dir = os.path.join(training_args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)

            peft_model_ref.save_pretrained(adapter_dir)
            rank0_print(f"  LoRA adapter → {adapter_dir}")

            merged_llm = peft_model_ref.merge_and_unload()
            model.get_model().language_model = merged_llm

            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, os.path.join(merged_dir, "pytorch_model.bin"))
            model.config.save_pretrained(merged_dir)
            rank0_print(f"  Merged model → {merged_dir}")

    if is_main:
        postflight_md5 = md5_prefix(base_weights)
        if postflight_md5 != preflight_md5:
            rank0_print(f"FATAL: base model modified! {preflight_md5} → {postflight_md5}")
        else:
            rank0_print(f"=== POST-FLIGHT: base model intact ({postflight_md5}) ===")

        rank0_print("Done.")


if __name__ == "__main__":
    train()
