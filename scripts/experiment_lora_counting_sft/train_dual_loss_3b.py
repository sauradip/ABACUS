#!/usr/bin/env python3
"""Dual-Loss SFT Training: CE loss + ObjectFocusedAttentionLoss.

Extends train_lora_counting_sft_3b_unfreezeconn.py with attention regularization
using ObjectFocusedAttentionLoss from UniLIP's HeadLens implementation.

Combines:
  1. Standard CE loss (language modeling from conversations)
  2. ObjectFocusedAttentionLoss (attention regularization from object_centers)

Data format (attn_regularizer_train.json):
  {
    "image": "...",
    "H": int,               # Image height in original pixel space
    "W": int,               # Image width in original pixel space
    "object_centers": [     # Normalized [0, 1] coordinates
      [x, y], ...
    ],
    "conversations": [      # Standard LLaVA format
      {"from": "system", "value": "..."},
      {"from": "human", "value": "<image>..."},
      {"from": "gpt", "value": "count"}
    ]
  }
"""
from __future__ import annotations

import copy
import json
import os
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

# Use UniLip_mod (with extract_token_indices_from_mask and improved ObjectFocusedAttentionLoss)
UNILIP_MOD_PATH = REPO_ROOT / "UniLip_mod"
if str(UNILIP_MOD_PATH) not in sys.path:
    sys.path.insert(0, str(UNILIP_MOD_PATH))

# Import core training components from base SFT script
from scripts.experiment_lora_counting_sft.train_lora_counting_sft import (  # noqa: E402
    ModelArguments,
    DataArguments,
    TrainingArguments,
    rank0_print,
    smart_tokenizer_resize,
    find_base_weights,
    md5_prefix,
    preprocess_multimodal,
    preprocess_internvl,
    SFTDataCollator,
    BucketBalancedSampler,
    IGNORE_INDEX,
    IMG_CONTEXT_TOKEN_ID,
)
from scripts.counting_grpo.train_hf_multi_image_count_sft import (  # noqa: E402
    apply_transformers_compat_shims,
    load_unilip_class,
)

# Import attention regularizer
try:
    from unilip.model.language_model.headlens import ObjectFocusedAttentionLoss
except ImportError:
    ObjectFocusedAttentionLoss = None

# peft >= 0.18 imports ALL_PARALLEL_STYLES from transformers.integrations.tensor_parallel
# which is absent in transformers 4.52 — neuter the call before any peft import
try:
    import peft.utils.save_and_load as _peft_sal
    _peft_sal._maybe_shard_state_dict_for_tp = lambda *a, **kw: None  # type: ignore[attr-defined]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Extend DataArguments with dual-loss controls
# ---------------------------------------------------------------------------

@dataclass
class DualLossDataArguments(DataArguments):
    """Extend DataArguments with AR loss parameters."""
    validation_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation dataset (JSON)"}
    )
    lambda_ar: float = field(
        default=0.1,
        metadata={"help": "Weight for ObjectFocusedAttentionLoss (ce_loss + lambda_ar * ar_loss)"}
    )
    ar_sigma: float = field(
        default=1.0,
        metadata={"help": "Gaussian sigma (in patch units) for ObjectFocusedAttentionLoss"}
    )
    ar_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for attention sharpening (lower = sharper)"}
    )
    ar_use_sharpening: bool = field(
        default=True,
        metadata={"help": "Enable attention sharpening in ObjectFocusedAttentionLoss"}
    )


# ---------------------------------------------------------------------------
# Dual-Loss Dataset
# ---------------------------------------------------------------------------

class DualLossCountingSFTDataset(Dataset):
    """Extends CountingSFTDataset to include AR loss fields (object_centers, H, W)."""

    def __init__(self, data_path: str, tokenizer, data_args: DataArguments) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        rank0_print(f"[Data] Loading {data_path} …")
        with open(data_path, encoding="utf-8") as fh:
            self.data = json.load(fh)
        rank0_print(f"[Data] {len(self.data):,} entries loaded.")

        # Check for AR loss fields in first record
        if self.data and "object_centers" in self.data[0]:
            rank0_print(
                f"[Data] AR loss fields detected (object_centers, H, W) "
                f"→ dual-loss mode enabled"
            )
        else:
            rank0_print(
                "[Data] WARNING: object_centers not found in data. "
                "AR loss will be skipped."
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        has_image = bool(item.get("image"))

        if has_image:
            try:
                img = Image.open(item["image"]).convert("RGB")
            except Exception as exc:
                rank0_print(f"[WARN] Cannot open {item['image']}: {exc}")
                img = Image.new("RGB", (448, 448), (255, 255, 255))

            pixel_values = self.data_args.image_processor.preprocess(
                [img], return_tensors="pt"
            )["pixel_values"][0]
            conv_sources = preprocess_multimodal(
                copy.deepcopy([item["conversations"]]), self.data_args
            )
        else:
            pixel_values = None
            conv_sources = copy.deepcopy([item["conversations"]])

        prep = preprocess_internvl(conv_sources, self.tokenizer, has_image=has_image)

        # ── NEW: Extract AR loss fields ────────────────────────────────────
        object_centers = item.get("object_centers", [])  # Already normalized [0, 1]
        H = item.get("H", 16)  # Default to 16×16 grid
        W = item.get("W", 16)

        return dict(
            input_ids       = prep["input_ids"][0],
            labels          = prep["labels"][0],
            pixel_values    = pixel_values,
            object_centers  = object_centers,  # ← NEW
            H               = H,               # ← NEW
            W               = W,               # ← NEW
        )


# ---------------------------------------------------------------------------
# Dual-Loss Collator
# ---------------------------------------------------------------------------

@dataclass
class DualLossDataCollator:
    """Extends SFTDataCollator to batch AR loss fields."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        max_len = self.tokenizer.model_max_length
        input_ids = [inst["input_ids"][:max_len] for inst in instances]
        labels = [inst["labels"][:max_len] for inst in instances]

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

        pixel_list = [
            inst["pixel_values"] for inst in instances
            if inst["pixel_values"] is not None
        ]
        batch["pixel_values"] = torch.stack(pixel_list) if pixel_list else None

        # ── NEW: Batch AR loss fields ──────────────────────────────────────
        object_centers_list = [
            inst.get("object_centers", []) for inst in instances
        ]
        batch["object_centers"] = object_centers_list
        batch["H"] = instances[0].get("H", 16)  # Assume all same grid
        batch["W"] = instances[0].get("W", 16)

        return batch


# ---------------------------------------------------------------------------
# Dual-Loss Trainer
# ---------------------------------------------------------------------------

class DualLossCountingTrainer(Trainer):
    """Extends CountingTrainer to compute dual loss (CE + AR)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids       = inputs["input_ids"]
        labels          = inputs["labels"]
        attention_mask  = inputs["attention_mask"]
        pixel_values    = inputs.get("pixel_values")
        object_centers  = inputs.get("object_centers")  # ← NEW
        H               = inputs.get("H", 16)           # ← NEW
        W               = inputs.get("W", 16)           # ← NEW

        mm = model.module if hasattr(model, "module") else model

        # Determine if we use AR loss
        use_ar = (
            object_centers is not None
            and any(len(centers) > 0 for centers in object_centers)
            and ObjectFocusedAttentionLoss is not None
        )

        # Forward pass with output_attentions if AR loss is needed
        # NOTE: Do NOT pass labels to skip the generative image-loss branch (DiT/connector)
        outputs = mm(
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            und_image        = pixel_values,
            # Note: Don't pass output_attentions here - rely on LLM forward patch
            # output_attentions = True if use_ar else None,
        )
        logits = outputs.logits  # (B, T, V)

        # If logits is None (can happen when output_attentions is used), compute from hidden_states
        if logits is None and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Reconstruct logits from last hidden state and lm_head
            last_hidden = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, (list, tuple)) else outputs.hidden_states
            logits = mm.lm_head(last_hidden.to(mm.lm_head.weight.dtype))

        if logits is None:
            raise RuntimeError(f"Could not obtain logits from model output. Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else dir(outputs)}")

        # ── Compute CE loss (unchanged) ────────────────────────────────────
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        loss = ce_loss

        # ── Compute AR loss (NEW) ──────────────────────────────────────────
        if use_ar and outputs.attentions:
            try:
                # Infer patch grid from attention shape (ignore H, W which are image dims)
                # attn shape: (batch, heads, q_len, k_len)
                # We want to work with the image patch tokens, typically the last N tokens
                attn_k_len = outputs.attentions[0].shape[-1]  # Total sequence length
                # Estimate patch grid: look for common sizes (256=16x16, 196=14x14, 64=8x8)
                patch_sizes = [(16, 16, 256), (14, 14, 196), (8, 8, 64)]
                grid_h, grid_w = 16, 16  # default
                for h, w, expected_k in patch_sizes:
                    if attn_k_len >= expected_k and (attn_k_len - expected_k) < 50:
                        grid_h, grid_w = h, w
                        break

                ar_loss_fn = ObjectFocusedAttentionLoss(
                    sigma=self.data_args.ar_sigma,
                    temperature=self.data_args.ar_temperature,
                    use_sharpening=self.data_args.ar_use_sharpening,
                )

                # Extract img_token_indices from input_ids (image patch token positions)
                img_token_indices = None
                if hasattr(mm, 'extract_token_indices_from_mask'):
                    try:
                        img_mask = (input_ids == IMG_CONTEXT_TOKEN_ID)
                        img_token_indices = mm.extract_token_indices_from_mask(img_mask)
                    except Exception as e:
                        rank0_print(f"[AR-LOSS] Could not extract token indices: {e}")

                # Debug: print attention shapes on first step
                if self.state.global_step == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"[DEBUG AR] num_layers: {len(outputs.attentions)}")
                    print(f"[DEBUG AR] attn[0] shape: {outputs.attentions[0].shape}")  # (batch, heads, q_len, k_len)
                    print(f"[DEBUG AR] Inferred grid: {grid_h}x{grid_w} (k_len={attn_k_len})")
                    print(f"[DEBUG AR] object_centers: {[len(c) for c in object_centers]}")
                    print(f"[DEBUG AR] img_token_indices: {img_token_indices is not None}")

                ar_loss = ar_loss_fn(
                    attentions=outputs.attentions,
                    object_centers=object_centers,
                    H=grid_h,
                    W=grid_w,
                    img_token_indices=img_token_indices,
                )
                ar_loss = ar_loss.to(ce_loss.device)  # Safety cast

                # Combine losses
                lambda_ar = float(self.data_args.lambda_ar)
                loss = ce_loss + lambda_ar * ar_loss

                # Log for monitoring
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    if self.state.global_step % max(1, self.args.logging_steps) == 0:
                        print(
                            f"[dual-loss] step={self.state.global_step:6d} "
                            f"ce={ce_loss.item():.4f} ar={float(ar_loss):.4f} "
                            f"λ={lambda_ar:.2f} total={loss.item():.4f}"
                        )
            except Exception as e:
                rank0_print(f"[AR-LOSS ERROR] {e}; falling back to CE loss only")
                loss = ce_loss

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, *args, **kwargs):
        """Support bucket-balanced sampler from v3s."""
        data_args = getattr(self, "data_args", None)
        if data_args is None or not getattr(data_args, "bucket_balanced", False):
            return super()._get_train_sampler(*args, **kwargs)

        num_replicas = max(1, int(getattr(self.args, "world_size", 1)))
        rank = int(getattr(self.args, "process_index", 0))
        return BucketBalancedSampler(
            dataset=self.train_dataset,
            n_per_bucket=data_args.n_per_bucket,
            num_replicas=num_replicas,
            rank=rank,
            seed=data_args.bucket_seed,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train() -> None:
    import json  # noqa: E402

    apply_transformers_compat_shims()

    # Use DualLossDataArguments instead of DataArguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DualLossDataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # ── Pre-flight integrity check ─────────────────────────────────────────
    base_weights = find_base_weights(model_args.model_name_or_path)
    preflight_md5 = md5_prefix(base_weights)
    rank0_print(
        f"=== PRE-FLIGHT MD5 (1MB prefix of {Path(base_weights).name}): "
        f"{preflight_md5} ==="
    )

    # ── Load model (from v3s unfrozen connector path) ──────────────────────
    rank0_print(f"[Model] Loading UniLIP-3B from {model_args.model_name_or_path}")
    model_cls = load_unilip_class()

    # Force eager attention if using AR loss
    _attn_impl = "eager"  # AR loss requires attention materialization
    rank0_print(f"[Model] AR loss enabled → forcing attn_implementation=eager")

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

    # ── LoRA setup (r=64, alpha=128) ──────────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        r            = model_args.lora_rank,
        lora_alpha   = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save = ["lm_head"],
        bias            = "none",
        task_type       = TaskType.FEATURE_EXTRACTION,
    )

    llm = model.get_model().language_model
    peft_llm = get_peft_model(llm, lora_cfg)
    model.get_model().language_model = peft_llm

    # Patch LLM forward to force output_attentions=True (AR loss requirement)
    _orig_llm_forward = peft_llm.forward
    def _llm_forward_force_attn(*a, **kw):
        kw["output_attentions"] = True
        return _orig_llm_forward(*a, **kw)
    peft_llm.forward = _llm_forward_force_attn
    rank0_print("[AR-Loss] Patched LLM forward to force output_attentions=True.")

    # Optional warm-start of adapter
    if model_args.init_adapter_from:
        from safetensors.torch import load_file as _safe_load
        sd_path = os.path.join(model_args.init_adapter_from, "adapter_model.safetensors")
        if not os.path.exists(sd_path):
            raise FileNotFoundError(f"init_adapter_from: {sd_path} not found")
        sd = _safe_load(sd_path)
        # Saved adapters use lora_A.weight; PEFT internal keys use lora_A.default.weight
        sd = {
            k.replace("lora_A.weight", "lora_A.default.weight")
             .replace("lora_B.weight", "lora_B.default.weight"): v
            for k, v in sd.items()
        }
        missing, unexpected = peft_llm.load_state_dict(sd, strict=False)
        n_missing, n_unexpected = len(missing), len(unexpected)
        rank0_print(
            f"[Warm-start] Loaded {len(sd)} tensors from {sd_path}  "
            f"(missing={n_missing}, unexpected={n_unexpected})"
        )

    # ── Unfreeze connector (from v3s) ──────────────────────────────────────
    inner = model.get_model()
    n_conn = 0
    if hasattr(inner, "multi_modal_projector"):
        inner.multi_modal_projector.train()
        for p in inner.multi_modal_projector.parameters():
            p.requires_grad = True
            n_conn += p.numel()
        rank0_print(f"[Unfreeze] multi_modal_projector trainable: {n_conn/1e6:.2f}M params")

        init_conn = os.environ.get("INIT_CONNECTOR_FROM", "").strip()
        if init_conn:
            if not os.path.exists(init_conn):
                raise FileNotFoundError(f"INIT_CONNECTOR_FROM: {init_conn} not found")
            sd_conn = torch.load(init_conn, map_location="cpu")
            res = inner.multi_modal_projector.load_state_dict(sd_conn, strict=False)
            n_miss = len(getattr(res, "missing_keys", []) or [])
            n_unex = len(getattr(res, "unexpected_keys", []) or [])
            rank0_print(
                f"[Warm-start] Loaded connector from {init_conn} "
                f"({len(sd_conn)} tensors, missing={n_miss}, unexpected={n_unex})"
            )
    else:
        rank0_print("[Unfreeze] WARNING: multi_modal_projector not found.")

    # Patch model's prepare_inputs_labels_for_multimodal to handle None labels
    # (when labels=None, "labels != -100" returns Python bool True instead of Tensor)
    _orig_prepare = model.prepare_inputs_labels_for_multimodal
    def _prepare_patched(input_ids, position_ids, attention_mask, past_key_values, labels,
                        gen_images, und_images, grid_thw, i_s_pos, image_sizes=None):
        # Ensure labels is a Tensor for comparison operations
        if labels is None:
            labels = torch.full_like(input_ids, -100, dtype=torch.long)
        return _orig_prepare(input_ids, position_ids, attention_mask, past_key_values, labels,
                           gen_images, und_images, grid_thw, i_s_pos, image_sizes)
    model.prepare_inputs_labels_for_multimodal = _prepare_patched
    rank0_print("[AR-Loss] Patched model.prepare_inputs_labels_for_multimodal to handle None labels.")

    # Make lm_head trainable (from v3s)
    for p in model.lm_head.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        f"[LoRA+Conn] Total={total/1e6:.1f}M  Trainable={trainable/1e6:.1f}M  "
        f"({100*trainable/total:.2f}%)"
    )
    peft_llm.print_trainable_parameters()

    # ── Gradient checkpointing ─────────────────────────────────────────────
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    # ── Tokenizer / processor ─────────────────────────────────────────────
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

    # ── Dataset / collator ────────────────────────────────────────────────
    train_dataset = DualLossCountingSFTDataset(
        data_path  = data_args.data_path,
        tokenizer  = tokenizer,
        data_args  = data_args,
    )
    collator = DualLossDataCollator(tokenizer=tokenizer)

    # Load validation dataset if provided
    eval_dataset = None
    if data_args.validation_data_path:
        eval_dataset = DualLossCountingSFTDataset(
            data_path  = data_args.validation_data_path,
            tokenizer  = tokenizer,
            data_args  = data_args,
        )
        rank0_print(f"[Data] Loaded validation dataset: {len(eval_dataset)} items")

    # ── Trainer ───────────────────────────────────────────────────────────
    import inspect as _inspect
    _trainer_kwargs = dict(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        data_collator = collator,
    )
    if eval_dataset is not None:
        _trainer_kwargs["eval_dataset"] = eval_dataset
    _trainer_sig = _inspect.signature(transformers.Trainer.__init__).parameters
    if "processing_class" in _trainer_sig:
        _trainer_kwargs["processing_class"] = tokenizer
    else:
        _trainer_kwargs["tokenizer"] = tokenizer

    trainer = DualLossCountingTrainer(**_trainer_kwargs)
    trainer.data_args = data_args
    trainer.tokenizer = tokenizer

    if data_args.bucket_balanced:
        rank0_print(
            f"[Sampler] Bucket-balanced sampling ENABLED "
            f"(n_per_bucket={data_args.n_per_bucket}, seed={data_args.bucket_seed})"
        )
    else:
        rank0_print("[Sampler] Standard (random / distributed) sampler.")

    rank0_print(
        f"\n[Dual-Loss] Configuration:\n"
        f"  λ_ar          = {data_args.lambda_ar}\n"
        f"  ar_sigma      = {data_args.ar_sigma}\n"
        f"  ar_temperature = {data_args.ar_temperature}\n"
        f"  ar_sharpening = {data_args.ar_use_sharpening}\n"
    )

    # Resume from checkpoint if available
    import pathlib
    ckpt_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_ckpt = any((c / "trainer_state.json").exists() for c in ckpt_dirs)
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_state()

    # ── Post-flight: confirm baseline ckpt unchanged ───────────────────────
    postflight_md5 = md5_prefix(base_weights)
    if postflight_md5 != preflight_md5:
        rank0_print(f"FATAL: base model modified! {preflight_md5} → {postflight_md5}")
    else:
        rank0_print(f"=== POST-FLIGHT: base model intact ({postflight_md5}) ===")

    rank0_print("Done.")


if __name__ == "__main__":
    train()
