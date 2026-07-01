#!/usr/bin/env python3
"""CRCO trainer — Variant B LoRA SFT with auxiliary ranking loss.

Total loss:
    L_total = mean_b( w_b · CE_b )      with w_b = lambda_rank if is_ranking_b else 1.0

Per-sample CE is computed by averaging token-level cross-entropy over the
unmasked (non-IGNORE_INDEX) target positions of each row in the batch.

This script mirrors ``scripts/experiment_lora_counting_sft/train_lora_counting_sft.py``
(Variant B) and adds:
    * ``MixedCountingRankingDataset`` + ``MixedSFTDataCollator``
    * weighted ``compute_loss`` with per-sample ranking flag
    * ``--ranking_json``, ``--p_rank``, ``--lambda_rank`` CLI flags
    * separate ``counting_loss`` and ``ranking_loss`` logging
"""
from __future__ import annotations

import hashlib
import inspect as _inspect
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor, Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse Variant B building blocks (LoRA setup, MD5 preflight, etc.)
from scripts.counting_grpo.train_hf_multi_image_count_sft import (  # noqa: E402
    apply_transformers_compat_shims,
    load_unilip_class,
)
from scripts.experiment_lora_counting_sft.train_lora_counting_sft import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_MLLM_HF,
    IGNORE_INDEX,
    find_base_weights,
    md5_prefix,
    smart_tokenizer_resize,
)
from crco.mixed_dataset import (  # noqa: E402
    MixedCountingRankingDataset,
    MixedSFTDataCollator,
)


DEFAULT_COUNTING_JSON = "outputs/experiment_lora_counting_sft/train/train_counting.json"
DEFAULT_RANKING_JSON = "crco/data/fsc147_crco_ranking.json"


def rank0_print(*args: Any) -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=DEFAULT_BASE_MODEL)
    mllm_hf_path: str = field(default=DEFAULT_MLLM_HF)
    lora_rank: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    counting_json: str = field(default=DEFAULT_COUNTING_JSON)
    ranking_json: str = field(default=DEFAULT_RANKING_JSON)
    p_rank: float = field(default=0.4)
    image_aspect_ratio: str = field(default="pad")
    is_multimodal: bool = field(default=True)
    image_processor: Optional[Any] = field(default=None)
    seed_dataset: int = field(default=1234)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=2048)
    remove_unused_columns: bool = field(default=False)
    lambda_rank: float = field(default=1.0)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class CRCOTrainer(Trainer):
    """Single forward, per-sample weighted CE.

    Logs aggregated, cross-rank-reduced per-component statistics that are
    commensurable with the trainer's rolling ``loss``:

    * ``counting_loss`` / ``ranking_loss`` — un-weighted CE averaged over the
      same window (= ``logging_steps × grad_accum`` microbatches) and across
      all data-parallel ranks. Tells you whether each objective is improving.
    * ``count_grad_share`` / ``rank_grad_share`` — fraction of the backward
      signal contributed by each objective in this window
      (``Σ w·CE`` per type, normalised). Useful for spotting gradient hijack.
    * ``frac_ranking`` — fraction of samples in the window that were ranking.
    """

    def __init__(self, *args: Any, lambda_rank: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_rank = float(lambda_rank)
        # Window accumulators (sums over microbatches × ranks since last log).
        self._sum_count_ce = torch.tensor(0.0, dtype=torch.float64)
        self._sum_rank_ce = torch.tensor(0.0, dtype=torch.float64)
        self._n_count = torch.tensor(0.0, dtype=torch.float64)
        self._n_rank = torch.tensor(0.0, dtype=torch.float64)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_ranking = inputs.pop("is_ranking")
        inputs.pop("image_counts", None)

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values")

        mm = model.module if hasattr(model, "module") else model
        outputs = mm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            und_image=pixel_values,
        )
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B, Tm1, V = shift_logits.shape

        token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(B, Tm1)
        valid = (shift_labels != IGNORE_INDEX).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        per_sample = (token_ce * valid).sum(dim=1) / denom  # (B,)

        is_rank_dev = is_ranking.to(per_sample.device)
        weights = torch.where(
            is_rank_dev,
            torch.full_like(per_sample, self.lambda_rank),
            torch.ones_like(per_sample),
        )
        total_loss = (per_sample * weights).mean()

        # Accumulate window statistics (sums; cross-rank reduce happens at log()).
        with torch.no_grad():
            count_mask = (~is_rank_dev).float()
            rank_mask = is_rank_dev.float()
            self._sum_count_ce += float((per_sample * count_mask).sum().item())
            self._sum_rank_ce += float((per_sample * rank_mask).sum().item())
            self._n_count += float(count_mask.sum().item())
            self._n_rank += float(rank_mask.sum().item())

        return (total_loss, outputs) if return_outputs else total_loss

    def _reset_window(self) -> None:
        self._sum_count_ce.zero_()
        self._sum_rank_ce.zero_()
        self._n_count.zero_()
        self._n_rank.zero_()

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = tensor.detach().clone().to(self.args.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            return t.cpu()
        return tensor.detach().clone()

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if "loss" in logs:
            sums = torch.stack([
                self._sum_count_ce, self._sum_rank_ce,
                self._n_count, self._n_rank,
            ])
            sums = self._all_reduce_sum(sums)
            sum_count_ce, sum_rank_ce, n_count, n_rank = (float(x.item()) for x in sums)

            logs["counting_loss"] = sum_count_ce / n_count if n_count > 0 else 0.0
            logs["ranking_loss"] = sum_rank_ce / n_rank if n_rank > 0 else 0.0
            n_total = n_count + n_rank
            logs["frac_ranking"] = (n_rank / n_total) if n_total > 0 else 0.0

            # Gradient share = Σ(w · CE) per type, normalised.
            count_signal = sum_count_ce  # weight = 1.0
            rank_signal = self.lambda_rank * sum_rank_ce
            denom = count_signal + rank_signal
            if denom > 0:
                logs["count_grad_share"] = count_signal / denom
                logs["rank_grad_share"] = rank_signal / denom
            else:
                logs["count_grad_share"] = 0.0
                logs["rank_grad_share"] = 0.0
            self._reset_window()
        super().log(logs, *args, **kwargs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def train() -> None:
    apply_transformers_compat_shims()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # ── Pre-flight integrity check ────────────────────────────────────────
    base_weights = find_base_weights(model_args.model_name_or_path)
    preflight_md5 = md5_prefix(base_weights)
    rank0_print(f"=== PRE-FLIGHT MD5 ({Path(base_weights).name}): {preflight_md5} ===")

    # ── Base model ─────────────────────────────────────────────────────────
    rank0_print(f"[Model] Loading UniLIP-3B from {model_args.model_name_or_path}")
    model_cls = load_unilip_class()
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

    # Freeze everything; LoRA + lm_head trainable.
    for p in model.parameters():
        p.requires_grad = False

    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["lm_head"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    llm = model.get_model().language_model
    peft_llm = get_peft_model(llm, lora_cfg)
    model.get_model().language_model = peft_llm
    for p in model.lm_head.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        f"[LoRA] Total={total/1e6:.1f}M  Trainable={trainable/1e6:.1f}M "
        f"({100*trainable/total:.2f}%)"
    )
    peft_llm.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    # ── Tokenizer / processor ──────────────────────────────────────────────
    rank0_print(f"[Proc] Loading processor from {model_args.mllm_hf_path}")
    proc_full = AutoProcessor.from_pretrained(model_args.mllm_hf_path, trust_remote_code=True)
    tokenizer = proc_full.tokenizer
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
    data_args.image_processor = proc_full.image_processor

    # ── Dataset & collator ─────────────────────────────────────────────────
    train_dataset = MixedCountingRankingDataset(
        counting_path=data_args.counting_json,
        ranking_path=data_args.ranking_json,
        tokenizer=tokenizer,
        data_args=data_args,
        p_rank=data_args.p_rank,
        seed=data_args.seed_dataset,
        verify_ranking_files=True,
    )
    collator = MixedSFTDataCollator(tokenizer=tokenizer)

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        lambda_rank=training_args.lambda_rank,
    )
    sig = _inspect.signature(transformers.Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = CRCOTrainer(**trainer_kwargs)

    ckpt_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_ckpt = any((c / "trainer_state.json").exists() for c in ckpt_dirs)
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_state()

    # ── Save adapter ───────────────────────────────────────────────────────
    _ds_engine = getattr(trainer, "deepspeed", None)
    _zero_stage = (
        _ds_engine.zero_optimization_stage()
        if (_ds_engine is not None and hasattr(_ds_engine, "zero_optimization_stage"))
        else 0
    )
    is_zero3 = (_zero_stage == 3)
    adapter_dir = os.path.join(training_args.output_dir, "adapter")
    if is_main:
        os.makedirs(adapter_dir, exist_ok=True)
    peft_model_ref = model.get_model().language_model

    if is_zero3:
        rank0_print("\n=== [ZeRO-3] Saving LoRA adapter (all-rank gather) ===")
        peft_model_ref.save_pretrained(adapter_dir)
        rank0_print(f"  LoRA adapter → {adapter_dir}")
    else:
        if is_main:
            rank0_print("\n=== Saving LoRA adapter ===")
            peft_model_ref.save_pretrained(adapter_dir)
            rank0_print(f"  LoRA adapter → {adapter_dir}")

    if is_main:
        postflight_md5 = md5_prefix(base_weights)
        if postflight_md5 != preflight_md5:
            rank0_print(f"FATAL: base model modified! {preflight_md5} → {postflight_md5}")
        else:
            rank0_print(f"=== POST-FLIGHT: base model intact ({postflight_md5}) ===")
        rank0_print("Done.")


if __name__ == "__main__":
    train()
