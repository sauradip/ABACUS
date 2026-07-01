#!/usr/bin/env python3
"""LoRA SFT (Variant B reproduction) — 3B + LoRA r=64/alpha=128 + connector unfrozen.

Reproduces ADAPTIVE_TILING_FULL_SPEC.md §A using the **3B** base, with the
following user-requested deltas vs. the original Variant B run:

  * LoRA rank/alpha doubled: r=64, alpha=128 (vs. 32/64 in spec).
  * Connector (`multi_modal_projector` aka mlp1, the vision→LLM MLP) is
    unfrozen and trained jointly with the LoRA adapters and lm_head.
  * (Also unfreezes `llm_connector` / `projector` / `latent_queries`
    mirroring `--fix_connect False` in the upstream UniLIP scripts; these
    are generation-side and won't see gradients in the understanding-only
    forward, but we follow the same convention.)

This is a thin wrapper around train_lora_counting_sft.py: identical data
pipeline, identical Trainer / collator / focus-reg path, only the LoRA
config + the additional unfreeze step differ.

All output paths are deliberately distinct from the original Variant B
run paths.  Default output dir: pass via `--output_dir`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_lora_counting_sft.train_lora_counting_sft import (  # noqa: E402
    CountingSFTDataset,
    CountingTrainer,
    DataArguments,
    ModelArguments,
    SFTDataCollator,
    TrainingArguments,
    find_base_weights,
    md5_prefix,
    rank0_print,
    smart_tokenizer_resize,
)
from scripts.counting_grpo.train_hf_multi_image_count_sft import (  # noqa: E402
    apply_transformers_compat_shims,
    load_unilip_class,
)

# PEFT 0.19 references transformers.integrations.tensor_parallel.ALL_PARALLEL_STYLES
# inside set_peft_model_state_dict via _maybe_shard_state_dict_for_tp; that symbol
# was removed in transformers 4.52. We aren't using TP — neutralize it.
try:
    import peft.utils.save_and_load as _peft_sal  # noqa: E402
    _peft_sal._maybe_shard_state_dict_for_tp = lambda *a, **kw: None  # type: ignore[attr-defined]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Override LoRA defaults to r=64 / alpha=128 (user request)
# ---------------------------------------------------------------------------
ModelArguments.__dataclass_fields__["lora_rank"].default = 64
ModelArguments.__dataclass_fields__["lora_alpha"].default = 128


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
    rank0_print(
        f"=== PRE-FLIGHT MD5 (1MB prefix of {Path(base_weights).name}): "
        f"{preflight_md5} ==="
    )

    # ── Load 3B base ───────────────────────────────────────────────────────
    rank0_print(f"[Model] Loading UniLIP-3B from {model_args.model_name_or_path}")
    model_cls = load_unilip_class()
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

    # ── Variant B LoRA, r=64 / alpha=128 ──────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        r            = model_args.lora_rank,        # 64
        lora_alpha   = model_args.lora_alpha,       # 128
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

    if data_args.attention_regularizer:
        _orig_llm_forward = peft_llm.forward
        def _llm_forward_force_attn(*a, **kw):
            kw["output_attentions"] = True
            return _orig_llm_forward(*a, **kw)
        peft_llm.forward = _llm_forward_force_attn
        rank0_print("[FocusReg] Patched LLM forward to force output_attentions=True.")

    # Optional warm-start (mirrors original script's --init_adapter_from)
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

    # lm_head trainable
    for p in model.lm_head.parameters():
        p.requires_grad = True

    # ── User request: ALSO unfreeze the connector ──────────────────────────
    # The "connector" in the understanding path is `multi_modal_projector`
    # (= internvl mlp1: vision-features → LLM hidden).  It IS exercised by
    # every forward pass that consumes pixel_values, so its gradients
    # actually update during this counting SFT.
    inner = model.get_model()
    n_conn = 0
    if hasattr(inner, "multi_modal_projector"):
        inner.multi_modal_projector.train()
        for p in inner.multi_modal_projector.parameters():
            p.requires_grad = True
            n_conn += p.numel()
        rank0_print(f"[Unfreeze] multi_modal_projector trainable: {n_conn/1e6:.2f}M params")

        # Optional warm-start of the connector from a previously saved
        # multi_modal_projector.bin (mirrors --init_adapter_from for LoRA).
        # Picked up from env var INIT_CONNECTOR_FROM to avoid touching the
        # shared ModelArguments dataclass.
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
        rank0_print("[Unfreeze] WARNING: multi_modal_projector not found — connector remains frozen.")

    # NOTE: We intentionally do NOT unfreeze the generation-side
    # `llm_connector`, `projector`, or `latent_queries` here, even though
    # the upstream UniLIP `--fix_connect False` flag would mark them
    # trainable.  In the understanding-only counting forward used by this
    # trainer, those modules are never executed, so they receive zero
    # gradient.  Marking them trainable causes:
    #   • Wasted optimizer state (~285M params for AdamW × 8 GPUs).
    #   • DDP / DeepSpeed desync risk on optimizer_step → NCCL hangs.
    # The vision→LLM connector that IS on the data path is
    # `multi_modal_projector` (mlp1), unfrozen above.
    rank0_print("[Unfreeze] Skipping gen-side llm_connector/projector/latent_queries (not on data path).")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        f"[LoRA+Conn] Total={total/1e6:.1f}M  Trainable={trainable/1e6:.1f}M  "
        f"({100*trainable/total:.2f}%)"
    )
    peft_llm.print_trainable_parameters()

    # ── Gradient checkpointing ────────────────────────────────────────────
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
    train_dataset = CountingSFTDataset(
        data_path  = data_args.data_path,
        tokenizer  = tokenizer,
        data_args  = data_args,
    )
    collator = SFTDataCollator(tokenizer=tokenizer)

    # ── Trainer ───────────────────────────────────────────────────────────
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
    trainer.data_args = data_args
    trainer.tokenizer = tokenizer

    import pathlib
    ckpt_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_ckpt  = any((c / "trainer_state.json").exists() for c in ckpt_dirs)
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_state()

    # ── Save adapter / merged ─────────────────────────────────────────────
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
            rank0_print("\n=== Saving LoRA adapter and merged model ===")

            merged_dir = os.path.join(training_args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)

            peft_model_ref.save_pretrained(adapter_dir)
            rank0_print(f"  LoRA adapter → {adapter_dir}")

            # Save the (now-trained) connector weights separately as well, so
            # downstream merge can re-apply them.  The full state_dict below
            # will also include them — this is just a redundant artifact for
            # convenience.
            try:
                conn_sd = {
                    k: v.detach().cpu()
                    for k, v in inner.multi_modal_projector.state_dict().items()
                }
                torch.save(conn_sd, os.path.join(adapter_dir, "multi_modal_projector.bin"))
                rank0_print(f"  Connector weights → {adapter_dir}/multi_modal_projector.bin")
            except Exception as exc:  # noqa: BLE001
                rank0_print(f"  [WARN] Could not save connector separately: {exc}")

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
