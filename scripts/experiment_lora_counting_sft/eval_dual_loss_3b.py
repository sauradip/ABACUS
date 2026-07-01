#!/usr/bin/env python3
"""
Evaluate dual-loss trained counter on attn_regularizer_val split — multi-GPU.

Adapted from eval_lora_counting_sft.py for dual-loss training variant:
  - Trains with CE + ObjectFocusedAttentionLoss (λ_ar = 0.1)
  - Evaluation unchanged: identical §A evaluation format
  - System: "You are a helpful counting assistant. Answer with only a number."
  - Prompt: "How many {category} are present in this image? Answer with only a number."
  - Greedy decoding, do_sample=False, max_new_tokens=8
  - fp16, sdpa attention

Validates on attn_regularizer_val.json (1,581 val images) from dual-loss training.
Compares MAE/RMSE against CE-only v3s baseline.

Usage (8 GPUs, dual-loss checkpoint):
    accelerate launch --num_processes=8 --mixed_precision=no \
        scripts/experiment_lora_counting_sft/eval_dual_loss_3b.py \
        --checkpoint_dir <path_to_attn_regularizer_ckpt_adapter> \
        [--connector_weights <path_to_multi_modal_projector.bin>] \
        [--val_json <path>] \
        [--out_json <path>]

Usage (single GPU):
    python scripts/experiment_lora_counting_sft/eval_dual_loss_3b.py

Defaults to checkpoint-2670 LoRA adapter from dual-loss full training run.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)

# ── PEFT/transformers compat shim ────────────────────────────────────────────
try:
    import peft.utils.save_and_load as _peft_sal  # noqa: PLC0415
    _peft_sal._maybe_shard_state_dict_for_tp = lambda *a, **kw: None  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# ── Defaults ──────────────────────────────────────────────────────────────────
_DUAL_LOSS_RUN = (
    "/data/amondal/unicount_runs/"
    "attn_regularizer_full_best_20260507_144336"
)
DEFAULT_BASE_MODEL = "/data/amondal/model_cache/UniLIP-3B"
DEFAULT_MLLM_HF = (
    "/data/amondal/UniCount/.hf_cache/hub/"
    "models--OpenGVLab--InternVL3-2B-hf/snapshots/"
    "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
)
DEFAULT_CHECKPOINT = f"{_DUAL_LOSS_RUN}/checkpoint-2670"
DEFAULT_VAL_JSON   = (
    "data/attn_regularizer_dataset/attn_regularizer_val.json"
)
DEFAULT_OUT_JSON   = (
    "outputs/experiment_lora_counting_sft/eval/dual_loss_val_mae.json"
)

# ── Constants (must match training exactly) ───────────────────────────────────
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
NUM_IMG_TOKENS    = 256

# Chat template — identical to training
CHAT_TEMPLATE = (
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

NUMERIC = re.compile(r"\d+")


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    base_model: str, checkpoint_dir: str, mllm_hf: str, device: torch.device,
    connector_weights: str | None = None,
):
    """Load dual-loss trained checkpoint or base + LoRA adapter."""
    apply_transformers_compat_shims()

    model_cls = load_unilip_class()

    # Try loading from full checkpoint first (dual-loss training with LoRA merged)
    try:
        import json as json_lib
        if (Path(checkpoint_dir) / "config.json").exists():
            model = model_cls.from_pretrained(
                checkpoint_dir,
                attn_implementation="sdpa",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            print(f"[Eval] loaded full checkpoint from {checkpoint_dir}")
        else:
            raise FileNotFoundError("No config.json in checkpoint")
    except Exception as e:  # noqa: BLE001
        print(f"[Eval] full checkpoint load failed: {e}, trying adapter approach")
        model = model_cls.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        from peft import PeftModel  # noqa: PLC0415
        llm = model.get_model().language_model
        llm = PeftModel.from_pretrained(llm, checkpoint_dir, is_trainable=False)
        model.get_model().language_model = llm

    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False

    if connector_weights:
        inner = model.get_model()
        if not hasattr(inner, "multi_modal_projector"):
            raise RuntimeError("Model has no multi_modal_projector to load into.")
        sd = torch.load(connector_weights, map_location="cpu")
        missing, unexpected = inner.multi_modal_projector.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[Eval] connector load: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"[Eval] loaded connector weights from {connector_weights}")

    model = model.to(device).eval()

    processor     = AutoProcessor.from_pretrained(mllm_hf, trust_remote_code=True)
    tokenizer     = processor.tokenizer
    img_processor = processor.image_processor

    return model, tokenizer, img_processor


# ── Prompt / tokenisation ──────────────────────────────────────────────────────
def build_input_ids(system: str, human: str, tokenizer) -> list[int]:
    """Tokenise system + user turns with image placeholder substitution."""
    img_placeholder = (
        f"{IMG_START_TOKEN}"
        f"{IMG_CONTEXT_TOKEN * NUM_IMG_TOKENS}"
        f"{IMG_END_TOKEN}"
    )
    human_filled = human.replace("<image>", img_placeholder).strip()

    tok = copy.deepcopy(tokenizer)
    tok.chat_template = CHAT_TEMPLATE

    input_ids: list[int] = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": human_filled},
        ],
        add_generation_prompt=True,
        return_dict=False,
    )
    return input_ids


# ── Eval loop ─────────────────────────────────────────────────────────────────
def run_eval(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    rank        = accelerator.process_index
    world_size  = accelerator.num_processes
    device      = accelerator.device
    is_main     = accelerator.is_main_process

    if is_main:
        print(f"[Eval] world_size    : {world_size}")
        print(f"[Eval] base_model    : {args.base_model}")
        print(f"[Eval] checkpoint    : {args.checkpoint_dir}")
        print(f"[Eval] val_json      : {args.val_json}")

    model, tokenizer, img_proc = load_model_and_tokenizer(
        args.base_model, args.checkpoint_dir, args.mllm_hf, device,
        connector_weights=args.connector_weights,
    )

    # Workaround: Ensure model has embed_tokens for generation (for PEFT compatibility)
    if not hasattr(model.get_model(), 'embed_tokens'):
        llm = model.get_model().language_model
        if hasattr(llm, 'model') and hasattr(llm.model, 'embed_tokens'):
            model.get_model().embed_tokens = llm.model.embed_tokens
        elif hasattr(llm, 'embed_tokens'):
            model.get_model().embed_tokens = llm.embed_tokens

    with open(args.val_json) as fh:
        val_data = json.load(fh)

    local_data = val_data[rank::world_size]
    if is_main:
        print(f"[Eval] {len(val_data):,} val images → {len(local_data)} per GPU\n")

    local_rows: list[dict] = []
    parse_failures = 0

    for item in tqdm(local_data, desc=f"rank{rank}", disable=(not is_main)):
        convs  = {c["from"]: c["value"] for c in item["conversations"]}
        system = convs.get(
            "system",
            "You are a helpful counting assistant. Answer with only a number.",
        )
        human  = convs["human"]
        gt     = int(convs["gpt"])

        try:
            pil = Image.open(item["image"]).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            if is_main:
                print(f"[WARN] Cannot open {item['image']}: {exc}")
            local_rows.append({
                "image": item["image"], "gt": gt, "pred": 0,
                "raw_output": "", "parse_ok": False,
            })
            parse_failures += 1
            continue

        pixel_values = img_proc.preprocess(
            [pil], return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.float16)

        input_ids = build_input_ids(system, human, tokenizer)
        ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)

        attn_mask = torch.ones_like(ids_t)
        with torch.no_grad():
            out = model.generate(
                input_ids      = ids_t,
                attention_mask = attn_mask,
                pixel_values   = pixel_values,
                do_sample      = False,
                max_new_tokens = 8,
            )

        new_toks = out[0]
        text = tokenizer.decode(new_toks, skip_special_tokens=True).strip()

        m = NUMERIC.search(text)
        if m:
            pred, parse_ok = int(m.group()), True
        else:
            pred, parse_ok = 0, False
            parse_failures += 1

        local_rows.append({
            "image"      : item["image"],
            "gt"         : gt,
            "pred"       : pred,
            "raw_output" : text,
            "parse_ok"   : parse_ok,
        })

    # ── Gather rows from all ranks to rank 0 ─────────────────────────────────
    if world_size > 1:
        all_rows_per_rank: list[list[dict]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(all_rows_per_rank, local_rows)
        rows: list[dict] = []
        max_len = max(len(r) for r in all_rows_per_rank)
        for i in range(max_len):
            for rk in range(world_size):
                if i < len(all_rows_per_rank[rk]):
                    rows.append(all_rows_per_rank[rk][i])
    else:
        rows = local_rows

    # ── Aggregate and write (rank 0 only) ────────────────────────────────────
    if is_main:
        preds = np.array([r["pred"] for r in rows], dtype=float)
        gts   = np.array([r["gt"]   for r in rows], dtype=float)
        mae   = float(np.mean(np.abs(preds - gts)))
        rmse  = float(np.sqrt(np.mean((preds - gts) ** 2)))
        total_parse_failures = sum(1 for r in rows if not r["parse_ok"])
        parse_rate = 1.0 - total_parse_failures / max(len(rows), 1)

        split_name = args.split_name or "attn_regularizer_val"
        dataset_name = args.dataset_name or "attention_regularizer"
        result = {
            "split"        : split_name,
            "dataset"      : dataset_name,
            "training_mode": "dual_loss (CE + attention_regularization)",
            "n"            : len(rows),
            "MAE"          : mae,
            "RMSE"         : rmse,
            "parse_rate"   : parse_rate,
            "checkpoint"   : args.checkpoint_dir,
            "world_size"   : world_size,
            "rows"         : rows,
        }

        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  dataset      : {dataset_name} {split_name}")
        print(f"  n_samples    : {len(rows)}")
        print(f"  GPUs used    : {world_size}")
        print(f"  MAE          : {mae:.2f}")
        print(f"  RMSE         : {rmse:.2f}")
        print(f"  parse rate   : {parse_rate:.1%}")
        print(f"  output       : {out_path}")
        print(f"{sep}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate dual-loss trained counter on attn_regularizer_val split"
    )
    ap.add_argument(
        "--base_model", default=DEFAULT_BASE_MODEL,
        help="Path to UniLIP-3B base weights directory",
    )
    ap.add_argument(
        "--mllm_hf", default=DEFAULT_MLLM_HF,
        help="InternVL3-2B-hf snapshot path",
    )
    ap.add_argument(
        "--checkpoint_dir", default=DEFAULT_CHECKPOINT,
        help="PEFT adapter directory (default: checkpoint-2670 from dual-loss run)",
    )
    ap.add_argument(
        "--connector_weights", default=None,
        help="Optional path to multi_modal_projector.bin from dual-loss training",
    )
    ap.add_argument(
        "--val_json", default=DEFAULT_VAL_JSON,
        help="attn_regularizer_val.json (1,581 entries)",
    )
    ap.add_argument(
        "--out_json", default=DEFAULT_OUT_JSON,
        help="Where to write per-image results + MAE/RMSE",
    )
    ap.add_argument("--dataset_name", default=None, help="Optional dataset label")
    ap.add_argument("--split_name", default=None, help="Optional split label")
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
