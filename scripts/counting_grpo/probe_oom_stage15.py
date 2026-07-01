#!/usr/bin/env python3
"""
OOM prober for Stage 1.5 SCAFFOLD-Rex Calibration SFT.

Progressively tries (context_length, batch_size) combinations
using the SAME model path, skeleton injection, and LoRA config
as train_stage15_sft.py.  Catches CUDA OOM at each step and
reports the last successful configuration to stdout as a
shell-sourceable KEY=VALUE block.

Usage (on an interactive GPU node):
    python3 scripts/counting_grpo/probe_oom_stage15.py \
        --base_model OpenGVLab/InternVL2-2B \
        --stage1_checkpoint checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140 \
        --sample_jsonl outputs/scaffold_rex_5k/train.jsonl

It will emit a final block like:
    # ── OOM probe result ──
    SAFE_MAX_LENGTH=12288
    SAFE_BATCH_SIZE=1
    SAFE_GRAD_ACCUM=16   # keeps effective batch = 16
"""

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_stage15_sft import (
    Stage15Dataset,
    Stage15Collator,
    apply_stability_overrides,
    inject_stage1_language_tensors,
    resolve_stage1_checkpoint,
    resolve_lora_targets,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

# Probe ladder: (max_length, batch_size)
# Starts conservative; walks toward GH200 target (16384, 1).
PROBE_LADDER = [
    (2048,  1),
    (4096,  1),
    (8192,  1),
    (12288, 1),
    (14336, 1),
    (16384, 1),
]


def build_single_batch(jsonl_path: str, tokenizer, image_processor, num_image_token: int, max_length: int):
    """Return one collated batch from the first record of the dataset."""
    from scripts.counting_grpo.train_stage15_sft import DataArguments  # not exported, define inline
    import dataclasses

    @dataclasses.dataclass
    class _DA:
        data_path: str = ""
        image_processor: object = None
        num_image_token: int = 256

    data_args = _DA(data_path=jsonl_path, image_processor=image_processor, num_image_token=num_image_token)
    ds = Stage15Dataset(jsonl_path, tokenizer, image_processor, num_image_token)
    collator = Stage15Collator(tokenizer=tokenizer)
    batch = collator([ds[0]])
    return batch


def try_forward_backward(model, batch, device, max_length: int):
    """Run one forward + backward pass; return True on success, False on OOM."""
    model.train()
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    # Truncate to target length to simulate the requested context window.
    for key in ["input_ids", "labels", "attention_mask"]:
        if key in batch and batch[key].shape[-1] > max_length:
            batch[key] = batch[key][..., :max_length]

    try:
        outputs = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_flags=batch.get("image_flags"),
        )
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return False
    except Exception as exc:
        print(f"  Non-OOM error at max_length={max_length}: {type(exc).__name__}: exc")
        traceback.print_exc()
        model.zero_grad(set_to_none=True)
        return False


def load_model(args, tokenizer, image_processor):
    checkpoint_dir = resolve_stage1_checkpoint(args.stage1_checkpoint)

    model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    injected = inject_stage1_language_tensors(model, checkpoint_dir)
    print(f"  Skeleton injection: {injected} tensors from {checkpoint_dir}")
    if injected < 120:
        print("  WARNING: low injection count — check checkpoint path.")

    apply_stability_overrides(model, vision_scale=args.vision_scale)

    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.mlp1.parameters():
        param.requires_grad = False

    lora_targets = resolve_lora_targets(model.language_model)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()

    model.config.use_cache = False
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="OpenGVLab/InternVL2-2B")
    p.add_argument("--stage1_checkpoint",
                   default="checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140")
    p.add_argument("--sample_jsonl", default="outputs/scaffold_rex_5k/train.jsonl")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--vision_scale", type=float, default=0.1)
    p.add_argument("--target_effective_batch", type=int, default=16,
                   help="Effective batch size; used to compute recommended grad_accum.")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available. Run on a GPU node via salloc.")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Device : {torch.cuda.get_device_name(device)}")
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"VRAM   : {total_vram:.1f} GB")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=False,
        padding_side="right", model_max_length=16384,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    print("Loading model + skeleton injection …")
    model = load_model(args, tokenizer, image_processor)
    model.to(device)
    print(f"Model loaded. Peak VRAM after load: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB\n")

    num_image_token = int(getattr(model, "num_image_token", 256))

    # Pre-build one real batch (images are loaded once; we re-use it).
    if not os.path.exists(args.sample_jsonl):
        print(f"WARNING: {args.sample_jsonl} not found. Generating a synthetic batch.")
        batch = _synthetic_batch(tokenizer, num_image_token, device)
    else:
        print(f"Loading one sample from {args.sample_jsonl} …")
        ds = Stage15Dataset(args.sample_jsonl, tokenizer, image_processor, num_image_token)
        collator = Stage15Collator(tokenizer=tokenizer)
        batch = collator([ds[0]])

    print("── OOM Probe Ladder ─────────────────────────────────────────")
    print(f"{'max_length':>12}  {'batch':>5}  {'result':>6}  peak VRAM")
    print("-" * 55)

    last_ok_length = None
    last_ok_batch = None

    for max_length, batch_size in PROBE_LADDER:
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        torch.cuda.empty_cache()

        ok = try_forward_backward(model, {k: v.clone() if torch.is_tensor(v) else v
                                          for k, v in batch.items()},
                                  device, max_length)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        status = "OK " if ok else "OOM"
        print(f"{max_length:>12}  {batch_size:>5}  {status:>6}  {peak:.2f} GB")

        if ok:
            last_ok_length = max_length
            last_ok_batch = batch_size
        else:
            print(f"\nOOM boundary hit at max_length={max_length}.")
            break

    print()
    if last_ok_length is None:
        print("ERROR: OOM on first rung. GPU may be too small or model load is failing.")
        sys.exit(2)

    safe_length = last_ok_length
    safe_batch = last_ok_batch
    safe_grad_accum = max(1, args.target_effective_batch // safe_batch)

    print("── OOM probe result ─────────────────────────────────────────")
    print(f"SAFE_MAX_LENGTH={safe_length}")
    print(f"SAFE_BATCH_SIZE={safe_batch}")
    print(f"SAFE_GRAD_ACCUM={safe_grad_accum}   # effective_batch={safe_batch * safe_grad_accum}")

    result_path = os.path.join(os.path.dirname(args.sample_jsonl) or ".", "oom_probe_result.env")
    with open(result_path, "w") as f:
        f.write(f"SAFE_MAX_LENGTH={safe_length}\n")
        f.write(f"SAFE_BATCH_SIZE={safe_batch}\n")
        f.write(f"SAFE_GRAD_ACCUM={safe_grad_accum}\n")
    print(f"\nResult also written to: {result_path}")
    print("Source it before sbatch:  source", result_path)


def _synthetic_batch(tokenizer, num_image_token, device):
    """Minimal synthetic batch for testing when no real JSONL exists yet."""
    from scripts.counting_grpo.train_stage15_sft import IGNORE_INDEX
    fake_text = f"<img>{'<IMG_CONTEXT>' * num_image_token}</img>\nCount objects.\n" + \
                '{"total_count":1,"anchors_summary":"(3,3)","clusters":[{"anchor":[3,3],"count":1,"region_bbox":[100,100,120,120]}]}'
    ids = tokenizer(fake_text, return_tensors="pt")["input_ids"][0]
    labels = ids.clone()
    labels[:10] = IGNORE_INDEX
    pv = torch.zeros(1, 3, 448, 448, dtype=torch.bfloat16)
    return {
        "input_ids": ids.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "pixel_values": pv,
        "image_flags": torch.tensor([[1]], dtype=torch.long),
    }


if __name__ == "__main__":
    main()
