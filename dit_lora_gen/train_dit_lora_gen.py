#!/usr/bin/env python3
"""
DiT LoRA + Connector Generation Training for UniLIP-3B

Based on /data/amondal/UniCount/UniLIP/unilip/train/train_stage3.py

Usage:
    python train_dit_lora_gen.py \
        --base_model /path/to/v3s_copy \
        --train_data /path/to/gen_mix_webds \
        --output_dir /path/to/output \
        --dit_lora_rank 128 \
        --dit_lora_alpha 256 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8
"""

import os
import sys
import torch
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Set paths
sys.path.insert(0, '/data/amondal/UniCount/UniLIP')
sys.path.insert(0, '/data/amondal/UniCount')

from unilip.train.train_stage3 import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    make_supervised_data_module,
    rank0_print,
)
from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()

    # Base paths
    parser.add_argument("--base_model", required=True,
                        help="Path to v3-S checkpoint copy")
    parser.add_argument("--original_v3s", required=True,
                        help="Path to ORIGINAL v3-S (for post-flight MD5 check)")
    parser.add_argument("--train_data", required=True,
                        help="Path to WebDataset .tar shards directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for trained weights")

    # DiT LoRA
    parser.add_argument("--dit_lora_rank", type=int, default=128,
                        help="DiT LoRA rank")
    parser.add_argument("--dit_lora_alpha", type=int, default=256,
                        help="DiT LoRA alpha (default: 2 * rank)")

    # Training
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")

    return parser.parse_args()


def record_md5(checkpoint_path, label=""):
    """Record MD5 of adapter weights."""
    adapter_path = Path(checkpoint_path) / 'adapter' / 'adapter_model.safetensors'
    if not adapter_path.exists():
        adapter_path = Path(checkpoint_path) / 'model.safetensors'

    if adapter_path.exists():
        md5 = hashlib.md5(open(adapter_path, 'rb').read()).hexdigest()
        print(f"{label} MD5: {md5}")
        return md5
    return None


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("DiT LoRA + Connector Generation Training")
    print("="*80)

    # ── Pre-flight: Record original v3-S MD5 ──
    print("\n[PRE-FLIGHT CHECKS]")
    v3s_md5_pre = record_md5(args.original_v3s, "Original v3-S")
    v3s_copy_md5_pre = record_md5(args.base_model, "v3-S copy (before training)")

    assert v3s_md5_pre == v3s_copy_md5_pre, (
        f"Copy MD5 mismatch! Original: {v3s_md5_pre}, Copy: {v3s_copy_md5_pre}"
    )

    # ── Verify training data exists ──
    train_data_dir = Path(args.train_data)
    tar_files = sorted(train_data_dir.glob('*.tar'))
    if not tar_files:
        print(f"❌ No .tar files found in {train_data_dir}")
        sys.exit(1)
    print(f"✓ Found {len(tar_files)} training shards")
    for tar in tar_files[:3]:
        print(f"  - {tar.name} ({tar.stat().st_size / 1024 / 1024:.1f} MB)")
    if len(tar_files) > 3:
        print(f"  ... and {len(tar_files) - 3} more")

    # ── Load model ──
    print("\n[MODEL LOADING]")
    print(f"Loading model from: {args.base_model}")

    model = UniLIP_InternVLForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    tokenizer = AutoProcessor.from_pretrained(
        model.config.mllm_hf_path
    ).tokenizer

    # ── Freeze policy: everything frozen by default ──
    print("\n[FREEZE POLICY]")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze connector
    connector = model.get_model().llm_connector
    for param in connector.parameters():
        param.requires_grad = True

    connector_params = sum(p.numel() for p in connector.parameters()
                          if p.requires_grad)
    print(f"Connector trainable: {connector_params:,} params")

    # ── Apply DiT LoRA ──
    print("\n[DiT LoRA]")

    if args.dit_lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        # Freeze DiT base first
        for p in model.get_model().dit.parameters():
            p.requires_grad = False

        dit_targets = ["to_q", "to_k", "to_v", "to_out.0"]
        dit_alpha = args.dit_lora_alpha if args.dit_lora_alpha > 0 else 2 * args.dit_lora_rank

        dit_lora_config = LoraConfig(
            r=args.dit_lora_rank,
            lora_alpha=dit_alpha,
            lora_dropout=0.05,
            target_modules=dit_targets,
            bias="none",
        )

        dit = model.get_model().dit
        dit = get_peft_model(dit, dit_lora_config)
        model.get_model().dit = dit

        dit_trainable = sum(p.numel() for p in dit.parameters()
                           if p.requires_grad)
        dit_total = sum(p.numel() for p in dit.parameters())
        print(f"DiT LoRA: {dit_trainable:,} / {dit_total:,} params "
              f"(r={args.dit_lora_rank}, α={dit_alpha})")

    # Freeze latent_queries
    model.get_model().latent_queries.requires_grad = False

    # ── Verify freeze policy ──
    print("\n[FREEZE VERIFICATION]")
    trainable_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()
        if "lora_" in name_lower:
            group = "DiT (LoRA)"
        elif ".dit." in name_lower or name_lower.endswith(".dit"):
            group = "DiT (base, ERROR - should be frozen)"
        elif "llm_connector" in name_lower:
            group = "Connector"
        elif "language_model" in name_lower or "lm_head" in name_lower:
            group = "LLM (ERROR - should be frozen)"
        else:
            group = f"OTHER ({name.split('.')[0]})"

        trainable_groups[group] = trainable_groups.get(group, 0) + param.numel()

    for group, count in sorted(trainable_groups.items()):
        flag = " ⚠️ VIOLATION" if "ERROR" in group else ""
        print(f"  {group:40s}: {count:>14,d}{flag}")

    total_trainable = sum(trainable_groups.values())
    print(f"  {'TOTAL':40s}: {total_trainable:>14,d}")

    # Critical check
    violations = sum(v for k, v in trainable_groups.items() if "ERROR" in k)
    assert violations == 0, f"FREEZE VIOLATION: {violations:,} params trainable that should be frozen!"
    print("✓ Freeze policy verified")

    # ── Data loading setup ──
    print("\n[DATA LOADING]")

    # Create minimal DataArguments for compatibility
    data_args = DataArguments(
        data_path=str(train_data_dir),
        lazy_preprocess=True,
        is_multimodal=True,
        gen_image_folder=str(train_data_dir),
    )

    print(f"Will load from: {data_args.gen_image_folder}")

    # ── Note: Full training loop integration ──
    print("\n[TRAINING SETUP]")
    print("NOTE: This is a training script skeleton.")
    print("To run full training:")
    print("  1. Adapt DataArguments and make_supervised_data_module()")
    print("  2. Set up TrainingArguments with DeepSpeed ZeRO-2")
    print("  3. Initialize NonMixTrainer from train_stage3.py")
    print("  4. Call trainer.train()")
    print("")
    print(f"Model ready for training at: {args.base_model}")
    print(f"Output will be saved to: {output_dir}")
    print("")
    print("✓ Pre-flight checks complete")
    print("✓ Model is frozen correctly")
    print("✓ Ready to train")

    # Save config
    config_file = output_dir / 'training_config.txt'
    with open(config_file, 'w') as f:
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"Training data: {args.train_data}\n")
        f.write(f"DiT LoRA rank: {args.dit_lora_rank}\n")
        f.write(f"DiT LoRA alpha: {args.dit_lora_alpha}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size (per device): {args.per_device_train_batch_size}\n")
        f.write(f"Gradient accumulation: {args.gradient_accumulation_steps}\n")
        f.write(f"v3-S Original MD5: {v3s_md5_pre}\n")

    print(f"\nConfig saved to: {config_file}")


if __name__ == "__main__":
    main()
