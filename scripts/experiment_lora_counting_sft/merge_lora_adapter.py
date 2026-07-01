#!/usr/bin/env python3
"""Offline merge of the Variant B LoRA adapter into the base UniLIP-3B model.

Run this AFTER training when ZeRO-3 was used (merge_and_unload cannot run
during a ZeRO-3 session because parameters are sharded across GPUs).

Usage:
    python scripts/experiment_lora_counting_sft/merge_lora_adapter.py \
        --adapter_dir /data/amondal/unicount_runs/lora_counting_sft_variantB_.../adapter \
        --base_model  /data/amondal/model_cache/UniLIP-3B \
        [--output_dir /data/amondal/unicount_runs/lora_counting_sft_variantB_.../merged]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", required=True,
                   help="Path to the saved PEFT adapter directory (adapter/)")
    p.add_argument("--base_model",  required=True,
                   help="Path to the original UniLIP-3B base model")
    p.add_argument("--output_dir",  default=None,
                   help="Where to save the merged model "
                        "(default: <adapter_dir>/../merged)")
    return p.parse_args()


def main():
    args = parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    base_model  = Path(args.base_model).resolve()
    output_dir  = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else adapter_dir.parent / "merged"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Merge] Loading base model from {base_model}")
    from scripts.counting_grpo.train_hf_multi_image_count_sft import (
        apply_transformers_compat_shims,
        load_unilip_class,
    )
    apply_transformers_compat_shims()

    model_cls = load_unilip_class()
    model = model_cls.from_pretrained(
        str(base_model),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    print(f"[Merge] Loading PEFT adapter from {adapter_dir}")
    from peft import PeftModel
    llm    = model.get_model().language_model
    peft_llm = PeftModel.from_pretrained(llm, str(adapter_dir))

    print("[Merge] Merging LoRA weights into base …")
    merged_llm = peft_llm.merge_and_unload()
    model.get_model().language_model = merged_llm

    print(f"[Merge] Saving merged model → {output_dir}")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    model.config.save_pretrained(str(output_dir))
    print("[Merge] Done.")


if __name__ == "__main__":
    main()
