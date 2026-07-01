#!/usr/bin/env python3
"""Merge LoRA adapter weights into base model and save merged checkpoint."""
from pathlib import Path
import sys
import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)

def merge_lora_checkpoint(base_model_path: str, lora_checkpoint_path: str, output_path: str) -> None:
    """Merge LoRA checkpoint into base model."""
    apply_transformers_compat_shims()
    model_cls = load_unilip_class()

    print(f"[Merge] Loading base model from {base_model_path}...")
    model = model_cls.from_pretrained(
        base_model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"[Merge] Wrapping with PEFT LoRA from {lora_checkpoint_path}...")
    llm = model.get_model().language_model
    llm = PeftModel.from_pretrained(llm, lora_checkpoint_path, is_trainable=False)

    print("[Merge] Merging LoRA weights into base model...")
    llm = llm.merge_and_unload()
    model.get_model().language_model = llm

    print(f"[Merge] Saving merged model to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    print(f"[Merge] Done! Merged model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Merge LoRA checkpoint into base model")
    ap.add_argument("--base_model", required=True, help="Base model path")
    ap.add_argument("--lora_checkpoint", required=True, help="LoRA checkpoint path")
    ap.add_argument("--output_path", required=True, help="Output path for merged model")
    args = ap.parse_args()

    merge_lora_checkpoint(args.base_model, args.lora_checkpoint, args.output_path)
