"""
Debug generation config and token IDs to diagnose immediate EOS termination.

Loads the native SFT checkpoint and inspects:
1. Generation config parameters (bos, eos, pad token IDs)
2. Tokenizer vocabulary for these special tokens
3. Raw token IDs produced by generation (before decoding)
4. Decoded output with and without special token stripping
"""

import argparse
import os
import pathlib
import sys
from typing import Optional

import torch
from peft import PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers.dynamic_module_utils as hf_dynamic_module_utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Patch module cache before imports
os.environ.setdefault("HF_MODULES_CACHE", str(REPO_ROOT / ".cache" / "native_sft_stage1" / "hf_modules"))
hf_dynamic_module_utils.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]

from scripts.counting_grpo.grpo_rewards import chamfer_distance


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    attn_implementation: str = "eager",
    torch_dtype=torch.float16,
):
    """Load model (possibly PEFT-wrapped) and tokenizer."""
    print(f"[DEBUG] Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"[DEBUG] Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map="cuda:0",
        local_files_only=True,
    )

    # Check if this is a PEFT checkpoint (has LoRA weights)
    checkpoint_dir = pathlib.Path(model_path)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_weights_path = checkpoint_dir / "adapter_model.bin"

    if adapter_config_path.exists() and adapter_weights_path.exists():
        print(f"[DEBUG] Detected PEFT checkpoint. Loading LoRA state dict...")
        state_dict = torch.load(adapter_weights_path, map_location="cpu")
        set_peft_model_state_dict(model, state_dict)
        model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
        model = model.merge_and_unload()
        print(f"[DEBUG] LoRA merged into base model.")

    # Ensure generate() method exists (v4.50+ compatibility)
    if not hasattr(model, "generate"):
        from transformers.generation import GenerationMixin
        model.__class__ = type(model.__class__.__name__, (GenerationMixin, model.__class__), {})
        print(f"[DEBUG] Patched GenerationMixin onto model for generate() support.")

    model.eval()
    return model, tokenizer


def inspect_generation_config(model, tokenizer):
    """Print generation config and special token IDs."""
    print("\n" + "=" * 60)
    print("GENERATION CONFIG INSPECTION")
    print("=" * 60)

    gen_config = model.generation_config
    print(f"\n[GEN CONFIG]")
    print(f"  bos_token_id:       {gen_config.bos_token_id}")
    print(f"  eos_token_id:       {gen_config.eos_token_id}")
    print(f"  pad_token_id:       {gen_config.pad_token_id}")
    print(f"  max_new_tokens:     {gen_config.max_new_tokens}")
    print(f"  max_length:         {gen_config.max_length}")
    print(f"  temperature:        {gen_config.temperature}")
    print(f"  top_p:              {gen_config.top_p}")
    print(f"  top_k:              {gen_config.top_k}")
    print(f"  do_sample:          {gen_config.do_sample}")
    print(f"  use_cache:          {gen_config.use_cache}")
    print(f"  num_beams:          {gen_config.num_beams}")

    print(f"\n[TOKENIZER SPECIAL TOKENS]")
    print(f"  bos_token:          {repr(tokenizer.bos_token)} (id={tokenizer.bos_token_id})")
    print(f"  eos_token:          {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  pad_token:          {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  unk_token:          {repr(tokenizer.unk_token)} (id={tokenizer.unk_token_id})")
    print(f"  vocab_size:         {tokenizer.vocab_size}")

    # Decode each special token to see what it looks like
    print(f"\n[SPECIAL TOKEN DECODING]")
    for token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id]:
        if token_id is not None and token_id >= 0:
            decoded = tokenizer.decode([token_id])
            print(f"  Token {token_id:>5d}: {repr(decoded)}")


def debug_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
):
    """Run generation and inspect raw token IDs."""
    print("\n" + "=" * 60)
    print("GENERATION DEBUG")
    print("=" * 60)

    print(f"\n[INPUT PROMPT]")
    print(f"  {repr(prompt)}\n")

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]
    print(f"[INPUT TOKENS]")
    print(f"  Length: {input_length}")
    print(f"  Input token IDs (first 20): {inputs.input_ids[0, :20].tolist()}")

    # Fix missing img_context_token_id (required by InternVL2 custom generate())
    if not hasattr(model, "img_context_token_id") or model.img_context_token_id is None:
        # Extract from tokenizer (usually "<|image|>" or similar)
        if hasattr(tokenizer, "img_context_token"):
            img_context_token = tokenizer.img_context_token
        else:
            img_context_token = "<|image|>"
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
        model.img_context_token_id = img_context_token_id
        print(f"[FIXED] Set model.img_context_token_id = {img_context_token_id} (token: {repr(img_context_token)})")

    # Override generation config to allow enough tokens
    print(f"\n[GENERATION CONFIG OVERRIDE]")
    print(f"  Original max_length: {model.generation_config.max_length}")
    model.generation_config.max_length = None  # Disable max_length constraint
    model.generation_config.max_new_tokens = max_new_tokens
    print(f"  Set max_length: None")
    print(f"  Set max_new_tokens: {max_new_tokens}")

    # Generate with output_scores to inspect logits
    print(f"\n[GENERATING...]")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract generated token IDs (everything after input)
    generated_ids = outputs.sequences[0, input_length:]
    print(f"  Generated {len(generated_ids)} tokens")
    print(f"  Generated token IDs: {generated_ids.tolist()}")

    # Decode with and without special tokens
    full_decoded = tokenizer.decode(outputs.sequences[0])
    trimmed_decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    clean_decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    print(f"\n[DECODED OUTPUT]")
    print(f"  Full (as-is):             {repr(full_decoded)}")
    print(f"  With special tokens:      {repr(trimmed_decoded)}")
    print(f"  Skip special tokens:      {repr(clean_decoded)}")
    print(f"  Length (chars):           {len(clean_decoded)}")

    # Check if all generated tokens are special tokens
    if tokenizer.pad_token_id is not None:
        is_pad = generated_ids == tokenizer.pad_token_id
        print(f"\n[SPECIAL TOKEN BREAKDOWN]")
        print(f"  Pad tokens: {is_pad.sum().item()} / {len(generated_ids)}")

    if tokenizer.eos_token_id is not None:
        is_eos = generated_ids == tokenizer.eos_token_id
        print(f"  EOS tokens: {is_eos.sum().item()} / {len(generated_ids)}")

    # Inspect scores (logits at each step)
    if hasattr(outputs, "scores") and len(outputs.scores) > 0:
        print(f"\n[FIRST 3 GENERATION STEP LOGITS]")
        for step_idx in range(min(3, len(outputs.scores))):
            logits = outputs.scores[step_idx][0]  # [vocab_size]
            top_k_logits, top_k_indices = torch.topk(logits, k=5)
            print(f"  Step {step_idx}:")
            for rank, (idx, logit) in enumerate(zip(top_k_indices.tolist(), top_k_logits.tolist())):
                token_text = tokenizer.decode([idx])
                print(f"    {rank+1}. Token {idx:>5d}: logit={logit:>8.3f}  text={repr(token_text)}")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to native SFT checkpoint")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer (usually OpenGVLab/InternVL2-2B)")
    parser.add_argument("--torch_dtype", default="float16", help="Torch dtype (float16 or float32)")
    parser.add_argument("--attn_implementation", default="eager", help="Attention implementation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--prompt", default=None, help="Custom prompt (if not provided, uses default)")
    args = parser.parse_args()

    torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.tokenizer_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype,
    )

    # Inspect config
    inspect_generation_config(model, tokenizer)

    # Build default prompt if not provided
    if args.prompt is None:
        prompt = (
            "You are a grounded counting assistant. "
            "Respond using the exact Thought -> Scaffold -> Count format.\n\n"
            "<image>\n"
            "How many objects are in this image? "
        )
    else:
        prompt = args.prompt

    # Run debug generation
    outputs = debug_generation(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
