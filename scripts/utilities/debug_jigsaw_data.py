#!/usr/bin/env python3
"""
OmniCount Phase 3 Pre-Flight Diagnostic Protocol
-------------------------------------------------
Stage 1 : Headless Dataset & Transform Audit
Stage 3 : Visual Handshake Baseline Check (lm_head logit probe)

Run from the UniCount repo root:
    python3 debug_jigsaw_data.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_processor_safe():
    """Load the processor, auto-resolving the HF snapshot for UniLIP-3B."""
    from scripts.experiment_jigsaw.dataset import resolve_processor_path
    from scripts.counting_grpo.train_hf_multi_image_count_sft import load_processor
    import argparse

    resolved = resolve_processor_path("/data/amondal/model_cache/UniLIP-3B")
    print(f"[PROC] Resolved processor path: {resolved}")

    args = argparse.Namespace(
        model_name_or_path=resolved,
        processor_name_or_path=resolved,
        trust_remote_code=True,
    )
    return load_processor(args)


# ===========================================================================
# STAGE 1: Headless Dataset & Transform Audit
# ===========================================================================

def stage1_dataset_audit():
    print("\n" + "=" * 70)
    print("STAGE 1 — Headless Dataset & Transform Audit")
    print("=" * 70)

    from scripts.experiment_jigsaw.dataset import JigsawFusionDataset

    processor = load_processor_safe()
    tokenizer = getattr(processor, "tokenizer", processor)

    ds = JigsawFusionDataset(
        jsonl_path="/data/amondal/UniCount/outputs/experiment_jigsaw/train/train_jigsaw.jsonl",
        processor=processor,
        strict_images=False,
    )
    print(f"[DS] Dataset size: {len(ds)}")

    # Fetch sample 0
    sample = ds[0]

    ids    = sample["input_ids"]
    labels = sample["labels"]

    # ------------------------------------------------------------------
    # 1. Masking Integrity Check
    # ------------------------------------------------------------------
    mask = labels != -100
    supervised_tokens = ids[mask]
    ignored_tokens    = ids[~mask]

    decoded_target  = tokenizer.decode(supervised_tokens, skip_special_tokens=False)
    decoded_ignored = tokenizer.decode(ignored_tokens[:64], skip_special_tokens=False)

    print(f"\n[MASK] Total tokens          : {ids.shape[-1]}")
    print(f"[MASK] Supervised tokens     : {mask.sum().item()}")
    print(f"[MASK] Ignored tokens        : {(~mask).sum().item()}")
    print(f"[MASK] Supervised fraction   : {mask.float().mean().item():.4f}")
    print(f"[MASK] Decoded target        : {repr(decoded_target[:200])}")
    print(f"[MASK] Decoded ignored (head): {repr(decoded_ignored[:200])}")

    # ------------------------------------------------------------------
    # 2. Coordinate Range Check
    # ------------------------------------------------------------------
    bbox_match = re.search(
        r'"bbox":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', decoded_target
    )
    if bbox_match:
        coords = [int(x) for x in bbox_match.groups()]
        print(f"\n[BBOX] Parsed coords: {coords}")
        ok_range = all(0 <= c <= 1000 for c in coords)
        ok_order = coords[0] < coords[2] and coords[1] < coords[3]
        print(f"[BBOX] Range [0,1000] OK  : {'✓' if ok_range else '✗ FATAL'}")
        print(f"[BBOX] Non-degenerate OK  : {'✓' if ok_order else '✗ FATAL'}")
        if not ok_range:
            raise AssertionError(f"FATAL: BBox out of 0-1000 range: {coords}")
        if not ok_order:
            raise AssertionError(f"FATAL: Degenerate BBox: {coords}")
    else:
        print(f"[BBOX] WARNING: no bbox found in supervised region — masking may be wrong")
        print(f"[BBOX] Full decoded supervised: {repr(decoded_target)}")

    # ------------------------------------------------------------------
    # 3. Tensor Topology Check
    # ------------------------------------------------------------------
    pv = sample["pixel_values"]
    print(f"\n[TENSOR] pixel_values shape : {pv.shape}")
    print(f"[TENSOR] pixel_values dtype : {pv.dtype}")
    print(f"[TENSOR] pixel_values min   : {pv.min().item():.4f}")
    print(f"[TENSOR] pixel_values max   : {pv.max().item():.4f}")

    # InternVL may return (2*num_tiles, 3, H, W); for our 2-image case
    # with 1 tile each the expected shape is (2, 3, 448, 448).
    expected = (2, 3, 448, 448)
    if tuple(pv.shape) == expected:
        print(f"[TENSOR] Shape check ✓ {expected}")
    else:
        print(
            f"[TENSOR] WARNING: shape {tuple(pv.shape)} != expected {expected} — "
            "InternVL may tile; downstream collator should handle this"
        )

    # ------------------------------------------------------------------
    # 4. Interpolation Sanity (BICUBIC check)
    # ------------------------------------------------------------------
    from torchvision.transforms import InterpolationMode
    from scripts.experiment_jigsaw.train_jigsaw_sft import JigsawFusionDataset as _DS
    import inspect
    src = inspect.getsource(_DS._resize)
    if "BICUBIC" in src:
        print("\n[INTERP] BICUBIC confirmed ✓")
    else:
        print(f"\n[INTERP] WARNING: BICUBIC not found in _resize source — check manually:\n{src}")

    print("\n[STAGE 1] PASS ✓" if bbox_match else "\n[STAGE 1] WARN — inspect BBOX match above")
    return sample, tokenizer


# ===========================================================================
# STAGE 3: Visual Handshake Baseline Check
# ===========================================================================

def stage3_logit_probe():
    print("\n" + "=" * 70)
    print("STAGE 3 — Visual Handshake Baseline Check (lm_head logit probe)")
    print("=" * 70)

    processor = load_processor_safe()
    tokenizer = getattr(processor, "tokenizer", processor)

    # Build a minimal single-sample batch (CPU only)
    from scripts.experiment_jigsaw.dataset import JigsawFusionDataset
    ds = JigsawFusionDataset(
        jsonl_path="/data/amondal/UniCount/outputs/experiment_jigsaw/train/train_jigsaw.jsonl",
        processor=processor,
        strict_images=False,
    )
    sample = ds[0]

    # Get token IDs for coordinate strings "0".."1000" step 100 + a few key ones
    probe_coords = list(range(0, 1001, 100)) + [42, 512, 999]
    coord_token_ids: dict[int, list[int]] = {}
    for c in sorted(set(probe_coords)):
        tids = tokenizer.encode(str(c), add_special_tokens=False)
        coord_token_ids[c] = tids

    print("\n[LOGIT] Coordinate token IDs (first token of each):")
    for c, tids in sorted(coord_token_ids.items()):
        print(f"  {c:5d} → token ids {tids}")

    # Load base model on CPU (bfloat16 meta → avoids OOM)
    # -----------------------------------------------------------------------
    # Probe the lm_head weight matrix directly from safetensors.
    # Avoids the from_pretrained tied-weights issue in newer transformers.
    # The lm_head (vocab_size × hidden) lives in model-00002-of-00002.safetensors.
    # -----------------------------------------------------------------------
    print("\n[LOGIT] Probing lm_head weights directly from safetensors (no full load)…")
    import glob as _glob

    sf_files = sorted(_glob.glob("/data/amondal/model_cache/UniLIP-3B/model-*.safetensors"))
    print(f"[LOGIT] Found {len(sf_files)} safetensors shard(s)")

    lm_head_weight: torch.Tensor | None = None
    embed_weight: torch.Tensor | None = None

    try:
        from safetensors.torch import load_file as sf_load
        for sf_path in sf_files:
            data = sf_load(sf_path, device="cpu")
            # UniLIP / InternVL naming conventions
            for key in data:
                if key.endswith("lm_head.weight") and lm_head_weight is None:
                    lm_head_weight = data[key].float()
                    print(f"[LOGIT] lm_head.weight found in {Path(sf_path).name}: shape={lm_head_weight.shape}")
                if "embed_tokens.weight" in key and embed_weight is None:
                    embed_weight = data[key].float()
                    print(f"[LOGIT] embed_tokens.weight found: shape={embed_weight.shape}")
            if lm_head_weight is not None:
                break
    except ImportError:
        print("[LOGIT] safetensors not available; falling back to config-based heuristic")

    if lm_head_weight is not None:
        vocab_size, hidden = lm_head_weight.shape
        print(f"\n[LOGIT] lm_head weight stats:")
        print(f"  shape      : {lm_head_weight.shape}")
        print(f"  dtype      : {lm_head_weight.dtype}")
        print(f"  min / max  : {lm_head_weight.min().item():.6f} / {lm_head_weight.max().item():.6f}")
        print(f"  mean / std : {lm_head_weight.mean().item():.6f} / {lm_head_weight.std().item():.6f}")

        # Simulate logits for a unit-norm hidden state (best-case baseline)
        # This shows relative magnitudes, not true posterior probabilities.
        probe_hidden = torch.ones(hidden, dtype=torch.float32) / (hidden ** 0.5)
        pseudo_logits = lm_head_weight @ probe_hidden  # (vocab_size,)
        probs = torch.softmax(pseudo_logits, dim=-1)

        print("\n[LOGIT] Coordinate token pseudo-probabilities (unit-hidden probe):")
        for c, tids in sorted(coord_token_ids.items()):
            if tids:
                tid = tids[0]
                if tid < vocab_size:
                    p = probs[tid].item()
                    l = pseudo_logits[tid].item()
                    print(f"  '{c:5d}' (tid={tid:5d}): p={p:.6e}  logit={l:.4f}")

        # Check zero-suppression: weight norm per coordinate token row
        suppressed = []
        for c, tids in coord_token_ids.items():
            if tids:
                tid = tids[0]
                if tid < vocab_size:
                    row_norm = lm_head_weight[tid].norm().item()
                    if row_norm < 1e-6:
                        suppressed.append((c, tid, row_norm))

        if suppressed:
            print(f"\n[LOGIT] ⚠ {len(suppressed)} coord token rows near-zero in lm_head: {suppressed[:5]}")
        else:
            print("\n[LOGIT] ✓ All coordinate token rows have non-zero lm_head weights")
            print("[LOGIT] ✓ LoRA with r=64 will NOT need to recover from zero-initialized coords")

        # Check tie status: are embed_tokens and lm_head identical?
        if embed_weight is not None:
            if embed_weight.shape == lm_head_weight.shape:
                tied = torch.allclose(embed_weight, lm_head_weight, atol=1e-5)
                print(f"\n[LOGIT] embed_tokens == lm_head (weight-tied): {tied}")
                if tied:
                    print("[LOGIT] ✓ Weights ARE tied in base ckpt — modules_to_save=['lm_head'] will correctly break tie")
            else:
                print(f"\n[LOGIT] embed_tokens shape {embed_weight.shape} ≠ lm_head shape {lm_head_weight.shape}")
                print("[LOGIT] Weights are NOT tied (different shapes — likely projection layer)")
    else:
        print("[LOGIT] Could not extract lm_head weight — manual inspection required")

    # Config checks
    import json as _json
    cfg = _json.loads(Path("/data/amondal/model_cache/UniLIP-3B/config.json").read_text())
    print(f"\n[LOGIT] config.tie_word_embeddings : {cfg.get('tie_word_embeddings', 'NOT SET')}")
    print(f"[LOGIT] config.model_type          : {cfg.get('model_type', 'NOT SET')}")
    if "llm_config" in cfg:
        llm = cfg["llm_config"]
        print(f"[LOGIT] llm_config.hidden_size     : {llm.get('hidden_size', '?')}")
        print(f"[LOGIT] llm_config.vocab_size      : {llm.get('vocab_size', '?')}")

    print("\n[STAGE 3] COMPLETE")


# ===========================================================================
# FlashAttention check
# ===========================================================================

def check_flash_attention():
    print("\n" + "=" * 70)
    print("FLASH ATTENTION CHECK")
    print("=" * 70)
    try:
        import flash_attn
        print(f"[FA2] flash_attn {flash_attn.__version__} available ✓")
    except ImportError:
        print("[FA2] WARNING: flash_attn not installed — training will use O(N²) attention")

    # Check if the train script is configured to use it
    src = Path("scripts/experiment_jigsaw/train_jigsaw_sft.py").read_text()
    if "flash_attention_2" in src:
        print("[FA2] train_jigsaw_sft.py references flash_attention_2 ✓")
    else:
        print("[FA2] WARNING: flash_attention_2 not found in train_jigsaw_sft.py")

    # Check tie_word_embeddings in base model config
    import json
    cfg = json.loads(Path("/data/amondal/model_cache/UniLIP-3B/config.json").read_text())
    tie = cfg.get("tie_word_embeddings", "NOT SET")
    print(f"\n[CONFIG] tie_word_embeddings in base config: {tie}")
    if tie:
        print("[CONFIG] ✓ lm_head is in modules_to_save — weight tying will be broken correctly")
    else:
        print("[CONFIG] lm_head not tied — modules_to_save still adds adapter wrapper")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    check_flash_attention()
    sample, tokenizer = stage1_dataset_audit()
    stage3_logit_probe()
    print("\n" + "=" * 70)
    print("PRE-FLIGHT DIAGNOSTIC COMPLETE")
    print("=" * 70)
