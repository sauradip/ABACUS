#!/usr/bin/env python3
"""Extract LoRA adapter from a Trainer checkpoint's model.safetensors.

The Variant-B trainer saves the FULL model state at each --save_steps
(not a PEFT adapter dir). To eval intermediate checkpoints with the
existing eval_lora_counting_sft.py path, we re-emit a synthetic adapter
dir matching the format produced by `peft_model_ref.save_pretrained()`.

Mapping rules (verified against final adapter/):
  checkpoint key:  language_model.model.base_model.model.<...>.lora_{A,B}.default.weight
  adapter   key:                base_model.model.<...>.lora_{A,B}.weight

I.e. strip leading 'language_model.model.' and the '.default' segment.
The trained adapter at adapter/ contains no lm_head weights (modules_to_save
was not propagated by the manual save_pretrained call), so we omit it here too
for parity with the deployed eval pipeline.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "language_model.model."


def convert_key(k: str) -> str | None:
    if not k.startswith(PREFIX):
        return None
    if "lora" not in k:
        return None
    s = k[len(PREFIX):]
    return s.replace(".default.weight", ".weight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True,
                    help="Path to checkpoint-N/ containing model.safetensors")
    ap.add_argument("--reference_adapter", required=True,
                    help="Path to a known-good adapter/ dir (for adapter_config.json).")
    ap.add_argument("--out_dir", required=True,
                    help="Destination dir for synthetic adapter")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    ref  = Path(args.reference_adapter)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Copy adapter_config.json + README from reference
    for f in ("adapter_config.json", "README.md"):
        src = ref / f
        if src.exists():
            shutil.copy(src, out / f)

    # Extract LoRA tensors
    src_path = ckpt / "model.safetensors"
    out_state: dict = {}
    with safe_open(src_path, framework="pt") as f:
        for k in f.keys():
            new_k = convert_key(k)
            if new_k is None:
                continue
            out_state[new_k] = f.get_tensor(k)

    # Sanity check vs reference adapter key set
    with safe_open(ref / "adapter_model.safetensors", framework="pt") as f:
        ref_keys = set(f.keys())
    out_keys = set(out_state.keys())
    missing = ref_keys - out_keys
    extra   = out_keys - ref_keys
    print(f"  extracted {len(out_state)} tensors")
    print(f"  reference adapter has {len(ref_keys)} keys")
    if missing:
        print(f"  MISSING ({len(missing)}): {sorted(missing)[:3]}")
    if extra:
        print(f"  EXTRA   ({len(extra)}): {sorted(extra)[:3]}")
    assert not missing and not extra, "Key set mismatch with reference adapter"

    save_file(out_state, str(out / "adapter_model.safetensors"))
    print(f"  wrote {out / 'adapter_model.safetensors'}")


if __name__ == "__main__":
    main()
