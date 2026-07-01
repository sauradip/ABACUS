#!/usr/bin/env python3
"""Evaluate dual-loss checkpoint on single GPU (simple, no accelerate)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)

NUMERIC = re.compile(r"\d+")

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate dual-loss model (single GPU)")
    ap.add_argument("--base_model", required=True, help="Base model directory")
    ap.add_argument("--checkpoint_dir", required=True, help="Model checkpoint directory")
    ap.add_argument("--mllm_hf", required=True, help="InternVL3 processor path")
    ap.add_argument("--val_json", required=True, help="Validation JSON")
    ap.add_argument("--out_json", required=True, help="Output JSON")
    ap.add_argument("--dataset_name", default="eval", help="Dataset name for output")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device: {device}")
    print(f"[Eval] base_model: {args.base_model}")
    print(f"[Eval] checkpoint: {args.checkpoint_dir}")

    # Load model with UniLIP loader
    apply_transformers_compat_shims()
    model_cls = load_unilip_class()

    model = model_cls.from_pretrained(
        args.checkpoint_dir,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(args.mllm_hf, trust_remote_code=True)
    tokenizer = processor.tokenizer
    img_proc = processor.image_processor

    # Load val data
    with open(args.val_json) as f:
        val_data = json.load(f)

    print(f"[Eval] evaluating {len(val_data)} samples\n")

    rows = []
    parse_failures = 0

    for item in tqdm(val_data, desc="evaluating"):
        convs  = {c["from"]: c["value"] for c in item["conversations"]}
        system = convs.get("system", "You are a helpful counting assistant. Answer with only a number.")
        human  = convs["human"]
        gt     = int(convs["gpt"])

        try:
            pil = Image.open(item["image"]).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Cannot open {item['image']}: {exc}")
            rows.append({
                "image": item["image"], "gt": gt, "pred": 0,
                "raw_output": "", "parse_ok": False,
            })
            parse_failures += 1
            continue

        pixel_values = img_proc.preprocess(
            [pil], return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.float16)

        # Build prompt
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        NUM_IMG_TOKENS = 256

        img_placeholder = (
            f"{IMG_START_TOKEN}"
            f"{IMG_CONTEXT_TOKEN * NUM_IMG_TOKENS}"
            f"{IMG_END_TOKEN}"
        )
        human_filled = human.replace("<image>", img_placeholder).strip()

        import copy
        tok = copy.deepcopy(tokenizer)
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
        tok.chat_template = CHAT_TEMPLATE

        input_ids = tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": human_filled},
            ],
            add_generation_prompt=True,
            return_dict=False,
        )
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

        rows.append({
            "image"      : item["image"],
            "gt"         : gt,
            "pred"       : pred,
            "raw_output" : text,
            "parse_ok"   : parse_ok,
        })

    # Compute metrics
    preds = np.array([r["pred"] for r in rows], dtype=float)
    gts   = np.array([r["gt"]   for r in rows], dtype=float)
    mae   = float(np.mean(np.abs(preds - gts)))
    rmse  = float(np.sqrt(np.mean((preds - gts) ** 2)))
    parse_rate = 1.0 - parse_failures / max(len(rows), 1)

    result = {
        "dataset"    : args.dataset_name,
        "n"          : len(rows),
        "MAE"        : mae,
        "RMSE"       : rmse,
        "parse_rate" : parse_rate,
        "rows"       : rows,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  {args.dataset_name:20s} | n={len(rows):4d} | MAE={mae:6.2f} | RMSE={rmse:6.2f} | Parse={parse_rate:5.1%}")
    print(f"{sep}")
    print(f"  Output: {args.out_json}\n")

if __name__ == "__main__":
    main()
