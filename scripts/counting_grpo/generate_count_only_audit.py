#!/usr/bin/env python3
"""Generate count-only predictions for Stage 3.2.1 MAE audit."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.grpo_reward_count_v3 import extract_total_count
from scripts.counting_grpo.train_internvl_grpo import (
    _load_model_and_tokenizer,
    _tokenize_prompt,
)


DEFAULT_DATASET = "outputs/scaffold_rex_5k_pca/grpo_full_count_only.jsonl"
DEFAULT_OUTPUT = "outputs/scaffold_rex_5k_pca/stage321_count_only_predictions.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_image_path(row: Dict[str, Any], data_dir: str) -> Optional[str]:
    value = row.get("pca_image") or row.get("image")
    if not value:
        return None
    path = str(value)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(data_dir, path))


def prompt_conversations(row: Dict[str, Any]) -> List[Dict[str, str]]:
    convs = row.get("conversations")
    if isinstance(convs, list) and convs:
        return convs
    prompt = row.get("problem") or row.get("instruction") or ""
    return [{"from": "human", "value": str(prompt)}]


def decode_new_tokens(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    ids = output_ids[0]
    if ids.numel() > prompt_len:
        ids = ids[prompt_len:]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return text


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    image_processor,
    num_image_token: int,
    row: Dict[str, Any],
    data_dir: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    image_path = resolve_image_path(row, data_dir)
    if image_path:
        image = Image.open(image_path).convert("RGB")
        pixel_values = image_processor.preprocess([image], return_tensors="pt")["pixel_values"]
        image_flags = torch.ones((1, 1), dtype=torch.long)
    else:
        pixel_values = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
        image_flags = torch.zeros((1, 1), dtype=torch.long)

    prompt_ids = _tokenize_prompt(prompt_conversations(row), tokenizer, num_image_token).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids)

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    image_flags = image_flags.to(device)

    generation_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    eos_ids: List[int] = []
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    if eos_ids:
        generation_kwargs["eos_token_id"] = sorted(set(eos_ids))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        try:
            output_ids = model.generate(**generation_kwargs)
        except TypeError:
            generation_kwargs.pop("image_flags", None)
            output_ids = model.generate(**generation_kwargs)
        except ValueError as exc:
            if "image_flags" not in str(exc):
                raise
            generation_kwargs.pop("image_flags", None)
            output_ids = model.generate(**generation_kwargs)
    return decode_new_tokens(tokenizer, output_ids, prompt_ids.shape[1])


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "low_lt30": [],
        "mid_30_100": [],
        "high_gt100": [],
    }
    missing = 0
    for row in rows:
        gt = int(row.get("gt_count", 0) or 0)
        pred = row.get("pred_count")
        if pred is None:
            missing += 1
            err = float(max(gt, 1))
        else:
            err = float(abs(gt - int(pred)))
        if gt < 30:
            buckets["low_lt30"].append(err)
        elif gt > 100:
            buckets["high_gt100"].append(err)
        else:
            buckets["mid_30_100"].append(err)
    all_errors = [x for vals in buckets.values() for x in vals]
    summary: Dict[str, Any] = {
        "rows": len(rows),
        "missing_predictions": missing,
        "overall_mae": mean(all_errors),
    }
    for name, values in buckets.items():
        summary[name] = {"n": len(values), "mae": mean(values)}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_jsonl", default=DEFAULT_DATASET)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--processor_name_or_path", default="OpenGVLab/InternVL2-2B")
    parser.add_argument("--attn_implementation", default=os.getenv("ATTN_IMPL", "eager"))
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--do_sample", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on a GPU node.")

    rows = load_jsonl(args.dataset_jsonl)
    data_dir = os.path.dirname(os.path.abspath(args.dataset_jsonl))
    if args.start_index:
        rows = rows[args.start_index :]
    if args.max_samples:
        rows = rows[: args.max_samples]

    model, tokenizer, image_processor, num_image_token = _load_model_and_tokenizer(
        args.model_path,
        processor_path=args.processor_name_or_path,
        attn_impl=args.attn_implementation,
    )
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = True
    if args.device.startswith("cuda"):
        model.to(torch.device(args.device))

    out_rows: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    tmp_output = args.output_jsonl + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                num_image_token=num_image_token,
                row=row,
                data_dir=data_dir,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
            )
            gt = int(row.get("gt_count", row.get("ground_truth_count", 0)) or 0)
            pred = extract_total_count(completion)
            record = {
                "id": row.get("id"),
                "image": row.get("image"),
                "pca_image": row.get("pca_image"),
                "prompt": row.get("problem", row.get("instruction", "")),
                "completion": completion,
                "prediction_text": completion,
                "pred_count": pred,
                "gt_count": gt,
                "abs_error": abs(gt - pred) if pred is not None else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            out_rows.append(record)
            if idx < 5:
                print(f"[{idx+1}/{len(rows)}] id={record['id']} pred={pred} gt={gt}")

    shutil.move(tmp_output, args.output_jsonl)
    print(json.dumps(summarize(out_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
