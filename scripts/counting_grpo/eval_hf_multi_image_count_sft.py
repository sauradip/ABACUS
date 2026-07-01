#!/usr/bin/env python3
"""Generate and audit count-only outputs for HF multi-image UniLIP SFT data."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.grpo_reward_count_v3 import extract_total_count
from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    expand_unilip_image_context,
    load_json_or_jsonl,
    load_model,
    load_pil_images,
    load_processor,
    module_device,
    module_dtype,
    validate_messages,
)


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def bucket_name(gt: int) -> str:
    if gt < 30:
        return "low_lt30"
    if gt > 100:
        return "high_gt100"
    return "mid_30_100"


def assistant_gt(row: Dict[str, Any]) -> int:
    if row.get("gt_count") is not None:
        return int(row["gt_count"])
    messages = row.get("messages") or []
    if len(messages) < 2:
        raise ValueError(f"Row {row.get('id')} does not contain an assistant target")
    content = messages[1].get("content") or []
    text_parts = [
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    payload = json.loads("".join(text_parts))
    return int(payload["total_count"])


def prompt_inputs(
    processor: Any,
    row: Dict[str, Any],
    max_seq_length: int,
    device: torch.device,
    vision_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    user_messages = row["messages"][:1]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    text = expand_unilip_image_context(text)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
    )
    images = load_pil_images(user_messages)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Processor has no image_processor")
    encoded.update(image_processor.preprocess(images, return_tensors="pt"))

    out: Dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if not torch.is_tensor(value):
            continue
        if key == "pixel_values":
            out[key] = value.to(device=device, dtype=vision_dtype)
        else:
            out[key] = value.to(device=device)
    return out


def configure_generation(model: Any, tokenizer: Any, max_new_tokens: int) -> None:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()
    model.config.use_cache = False
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.use_cache = False
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if isinstance(img_ctx_id, int) and img_ctx_id >= 0:
        model.img_context_token_id = img_ctx_id


def decode_completion(tokenizer: Any, output_ids: torch.Tensor, prompt_len: int) -> str:
    ids = output_ids[0] if output_ids.ndim == 2 else output_ids
    full = tokenizer.decode(ids, skip_special_tokens=True).strip()
    trimmed = ""
    if ids.numel() > prompt_len:
        trimmed = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
    if trimmed and (extract_total_count(trimmed) is not None or len(trimmed) < len(full)):
        return trimmed
    return full


@torch.no_grad()
def generate_one(
    model: Any,
    processor: Any,
    row: Dict[str, Any],
    max_seq_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    device = module_device(model)
    model_module = model.module if hasattr(model, "module") else model
    vision_tower = getattr(model_module, "vision_tower", None)
    if vision_tower is None and hasattr(model_module, "get_model"):
        vision_tower = getattr(model_module.get_model(), "vision_tower", None)
    vision_dtype = module_dtype(vision_tower) if vision_tower is not None else torch.bfloat16
    inputs = prompt_inputs(processor, row, max_seq_length, device, vision_dtype)

    eos_ids: List[int] = []
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    kwargs: Dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs["pixel_values"],
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if eos_ids:
        kwargs["eos_token_id"] = sorted(set(eos_ids))
    if do_sample:
        kwargs["temperature"] = temperature

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output_ids = model.generate(**kwargs)
    return decode_completion(tokenizer, output_ids, int(inputs["input_ids"].shape[-1]))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[float]] = {"low_lt30": [], "mid_30_100": [], "high_gt100": []}
    missing = 0
    for row in rows:
        gt = int(row["gt_count"])
        pred = row.get("pred_count")
        if pred is None:
            missing += 1
            err = float(max(gt, 1))
        else:
            err = float(abs(gt - int(pred)))
        buckets[bucket_name(gt)].append(err)
    all_errors = [err for values in buckets.values() for err in values]
    summary: Dict[str, Any] = {
        "rows": len(rows),
        "missing_predictions": missing,
        "overall": {"n": len(all_errors), "mae": mean(all_errors)},
    }
    for name, values in buckets.items():
        summary[name] = {"n": len(values), "mae": mean(values)}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPL", "flash_attention_2"))
    parser.add_argument("--allow_attn_fallback", type=int, default=int(os.environ.get("ALLOW_ATTN_FALLBACK", "0")))
    parser.add_argument("--bf16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--trust_remote_code", type=int, default=1)
    parser.add_argument("--strict_images", type=int, default=1)
    parser.add_argument("--do_sample", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_transformers_compat_shims()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UniLIP generation")

    rows = load_json_or_jsonl(Path(args.data_path))
    rows = rows[args.start_index :]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows selected for eval")
    for row in rows:
        validate_messages(row, strict_images=bool(args.strict_images))

    processor = load_processor(args)
    model = load_model(args)
    tokenizer = getattr(processor, "tokenizer", processor)
    configure_generation(model, tokenizer, args.max_new_tokens)

    out_rows: List[Dict[str, Any]] = []
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            gt = assistant_gt(row)
            completion = generate_one(
                model=model,
                processor=processor,
                row=row,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
            )
            pred = extract_total_count(completion)
            record = {
                "id": row.get("id"),
                "completion": completion,
                "pred_count": pred,
                "gt_count": gt,
                "abs_error": abs(gt - pred) if pred is not None else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            out_rows.append(record)
            if idx <= 20 or idx == len(rows) or idx % 100 == 0:
                print(
                    f"[{idx}/{len(rows)}] id={record['id']} pred={pred} gt={gt} "
                    f"text={completion[:120]!r}",
                    flush=True,
                )

    tmp_path.replace(output_path)
    print(json.dumps({"summary": summarize(out_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
