#!/usr/bin/env python3
"""
Run UniLIP image-understanding counting on FSC-147 and populate `uniclip_count`.

It asks the model:
  "How many {class_name} are present in this image? Answer with only a number."

Then applies robust regex cleanup/parsing to extract a numeric count from generated text.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor

from unilip.constants import IMAGE_TOKEN_IDX
from unilip.mm_utils import tokenizer_image_token
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init


WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate FSC-147 JSON with UniLIP counts")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_ft_yolocount_final.json"),
        help="JSON file with an `entries` object to be augmented with uniclip_count.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output path. If omitted, input file is updated in place.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to UniLIP checkpoint directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/projects/u6bl/myprojects/Datasets/FSC-147/images_384_VarV2"),
        help="Directory for FSC-147 images keyed by entry key (e.g., 2.jpg).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means process all remaining entries")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Skip entries that already have uniclip_count")
    parser.add_argument("--store-raw", action="store_true", help="Store raw model output in uniclip_raw")
    return parser.parse_args()


def build_prompt(class_name: str) -> str:
    return (
        "<|im_start|>user\n"
        f"<image>\nHow many {class_name} are present in this image? "
        "Answer with only a number.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt_with_image_tokens(class_name: str, n_image_tokens: int) -> str:
    image_block = "<image>" * n_image_tokens
    return (
        "<|im_start|>user\n"
        f"{image_block}\nHow many {class_name} are present in this image? "
        "Answer with only a number.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_image_tensor(image: Image.Image, model) -> torch.Tensor:
    processor = None
    for source in (model.config.mllm_path, model.config.mllm_hf_path):
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
            break
        except Exception:
            continue

    image_processor = getattr(processor, "image_processor", processor)
    if image_processor is not None and hasattr(image_processor, "preprocess"):
        return image_processor.preprocess(image, return_tensors="pt")["pixel_values"]

    image_size = model.config.vision_config.image_size
    if isinstance(image_size, (list, tuple)):
        height, width = image_size
    else:
        height = width = int(image_size)

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def sanitize_text(text: str) -> str:
    # Remove control/special garbage while keeping numeric separators and words.
    cleaned = re.sub(r"[^\w\s,.:;\-+]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_numeric_count(text: str) -> Tuple[Optional[int], str]:
    cleaned = sanitize_text(text)
    lower = cleaned.lower()

    # 1) Word-number fallback for very short answers.
    if lower in WORD_TO_NUM:
        return WORD_TO_NUM[lower], cleaned

    # 2) Prefer patterns near explicit answer/count markers.
    marker_patterns = [
        r"\b(?:answer|count|number(?:\s+of)?|there\s+(?:are|is))\b[^\d\-+]{0,20}([\-+]?\d[\d,]*(?:\.\d+)?)",
        r"\b([\-+]?\d[\d,]*(?:\.\d+)?)\b\s*(?:objects?|items?|instances?|people|persons|cars|animals)?\b",
    ]
    for pat in marker_patterns:
        m = re.search(pat, lower, flags=re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                parsed = int(round(float(val)))
                return max(0, parsed), cleaned
            except ValueError:
                pass

    # 3) Last-resort: first standalone number in text.
    m = re.search(r"\b([\-+]?\d[\d,]*(?:\.\d+)?)\b", lower)
    if m:
        val = m.group(1).replace(",", "")
        try:
            parsed = int(round(float(val)))
            return max(0, parsed), cleaned
        except ValueError:
            pass

    # 4) Last-resort: any word-number token present in text.
    for token, number in WORD_TO_NUM.items():
        if re.search(rf"\b{re.escape(token)}\b", lower):
            return number, cleaned

    return None, cleaned


def ask_count(tokenizer, model, image_path: Path, class_name: str, max_new_tokens: int) -> str:
    prompt = build_prompt(class_name)
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = build_image_tensor(image, model).to(device=model.device, dtype=model.dtype)

    try:
        output_ids = model.generate(
            inputs=input_ids,
            images=pixel_values,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][input_ids.shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    except AttributeError as exc:
        if "prepare_inputs_labels_for_understanding" not in str(exc):
            raise

        vision_feature_layer = model.config.vision_feature_layer
        vision_feature_select_strategy = model.config.vision_feature_select_strategy
        image_embeds = model.model.get_image_features(
            pixel_values=pixel_values.type(model.model.vision_tower.dtype),
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=None,
        )

        n_image_tokens = image_embeds.shape[1]
        prompt = build_prompt_with_image_tokens(class_name, n_image_tokens)
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)

        text_embeds = model.get_model().language_model.embed_tokens(input_ids)
        image_token_mask = input_ids == IMAGE_TOKEN_IDX
        if int(image_token_mask.sum().item()) != n_image_tokens:
            raise RuntimeError("Image token count does not match image embedding length in fallback path.")

        text_embeds = text_embeds.clone()
        text_embeds[image_token_mask] = image_embeds.to(
            device=text_embeds.device, dtype=text_embeds.dtype
        ).flatten(0, 1)

        language_model = model.get_model().language_model
        attention_mask = torch.ones(
            (text_embeds.shape[0], text_embeds.shape[1]), device=text_embeds.device, dtype=torch.long
        )

        generated = []
        for _ in range(max_new_tokens):
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids[position_ids < 0] = 0
            outputs = language_model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask.bool(),
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            next_token_logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

            next_embed = language_model.embed_tokens(next_token)
            text_embeds = torch.cat([text_embeds, next_embed], dim=1)
            next_attention = torch.ones(
                (attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)

        generated_ids = torch.cat(generated, dim=1)[0] if generated else torch.empty((0,), device=text_embeds.device)
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def resolve_entries(payload: Dict) -> Dict:
    if "entries" in payload and isinstance(payload["entries"], dict):
        return payload["entries"]
    # Allow direct mapping JSON as fallback.
    return payload


def main() -> None:
    args = parse_args()
    disable_torch_init()

    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    with args.input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = resolve_entries(payload)
    all_items = sorted(entries.items(), key=lambda kv: int(Path(kv[0]).stem))

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    sliced = all_items[args.start_index :]
    if args.limit > 0:
        sliced = sliced[: args.limit]

    tokenizer, model, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM",
        os.path.expanduser(args.model_path),
    )

    processed = 0
    parsed_ok = 0
    missing_images = 0
    parse_failed = 0

    output_json = args.output_json if args.output_json is not None else args.input_json

    for image_key, row in sliced:
        if args.resume and "uniclip_count" in row:
            continue

        image_path = args.image_dir / image_key
        if not image_path.exists():
            missing_images += 1
            row["uniclip_count"] = None
            if args.store_raw:
                row["uniclip_raw"] = ""
            continue

        class_name = str(row.get("class_name", "objects")).strip() or "objects"
        raw_text = ask_count(tokenizer, model, image_path, class_name, args.max_new_tokens)
        count, cleaned = extract_numeric_count(raw_text)

        if count is None:
            parse_failed += 1
            row["uniclip_count"] = None
        else:
            parsed_ok += 1
            row["uniclip_count"] = int(count)

        if args.store_raw:
            row["uniclip_raw"] = cleaned

        processed += 1
        if processed % args.save_every == 0:
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(
                f"checkpoint save: processed={processed} parsed_ok={parsed_ok} "
                f"parse_failed={parse_failed} missing_images={missing_images}"
            , flush=True)

        if processed % args.log_every == 0:
            print(
                f"progress: processed={processed} parsed_ok={parsed_ok} "
                f"parse_failed={parse_failed} missing_images={missing_images}",
                flush=True,
            )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== UniLIP FSC-147 counting summary ===", flush=True)
    print(f"input_json: {args.input_json}", flush=True)
    print(f"output_json: {output_json}", flush=True)
    print(f"start_index: {args.start_index}", flush=True)
    print(f"limit: {args.limit}", flush=True)
    print(f"processed: {processed}", flush=True)
    print(f"parsed_ok: {parsed_ok}", flush=True)
    print(f"parse_failed: {parse_failed}", flush=True)
    print(f"missing_images: {missing_images}", flush=True)


if __name__ == "__main__":
    main()
