"""
Evaluate counting MAE on FSC-147 test split.

Uses the CORRECT prompt format (image at END, matching SFT training).
Uses median MAE as primary metric (robust to outliers).
Caps individual MAE at 1000 to prevent mean inflation.
"""

import json
import os
import re
import argparse
import statistics
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROMPT_TEMPLATE = "How many {category} are present in this image? Answer with only a number."


def parse_count(response):
    """Extract integer count from model response."""
    response = response.strip()
    try:
        return int(response)
    except ValueError:
        match = re.search(r'\b(\d+)\b', response)
        if match:
            return int(match.group(1))
        return None


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def get_model_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if model_device is not None:
        return torch.device(model_device)

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vision_dtype(model, default_dtype: torch.dtype) -> torch.dtype:
    try:
        return model.get_model().vision_tower.dtype
    except Exception:
        return default_dtype


def load_model(checkpoint_path, tokenizer_path=None, torch_dtype_name="float16",
               load_4bit=False, load_8bit=False, attn_implementation="sdpa"):
    """Load UniLIP/InternVL checkpoint."""
    try:
        from unilip.model.language_model.unilip_internvl import (
            UniLIP_InternVLForCausalLM, UniLIP_InternVLConfig,
        )
        AutoConfig.register("unilip_internvl", UniLIP_InternVLConfig)
        AutoModel.register(UniLIP_InternVLConfig, UniLIP_InternVLForCausalLM)
    except Exception as e:
        print(f"[WARN] UniLIP registration skipped: {e}")

    tok_path = tokenizer_path if tokenizer_path else checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    if load_8bit:
        load_kwargs["load_in_8bit"] = True
    elif load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = parse_torch_dtype(torch_dtype_name)

    model = AutoModel.from_pretrained(checkpoint_path, **load_kwargs)
    model.eval()
    return model, tokenizer


def maybe_load_connector(model, connector_weights: str | None) -> bool:
    if not connector_weights:
        return False
    path = os.path.abspath(os.path.expanduser(connector_weights))
    if not os.path.exists(path):
        raise FileNotFoundError(f"connector weights not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.get_model().llm_connector.load_state_dict(state)
    return True


def preprocess_image(image_path, model, target_dtype: torch.dtype):
    """Build pixel_values tensor for InternVL."""
    from torchvision import transforms

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    device = get_model_device(model)
    pixel_values = transform(img).unsqueeze(0).to(device=device, dtype=target_dtype)
    return pixel_values


def run_inference(model, tokenizer, image_path, prompt, max_new_tokens=20,
                  target_dtype=torch.float16):
    """
    Run greedy generation for one sample.
    Image goes at END of prompt — matches SFT training format.
    Prompt is wrapped in the model's chat template (apply_chat_template) so
    the format matches training exactly.
    Do NOT pass image_flags to model.generate() (UniLIP returns only new tokens).
    """
    device = get_model_device(model)
    pixel_values = preprocess_image(image_path, model, target_dtype)
    num_image_token = model.num_image_token if hasattr(model, "num_image_token") else 256

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_tokens = IMG_CONTEXT_TOKEN * num_image_token
    img_tag = f"<img>{img_tokens}</img>"

    # Image at END — matches how SFT model was trained
    full_prompt = f"{prompt}\n{img_tag}"

    # Wrap in chat template to match training format exactly.
    # Without this, the model sees a raw prompt instead of the <|im_start|>user...<|im_end|>
    # format it was trained on, inflating MAE by ~4.6x.
    messages = [{"role": "user", "content": full_prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # UniLIP returns only new tokens — decode output_ids[0] directly
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def evaluate(model_path, tokenizer_path, fsc_root, split="test", num_samples=None,
             output_path=None, max_new_tokens=20, torch_dtype_name="float16",
             load_4bit=False, load_8bit=False, attn_implementation="sdpa",
             start_index=0, end_index=None, connector_weights=None):
    # Load metadata
    ann_path = os.path.join(fsc_root, "annotation_FSC147_384.json")
    cls_path = os.path.join(fsc_root, "ImageClasses_FSC147.txt")
    splits_path = os.path.join(fsc_root, "Train_Test_Val_FSC_147.json")
    img_dir = os.path.join(fsc_root, "images_384_VarV2")

    with open(ann_path) as f:
        annotations = json.load(f)

    image_classes = {}
    with open(cls_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                image_classes[parts[0].strip()] = parts[1].strip().lower()

    with open(splits_path) as f:
        splits = json.load(f)

    full_image_names = splits[split]
    split_total = len(full_image_names)
    if end_index is None:
        end_index = split_total
    image_names = full_image_names[start_index:end_index]
    if num_samples:
        image_names = image_names[:num_samples]

    model, tokenizer = load_model(
        model_path,
        tokenizer_path,
        torch_dtype_name=torch_dtype_name,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        attn_implementation=attn_implementation,
    )
    connector_loaded = maybe_load_connector(model, connector_weights)
    target_dtype = get_vision_dtype(model, parse_torch_dtype(torch_dtype_name))
    print(f"Connector loaded: {connector_loaded}")

    maes = []
    valid = 0
    buckets = {"7-20": [], "21-50": [], "51-100": [], "100+": []}
    results = []

    for i, img_name in enumerate(image_names):
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path) or img_name not in annotations:
            continue

        gt_count = len(annotations[img_name]["points"])
        category = image_classes.get(img_name, "objects")
        prompt = PROMPT_TEMPLATE.format(category=category)

        response = run_inference(
            model,
            tokenizer,
            img_path,
            prompt,
            max_new_tokens=max_new_tokens,
            target_dtype=target_dtype,
        )
        pred_count = parse_count(response)

        result = {
            "image": img_name,
            "category": category,
            "ground_truth": gt_count,
            "response": response,
            "prediction": pred_count,
        }

        if pred_count is not None:
            valid += 1
            mae = min(abs(pred_count - gt_count), 1000)  # cap outliers
            maes.append(mae)
            result["error"] = mae

            if gt_count <= 20:
                buckets["7-20"].append(mae)
            elif gt_count <= 50:
                buckets["21-50"].append(mae)
            elif gt_count <= 100:
                buckets["51-100"].append(mae)
            else:
                buckets["100+"].append(mae)
        else:
            result["error"] = None

        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(image_names)}] running... valid={valid}")

    mean_mae = sum(maes) / len(maes) if maes else float('inf')
    median_mae = statistics.median(maes) if maes else float('inf')

    print(f"\n=== Counting Evaluation ({len(maes)} valid / {len(image_names)} total) ===")
    print(f"  Mean MAE (capped):  {mean_mae:.2f}")
    print(f"  Median MAE:         {median_mae:.1f}")
    print(f"  Valid parse rate:   {valid / len(image_names) * 100:.1f}%")

    print("\n  Bucketed MAE:")
    for bucket, vals in buckets.items():
        if vals:
            print(f"    {bucket}: mean={sum(vals)/len(vals):.2f}, median={statistics.median(vals):.1f}, n={len(vals)}")

    summary = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path if tokenizer_path else model_path,
        "fsc_root": fsc_root,
        "split": split,
        "split_total": split_total,
        "start_index": start_index,
        "end_index": end_index,
        "num_requested": len(image_names),
        "num_valid": valid,
        "valid_parse_rate": valid / len(image_names) if image_names else 0.0,
        "mean_mae": mean_mae,
        "median_mae": median_mae,
        "torch_dtype": torch_dtype_name,
        "load_4bit": bool(load_4bit),
        "load_8bit": bool(load_8bit),
        "attn_implementation": attn_implementation,
        "max_new_tokens": max_new_tokens,
        "bucketed_mae": {
            bucket: {
                "mean": (sum(vals) / len(vals)) if vals else None,
                "median": (statistics.median(vals) if vals else None),
                "n": len(vals),
            }
            for bucket, vals in buckets.items()
        },
        "results": results,
        "connector_weights": connector_weights,
        "connector_loaded": connector_loaded,
    }

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="Tokenizer path if model_path lacks tokenizer files")
    parser.add_argument("--fsc_root", default="/projects/u6bl/myprojects/Datasets/FSC-147")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--connector_weights", default=None)
    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.tokenizer_path,
        args.fsc_root,
        args.split,
        args.num_samples,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        torch_dtype_name=args.torch_dtype,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        attn_implementation=args.attn_implementation,
        start_index=args.start_index,
        end_index=args.end_index,
        connector_weights=args.connector_weights,
    )
