#!/usr/bin/env python3
"""
Stage 1.5 calibration SFT trainer for SCAFFOLD-Rex.

Key behaviors:
- Loads raw InternVL2-2B base in bf16.
- Reconstructs Stage-1 language weights via manual tensor copy (skeleton injection).
- Applies FP32 stability overrides to output embeddings + RMSNorm layers.
- Trains a fresh LoRA adapter on scaffolded JSON targets.
"""

import argparse
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import transformers
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file as safetensors_load_file
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer, Trainer


IGNORE_INDEX = -100
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
DEFAULT_IMAGE_TOKEN = "<image>"
SYSTEM_PROMPT = (
    "You are a counting assistant. The image contains a 6x6 anchor grid. "
    "Return strict JSON only with keys total_count, anchors_summary, and clusters."
)


@dataclass
class ScriptArgs:
    data_path: str
    output_dir: str
    base_model: str = "OpenGVLab/InternVL2-2B"
    stage1_checkpoint: str = "checkpoints/checkpoint-1140"
    attn_implementation: str = "eager"
    model_max_length: int = 16384
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "none"
    vision_scale: float = 1.0
    min_dynamic_patch: int = 2
    max_dynamic_patch: int = 12
    force_manual_tiling: int = 1
    debug_patch_shapes: int = 1
    expected_injected_tensors: int = 171
    min_injected_tensors: int = 120
    image_field: str = "image"
    pca_image_fields: str = "pca_image,image_pca,composite_image,dino_pca_image,image"
    use_pca_images: int = 0
    fail_on_missing_image: int = 1
    require_unique_output_dir: int = 1


def rank0_print(*args: Any) -> None:
    if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0:
        print(*args)


def load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        rows: List[dict] = []
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def parse_field_list(raw_fields: str) -> List[str]:
    return [field.strip() for field in raw_fields.split(",") if field.strip()]


def resolve_row_image_path(
    row: Dict[str, Any],
    data_dir: str,
    image_field: str,
    use_pca_images: bool,
    pca_image_fields: List[str],
) -> Optional[str]:
    fields: List[str] = []
    if use_pca_images:
        fields.extend(pca_image_fields)
        if image_field not in fields:
            fields.append(image_field)
    else:
        fields.append(image_field)

    for field in fields:
        value = row.get(field)
        if not value:
            continue
        path = str(value)
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(data_dir, path))
    return None


def resolve_stage1_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        return path

    parent = os.path.dirname(path)
    if not parent or not os.path.isdir(parent):
        raise FileNotFoundError(f"Stage-1 checkpoint path does not exist: {path}")

    candidates: List[tuple[int, str]] = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if not match:
            continue
        if os.path.exists(os.path.join(full, "model.safetensors")):
            candidates.append((int(match.group(1)), full))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint-* folders with model.safetensors found under {parent}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    resolved = candidates[0][1]
    rank0_print(f"Requested checkpoint missing: {path}. Auto-resolved to {resolved}")
    return resolved


def _resolve_submodule_attr(root: Any, dotted_path: str) -> tuple[Any, str]:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_stage1_language_tensors(model: torch.nn.Module, checkpoint_dir: str) -> int:
    full_weights = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(full_weights):
        raise FileNotFoundError(f"Missing Stage-1 checkpoint weights: {full_weights}")

    state = safetensors_load_file(full_weights, device="cpu")
    mappings = (
        ("language_model.base_model.model.model.", model.language_model.model),
        ("model.language_model.base_model.model.model.", model.language_model.model),
        ("language_model.base_model.model.output.", model.language_model.output),
        ("model.language_model.base_model.model.output.", model.language_model.output),
    )

    injected = 0
    with torch.no_grad():
        for key, value in state.items():
            if "lora_" in key:
                continue

            target_key = None
            target_root = None
            for prefix, root in mappings:
                if key.startswith(prefix):
                    target_key = key[len(prefix) :]
                    target_root = root
                    break
            if target_key is None or target_root is None:
                continue

            # LoRA-wrapped checkpoints store the frozen backbone under base_layer.
            # Strip that wrapper path so we can address the underlying InternLM2 blocks.
            target_key = target_key.replace(".base_layer.", ".")
            if target_key.endswith(".base_layer"):
                target_key = target_key[: -len(".base_layer")]

            try:
                parent, leaf = _resolve_submodule_attr(target_root, target_key)
                param_or_buf = getattr(parent, leaf)
                if isinstance(param_or_buf, torch.nn.Parameter):
                    param_or_buf.data.copy_(value.to(device=param_or_buf.device, dtype=param_or_buf.dtype))
                    injected += 1
                elif torch.is_tensor(param_or_buf):
                    param_or_buf.copy_(value.to(device=param_or_buf.device, dtype=param_or_buf.dtype))
                    injected += 1
            except Exception:
                continue

    return injected


def _wrap_fp32_forward(module: torch.nn.Module) -> None:
    if getattr(module, "_stage15_fp32_wrapped", False):
        return

    original_forward = module.forward

    def _forward_cast(x, *args, **kwargs):
        input_dtype = x.dtype if torch.is_tensor(x) else None
        if torch.is_tensor(x):
            x = x.float()
        out = original_forward(x, *args, **kwargs)
        if input_dtype is not None and torch.is_tensor(out):
            return out.to(input_dtype)
        return out

    module.forward = _forward_cast
    module._stage15_fp32_wrapped = True


def apply_stability_overrides(model: torch.nn.Module, vision_scale: float = 0.1) -> None:
    output_embeddings = None
    if hasattr(model.language_model, "get_output_embeddings"):
        output_embeddings = model.language_model.get_output_embeddings()
    if output_embeddings is None and hasattr(model.language_model, "output"):
        output_embeddings = model.language_model.output
    if output_embeddings is not None:
        output_embeddings.to(torch.float32)

    for _, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "rmsnorm" in class_name:
            module.to(torch.float32)
            _wrap_fp32_forward(module)

    if hasattr(model, "mlp1") and hasattr(model.mlp1, "forward"):
        if not getattr(model.mlp1, "_stage15_scaled", False):
            original_forward = model.mlp1.forward

            def _scaled_forward(*args, **kwargs):
                return original_forward(*args, **kwargs) * vision_scale

            model.mlp1.forward = _scaled_forward
            model.mlp1._stage15_scaled = True


def preprocess_multimodal(conversations: List[dict], num_image_token: int) -> List[dict]:
    image_placeholder = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}"
    out = copy.deepcopy(conversations)
    for sentence in out:
        if sentence["from"] == "human" and DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, image_placeholder).strip()
        elif sentence["from"] == "gpt" and DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
    return out


def _choose_tile_grid(width: int, height: int, min_tiles: int, max_tiles: int) -> tuple[int, int]:
    aspect = float(width) / float(max(1, height))
    best: Optional[tuple[float, int, int, int]] = None
    for tiles in range(max(1, min_tiles), max_tiles + 1):
        for rows in range(1, tiles + 1):
            if tiles % rows != 0:
                continue
            cols = tiles // rows
            grid_aspect = float(cols) / float(rows)
            aspect_err = abs(grid_aspect - aspect)
            score = (aspect_err, -tiles)
            if best is None or score < (best[0], best[1]):
                best = (aspect_err, -tiles, rows, cols)

    if best is None:
        return 1, 1
    return best[2], best[3]


def _dynamic_tile_image(image: Image.Image, min_dynamic_patch: int, max_dynamic_patch: int) -> List[Image.Image]:
    width, height = image.size
    rows, cols = _choose_tile_grid(width, height, min_dynamic_patch, max_dynamic_patch)
    target_w = cols * 448
    target_h = rows * 448
    resized = image.resize((target_w, target_h), resample=Image.BICUBIC)

    tiles: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * 448
            y1 = r * 448
            x2 = x1 + 448
            y2 = y1 + 448
            tiles.append(resized.crop((x1, y1, x2, y2)))
    return tiles


def preprocess_chat(conversations: List[dict], tokenizer) -> Dict[str, torch.Tensor]:
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = (
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

    source = list(conversations)
    if source and source[0].get("from") != "system":
        source = [{"from": "system", "value": SYSTEM_PROMPT}] + source

    sample_ids: List[int] = []
    sample_labels: List[int] = []
    for turn in source:
        role = roles.get(turn.get("from", "user"), turn.get("from", "user"))
        content = turn.get("value", "")
        encoded = tokenizer.apply_chat_template([{"role": role, "content": content}])
        sample_ids.extend(encoded)
        if role in ["user", "system"]:
            sample_labels.extend([IGNORE_INDEX] * len(encoded))
        else:
            sample_labels.extend(encoded)

    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and max_len > 0 and len(sample_ids) > max_len:
        sample_ids = sample_ids[:max_len]
        sample_labels = sample_labels[:max_len]

    return {
        "input_ids": torch.tensor(sample_ids, dtype=torch.long),
        "labels": torch.tensor(sample_labels, dtype=torch.long),
    }


class Stage15Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        image_processor,
        num_image_token: int,
        min_dynamic_patch: int,
        max_dynamic_patch: int,
        force_manual_tiling: bool,
        debug_patch_shapes: bool,
        image_field: str,
        pca_image_fields: List[str],
        use_pca_images: bool,
        fail_on_missing_image: bool,
    ):
        self.rows = load_json_or_jsonl(data_path)
        self.data_dir = os.path.dirname(os.path.abspath(data_path))
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = image_processor
        self.num_image_token = num_image_token
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.force_manual_tiling = force_manual_tiling
        self.debug_patch_shapes = debug_patch_shapes
        self.image_field = image_field
        self.pca_image_fields = pca_image_fields
        self.use_pca_images = use_pca_images
        self.fail_on_missing_image = fail_on_missing_image
        rank0_print(f"Loaded {len(self.rows)} records from {data_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = resolve_row_image_path(
            row=row,
            data_dir=self.data_dir,
            image_field=self.image_field,
            use_pca_images=self.use_pca_images,
            pca_image_fields=self.pca_image_fields,
        )
        if image_path:
            if self.fail_on_missing_image and not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image file for row index {idx}: {image_path}")
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                if self.fail_on_missing_image:
                    raise
                image = Image.new("RGB", (448, 448), (255, 255, 255))

            if self.force_manual_tiling:
                tiles = _dynamic_tile_image(
                    image,
                    min_dynamic_patch=self.min_dynamic_patch,
                    max_dynamic_patch=self.max_dynamic_patch,
                )
                pixel_values = self.image_processor.preprocess(tiles, return_tensors="pt")["pixel_values"]
            else:
                try:
                    processed = self.processor(images=[image], return_tensors="pt", max_dynamic_patch=self.max_dynamic_patch)
                    pixel_values = processed["pixel_values"]
                except Exception:
                    try:
                        pixel_values = self.image_processor.preprocess(
                            [image], return_tensors="pt", max_dynamic_patch=self.max_dynamic_patch
                        )["pixel_values"]
                    except TypeError:
                        pixel_values = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"]

            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            elif pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values[0]

            num_patches = int(pixel_values.shape[0])
            conv = preprocess_multimodal(row["conversations"], self.num_image_token * max(1, num_patches))
            image_flags = torch.ones((num_patches, 1), dtype=torch.long)
            if self.debug_patch_shapes and idx < 3:
                rank0_print(
                    f"[stage15 debug] idx={idx} image={os.path.basename(str(image_path))} "
                    f"pixel_values.shape={tuple(pixel_values.shape)} num_patches={num_patches}"
                )
        else:
            if self.fail_on_missing_image:
                raise ValueError(
                    f"No image path resolved for row index {idx}. "
                    f"Checked image_field='{self.image_field}' and pca fields={self.pca_image_fields}."
                )
            pixel_values = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
            conv = copy.deepcopy(row["conversations"])
            image_flags = torch.zeros((1, 1), dtype=torch.long)

        prep = preprocess_chat(conv, self.tokenizer)
        return {
            "input_ids": prep["input_ids"],
            "labels": prep["labels"],
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }


@dataclass
class Stage15Collator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: List[dict]) -> Dict[str, torch.Tensor]:
        max_len = self.tokenizer.model_max_length
        input_ids = [ins["input_ids"][:max_len] for ins in instances]
        labels = [ins["labels"][:max_len] for ins in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        pixel_values = torch.cat([ins["pixel_values"] for ins in instances], dim=0).to(torch.bfloat16)
        image_flags = torch.cat([ins["image_flags"] for ins in instances], dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }


def resolve_lora_targets(language_model: torch.nn.Module) -> List[str]:
    suffixes = {
        name.split(".")[-1]
        for name, module in language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    candidates = ["wqkv", "wo", "w1", "w2", "w3"]
    targets = [name for name in candidates if name in suffixes]
    if not targets:
        raise RuntimeError(
            "Could not resolve InternLM2 LoRA targets. "
            f"Linear suffix sample={sorted(list(suffixes))[:40]}"
        )
    return targets


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="OpenGVLab/InternVL2-2B")
    parser.add_argument("--stage1_checkpoint", default="checkpoints/checkpoint-1140")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--model_max_length", type=int, default=16384)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--vision_scale", type=float, default=1.0)
    parser.add_argument("--min_dynamic_patch", type=int, default=2)
    parser.add_argument("--max_dynamic_patch", type=int, default=12)
    parser.add_argument("--force_manual_tiling", type=int, default=1)
    parser.add_argument("--debug_patch_shapes", type=int, default=1)
    parser.add_argument("--expected_injected_tensors", type=int, default=171)
    parser.add_argument("--min_injected_tensors", type=int, default=120)
    parser.add_argument("--image_field", type=str, default="image")
    parser.add_argument(
        "--pca_image_fields",
        type=str,
        default="pca_image,image_pca,composite_image,dino_pca_image,image",
    )
    parser.add_argument("--use_pca_images", type=int, default=0)
    parser.add_argument("--fail_on_missing_image", type=int, default=1)
    parser.add_argument("--require_unique_output_dir", type=int, default=1)
    args = parser.parse_args()
    return ScriptArgs(**vars(args))


def main() -> None:
    args = parse_args()
    if bool(args.require_unique_output_dir):
        if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
            raise RuntimeError(
                f"Output directory already exists and is non-empty: {args.output_dir}. "
                "Use a fresh STAGE15_OUT_DIR to avoid merging checkpoints."
            )
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_dir = resolve_stage1_checkpoint(args.stage1_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right",
        model_max_length=args.model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    injected = inject_stage1_language_tensors(model, checkpoint_dir)
    rank0_print(f"Injected {injected} Stage-1 LM tensors from {checkpoint_dir}")
    if injected < args.min_injected_tensors:
        raise RuntimeError(
            f"Skeleton injection too small: injected={injected}, min_required={args.min_injected_tensors}"
        )
    if args.expected_injected_tensors > 0 and injected != args.expected_injected_tensors:
        rank0_print(
            "WARNING: injected tensor count differs from expected "
            f"({injected} vs {args.expected_injected_tensors})."
        )

    apply_stability_overrides(model, vision_scale=args.vision_scale)

    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.mlp1.parameters():
        param.requires_grad = False

    lora_targets = resolve_lora_targets(model.language_model)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)

    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()

    model.config.use_cache = False
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if model.img_context_token_id is None or model.img_context_token_id < 0:
        raise RuntimeError(f"Could not resolve token id for {IMG_CONTEXT_TOKEN}")

    num_image_token = int(getattr(model, "num_image_token", 256))
    rank0_print(f"num_image_token={num_image_token}, img_context_token_id={model.img_context_token_id}")
    rank0_print(
        f"Stage1.5 vision config: vision_scale={args.vision_scale}, "
        f"min_dynamic_patch={args.min_dynamic_patch}, max_dynamic_patch={args.max_dynamic_patch}, "
        f"force_manual_tiling={bool(args.force_manual_tiling)}"
    )
    rank0_print(
        f"Stage1.5 data config: use_pca_images={bool(args.use_pca_images)}, "
        f"image_field={args.image_field}, pca_image_fields={parse_field_list(args.pca_image_fields)}, "
        f"fail_on_missing_image={bool(args.fail_on_missing_image)}"
    )

    train_dataset = Stage15Dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_processor=image_processor,
        num_image_token=num_image_token,
        min_dynamic_patch=args.min_dynamic_patch,
        max_dynamic_patch=args.max_dynamic_patch,
        force_manual_tiling=bool(args.force_manual_tiling),
        debug_patch_shapes=bool(args.debug_patch_shapes),
        image_field=args.image_field,
        pca_image_fields=parse_field_list(args.pca_image_fields),
        use_pca_images=bool(args.use_pca_images),
        fail_on_missing_image=bool(args.fail_on_missing_image),
    )
    collator = Stage15Collator(tokenizer=tokenizer)

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        bf16=True,
        fp16=False,
        do_train=True,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
