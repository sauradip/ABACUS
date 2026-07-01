#!/usr/bin/env python3
"""
Generate Stage 1.5 self-audit for SCAFFOLD-Rex RankDPO.

Outputs JSON payload with `rows` entries compatible with prepare_rankdpo_data.py.
Each row contains at minimum:
  - image (id key used for matching)
  - prompt
  - prediction_text
  - pred_count
  - gt_count

Typical usage:
  python scripts/counting_grpo/generate_scaffold_rex_audit_v2.py \
    --model_path checkpoints/scaffold_rex_stage15_4342903 \
    --scaffold_jsonl outputs/scaffold_rex_5k/all.jsonl \
    --output_json checkpoints/scaffold_rex_stage15_4342903/scaffold_rex_audit_v2.json

One-image sanity check (before full run):
  python scripts/counting_grpo/generate_scaffold_rex_audit_v2.py \
    --model_path checkpoints/scaffold_rex_stage15_4342903 \
    --scaffold_jsonl outputs/scaffold_rex_5k/all.jsonl \
    --output_json /tmp/audit_sanity.json \
    --max_samples 1 --print_samples 1
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
DEFAULT_IMAGE_TOKEN = "<image>"

SYSTEM_PROMPT = (
    "You are a grounded counting assistant. "
    "The image has a 6x6 anchor matrix where x increases top-to-bottom and y increases left-to-right. "
    "Return strict JSON only with keys total_count, anchors_summary, and clusters."
)

SCAFFOLD_PREFIX = (
    "The image is overlaid with a 6x6 dot matrix. Dots are labeled (x,y). "
    "Within columns, x increases top-to-bottom. Within rows, y increases left-to-right. "
    "Identify objects near their nearest coordinates and return strict JSON only."
)


def grid_anchor_pixels(width: int, height: int, grid_size: int = 6) -> Dict[tuple[int, int], tuple[float, float]]:
    anchors: Dict[tuple[int, int], tuple[float, float]] = {}
    for row in range(1, grid_size + 1):
        py = (row / float(grid_size + 1)) * float(height)
        for col in range(1, grid_size + 1):
            px = (col / float(grid_size + 1)) * float(width)
            anchors[(row, col)] = (px, py)
    return anchors


def local_luminance(gray_img: Image.Image, x_center: int, y_center: int, radius: int = 6) -> float:
    x1 = max(0, x_center - radius)
    y1 = max(0, y_center - radius)
    x2 = min(gray_img.width, x_center + radius + 1)
    y2 = min(gray_img.height, y_center + radius + 1)
    patch = gray_img.crop((x1, y1, x2, y2))
    hist = patch.histogram()
    total = sum(hist)
    if total <= 0:
        return 255.0
    weighted = sum(i * c for i, c in enumerate(hist))
    return float(weighted) / float(total)


def draw_scaffold_overlay(image: Image.Image, grid_size: int = 6, dot_radius: int = 4) -> Image.Image:
    overlaid = image.copy().convert("RGB")
    gray = overlaid.convert("L")
    draw = ImageDraw.Draw(overlaid)
    anchors = grid_anchor_pixels(overlaid.width, overlaid.height, grid_size=grid_size)

    for (_, _), (px, py) in anchors.items():
        xi = int(round(px))
        yi = int(round(py))
        lum = local_luminance(gray, xi, yi)
        color = (255, 255, 255) if lum < 128.0 else (0, 0, 0)
        draw.ellipse(
            [xi - dot_radius, yi - dot_radius, xi + dot_radius, yi + dot_radius],
            fill=color,
            outline=(128, 128, 128),
            width=1,
        )
    return overlaid


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


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_chat_prompt(problem: str, image_context_token_count: int) -> str:
    image_placeholder = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * image_context_token_count}{IMG_END_TOKEN}"
    user_text = problem.replace(DEFAULT_IMAGE_TOKEN, image_placeholder)
    if SCAFFOLD_PREFIX.lower() not in user_text.lower():
        user_text = f"{SCAFFOLD_PREFIX}\n{user_text}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{{"
    )


def robust_parse(text: str) -> Optional[dict]:
    candidate = text.strip()

    # Greedy extraction: from first '{' to last '}' to tolerate prose drift
    # around the JSON body.
    match = re.search(r"(\{.*\})", candidate, re.DOTALL)
    if not match:
        # Prefilled assistant mode may start right after '{'. Reconstruct it.
        if '"total_count"' in candidate:
            candidate = "{" + candidate
            match = re.search(r"(\{.*\})", candidate, re.DOTALL)
        if not match:
            return None

    clean_json = match.group(1)
    try:
        payload = json.loads(clean_json)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        # Common recovery for truncated cluster lists.
        try:
            payload = json.loads(clean_json + "]}")
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None


def _wrap_fp32_forward(module: torch.nn.Module) -> None:
    """Wrap forward pass to cast through float32 for numerical stability."""
    if getattr(module, "_stage15_fp32_wrapped", False):
        return
    original_forward = module.forward

    def _forward_cast(x: Any, *args: Any, **kwargs: Any) -> Any:
        input_dtype = x.dtype if torch.is_tensor(x) else None
        if torch.is_tensor(x):
            x = x.float()
        out = original_forward(x, *args, **kwargs)
        if input_dtype is not None and torch.is_tensor(out):
            return out.to(input_dtype)
        return out

    module.forward = _forward_cast
    module._stage15_fp32_wrapped = True


def apply_stability_overrides(model: torch.nn.Module, vision_scale: float = 1.0) -> None:
    """Apply Stage 1.5 stability overrides: FP32 casting + vision scaling."""
    output_embeddings = None
    if hasattr(model.language_model, "get_output_embeddings"):
        output_embeddings = model.language_model.get_output_embeddings()
    if output_embeddings is None and hasattr(model.language_model, "output"):
        output_embeddings = model.language_model.output
    # NOTE: do NOT cast output_embeddings to float32 here.
    # The RMSNorm wrapper restores bfloat16 on output, so the LM head
    # must also stay bfloat16 at inference (unlike training which has autocast).

    for _, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "rmsnorm" in class_name:
            module.to(torch.float32)
            _wrap_fp32_forward(module)

    if hasattr(model, "mlp1") and hasattr(model.mlp1, "forward"):
        if not getattr(model.mlp1, "_stage15_scaled", False):
            original_forward = model.mlp1.forward

            def _scaled_forward(*args: Any, **kwargs: Any) -> Any:
                return original_forward(*args, **kwargs) * vision_scale

            model.mlp1.forward = _scaled_forward
            model.mlp1._stage15_scaled = True
    
    print(f"[info] Applied stability overrides: vision_scale={vision_scale}")


def parse_gt_count(record: dict) -> int:
    if isinstance(record.get("ground_truth_count"), int):
        return int(record["ground_truth_count"])
    if isinstance(record.get("count"), int):
        return int(record["count"])

    solution = str(record.get("solution", ""))
    match = re.search(r'"total_count"\s*:\s*(\d+)', solution)
    if match:
        return int(match.group(1))
    return 0


def resolve_base_model_id(model_path: str) -> Optional[str]:
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except Exception:
        return None

    auto_map = cfg.get("auto_map")
    if isinstance(auto_map, dict):
        for _, value in auto_map.items():
            if isinstance(value, str) and "--" in value:
                candidate = value.split("--", 1)[0].strip()
                if candidate:
                    return candidate

    name_or_path = cfg.get("_name_or_path")
    if isinstance(name_or_path, str) and name_or_path.strip():
        return name_or_path.strip()

    return None


def _resolve_submodule_attr(root: Any, dotted_path: str) -> tuple[Any, str]:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _resolve_checkpoint_dir_with_weights(checkpoint_dir: str) -> str:
    direct_has_weights = bool(glob.glob(os.path.join(checkpoint_dir, "model*.safetensors")))
    if direct_has_weights:
        return checkpoint_dir

    candidates: List[tuple[int, str]] = []
    for path in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if not match:
            continue
        has_weights = bool(glob.glob(os.path.join(path, "model*.safetensors")))
        if has_weights:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return checkpoint_dir

    candidates.sort(key=lambda x: x[0], reverse=True)
    resolved = candidates[0][1]
    print(f"[info] Auto-resolved Stage-1 weights dir: {resolved}")
    return resolved


def _list_model_safetensor_files(checkpoint_dir: str) -> List[str]:
    single = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    shard_glob = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
    if shard_glob:
        return shard_glob

    fallback = sorted(glob.glob(os.path.join(checkpoint_dir, "model*.safetensors")))
    fallback = [p for p in fallback if os.path.basename(p) != "adapter_model.safetensors"]
    return fallback


def inject_stage1_backbone_from_safetensors(model: torch.nn.Module, checkpoint_dir: str) -> int:
    resolved_dir = _resolve_checkpoint_dir_with_weights(checkpoint_dir)
    weight_files = _list_model_safetensor_files(resolved_dir)
    if not weight_files:
        raise FileNotFoundError(
            "Expected Stage-1 checkpoint safetensors under "
            f"{checkpoint_dir} (resolved={resolved_dir}), but found none."
        )

    mappings = (
        ("language_model.base_model.model.model.", model.language_model.model),
        ("model.language_model.base_model.model.model.", model.language_model.model),
        ("language_model.base_model.model.output.", model.language_model.output),
        ("model.language_model.base_model.model.output.", model.language_model.output),
    )

    injected = 0
    skipped = 0
    with torch.no_grad():
        for file_path in weight_files:
            state = safetensors_load_file(file_path, device="cpu")
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

                target_key = target_key.replace(".base_layer.", ".")
                if target_key.endswith(".base_layer"):
                    target_key = target_key[: -len(".base_layer")]

                try:
                    parent, leaf = _resolve_submodule_attr(target_root, target_key)
                    param_or_buf = getattr(parent, leaf)
                    if isinstance(param_or_buf, torch.nn.Parameter):
                        param_or_buf.data.copy_(value.to(device=param_or_buf.device, dtype=param_or_buf.dtype))
                    elif torch.is_tensor(param_or_buf):
                        param_or_buf.copy_(value.to(device=param_or_buf.device, dtype=param_or_buf.dtype))
                    else:
                        skipped += 1
                        continue
                    injected += 1
                except Exception:
                    skipped += 1

    print(
        f"Injection Complete: Reconstructed {injected} Stage-1 backbone tensors "
        f"from {len(weight_files)} file(s) (skipped={skipped})."
    )
    if injected == 0:
        raise RuntimeError(
            "Manual Stage-1 backbone injection found zero matching tensors. "
            "Check Stage-1 checkpoint format and key prefixes."
        )
    return injected


def merge_lora_deltas_from_safetensors(
    model: torch.nn.Module,
    checkpoint_dir: str,
    lora_rank: int = 64,
    lora_alpha: float = 128.0,
) -> int:
    """Read lora_A / lora_B pairs from the checkpoint safetensors and add the
    LoRA delta (lora_B @ lora_A * scaling) directly to the corresponding base
    weight in *model*.  Returns the number of LoRA layers merged."""
    resolved_dir = _resolve_checkpoint_dir_with_weights(checkpoint_dir)
    weight_files = _list_model_safetensor_files(resolved_dir)
    if not weight_files:
        return 0

    scaling = lora_alpha / lora_rank

    # Collect all tensors keyed by their full checkpoint name.
    full_state: Dict[str, Any] = {}
    for file_path in weight_files:
        full_state.update(safetensors_load_file(file_path, device="cpu"))

    # Group lora_A and lora_B pairs by their "stem" (everything before .lora_A / .lora_B).
    # Expected key pattern:
    #   language_model.base_model.model.model.<rest>.lora_A.default.weight
    #   language_model.base_model.model.model.<rest>.lora_B.default.weight
    lora_A_map: Dict[str, Any] = {}
    lora_B_map: Dict[str, Any] = {}
    for key, val in full_state.items():
        if ".lora_A." in key:
            # stem = everything before ".lora_A.<adapter>.weight"
            stem = key[: key.index(".lora_A.")]
            lora_A_map[stem] = val
        elif ".lora_B." in key:
            stem = key[: key.index(".lora_B.")]
            lora_B_map[stem] = val

    # Base-weight key prefix → model sub-module root (same as injection function)
    prefixes = (
        ("language_model.base_model.model.model.", model.language_model.model),
        ("model.language_model.base_model.model.model.", model.language_model.model),
        ("language_model.base_model.model.output.", model.language_model.output),
        ("model.language_model.base_model.model.output.", model.language_model.output),
    )

    merged = 0
    with torch.no_grad():
        for stem, lora_A in lora_A_map.items():
            if stem not in lora_B_map:
                continue
            lora_B = lora_B_map[stem]

            # Compute delta = lora_B @ lora_A  (both are 2-D weight matrices)
            delta = (lora_B.float() @ lora_A.float()) * scaling

            # Find which prefix the stem belongs to.
            target_key = None
            target_root = None
            for prefix, root in prefixes:
                if stem.startswith(prefix):
                    target_key = stem[len(prefix):]
                    target_root = root
                    break
            if target_key is None or target_root is None:
                continue

            # Strip ".base_layer" suffix if present (PEFT stores base weight there).
            target_key = target_key.replace(".base_layer", "")
            # The stem points to the Linear *module*; the actual parameter is .weight
            weight_key = target_key + ".weight"

            try:
                parent, leaf = _resolve_submodule_attr(target_root, weight_key)
                param = getattr(parent, leaf)
                if isinstance(param, torch.nn.Parameter):
                    param.data.add_(delta.to(device=param.device, dtype=param.dtype))
                elif torch.is_tensor(param):
                    param.add_(delta.to(device=param.device, dtype=param.dtype))
                else:
                    continue
                merged += 1
            except Exception:
                continue

    print(f"LoRA Merge Complete: applied {merged} LoRA delta(s) (scaling={scaling:.3f}).")
    return merged


def payload_to_points(payload: Optional[dict]) -> List[List[int]]:
    if not isinstance(payload, dict):
        return []

    points: List[List[int]] = []
    clusters = payload.get("clusters", [])
    if isinstance(clusters, list):
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            anchor = cluster.get("anchor")
            if not (isinstance(anchor, list) and len(anchor) == 2):
                continue
            try:
                x_val = int(anchor[0])
                y_val = int(anchor[1])
            except Exception:
                continue
            count = cluster.get("count", 1)
            try:
                n = max(1, int(count))
            except Exception:
                n = 1
            for _ in range(n):
                points.append([x_val, y_val])
    return points


def run_generate(
    model: torch.nn.Module,
    tokenizer,
    processor,
    image_processor,
    image: Image.Image,
    problem: str,
    num_image_token: int,
    max_new_tokens: int,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    force_manual_tiling: bool,
    debug_pixel_shape: bool,
) -> str:
    if force_manual_tiling:
        tile_images = _dynamic_tile_image(image, min_dynamic_patch=min_dynamic_patch, max_dynamic_patch=max_dynamic_patch)
        pixel_values = image_processor.preprocess(tile_images, return_tensors="pt")["pixel_values"]
    else:
        try:
            processed = processor(images=[image], return_tensors="pt", max_dynamic_patch=max_dynamic_patch)
            pixel_values = processed["pixel_values"]
        except Exception:
            try:
                pixel_values = image_processor.preprocess(
                    [image], return_tensors="pt", max_dynamic_patch=max_dynamic_patch
                )["pixel_values"]
            except TypeError:
                pixel_values = image_processor.preprocess([image], return_tensors="pt")["pixel_values"]

    if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values[0]
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)

    num_patches = int(pixel_values.shape[0])
    prompt = build_chat_prompt(problem, num_image_token * max(1, num_patches))

    if debug_pixel_shape:
        print(
            f"[debug] pixel_values.shape={tuple(pixel_values.shape)}, "
            f"num_patches={num_patches}, min_dynamic_patch={min_dynamic_patch}, "
            f"max_dynamic_patch={max_dynamic_patch}, force_manual_tiling={force_manual_tiling}"
        )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

    generation_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    eos_ids: List[int] = []
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    if eos_ids:
        generation_kwargs["eos_token_id"] = sorted(set(eos_ids))

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(**generation_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    assistant_split = decoded.split("<|im_start|>assistant\n")
    assistant_text = assistant_split[-1] if assistant_split else decoded
    assistant_text = assistant_text.split("<|im_end|>")[0].strip()

    # We prime the assistant with "{" in the prompt. Some decodes omit that
    # leading token in the returned text, so reconstruct JSON boundaries.
    if not assistant_text.startswith("{") and '"total_count"' in assistant_text:
        assistant_text = "{" + assistant_text
    if assistant_text.startswith("{") and not assistant_text.endswith("}"):
        assistant_text = assistant_text + "}"
    return assistant_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scaffold_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--print_samples", type=int, default=3)
    parser.add_argument("--min_dynamic_patch", type=int, default=2)
    parser.add_argument("--max_dynamic_patch", type=int, default=12)
    parser.add_argument("--force_manual_tiling", type=int, default=1)
    parser.add_argument("--force_scaffold_overlay", type=int, default=1)
    parser.add_argument("--dot_radius", type=int, default=4)
    parser.add_argument("--debug_pixel_shape", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vision_scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on a GPU node.")

    rows = load_jsonl(args.scaffold_jsonl)
    if args.start_index > 0:
        rows = rows[args.start_index :]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right",
        model_max_length=16384,
    )
    base_model_id = resolve_base_model_id(args.model_path)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as exc:
        print(f"[warn] AutoProcessor load from checkpoint failed: {exc}")
        if base_model_id:
            print(f"[info] Falling back AutoProcessor to base model: {base_model_id}")
            processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        else:
            raise

    image_processor = getattr(processor, "image_processor", None) if processor is not None else None
    if image_processor is None:
        try:
            image_processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        except Exception as exc:
            if not base_model_id:
                raise
            print(f"[warn] AutoImageProcessor load from checkpoint failed: {exc}")
            print(f"[info] Falling back AutoImageProcessor to base model: {base_model_id}")
            image_processor = AutoImageProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    model_load_path = base_model_id or args.model_path
    print(f"[info] Model init path: {model_load_path}")
    model = AutoModel.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    # Materialize Stage-1 language backbone tensors into base model to avoid
    # prefix-mismatch silent failures (zombie model with random LM weights).
    injected = inject_stage1_backbone_from_safetensors(model, args.model_path)
    if injected < 120:
        raise RuntimeError(f"Audit Loading Failed: only {injected} tensors injected from Stage-1 checkpoint")

    # Apply Stage-1.5 LoRA deltas (lora_B @ lora_A * scaling) on top of backbone.
    # The checkpoint stores PEFT weights unmerged; we merge in-place here.
    lora_merged = merge_lora_deltas_from_safetensors(
        model, args.model_path, lora_rank=64, lora_alpha=128.0
    )
    if lora_merged == 0:
        print("[warn] No LoRA deltas found in checkpoint — running without Stage-1.5 LoRA.")

    # Apply stability overrides (FP32 casting + vision scaling) to match training.
    apply_stability_overrides(model, vision_scale=args.vision_scale)
    
    model.eval()
    model.config.use_cache = False

    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()

    if args.device.startswith("cuda"):
        model.to(torch.device(args.device))

    num_image_token = int(getattr(model, "num_image_token", 256))
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print("[info] Vision scaling: using default model projector scale (1.0)")

    out_rows: List[dict] = []
    parse_ok = 0
    zero_count = 0

    for idx, record in enumerate(rows):
        image_path = str(record["image"])
        source_image_path = str(record.get("source_image", image_path))
        image_id = str(record.get("id", os.path.basename(image_path)))
        problem = str(record["problem"])

        load_path = source_image_path if bool(args.force_scaffold_overlay) else image_path
        image = Image.open(load_path).convert("RGB")
        if bool(args.force_scaffold_overlay):
            image = draw_scaffold_overlay(image, grid_size=6, dot_radius=args.dot_radius)

        prediction_text = run_generate(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            image=image,
            problem=problem,
            num_image_token=num_image_token,
            max_new_tokens=args.max_new_tokens,
            min_dynamic_patch=args.min_dynamic_patch,
            max_dynamic_patch=args.max_dynamic_patch,
            force_manual_tiling=bool(args.force_manual_tiling),
            debug_pixel_shape=(bool(args.debug_pixel_shape) and idx < args.print_samples),
        )

        payload = robust_parse(prediction_text)
        pred_count = 0
        if isinstance(payload, dict) and isinstance(payload.get("total_count"), int):
            pred_count = int(payload["total_count"])
            parse_ok += 1
        else:
            match = re.search(r'"total_count"\s*:\s*(\d+)', prediction_text)
            if match:
                pred_count = int(match.group(1))

        if pred_count == 0:
            zero_count += 1

        out_rows.append(
            {
                "image": image_id,
                "image_path": image_path,
                "prompt": str(record["problem"]),
                "prediction_text": prediction_text,
                "pred_points": payload_to_points(payload),
                "pred_count": pred_count,
                "gt_count": parse_gt_count(record),
            }
        )

        if idx < args.print_samples:
            print(
                f"[{idx+1}/{len(rows)}] {image_id} pred={pred_count} gt={parse_gt_count(record)} "
                f"parsed_json={isinstance(payload, dict)}"
            )

    summary = {
        "model_path": args.model_path,
        "scaffold_jsonl": args.scaffold_jsonl,
        "num_rows": len(out_rows),
        "json_parse_success": parse_ok,
        "json_parse_success_rate": (100.0 * parse_ok / max(1, len(out_rows))),
        "zero_count_rows": zero_count,
        "zero_count_rate": (100.0 * zero_count / max(1, len(out_rows))),
        "rows": out_rows,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Scaffold-Rex Self-Audit Complete ===")
    print(f"Rows: {len(out_rows)}")
    print(f"JSON parse success: {parse_ok}/{len(out_rows)} ({summary['json_parse_success_rate']:.2f}%)")
    print(f"Zero-count rows: {zero_count}/{len(out_rows)} ({summary['zero_count_rate']:.2f}%)")
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
