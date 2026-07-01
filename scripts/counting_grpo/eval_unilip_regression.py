#!/usr/bin/env python3
"""Evaluate UniLIP CE/MSE/Hybrid checkpoints with dual metrics.

Reports:
1) token-decoded count MAE from generated completion text
2) regression-head count MAE from scalar prediction head (if present)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.grpo_reward_count_v3 import extract_total_count
from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    embed_tokens,
    expand_unilip_image_context,
    load_json_or_jsonl,
    load_model,
    load_pil_images,
    load_processor,
    load_unilip_constants,
    module_device,
    module_dtype,
    validate_messages,
)


def load_fsc147_annotation_mapping(annotation_json: Optional[str]) -> Dict[str, int]:
    if not annotation_json:
        return {}
    path = Path(annotation_json)
    if not path.exists():
        raise FileNotFoundError(f"Missing FSC147 annotations: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in FSC147 annotation file: {path}")
    out: Dict[str, int] = {}
    for image_name, ann in payload.items():
        key = str(image_name).split(".")[0]
        if isinstance(ann, dict):
            points = ann.get("points")
            if not isinstance(points, list):
                raise ValueError(f"Annotation for {image_name} is missing list field 'points'")
            out[key] = int(len(points))
        elif isinstance(ann, list):
            out[key] = int(len(ann))
        else:
            raise ValueError(f"Unsupported annotation entry type for {image_name}: {type(ann).__name__}")
    return out


def normalize_eval_row(
    row: Dict[str, Any],
    gt_map: Dict[str, int],
    regression_target_key: str,
) -> Dict[str, Any]:
    if isinstance(row.get("messages"), list) and len(row["messages"]) >= 2:
        return row

    qid_raw = str(row.get("question_id", row.get("id", "")))
    qid = qid_raw.split("_")[0].split(".")[0]
    if not qid:
        raise ValueError("Eval row missing question_id/id")

    gt = row.get(regression_target_key)
    if gt is None:
        gt = row.get("gt_count")
    if gt is None:
        gt = gt_map.get(qid)
    if gt is None:
        raise ValueError(f"Missing GT for eval row qid={qid}")

    image_paths = row.get("image_paths")
    if not image_paths:
        image = row.get("image")
        if image:
            image_paths = [str(image), str(image)]
    if not isinstance(image_paths, list) or len(image_paths) != 2:
        raise ValueError(f"Eval row qid={qid} must have two image paths (image_paths or image)")

    system_text = ""
    history = row.get("history") or []
    if history and isinstance(history[0], dict):
        content = history[0].get("content") or []
        if content and isinstance(content[0], dict):
            system_text = str(content[0].get("text", ""))
    question = str(row.get("question") or row.get("instruction") or row.get("problem") or "")
    if not question:
        raise ValueError(f"Eval row qid={qid} missing question/instruction/problem text")
    if system_text and question:
        user_text = f"{system_text}\n\n{question}"
    else:
        user_text = system_text or question
    return {
        "id": qid,
        "question_id": qid,
        "gt_count": int(gt),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_paths[0])},
                    {"type": "image", "url": str(image_paths[1])},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps({"total_count": int(gt)})}],
            },
        ],
    }


def parse_int_set(text: str) -> set[int]:
    values: set[int] = set()
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.add(int(piece))
    return values


def fused_count(
    token_pred: Optional[int],
    reg_pred: Optional[float],
    fusion_mode: str,
    token_trust_max: int,
    bucket_values: set[int],
    token_max_sane: int,
) -> Optional[float]:
    if fusion_mode == "none":
        return float(token_pred) if token_pred is not None else reg_pred
    if fusion_mode != "gated":
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    if token_pred is not None and 0 <= int(token_pred) <= int(token_max_sane):
        if int(token_pred) < int(token_trust_max):
            return float(token_pred)
        if int(token_pred) not in bucket_values:
            return float(token_pred)

    return reg_pred if reg_pred is not None else (float(token_pred) if token_pred is not None else None)


class UniLIPRegressionHead(nn.Module):
    def __init__(self, hidden_size: int, output_activation: str = "linear"):
        super().__init__()
        if output_activation not in {"linear", "relu", "softplus"}:
            raise ValueError(f"Unsupported regression output activation: {output_activation}")
        self.output_activation = output_activation
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.output_activation == "relu":
            return torch.relu(out)
        if self.output_activation == "softplus":
            return torch.nn.functional.softplus(out)
        return out


def maybe_load_peft_eval_model(args: argparse.Namespace) -> Any:
    model_path = Path(args.model_name_or_path)
    adapter_cfg_path = model_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        return load_model(args)

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Adapter checkpoint detected (adapter_config.json) but `peft` is not installed."
        ) from exc

    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    base_model_path = adapter_cfg.get("base_model_name_or_path")
    if not base_model_path:
        raise RuntimeError(f"Missing base_model_name_or_path in {adapter_cfg_path}")

    base_args = argparse.Namespace(**vars(args))
    base_args.model_name_or_path = str(base_model_path)
    base_model = load_model(base_args)
    model = PeftModel.from_pretrained(base_model, str(model_path), is_trainable=False)
    model = model.to(module_device(base_model))
    return model


def maybe_load_regression_head_sidecar(model: Any, model_name_or_path: str) -> Optional[float]:
    if resolve_regression_head(model) is not None:
        return None
    model_path = Path(model_name_or_path)
    # Support both:
    # 1) model_name_or_path=<run_dir> containing regression_head.pt
    # 2) model_name_or_path=<run_dir>/checkpoint-XXXX, with sidecar at run_dir/regression_head.pt
    candidates = [model_path / "regression_head.pt", model_path.parent / "regression_head.pt"]
    sidecar_path = next((p for p in candidates if p.exists()), None)
    if sidecar_path is None:
        return None

    payload = torch.load(str(sidecar_path), map_location="cpu")
    raw_state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        new_key = key
        for prefix in ("original_module.", "modules_to_save.default."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        # Prefer first occurrence if duplicate keys appear from PEFT wrappers.
        if new_key not in state_dict:
            state_dict[new_key] = value

    fc1_weight = state_dict.get("net.0.weight")
    if fc1_weight is None:
        raise RuntimeError(f"Invalid regression head sidecar: missing net.0.weight in {sidecar_path}")
    hidden_size = int(fc1_weight.shape[1])
    output_activation = "linear"
    if isinstance(payload, dict):
        output_activation = str(payload.get("regression_output_activation", "linear"))
    head = UniLIPRegressionHead(hidden_size, output_activation=output_activation)
    head.load_state_dict(state_dict, strict=True)
    head = head.to(module_device(model))
    head.eval()
    model.regression_head = head
    return float(payload.get("count_norm_factor")) if isinstance(payload, dict) and payload.get("count_norm_factor") else None


def resolve_regression_head(model: Any) -> Optional[torch.nn.Module]:
    """Find regression head across plain / PEFT / DDP wrappers."""
    if hasattr(model, "regression_head"):
        return model.regression_head
    wrapped = getattr(model, "module", None)
    if wrapped is not None and hasattr(wrapped, "regression_head"):
        return wrapped.regression_head
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        if hasattr(base_model.model, "regression_head"):
            return base_model.model.regression_head
    if wrapped is not None:
        base_model = getattr(wrapped, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            if hasattr(base_model.model, "regression_head"):
                return base_model.model.regression_head
    return None


def resolve_internvl_core(model_module: Any) -> Any:
    """Return InternVL core (vision_tower/pixel_shuffle/projector owner)."""
    if hasattr(model_module, "model"):
        core = model_module.model
        if hasattr(core, "pixel_shuffle"):
            return core
        if hasattr(core, "model") and hasattr(core.model, "pixel_shuffle"):
            return core.model
    base_model = getattr(model_module, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        if hasattr(base_model.model, "pixel_shuffle"):
            return base_model.model
        if hasattr(base_model.model, "model") and hasattr(base_model.model.model, "pixel_shuffle"):
            return base_model.model.model
    raise AttributeError(f"Could not resolve InternVL core from {type(model_module).__name__}")


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def assistant_gt(row: Dict[str, Any]) -> int:
    if row.get("gt_count") is not None:
        return int(row["gt_count"])
    messages = row.get("messages") or []
    if len(messages) < 2:
        raise ValueError(f"Row {row.get('id')} missing assistant target")
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


@torch.no_grad()
def regression_pred_one(
    model: Any,
    processor: Any,
    row: Dict[str, Any],
    max_seq_length: int,
    count_norm_factor: float,
) -> Optional[float]:
    model_module = model.module if hasattr(model, "module") else model
    reg_head = resolve_regression_head(model)
    if reg_head is None:
        return None

    device = module_device(model)
    vision_tower = getattr(model_module, "vision_tower", None)
    if vision_tower is None and hasattr(model_module, "get_model"):
        vision_tower = getattr(model_module.get_model(), "vision_tower", None)
    vision_dtype = module_dtype(vision_tower) if vision_tower is not None else torch.bfloat16
    inputs = prompt_inputs(processor, row, max_seq_length, device, vision_dtype)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")

    language_model = model_module.get_model().language_model
    text_embeds = embed_tokens(language_model, input_ids)

    if pixel_values is not None:
        feature_layer = getattr(model_module.config, "vision_feature_layer", None)
        feature_strategy = getattr(model_module.config, "vision_feature_select_strategy", None)
        internvl_core = resolve_internvl_core(model_module)
        output_hidden_states = feature_layer != -1
        vision_outputs = internvl_core.vision_tower(
            pixel_values=pixel_values,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
        if feature_layer == -1:
            vision_features = vision_outputs.last_hidden_state
        else:
            vision_features = vision_outputs.hidden_states[feature_layer]
        if feature_strategy == "default":
            vision_features = vision_features[:, 1:, :]
        channels = vision_features.shape[1]
        feature_size = int(channels ** 0.5)
        batch_size = vision_features.shape[0]
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)
        vision_features = internvl_core.pixel_shuffle(
            vision_features,
            scale_factor=internvl_core.config.downsample_ratio,
        )
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])
        image_embeds = internvl_core.multi_modal_projector(vision_features)

        constants = load_unilip_constants()
        image_token_id = constants["UND_IMAGE_TOKEN_IDX"]
        image_token_mask = input_ids == image_token_id
        flat_embeds = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)
        expected = int(image_token_mask.sum().item())
        if expected == int(flat_embeds.shape[0]):
            text_embeds = text_embeds.clone()
            text_embeds[image_token_mask] = flat_embeds

    position_ids = torch.cumsum(attention_mask.int(), dim=1) - 1
    position_ids[position_ids < 0] = 0
    outputs = language_model(
        inputs_embeds=text_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    pooled_state = outputs.last_hidden_state[:, -1, :]
    head_param = next(reg_head.parameters())
    pooled_state = pooled_state.to(dtype=head_param.dtype)
    pred_norm = reg_head(pooled_state).squeeze(-1).float()
    pred = float((pred_norm * float(count_norm_factor)).detach().cpu().item())
    return pred


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_errors: List[float] = []
    reg_errors: List[float] = []
    fused_errors: List[float] = []
    token_missing = 0
    reg_missing = 0
    fused_missing = 0
    for row in rows:
        gt = int(row["gt_count"])
        pred_token = row.get("pred_count")
        pred_reg = row.get("reg_pred_count")
        pred_fused = row.get("fused_pred_count")
        if pred_token is None:
            token_missing += 1
        else:
            token_errors.append(float(abs(gt - int(pred_token))))
        if pred_reg is None:
            reg_missing += 1
        else:
            reg_errors.append(float(abs(gt - float(pred_reg))))
        if pred_fused is None:
            fused_missing += 1
        else:
            fused_errors.append(float(abs(gt - float(pred_fused))))
    return {
        "rows": len(rows),
        "token_metrics": {
            "missing": token_missing,
            "n": len(token_errors),
            "mae": mean(token_errors),
        },
        "regression_metrics": {
            "missing": reg_missing,
            "n": len(reg_errors),
            "mae": mean(reg_errors),
        },
        "fused_metrics": {
            "missing": fused_missing,
            "n": len(fused_errors),
            "mae": mean(fused_errors),
        },
    }


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
    parser.add_argument("--count_norm_factor", type=float, default=1000.0)
    parser.add_argument("--regression_target_key", default="gt_count")
    parser.add_argument("--fsc147_annotations", default="/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json")
    parser.add_argument(
        "--regression_output_activation",
        choices=["linear", "relu", "softplus"],
        default="linear",
    )
    parser.add_argument("--fusion_mode", choices=["none", "gated"], default="none")
    parser.add_argument("--fusion_token_trust_max", type=int, default=100)
    parser.add_argument("--fusion_bucket_values", default="100,128,1000")
    parser.add_argument("--fusion_token_max_sane", type=int, default=1000000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_transformers_compat_shims()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UniLIP generation/eval")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_rank)

    rows = load_json_or_jsonl(Path(args.data_path))
    gt_map = load_fsc147_annotation_mapping(args.fsc147_annotations)
    rows = [normalize_eval_row(row, gt_map, args.regression_target_key) for row in rows]
    rows = rows[args.start_index :]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows selected for eval")
    for row in rows:
        validate_messages(row, strict_images=bool(args.strict_images))

    processor = load_processor(args)
    model = maybe_load_peft_eval_model(args)
    sidecar_norm = maybe_load_regression_head_sidecar(model, args.model_name_or_path)
    if sidecar_norm is not None:
        args.count_norm_factor = float(sidecar_norm)
        print(f"Loaded regression head sidecar; using count_norm_factor={args.count_norm_factor}", flush=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    configure_generation(model, tokenizer, args.max_new_tokens)
    bucket_values = parse_int_set(args.fusion_bucket_values)

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
            reg_pred = regression_pred_one(
                model=model,
                processor=processor,
                row=row,
                max_seq_length=args.max_seq_length,
                count_norm_factor=args.count_norm_factor,
            )
            fused_pred = fused_count(
                token_pred=pred,
                reg_pred=reg_pred,
                fusion_mode=args.fusion_mode,
                token_trust_max=args.fusion_token_trust_max,
                bucket_values=bucket_values,
                token_max_sane=args.fusion_token_max_sane,
            )
            record = {
                "id": row.get("id"),
                "completion": completion,
                "pred_count": pred,
                "reg_pred_count": reg_pred,
                "fused_pred_count": fused_pred,
                "gt_count": gt,
                "abs_error_token": abs(gt - pred) if pred is not None else None,
                "abs_error_reg": abs(gt - reg_pred) if reg_pred is not None else None,
                "abs_error_fused": abs(gt - fused_pred) if fused_pred is not None else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            out_rows.append(record)
            if idx <= 20 or idx == len(rows) or idx % 100 == 0:
                print(
                    f"[{idx}/{len(rows)}] id={record['id']} token_pred={pred} "
                    f"reg_pred={None if reg_pred is None else round(reg_pred, 3)} "
                    f"fused_pred={None if fused_pred is None else round(fused_pred, 3)} gt={gt} "
                    f"text={completion[:100]!r}",
                    flush=True,
                )

    tmp_path.replace(output_path)
    print(json.dumps({"summary": summarize(out_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
