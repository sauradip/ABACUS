#!/usr/bin/env python3
"""Dual-head UniLIP trainer for CE/MSE counting ablations.

This script is intentionally separate from the baseline SFT trainer so that
experiments do not mix checkpoints/logs with existing runs.
"""

import argparse
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    IGNORE_INDEX,
    HFMultiImageCollator,
    HFMultiImageCountDataset,
    apply_transformers_compat_shims,
    assert_clean_model_source,
    assert_clean_processor_source,
    embed_tokens,
    force_default_cpu_device,
    load_model,
    load_processor,
    load_unilip_constants,
    module_device,
    module_dtype,
    rank0_print,
    validate_messages,
)


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
            return F.relu(out)
        if self.output_activation == "softplus":
            return F.softplus(out)
        return out


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


def normalize_prompt_only_row(
    row: Dict[str, Any],
    gt_map: Dict[str, int],
    regression_target_key: str,
) -> Dict[str, Any]:
    # Already canonical supervised format.
    if isinstance(row.get("messages"), list) and len(row["messages"]) >= 2:
        return row

    qid_raw = str(row.get("question_id", row.get("id", "")))
    qid = qid_raw.split("_")[0].split(".")[0]
    if not qid:
        raise ValueError("Prompt-only row missing question_id/id")

    gt = row.get(regression_target_key)
    if gt is None:
        gt = row.get("gt_count")
    if gt is None:
        gt = gt_map.get(qid)
    if gt is None:
        raise ValueError(f"Missing GT for row qid={qid} (target_key={regression_target_key})")

    image_paths = row.get("image_paths") or []
    if not isinstance(image_paths, list) or len(image_paths) != 2:
        raise ValueError(f"Row qid={qid} must have exactly two image_paths")

    system_text = ""
    history = row.get("history") or []
    if history and isinstance(history[0], dict):
        content = history[0].get("content") or []
        if content and isinstance(content[0], dict):
            system_text = str(content[0].get("text", ""))
    question = str(row.get("question", ""))
    if not question:
        raise ValueError(f"Row qid={qid} missing question text")

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


class RegressionDataset(HFMultiImageCountDataset):
    """Extends baseline dataset to emit numeric count labels for MSE."""

    def __init__(
        self,
        *args: Any,
        regression_target_key: str = "gt_count",
        count_norm_factor: float = 1000.0,
        fsc147_annotations: Optional[str] = None,
        **kwargs: Any,
    ):
        if "data_path" not in kwargs:
            raise ValueError("RegressionDataset requires data_path")
        data_path = Path(kwargs["data_path"])
        processor = kwargs.get("processor")
        max_seq_length = int(kwargs.get("max_seq_length"))
        validate_only = bool(kwargs.get("validate_only", False))
        strict_images = bool(kwargs.get("strict_images", True))

        self.data_path = data_path
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.validate_only = validate_only
        self.strict_images = strict_images
        self.regression_target_key = regression_target_key
        self.count_norm_factor = float(count_norm_factor)

        gt_map = load_fsc147_annotation_mapping(fsc147_annotations)
        loaded_rows = []
        for raw in self._load_rows(data_path):
            loaded_rows.append(normalize_prompt_only_row(raw, gt_map, regression_target_key))
        self.raw_data = loaded_rows
        for row in self.raw_data:
            validate_messages(row, strict_images=strict_images)
        self.gt_counts = [self._extract_count(row) for row in self.raw_data]
        rank0_print(f"Loaded {len(self.raw_data)} regression rows from {self.data_path}")

    @staticmethod
    def _load_rows(path: Path) -> list[Dict[str, Any]]:
        if path.suffix == ".jsonl":
            rows: list[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported dataset format at {path}")

    def _extract_count(self, row: Dict[str, Any]) -> int:
        # Priority 1: explicit key from row (for alternative dataset layouts).
        if self.regression_target_key in row and row[self.regression_target_key] is not None:
            return int(row[self.regression_target_key])

        # Priority 2: assistant JSON payload from canonical SFT rows.
        messages = row.get("messages") or []
        if len(messages) >= 2:
            content = messages[1].get("content") or []
            if content and isinstance(content[0], dict):
                text = content[0].get("text", "")
                payload = json.loads(text)
                if isinstance(payload, dict) and "total_count" in payload:
                    return int(payload["total_count"])
        raise ValueError(f"Row {row.get('id')} missing regression target '{self.regression_target_key}'")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        if self.validate_only:
            return item
        gt = float(self.gt_counts[index])
        item["gt_counts"] = torch.tensor(gt, dtype=torch.float32)
        item["gt_counts_norm"] = torch.tensor(gt / self.count_norm_factor, dtype=torch.float32)
        return item


def resolve_internvl_core(model_module: Any) -> Any:
    """Return the InternVL core that owns vision_tower/pixel_shuffle/projector.

    Handles plain UniLIP models and PEFT-wrapped variants.
    """
    # Common path for plain UniLIP_InternVLForCausalLM.
    if hasattr(model_module, "model"):
        core = model_module.model
        if hasattr(core, "pixel_shuffle"):
            return core
        if hasattr(core, "model") and hasattr(core.model, "pixel_shuffle"):
            return core.model

    # PEFT wrappers often keep the base under `.base_model.model`.
    base_model = getattr(model_module, "base_model", None)
    if base_model is not None:
        if hasattr(base_model, "model") and hasattr(base_model.model, "pixel_shuffle"):
            return base_model.model
        if hasattr(base_model, "model") and hasattr(base_model.model, "model"):
            nested = base_model.model.model
            if hasattr(nested, "pixel_shuffle"):
                return nested

    raise AttributeError(
        f"Could not resolve InternVL core with pixel_shuffle from type {type(model_module).__name__}"
    )


def _compute_multimodal_forward(model: Any, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")

    model_module = model.module if hasattr(model, "module") else model
    language_model = model_module.get_model().language_model
    text_embeds = embed_tokens(language_model, input_ids)

    if pixel_values is not None:
        vision_tower = getattr(model_module, "vision_tower", None)
        if vision_tower is None and hasattr(model_module, "get_model"):
            vision_tower = getattr(model_module.get_model(), "vision_tower", None)
        vision_dtype = module_dtype(vision_tower) if vision_tower is not None else text_embeds.dtype
        pixel_values = pixel_values.to(device=text_embeds.device, dtype=vision_dtype)

        feature_layer = getattr(model_module.config, "vision_feature_layer", None)
        feature_strategy = getattr(model_module.config, "vision_feature_select_strategy", None)

        with torch.no_grad():
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
        if expected != int(flat_embeds.shape[0]):
            raise RuntimeError(
                "Image token count does not match vision features: "
                f"tokens={expected}, embeds={tuple(flat_embeds.shape)}"
            )
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
    logits = model_module.lm_head(outputs.last_hidden_state)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return {
        "ce_loss": ce_loss,
        "last_hidden_state": outputs.last_hidden_state,
        "logits": logits,
    }


def _pool_assistant_state(last_hidden_state: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Use final supervised token representation for regression signal.
    supervised_mask = labels.ne(IGNORE_INDEX)
    pooled = []
    for b in range(last_hidden_state.shape[0]):
        valid_pos = torch.where(supervised_mask[b])[0]
        idx = int(valid_pos[-1].item()) if valid_pos.numel() > 0 else int(last_hidden_state.shape[1] - 1)
        pooled.append(last_hidden_state[b, idx, :])
    return torch.stack(pooled, dim=0)


def configure_regression_head(model: Any, hidden_size: int, output_activation: str) -> None:
    if hasattr(model, "regression_head"):
        return
    model.regression_head = UniLIPRegressionHead(
        hidden_size,
        output_activation=output_activation,
    ).to(module_device(model))


def resolve_regression_head(model: Any) -> nn.Module:
    """Find regression head across plain / PEFT / DDP wrappers."""
    if hasattr(model, "regression_head"):
        return model.regression_head
    wrapped = getattr(model, "module", None)
    if wrapped is not None and hasattr(wrapped, "regression_head"):
        return wrapped.regression_head
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        if hasattr(base_model, "model") and hasattr(base_model.model, "regression_head"):
            return base_model.model.regression_head
    if wrapped is not None:
        base_model = getattr(wrapped, "base_model", None)
        if base_model is not None:
            if hasattr(base_model, "model") and hasattr(base_model.model, "regression_head"):
                return base_model.model.regression_head
    raise AttributeError(
        f"Could not resolve regression_head from model type {type(model).__name__}"
    )


def unwrap_model(model: Any) -> Any:
    module = model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    while hasattr(module, "module"):
        module = module.module
    return module


def save_regression_head_sidecar(
    model: Any,
    output_dir: str,
    count_norm_factor: float,
    output_activation: str,
) -> Optional[str]:
    try:
        reg_head = resolve_regression_head(model)
    except AttributeError:
        return None
    target_path = Path(output_dir) / "regression_head.pt"
    payload = {
        "state_dict": reg_head.state_dict(),
        "count_norm_factor": float(count_norm_factor),
        "regression_output_activation": str(output_activation),
    }
    torch.save(payload, target_path)
    return str(target_path)


def maybe_apply_lora(model: Any, args: argparse.Namespace, modules_to_save: Optional[list[str]] = None) -> Any:
    if args.lora_rank <= 0:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested but `peft` is not installed.") from exc

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    rank0_print(
        f"Applying LoRA: r={args.lora_rank}, alpha={args.lora_alpha}, modules={target_modules}"
    )
    return get_peft_model(model, lora_cfg)


def maybe_toggle_query_freeze(model: Any, unfreeze_queries: bool) -> None:
    if not unfreeze_queries:
        return
    # Best-effort: enable grads for known query-style parameters in the bridge.
    toggled = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if "query" in lname or "latent_queries" in lname or "resampler" in lname:
            if not param.requires_grad:
                param.requires_grad = True
            toggled += 1
    rank0_print(f"unfreeze_queries enabled; query-like params matched: {toggled}")


def ensure_isolated_output_paths(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output_dir:
        raise RuntimeError(
            f"Output dir already exists and is non-empty: {output_dir}. "
            "Use a new run directory to avoid mixing checkpoints/logs, or set "
            "--allow_existing_output_dir true if you intentionally resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


class DualHeadTrainer(transformers.Trainer):
    def __init__(
        self,
        *args: Any,
        loss_type: str,
        mse_weight: float,
        count_norm_factor: float,
        mse_variant: str,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.mse_weight = float(mse_weight)
        self.count_norm_factor = float(count_norm_factor)
        self.mse_variant = mse_variant
        self._last_mse = None
        self._last_ce = None

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        gt_counts_norm = inputs.pop("gt_counts_norm", None)
        _ = inputs.pop("gt_counts", None)  # Kept in dataset for debugging/inspection.

        forward = _compute_multimodal_forward(model, inputs)
        ce_loss = forward["ce_loss"]
        last_hidden_state = forward["last_hidden_state"]
        labels = inputs["labels"]

        loss = ce_loss
        mse_loss = None
        if self.loss_type in {"mse", "hybrid"}:
            pooled = _pool_assistant_state(last_hidden_state, labels)
            reg_head = resolve_regression_head(model)
            head_param = next(reg_head.parameters())
            pooled = pooled.to(dtype=head_param.dtype)
            pred_counts_norm = reg_head(pooled).squeeze(-1)
            if gt_counts_norm is None:
                raise RuntimeError("Missing gt_counts_norm in batch for MSE mode")
            # Compute MSE in fp32 for numeric stability.
            pred_counts_norm = pred_counts_norm.float()
            gt_counts_norm = gt_counts_norm.to(pred_counts_norm.device, dtype=torch.float32)
            if self.mse_variant == "log":
                pred_counts = pred_counts_norm * self.count_norm_factor
                gt_counts = gt_counts_norm * self.count_norm_factor
                pred_log = torch.log1p(torch.clamp(pred_counts, min=0.0))
                gt_log = torch.log1p(torch.clamp(gt_counts, min=0.0))
                mse_loss = F.mse_loss(pred_log, gt_log)
            else:
                mse_loss = F.mse_loss(pred_counts_norm, gt_counts_norm)
            if self.loss_type == "mse":
                loss = mse_loss
            else:
                loss = ce_loss + (self.mse_weight * mse_loss)

        self._last_ce = float(ce_loss.detach().cpu())
        self._last_mse = float(mse_loss.detach().cpu()) if mse_loss is not None else None

        out = {"logits": forward["logits"], "ce_loss": ce_loss}
        if mse_loss is not None:
            out["mse_loss"] = mse_loss
        return (loss, out) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self._last_ce is not None and "ce_loss" not in logs:
            logs["ce_loss"] = round(self._last_ce, 6)
        if self._last_mse is not None and "mse_loss" not in logs:
            logs["mse_loss"] = round(self._last_mse, 6)
        super().log(logs, start_time=start_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--bf16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPL", "flash_attention_2"))
    parser.add_argument("--allow_attn_fallback", type=int, default=int(os.environ.get("ALLOW_ATTN_FALLBACK", "0")))
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--eval_strategy", default="no")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_only_model", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--load_best_model_at_end", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--strict_images", type=int, default=1)
    parser.add_argument("--trust_remote_code", type=int, default=1)
    parser.add_argument("--report_to", default="none")

    # Regression/ablation args
    parser.add_argument("--loss_type", choices=["ce", "mse", "hybrid"], default="ce")
    parser.add_argument("--mse_variant", choices=["raw", "log"], default="raw")
    parser.add_argument("--regression_target_key", default="gt_count")
    parser.add_argument("--fsc147_annotations", default="/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json")
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--count_norm_factor", type=float, default=1000.0)
    parser.add_argument(
        "--regression_output_activation",
        choices=["linear", "relu", "softplus"],
        default="linear",
    )
    parser.add_argument("--unfreeze_queries", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--allow_existing_output_dir", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"regression_sft_{args.loss_type}_{stamp}"
    return str(Path("checkpoints") / "experiment_regression_sft" / run_name)


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args)
    apply_transformers_compat_shims()

    assert_clean_model_source(args.model_name_or_path)
    if args.processor_name_or_path:
        assert_clean_processor_source(args.processor_name_or_path)
    ensure_isolated_output_paths(args)

    processor = load_processor(args)
    train_dataset = RegressionDataset(
        data_path=args.data_path,
        processor=processor,
        max_seq_length=args.max_seq_length,
        strict_images=bool(args.strict_images),
        regression_target_key=args.regression_target_key,
        count_norm_factor=args.count_norm_factor,
        fsc147_annotations=args.fsc147_annotations,
    )
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = RegressionDataset(
            data_path=args.eval_data_path,
            processor=processor,
            max_seq_length=args.max_seq_length,
            strict_images=bool(args.strict_images),
            regression_target_key=args.regression_target_key,
            count_norm_factor=args.count_norm_factor,
            fsc147_annotations=args.fsc147_annotations,
        )
    collator = HFMultiImageCollator(processor)

    with force_default_cpu_device():
        model = load_model(args)
    model.config.use_cache = False

    hidden_size = int(getattr(getattr(model.config, "text_config", model.config), "hidden_size"))
    if args.loss_type in {"mse", "hybrid"}:
        configure_regression_head(model, hidden_size, args.regression_output_activation)
        rank0_print("Regression head initialized")
        model = maybe_apply_lora(model, args, modules_to_save=["regression_head"])
    else:
        model = maybe_apply_lora(model, args)
    maybe_toggle_query_freeze(model, args.unfreeze_queries)

    if args.dry_run:
        batch = collator([train_dataset[0]])
        device = module_device(model)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        trainer = DualHeadTrainer(
            model=model,
            args=transformers.TrainingArguments(output_dir=args.output_dir, do_train=False, report_to="none"),
            train_dataset=train_dataset,
            data_collator=collator,
            loss_type=args.loss_type,
            mse_weight=args.mse_weight,
            count_norm_factor=args.count_norm_factor,
            mse_variant=args.mse_variant,
        )
        loss = trainer.compute_loss(model, batch)
        rank0_print(f"Dry-run {args.loss_type} loss={float(loss.detach().cpu())}")
        return

    run_name = args.run_name or Path(args.output_dir).name
    logging_dir = str(Path(args.output_dir) / "logs")

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=logging_dir,
        run_name=run_name,
        do_train=True,
        do_eval=eval_dataset is not None,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        load_best_model_at_end=args.load_best_model_at_end if eval_dataset is not None else False,
        metric_for_best_model=args.metric_for_best_model if eval_dataset is not None else None,
        greater_is_better=args.greater_is_better if eval_dataset is not None else None,
        report_to=args.report_to,
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "loss_type": args.loss_type,
        "mse_variant": args.mse_variant,
        "mse_weight": args.mse_weight,
        "count_norm_factor": args.count_norm_factor,
    }
    trainer_init = inspect.signature(DualHeadTrainer.__init__)
    if "tokenizer" in trainer_init.parameters:
        trainer_kwargs["tokenizer"] = getattr(processor, "tokenizer", None)
    elif "processing_class" in trainer_init.parameters:
        trainer_kwargs["processing_class"] = getattr(processor, "tokenizer", None)

    rank0_print(
        "Starting dual-head run "
        f"(loss_type={args.loss_type}, mse_variant={args.mse_variant}, "
        f"output_dir={args.output_dir}, logging_dir={logging_dir})"
    )
    trainer = DualHeadTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    sidecar_path = save_regression_head_sidecar(
        trainer.model,
        args.output_dir,
        args.count_norm_factor,
        args.regression_output_activation,
    )

    rank0_print("Training complete.")
    if sidecar_path:
        rank0_print(f"regression_head: {sidecar_path}")
    rank0_print(f"checkpoints: {args.output_dir}")
    rank0_print(f"logs: {logging_dir}")


if __name__ == "__main__":
    main()
