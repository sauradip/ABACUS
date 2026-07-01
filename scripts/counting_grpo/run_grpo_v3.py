#!/usr/bin/env python3
"""Stage-2 GRPO refinement runner (dual-head consensus reward).

New partitioned entrypoint to keep existing GRPO scripts untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import (
    apply_chat_template,
    gather,
    is_conversational,
    profiling_context,
    selective_log_softmax,
    unwrap_model_for_generation,
)

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.grpo_reward_consensus_v1 import (  # noqa: E402
    consensus_quadratic_v1,
    consensus_quadratic_v1_components,
)
from scripts.counting_grpo.train_hf_multi_image_count_sft import (  # noqa: E402
    embed_tokens,
    expand_unilip_image_context,
    load_model,
    load_pil_images,
    load_processor,
    load_unilip_constants,
    module_dtype,
)


def precision_counting_reward(prompts, completions, gt_count, **kwargs):
    rewards = []
    lazy_buckets = {100, 108, 120, 128, 139, 150, 200, 250, 300, 350, 400, 500}
    for completion, gt in zip(completions, gt_count):
        text = completion[0].get("content", "") if isinstance(completion, list) else str(completion)

        match = re.search(r'\{\s*"total_count"\s*:\s*(\d+)\s*\}', text.strip())
        if not match:
            rewards.append(-1.0)
            continue

        pred_count = int(match.group(1))
        gt = int(gt)
        error = abs(gt - pred_count)
        error_ratio = error / max(1, gt)

        base_accuracy = max(0.0, 1.0 - (error_ratio**2))

        precision_bonus = 0.0
        if gt >= 40:
            if error_ratio == 0:
                precision_bonus = 1.0
            elif error_ratio <= 0.05:
                precision_bonus = 0.5
            elif error_ratio <= 0.10:
                precision_bonus = 0.2

        bucket_penalty = -0.5 if pred_count in lazy_buckets and error_ratio > 0.10 else 0.0

        if gt < 40 and error > 2:
            base_accuracy -= 0.1

        rewards.append(base_accuracy + precision_bonus + bucket_penalty + 0.1)

    return rewards


REWARD_REGISTRY = {
    "precision_counting_reward": precision_counting_reward,
    "consensus_quadratic_v1": consensus_quadratic_v1,
}

try:
    _BOUND_REWARD_DIR = REPO_DIR / "bound_reward"
    import sys as _sys
    if str(_BOUND_REWARD_DIR) not in _sys.path:
        _sys.path.insert(0, str(_BOUND_REWARD_DIR))
    from boundary_reward_adapter import boundary_decomposed_reward  # noqa: E402
    REWARD_REGISTRY["boundary_decomposed_reward"] = boundary_decomposed_reward
except Exception as _e:
    import warnings as _warnings
    _warnings.warn(f"boundary_decomposed_reward not available: {_e}")


def pairwise_overlap_reward(prompts, completions, gt_count, **kwargs):
    """Reward for pairwise overlap counting. Model outputs {\"overlap\": N}."""
    rewards = []
    for completion, gt in zip(completions, gt_count):
        text = completion[0].get("content", "") if isinstance(completion, list) else str(completion)
        match = re.search(r'\{\s*"overlap"\s*:\s*(\d+)\s*\}', text.strip())
        if not match:
            rewards.append(-1.0)
            continue
        pred = int(match.group(1))
        gt   = int(gt)
        if gt == 0:
            rewards.append(1.0 if pred == 0 else max(-1.0, -0.5 * pred))
        else:
            err = abs(pred - gt) / gt
            rewards.append(max(-1.0, 1.0 - 2.0 * err))
    return rewards


REWARD_REGISTRY["pairwise_overlap_reward"] = pairwise_overlap_reward


def _maybe_apply_chat_template(example: Dict[str, Any], processing_class: Any) -> Dict[str, Any]:
    # TRL API compatibility: older versions expose apply_chat_template, newer add maybe_apply_chat_template.
    try:
        from trl.trainer.grpo_trainer import maybe_apply_chat_template as _impl  # type: ignore

        return _impl(example, processing_class)
    except Exception:
        return apply_chat_template(example, processing_class)


class UniLIPRegressionHead(nn.Module):
    def __init__(self, hidden_size: int, output_activation: str = "softplus"):
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


class GRPOProcessor:
    def __init__(self, processor_bundle: Any):
        self.tokenizer = getattr(processor_bundle, "tokenizer", processor_bundle)
        self.image_processor = getattr(processor_bundle, "image_processor", None)
        if self.image_processor is None:
            raise RuntimeError("GRPOProcessor requires an image_processor")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def save_pretrained(self, output_dir: str) -> None:
        self.tokenizer.save_pretrained(output_dir)
        if hasattr(self.image_processor, "save_pretrained"):
            self.image_processor.save_pretrained(output_dir)


def _load_fsc147_annotation_mapping(annotation_json: Optional[str]) -> Dict[str, int]:
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
                raise ValueError(f"Annotation for {image_name} missing list field 'points'")
            out[key] = int(len(points))
        elif isinstance(ann, list):
            out[key] = int(len(ann))
        else:
            raise ValueError(f"Unsupported annotation entry type for {image_name}: {type(ann).__name__}")
    return out


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported data format: {path}")


def _normalize_grpo_row(row: Dict[str, Any], gt_map: Dict[str, int]) -> Dict[str, Any]:
    if "prompt" in row and "gt_count" in row:
        return row

    qid_raw = str(row.get("question_id", row.get("id", "")))
    qid = qid_raw.split("_")[0].split(".")[0]
    if not qid:
        raise ValueError("Row missing question_id/id")

    gt = row.get("gt_count")
    if gt is None:
        gt = gt_map.get(qid)
    if gt is None:
        raise ValueError(f"Missing gt_count for row qid={qid}")

    image_paths = row.get("image_paths") or []
    if not isinstance(image_paths, list) or len(image_paths) != 2:
        raise ValueError(f"Row qid={qid} must contain exactly two image_paths")

    system_text = ""
    history = row.get("history") or []
    if history and isinstance(history[0], dict):
        content = history[0].get("content") or []
        if content and isinstance(content[0], dict):
            system_text = str(content[0].get("text", ""))

    question = str(row.get("question") or row.get("instruction") or "")
    if not question:
        raise ValueError(f"Row qid={qid} missing question text")

    if system_text and question:
        user_text = f"{system_text}\n\n{question}"
    else:
        user_text = system_text or question

    return {
        "id": qid,
        "gt_count": int(gt),
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_paths[0])},
                    {"type": "image", "url": str(image_paths[1])},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    }


def _resolve_regression_head(model: Any) -> Optional[nn.Module]:
    if hasattr(model, "regression_head"):
        return model.regression_head
    wrapped = getattr(model, "module", None)
    if wrapped is not None and hasattr(wrapped, "regression_head"):
        return wrapped.regression_head
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model") and hasattr(base_model.model, "regression_head"):
        return base_model.model.regression_head
    if wrapped is not None:
        base_model = getattr(wrapped, "base_model", None)
        if base_model is not None and hasattr(base_model, "model") and hasattr(base_model.model, "regression_head"):
            return base_model.model.regression_head
    return None


def _configure_regression_head(model: Any, output_activation: str) -> nn.Module:
    reg_head = _resolve_regression_head(model)
    if reg_head is not None:
        return reg_head
    hidden_size = int(getattr(getattr(model.config, "text_config", model.config), "hidden_size"))
    head = UniLIPRegressionHead(hidden_size, output_activation=output_activation).to(next(model.parameters()).device)
    model.regression_head = head
    return head


def _load_sidecar(model: Any, args: argparse.Namespace) -> float:
    sidecar_path = Path(args.sidecar_weights_path) if args.sidecar_weights_path else None
    if sidecar_path is None:
        model_path = Path(args.model_name_or_path)
        cands = [model_path / "regression_head.pt", model_path.parent / "regression_head.pt"]
        sidecar_path = next((p for p in cands if p.exists()), None)

    _configure_regression_head(model, args.regression_output_activation)

    if sidecar_path is None or not sidecar_path.exists():
        print("[warn] regression_head.pt not found; continuing without sidecar load")
        return float(args.count_norm_factor)

    payload = torch.load(str(sidecar_path), map_location="cpu")
    raw_state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        new_key = key
        for prefix in ("original_module.", "modules_to_save.default."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key not in state_dict:
            state_dict[new_key] = value

    reg_head = _resolve_regression_head(model)
    if reg_head is None:
        raise RuntimeError("Regression head missing while loading sidecar")
    reg_head.load_state_dict(state_dict, strict=False)
    print(f"[info] loaded regression head sidecar: {sidecar_path}")

    sidecar_norm = payload.get("count_norm_factor") if isinstance(payload, dict) else None
    if sidecar_norm is not None:
        try:
            sidecar_norm_f = float(sidecar_norm)
            cli_norm_f = float(args.count_norm_factor)
            if abs(sidecar_norm_f - cli_norm_f) > 1e-9:
                print(
                    f"[warn] sidecar count_norm_factor={sidecar_norm_f} differs from CLI "
                    f"count_norm_factor={cli_norm_f}; using CLI value."
                )
        except Exception:
            pass
    return float(args.count_norm_factor)


def _load_policy_model(args: argparse.Namespace) -> Any:
    model_path = Path(args.model_name_or_path)
    adapter_cfg_path = model_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        return load_model(args)

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Adapter checkpoint detected but peft is not installed") from exc

    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    base_model_path = args.base_model_name_or_path or adapter_cfg.get("base_model_name_or_path")
    if not base_model_path:
        raise RuntimeError(f"Missing base_model_name_or_path in {adapter_cfg_path}")

    base_args = argparse.Namespace(**vars(args))
    base_args.model_name_or_path = str(base_model_path)
    base_model = load_model(base_args)
    model = PeftModel.from_pretrained(base_model, str(model_path), is_trainable=True)

    # Load fine-tuned connector saved alongside the adapter (adapter_extracted workflow)
    conn_path = model_path / "multi_modal_projector.bin"
    if conn_path.exists():
        inner = model.get_model() if hasattr(model, "get_model") else (
            model.base_model.model.get_model()
            if hasattr(model.base_model.model, "get_model") else None
        )
        if inner is not None and hasattr(inner, "multi_modal_projector"):
            sd_conn = torch.load(str(conn_path), map_location="cpu")
            inner.multi_modal_projector.load_state_dict(sd_conn, strict=False)
            print(f"[info] Loaded fine-tuned connector from {conn_path}")

    return model


def _freeze_vision_tower(model: Any) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if "vision_tower" in name.lower():
            if param.requires_grad:
                param.requires_grad = False
            frozen += 1
    return frozen


def _unfreeze_query_params(model: Any) -> int:
    matched = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if "query" in lname or "latent_queries" in lname or "resampler" in lname:
            if not param.requires_grad:
                param.requires_grad = True
            matched += 1
    return matched


class UniLIPMultiImageGRPOTrainerV3(GRPOTrainer):
    def __init__(self, *args: Any, count_norm_factor: float, **kwargs: Any):
        # trl 1.3.0 enforces processing_class must be PreTrainedTokenizerBase or ProcessorMixin.
        # GRPOProcessor is a custom wrapper — pass its underlying tokenizer to super(), then
        # restore our wrapper so _prompt_batch can access .image_processor.
        proc = kwargs.get("processing_class", None)
        if proc is not None and isinstance(proc, GRPOProcessor):
            kwargs["processing_class"] = proc.tokenizer
        super().__init__(*args, **kwargs)
        if proc is not None and isinstance(proc, GRPOProcessor):
            self.processing_class = proc
        self.count_norm_factor = float(count_norm_factor)
        # trl 1.3.0 removed max_prompt_length from GRPOConfig; our _generate_and_score_completions
        # still references it, so set to None (no truncation) if not set by parent.
        if not hasattr(self, "max_prompt_length"):
            self.max_prompt_length = None
        # trl 1.3.0 references this attr in training_step but transformers 4.52.1 never sets it.
        if not hasattr(self, "current_gradient_accumulation_steps"):
            self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps

    def _get_train_sampler(self, train_dataset=None):
        # Compatibility shim: some TRL versions define _get_train_sampler(self)
        # while newer Transformers calls it as sampler_fn(dataset).
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

    def _prompt_batch(self, inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts_text = [
            expand_unilip_image_context(_maybe_apply_chat_template(example, self.processing_class)["prompt"])
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        pixel_batches: List[torch.Tensor] = []
        for example in inputs:
            images = load_pil_images(example["prompt"])
            pixels = self.processing_class.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
            pixel_batches.append(pixels)
        prompt_inputs["pixel_values"] = torch.cat(pixel_batches, dim=0)
        return Trainer._prepare_inputs(self, prompt_inputs)

    def _model_last_hidden(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
            # Keep this path aligned with the working SFT forward used in train_hf_multi_image_count_sft.py
            # to avoid API drift in get_image_features across different model wrappers.
            # In PEFT-wrapped runs, model_module.model is the CausalLM wrapper,
            # while the internals with pixel_shuffle/projector live on get_model().
            internvl_core = model_module.get_model() if hasattr(model_module, "get_model") else model_module.model
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

            image_token_id = load_unilip_constants()["UND_IMAGE_TOKEN_IDX"]
            image_token_mask = input_ids == image_token_id
            flat_embeds = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)
            expected = int(image_token_mask.sum().item())
            if expected != int(flat_embeds.shape[0]):
                raise RuntimeError(
                    "Image token count does not match vision features: "
                    f"tokens={expected}, embeds={tuple(flat_embeds.shape)}, pixel_values={tuple(pixel_values.shape)}"
                )
            text_embeds = text_embeds.clone()
            text_embeds[image_token_mask] = flat_embeds

        position_ids = torch.cumsum(attention_mask.int(), dim=1) - 1
        position_ids[position_ids < 0] = 0
        outputs = language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def _model_logits(self, model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: Optional[torch.Tensor]) -> torch.Tensor:
        hidden = self._model_last_hidden(model, input_ids, attention_mask, pixel_values)
        model_module = model.module if hasattr(model, "module") else model
        return model_module.lm_head(hidden)

    def _predict_regression_counts(
        self,
        model: Any,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
    ) -> List[Optional[float]]:
        reg_head = _resolve_regression_head(model)
        if reg_head is None:
            return [None] * int(prompt_completion_ids.shape[0])

        hidden = self._model_last_hidden(model, prompt_completion_ids, attention_mask, pixel_values)
        prompt_len = int(prompt_completion_ids.shape[1] - completion_mask.shape[1])
        completion_lens = completion_mask.sum(dim=1).long()
        idx = torch.clamp(prompt_len + completion_lens - 1, min=0, max=prompt_completion_ids.shape[1] - 1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx, :]
        head_param = next(reg_head.parameters())
        pooled = pooled.to(dtype=head_param.dtype)
        pred_norm = reg_head(pooled).squeeze(-1).float()
        pred = pred_norm * float(self.count_norm_factor)
        return [float(v) for v in pred.detach().cpu().tolist()]

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, pixel_values: Optional[torch.Tensor] = None):
        logits = self._model_logits(model, input_ids, attention_mask, pixel_values)
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        device = self.accelerator.device
        prompts = [copy.deepcopy(x["prompt"]) for x in inputs]
        prompt_inputs = self._prompt_batch(inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        # Cast pixel_values to model dtype (bfloat16 when --bf16 is set) to avoid conv dtype mismatch.
        model_module = self.model.module if hasattr(self.model, "module") else self.model
        _first_param = next(iter(model_module.parameters()), None)
        if _first_param is not None and pixel_values.dtype != _first_param.dtype:
            pixel_values = pixel_values.to(_first_param.dtype)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            raise ValueError("vLLM is disabled for HF-native multi-image UniLIP GRPO")

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            generated_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                pixel_values=pixel_values,
                generation_config=self.generation_config,
            )

        prompt_length = prompt_ids.size(1)
        if generated_ids.shape[1] > prompt_length and torch.equal(generated_ids[:, :prompt_length], prompt_ids):
            prompt_completion_ids = generated_ids
            completion_ids = generated_ids[:, prompt_length:]
        else:
            completion_ids = generated_ids
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    pixel_values=pixel_values,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    pixel_values=pixel_values,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        pixel_values=pixel_values,
                    )

            reg_pred_count = self._predict_regression_counts(
                self.model,
                prompt_completion_ids,
                attention_mask,
                completion_mask,
                pixel_values,
            )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt and prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        reward_kwargs = {}
        consensus_means: Optional[Dict[str, float]] = None

        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ if not isinstance(reward_func, nn.Module) else str(reward_func)
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                reward_kwargs["reg_pred_count"] = reg_pred_count

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

                if reward_func_name == "consensus_quadratic_v1":
                    comps = consensus_quadratic_v1_components(
                        completions=completions,
                        gt_count=reward_kwargs.get("gt_count", []),
                        reg_pred_count=reg_pred_count,
                    )
                    if comps:
                        consensus_means = {
                            "reward_components/r_form": float(sum(x["r_form"] for x in comps) / len(comps)),
                            "reward_components/r_tok": float(sum(x["r_tok"] for x in comps) / len(comps)),
                            "reward_components/r_reg": float(sum(x["r_reg"] for x in comps) / len(comps)),
                            "reward_components/p_con": float(sum(x["p_con"] for x in comps) / len(comps)),
                        }

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        self._metrics[mode]["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ if not isinstance(reward_func, nn.Module) else str(reward_func)
            self._metrics[mode][f"rewards/{reward_func_name}"].append(torch.nanmean(rewards_per_func[:, i]).item())
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if consensus_means is not None:
            for key, value in consensus_means.items():
                self._metrics[mode][key].append(value)

        # Reshape pixel_values from (B*N_imgs, C, H, W) → (B, N_imgs, C, H, W) so that
        # TRL's shuffle_sequence_dict sees batch dim=B (same as prompt_ids) and doesn't
        # truncate the image batch when it applies a length-B permutation.
        n_examples = len(inputs)
        if pixel_values is not None and pixel_values.ndim == 4:
            n_imgs_per = pixel_values.shape[0] // n_examples
            pixel_values_stored = pixel_values.view(n_examples, n_imgs_per, *pixel_values.shape[1:])
        else:
            pixel_values_stored = pixel_values

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values_stored,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        pixel_values = inputs.get("pixel_values")
        # Unflatten (B, N_imgs, C, H, W) → (B*N_imgs, C, H, W) for the vision encoder.
        if pixel_values is not None and pixel_values.ndim == 5:
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(B * N, C, H, W)

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            pixel_values=pixel_values,
        )

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss


def _validate_grpo_row(row: Dict[str, Any]) -> None:
    if "prompt" not in row or "gt_count" not in row:
        raise ValueError(f"GRPO row must contain prompt and gt_count keys: {row.keys()}")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
        raise ValueError(f"Row {row.get('id')} prompt must contain one user message")
    content = prompt[0].get("content", [])
    images = [item for item in content if isinstance(item, dict) and item.get("type") == "image"]
    texts = [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if len(images) < 1 or len(texts) != 1:
        raise ValueError(f"Row {row.get('id')} must have at least one image and exactly one text item")


def _extract_metric_from_trainer_state(trainer_state_path: Path, metric_name: str) -> Optional[float]:
    if not trainer_state_path.exists():
        return None
    try:
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    history = payload.get("log_history", [])
    if not isinstance(history, list):
        return None
    for row in reversed(history):
        if isinstance(row, dict) and metric_name in row:
            try:
                return float(row[metric_name])
            except Exception:
                continue
    return None


def _select_best_checkpoint(output_dir: str, metric_name: str, greater_is_better: bool) -> Optional[Path]:
    out = Path(output_dir)
    ckpts = sorted(
        [p for p in out.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if not ckpts:
        return None

    scored: List[tuple[Path, float]] = []
    for ckpt in ckpts:
        metric = _extract_metric_from_trainer_state(ckpt / "trainer_state.json", metric_name)
        if metric is not None:
            scored.append((ckpt, metric))

    if not scored:
        return None

    best_ckpt, best_metric = max(scored, key=lambda x: x[1]) if greater_is_better else min(scored, key=lambda x: x[1])
    (out / "best_checkpoint.txt").write_text(
        f"{best_ckpt}\nmetric={metric_name}\nvalue={best_metric}\n",
        encoding="utf-8",
    )
    return best_ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--base_model_name_or_path", default=None)
    p.add_argument("--sidecar_weights_path", default=None)
    p.add_argument("--processor_name_or_path", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--fsc147_annotations", default="/home/nvidia/amondal/FSC147_hf/annotation_FSC147_384.json")
    p.add_argument("--reward_funcs", default="consensus_quadratic_v1")
    p.add_argument("--regression_output_activation", choices=["linear", "relu", "softplus"], default="softplus")
    p.add_argument("--count_norm_factor", type=float, default=100.0)
    p.add_argument("--unfreeze_queries", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--freeze_vision_tower", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--bf16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPL", "eager"))
    p.add_argument("--allow_attn_fallback", type=int, default=int(os.environ.get("ALLOW_ATTN_FALLBACK", "1")))
    p.add_argument("--trust_remote_code", type=int, default=1)
    p.add_argument("--report_to", default="none")
    p.add_argument("--metric_for_best_model", default="reward")
    p.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    p.add_argument("--init_only", action="store_true")
    return p.parse_args()


def set_local_cuda_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def main() -> None:
    args = parse_args()
    set_local_cuda_device()

    raw_rows = _load_rows(Path(args.data_path))
    gt_map = _load_fsc147_annotation_mapping(args.fsc147_annotations)
    rows = [_normalize_grpo_row(row, gt_map) for row in raw_rows]

    if args.max_samples > 0:
        rows = rows[: min(args.max_samples, len(rows))]

    for row in rows[: min(16, len(rows))]:
        _validate_grpo_row(row)

    dataset = Dataset.from_list(rows)

    processor = GRPOProcessor(load_processor(args))
    model = _load_policy_model(args)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
        model.generation_config.pad_token_id = processor.pad_token_id
        model.generation_config.eos_token_id = processor.eos_token_id

    args.count_norm_factor = _load_sidecar(model, args)
    print(f"[info] count_norm_factor={args.count_norm_factor}")
    if args.freeze_vision_tower:
        frozen = _freeze_vision_tower(model)
        print(f"[info] vision_tower params frozen: {frozen}")
    if args.unfreeze_queries:
        matched = _unfreeze_query_params(model)
        print(f"[info] query-like params unfrozen: {matched}")

    reward_names = [x.strip() for x in str(args.reward_funcs).split(",") if x.strip()]
    reward_funcs = []
    for name in reward_names:
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward func: {name}. Available: {sorted(REWARD_REGISTRY.keys())}")
        reward_funcs.append(REWARD_REGISTRY[name])

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        logging_steps=10,
        save_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=args.report_to,
        use_vllm=False,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = UniLIPMultiImageGRPOTrainerV3(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        count_norm_factor=float(args.count_norm_factor),
    )

    if args.init_only:
        print("GRPO v3 init-only preflight passed.")
        return

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)
    best_ckpt = _select_best_checkpoint(
        output_dir=args.output_dir,
        metric_name=args.metric_for_best_model,
        greater_is_better=bool(args.greater_is_better),
    )
    if best_ckpt is None:
        print("[warn] Could not determine best checkpoint from trainer_state logs.")
    else:
        print(f"[info] best_checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
