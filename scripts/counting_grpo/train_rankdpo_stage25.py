#!/usr/bin/env python3
"""
Stage 2.5 RankDPO trainer for grounded counting with InternVL2-2B.

Implements the required pivot:
  - Hard-merge Stage-1 adapter into base model before DPO adapter init.
  - Fresh LoRA adapter (rank 64) for DPO phase.
  - TRL DPOTrainer with GH200-oriented defaults (16k context, batch size 4,
    gradient checkpointing disabled).
  - Custom multimodal data flow for pixel_values and image_flags.
"""

import argparse
import copy
import glob
import json
import os
import re
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import flush_left, pad_to_length, selective_log_softmax


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


def rank0_print(*args: Any) -> None:
    if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0:
        print(*args)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_targets_for_lora(language_model: torch.nn.Module) -> List[str]:
    suffixes = {
        name.split(".")[-1]
        for name, module in language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    wanted = ["wqkv", "wo", "w1", "w2", "w3"]
    resolved = [name for name in wanted if name in suffixes]
    if not resolved:
        raise RuntimeError(
            "Could not resolve InternLM2 LoRA targets. "
            f"Linear suffix sample={sorted(list(suffixes))[:40]}"
        )
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


def _check_special_token_rows(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    row_start: int = 92544,
    row_end_exclusive: int = 92558,
) -> None:
    output_key_candidates = [
        "language_model.base_model.model.output.weight",
        "model.language_model.base_model.model.output.weight",
    ]
    ckpt_key = next((k for k in output_key_candidates if k in state), None)
    if ckpt_key is None:
        rank0_print("Special-token row check skipped: output.weight not found in checkpoint state.")
        return

    live_weight = getattr(model.language_model.output, "weight", None)
    if live_weight is None:
        rank0_print("Special-token row check skipped: model.language_model.output.weight not found.")
        return

    ckpt_weight = state[ckpt_key]
    upper = min(row_end_exclusive, ckpt_weight.shape[0], live_weight.shape[0])
    lower = min(row_start, upper)
    if lower >= upper:
        rank0_print(
            f"Special-token row check skipped: invalid range [{row_start}:{row_end_exclusive}) "
            f"for vocab sizes ckpt={ckpt_weight.shape[0]}, live={live_weight.shape[0]}."
        )
        return

    ck_slice = ckpt_weight[lower:upper].to(device=live_weight.device, dtype=live_weight.dtype)
    live_slice = live_weight.data[lower:upper]
    max_abs_diff = (live_slice - ck_slice).abs().max().item()
    rank0_print(
        f"Special-token row check [{lower}:{upper}) max_abs_diff={max_abs_diff:.6e} "
        f"(source={ckpt_key})"
    )


def _resolve_checkpoint_dir_with_weights(checkpoint_dir: str) -> str:
    """Return a directory that directly contains model safetensors (single or sharded)."""
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
    rank0_print(f"Auto-resolved Stage-1 weights dir: {resolved}")
    return resolved


def _list_model_safetensor_files(checkpoint_dir: str) -> List[str]:
    """List full checkpoint safetensors, supporting both monolithic and sharded layouts."""
    single = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    shard_glob = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
    if shard_glob:
        return shard_glob

    # Fallback to any model*.safetensors naming convention.
    fallback = sorted(glob.glob(os.path.join(checkpoint_dir, "model*.safetensors")))
    # Exclude adapter files if present in the same directory tree.
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
    check_state: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for file_path in weight_files:
            state = safetensors_load_file(file_path, device="cpu")
            for output_key in (
                "language_model.base_model.model.output.weight",
                "model.language_model.base_model.model.output.weight",
            ):
                if output_key in state and output_key not in check_state:
                    check_state[output_key] = state[output_key]

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

                # Handle LoRA-wrapped backbone keys embedded in full safetensors.
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

    rank0_print(
        f"Injection Complete: Reconstructed {injected} Stage-1 backbone tensors "
        f"from {len(weight_files)} file(s) (skipped={skipped})."
    )
    if injected == 0:
        raise RuntimeError(
            "Manual Stage-1 backbone injection found zero matching tensors. "
            "Check Stage-1 checkpoint format and key prefixes."
        )
    if check_state:
        _check_special_token_rows(model, check_state)
    return injected


def _wrap_fp32_forward(module: torch.nn.Module) -> None:
    if getattr(module, "_rankdpo_fp32_wrapped", False):
        return

    original_forward = module.forward

    def _forward_cast(x, *args, **kwargs):
        x_dtype = x.dtype if torch.is_tensor(x) else None
        if torch.is_tensor(x):
            x = x.float()
        out = original_forward(x, *args, **kwargs)
        if x_dtype is not None and torch.is_tensor(out):
            return out.to(x_dtype)
        return out

    module.forward = _forward_cast
    module._rankdpo_fp32_wrapped = True


def apply_numerical_stability_guards(model: torch.nn.Module, vision_scale: float = 0.1) -> None:
    for name, module in model.named_modules():
        lname = name.lower()
        if "norm" in lname or "output_embeddings" in lname:
            module.to(torch.float32)
            _wrap_fp32_forward(module)

    if hasattr(model, "mlp1") and hasattr(model.mlp1, "forward"):
        if not getattr(model.mlp1, "_rankdpo_scaled", False):
            original_forward = model.mlp1.forward

            def _scaled_forward(*args, **kwargs):
                return original_forward(*args, **kwargs) * vision_scale

            model.mlp1.forward = _scaled_forward
            model.mlp1._rankdpo_scaled = True


def _normalize_lora_key_namespace(raw_key: str) -> str | None:
    key = raw_key
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    if ".lora_A." not in key and ".lora_B." not in key:
        return None

    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")
    if key.startswith("model.layers."):
        key = f"base_model.model.{key}"
    return key


def materialize_native_adapter_from_full_checkpoint(checkpoint_dir: str, base_model_name: str) -> str:
    adapter_config = os.path.join(checkpoint_dir, "native_peft_adapter", "adapter_config.json")
    adapter_weights = os.path.join(checkpoint_dir, "native_peft_adapter", "adapter_model.safetensors")
    if os.path.exists(adapter_config) and os.path.exists(adapter_weights):
        return os.path.join(checkpoint_dir, "native_peft_adapter")

    resolved_dir = _resolve_checkpoint_dir_with_weights(checkpoint_dir)
    weight_files = _list_model_safetensor_files(resolved_dir)
    if not weight_files:
        raise FileNotFoundError(
            "Could not find native adapter or full checkpoint weights under "
            f"{checkpoint_dir}. Expected native_peft_adapter/* or model*.safetensors."
        )

    adapter_state: Dict[str, torch.Tensor] = {}
    rank = None
    targets = set()
    for file_path in weight_files:
        full_state = safetensors_load_file(file_path, device="cpu")
        for key, tensor in full_state.items():
            normalized = _normalize_lora_key_namespace(key)
            if normalized is None:
                continue
            adapter_state[normalized] = tensor

            parts = normalized.split(".")
            if "lora_A" in parts and tensor.ndim == 2 and rank is None:
                rank = int(tensor.shape[0])
            if "lora_A" in parts or "lora_B" in parts:
                marker = "lora_A" if "lora_A" in parts else "lora_B"
                marker_idx = parts.index(marker)
                if marker_idx > 0:
                    targets.add(parts[marker_idx - 1])

    if not adapter_state:
        raise RuntimeError(f"No LoRA keys found in {resolved_dir}")
    if rank is None:
        raise RuntimeError(f"Could not infer LoRA rank from {resolved_dir}")

    adapter_dir = os.path.join(checkpoint_dir, "native_peft_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    safetensors_save_file(adapter_state, os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model_name_or_path": base_model_name,
                "bias": "none",
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "r": rank,
                "target_modules": sorted(targets),
                "task_type": "CAUSAL_LM",
            },
            handle,
            indent=2,
        )
    rank0_print(
        f"Materialized native_peft_adapter at {adapter_dir} "
        f"(rank={rank}, targets={sorted(targets)}, keys={len(adapter_state)})"
    )
    return adapter_dir


def resolve_stage1_checkpoint(requested_path: str) -> str:
    if os.path.exists(requested_path):
        return requested_path

    parent = os.path.dirname(requested_path) if requested_path else ""
    if not parent:
        raise FileNotFoundError(f"Stage-1 checkpoint path does not exist: {requested_path}")
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"Stage-1 checkpoint path does not exist and parent directory is missing: {requested_path}"
        )

    candidates: List[tuple[int, str]] = []
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if not match:
            continue

        has_adapter = os.path.exists(os.path.join(full, "native_peft_adapter", "adapter_model.safetensors"))
        has_full = os.path.exists(os.path.join(full, "model.safetensors"))
        if has_adapter or has_full:
            candidates.append((int(match.group(1)), full))

    if not candidates:
        raise FileNotFoundError(
            "Could not resolve a valid Stage-1 checkpoint under "
            f"{parent}. Expected checkpoint-*/native_peft_adapter or checkpoint-*/model.safetensors"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    resolved = candidates[0][1]
    rank0_print(f"Requested stage1 checkpoint missing: {requested_path}. Auto-resolved to {resolved}")
    return resolved


def build_chat_prompt(
    problem: str,
    num_image_token: int,
    assistant_prefill: str = "{",
    prepend_scaffold_prefix: bool = True,
) -> str:
    image_placeholder = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}"
    user_text = problem.replace(DEFAULT_IMAGE_TOKEN, image_placeholder)
    if prepend_scaffold_prefix and SCAFFOLD_PREFIX.lower() not in user_text.lower():
        user_text = f"{SCAFFOLD_PREFIX}\n{user_text}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefill}"
    )


def _normalize_json_completion(text: str) -> str:
    normalized = str(text).strip()
    if normalized.startswith("<|im_start|>"):
        return normalized
    if not normalized.startswith("{"):
        start = normalized.find("{")
        if start >= 0:
            normalized = normalized[start:]
    return normalized


def run_scaffold_syntax_verify(
    model: torch.nn.Module,
    tokenizer,
    image_processor,
    row: Dict[str, Any],
    num_image_token: int,
    max_new_tokens: int = 256,
) -> None:
    model.eval()
    verify_prompt = (
        "<image>\n"
        "The image is overlaid with a 6x6 dot matrix. Dots are labeled (x,y). "
        "Within columns, x increases top-to-bottom. Within rows, y increases left-to-right. "
        "Count objects by nearest anchors and output strict JSON only."
    )
    prompt = build_chat_prompt(verify_prompt, num_image_token, assistant_prefill="{", prepend_scaffold_prefix=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device

    image = Image.open(str(row["image"])).convert("RGB")
    pixel_values = image_processor.preprocess([image], return_tensors="pt")["pixel_values"]

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

    generation_kwargs: Dict[str, Any] = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids: List[int] = []
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    if eos_ids:
        generation_kwargs["eos_token_id"] = sorted(set(eos_ids))

    if hasattr(model, "generation_config") and hasattr(model.generation_config, "stop_strings"):
        model.generation_config.stop_strings = ["<|im_end|>"]

    with torch.no_grad():
        output_ids = model.generate(
            **generation_kwargs,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    assistant_split = generated.split("<|im_start|>assistant\n")
    assistant_text = assistant_split[-1] if assistant_split else generated
    assistant_text = assistant_text.split("<|im_end|>")[0].strip()
    if not assistant_text.startswith("{"):
        brace_idx = assistant_text.find("{")
        if brace_idx >= 0:
            assistant_text = assistant_text[brace_idx:]

    parse_ok = False
    try:
        payload = json.loads(assistant_text)
        parse_ok = isinstance(payload, dict) and "total_count" in payload and "clusters" in payload
    except Exception:
        parse_ok = False

    if not parse_ok:
        snippet = generated[-1200:]
        raise RuntimeError(
            "Merge verification failed: generated output is not valid schema-aligned JSON. "
            f"Tail snippet: {snippet}"
        )
    rank0_print("Merge verification passed: output contains valid schema-aligned JSON.")


def build_dataset_rows(
    data_path: str,
    num_image_token: int,
    assistant_prefill: str = "{",
    prepend_scaffold_prefix: bool = True,
) -> List[dict]:
    raw = load_jsonl(data_path)
    rows = []
    for item in raw:
        problem = str(item.get("prompt", item.get("problem", "")))
        chosen = _normalize_json_completion(str(item.get("chosen", item.get("solution", ""))))
        rejected = _normalize_json_completion(str(item.get("rejected", "")))
        rows.append(
            {
                "id": item.get("id", ""),
                "image": item["image"],
                "prompt": build_chat_prompt(
                    problem,
                    num_image_token,
                    assistant_prefill=assistant_prefill,
                    prepend_scaffold_prefix=prepend_scaffold_prefix,
                ),
                "chosen": chosen + "<|im_end|>\n",
                "rejected": rejected + "<|im_end|>\n",
                "pair_type": item.get("pair_type", "unknown"),
            }
        )
    return rows


class InternVLDPOCollator:
    """Pad preference tokens and lazily materialize image tensors + image flags."""

    def __init__(self, tokenizer, image_processor):
        self.base = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)
        self.image_processor = image_processor

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        batch = self.base.torch_call(features)

        pixel_values = []
        image_flags = []
        for feature in features:
            image_path = feature["image"]
            image = Image.open(image_path).convert("RGB")
            processed = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"][0]
            pixel_values.append(processed)
            image_flags.append([1])

        batch["pixel_values"] = torch.stack(pixel_values).to(torch.bfloat16)
        batch["image_flags"] = torch.tensor(image_flags, dtype=torch.long)
        return batch


class InternVLDPOTrainer(DPOTrainer):
    """DPOTrainer extension that forwards InternVL multimodal image_flags."""

    @staticmethod
    def concatenated_inputs(batch: dict[str, torch.Tensor], padding_value: int) -> dict[str, torch.Tensor]:
        output = DPOTrainer.concatenated_inputs(batch, padding_value)
        if "image_flags" in batch:
            output["image_flags"] = torch.cat([batch["image_flags"], batch["image_flags"]], dim=0)
        return output

    def concatenated_forward(self, model: torch.nn.Module, batch: dict[str, torch.Tensor]):
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs: Dict[str, Any] = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]
        if "image_flags" in concatenated_batch:
            model_kwargs["image_flags"] = concatenated_batch["image_flags"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1)

        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        if self.max_length is not None:
            if self.truncation_mode == "keep_end":
                input_ids = input_ids[:, -self.max_length :]
                attention_mask = attention_mask[:, -self.max_length :]
                loss_mask = loss_mask[:, -self.max_length :]
            elif self.truncation_mode == "keep_start":
                input_ids = input_ids[:, : self.max_length]
                attention_mask = attention_mask[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        if self.use_logits_to_keep:
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
            model_kwargs["logits_to_keep"] = logits_to_keep

        if self.padding_free:
            input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
            position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
            model_kwargs["position_ids"] = position_ids
        else:
            model_kwargs["attention_mask"] = attention_mask

        # Some wrappers may not expose image_flags in signature; retry without it.
        try:
            outputs = model(input_ids=input_ids, **model_kwargs)
        except TypeError as exc:
            if "image_flags" in model_kwargs and "image_flags" in str(exc):
                model_kwargs.pop("image_flags")
                outputs = model(input_ids=input_ids, **model_kwargs)
            else:
                raise

        logits = outputs.logits.clamp(min=-50.0, max=50.0)
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if self.use_logits_to_keep:
            labels = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype)
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)
        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output: Dict[str, torch.Tensor] = {
            "chosen_logps": all_logps[:num_examples],
            "rejected_logps": all_logps[num_examples:],
        }

        if self.padding_free:
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.args.rpo_alpha is not None:
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss
        return output

    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        found_non_finite = False
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                found_non_finite = True
                break

        if found_non_finite:
            rank0_print("Gradient firewall triggered: non-finite gradients detected, skipping optimizer step.")
            model.zero_grad(set_to_none=True)
            return torch.zeros_like(loss)

        return loss


def freeze_non_lora_policy_modules(model: torch.nn.Module) -> None:
    for module_name in ["vision_model", "mlp1"]:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False


def freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def build_models(args) -> tuple[torch.nn.Module, torch.nn.Module, AutoTokenizer, Any, int]:
    args.stage1_checkpoint = resolve_stage1_checkpoint(args.stage1_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right",
        model_max_length=args.max_length,
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

    if args.init_mode == "manual_inject":
        inject_stage1_backbone_from_safetensors(model, args.stage1_checkpoint)
    else:
        adapter_dir = materialize_native_adapter_from_full_checkpoint(args.stage1_checkpoint, args.base_model)
        rank0_print(f"Loading Stage-1 adapter from {adapter_dir}")
        model.language_model = PeftModel.from_pretrained(model.language_model, adapter_dir, is_trainable=False)
        model.language_model = model.language_model.merge_and_unload()

    apply_numerical_stability_guards(model, vision_scale=args.vision_scale)

    # Build reference model from hard-merged weights before adding fresh DPO LoRA.
    ref_model = copy.deepcopy(model)
    freeze_all(ref_model)
    ref_model.eval()

    lora_targets = _resolve_targets_for_lora(model.language_model)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)

    # Required gradient anchor for stable graph connectivity.
    model.language_model.get_input_embeddings().weight.requires_grad_(True)

    freeze_non_lora_policy_modules(model)
    model.config.use_cache = False
    ref_model.config.use_cache = True

    if hasattr(model.language_model, "print_trainable_parameters"):
        model.language_model.print_trainable_parameters()

    num_image_token = int(getattr(model, "num_image_token", 256))
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    ref_model.img_context_token_id = model.img_context_token_id
    if model.img_context_token_id is None or model.img_context_token_id < 0:
        raise RuntimeError(f"Could not resolve token id for {IMG_CONTEXT_TOKEN}")

    return model, ref_model, tokenizer, image_processor, num_image_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="outputs/rankdpo/preference_pairs.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="OpenGVLab/InternVL2-2B")
    parser.add_argument("--stage1_checkpoint", default="checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1140")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--init_mode", default="manual_inject", choices=["manual_inject", "merge_adapter"])
    parser.add_argument("--vision_scale", type=float, default=1.0)

    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--max_prompt_length", type=int, default=12288)
    parser.add_argument("--max_completion_length", type=int, default=4096)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", default="sigmoid", choices=["sigmoid", "ipo"])
    parser.add_argument("--learning_rate", type=float, default=5e-7)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--skip_merge_verify", action="store_true")
    parser.add_argument("--verify_max_new_tokens", type=int, default=256)
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--assistant_prefill", default="{")
    parser.add_argument("--disable_scaffold_prefix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, ref_model, tokenizer, image_processor, num_image_token = build_models(args)

    rows = build_dataset_rows(
        args.data_path,
        num_image_token=num_image_token,
        assistant_prefill=args.assistant_prefill,
        prepend_scaffold_prefix=not args.disable_scaffold_prefix,
    )
    train_dataset = Dataset.from_list(rows)
    rank0_print(f"Loaded {len(train_dataset)} RankDPO rows from {args.data_path}")

    if not rows:
        raise RuntimeError("No training rows found. Check --data_path and dataset generation.")

    if not args.skip_merge_verify:
        rank0_print("Running one-sample merged-model scaffold syntax verification...")
        try:
            run_scaffold_syntax_verify(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                row=rows[0],
                num_image_token=num_image_token,
                max_new_tokens=args.verify_max_new_tokens,
            )
        except RuntimeError as e:
            rank0_print(f"WARNING: Verify check failed, but continuing to training: {str(e)[:500]}")

    if args.verify_only:
        rank0_print("Verify-only mode enabled; exiting before trainer initialization.")
        return

    training_args = DPOConfig(
        output_dir=args.output_dir,
        bf16=True,
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
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        loss_type=args.loss_type,
        dataloader_num_workers=4,
    )

    collator = InternVLDPOCollator(tokenizer=tokenizer, image_processor=image_processor)

    trainer = InternVLDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
