"""
Stage-2 GRPO trainer for InternVL2-2B.

Subclasses HF Trainer (not TRL GRPOTrainer) and implements the full
GRPO update loop using InternVL2's pixel_values / image_flags forward
interface.  Pattern adapted from VLM-R1 Qwen2VLGRPOTrainer.
"""

import argparse
import contextlib
import copy
import glob
import json
import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sized, Union

import torch
import torch.utils.data
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
)
from accelerate.utils import is_peft_model, set_seed
from peft import PeftModel
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig

# ---------------------------------------------------------------------------
# Image / token constants  (mirrors train_native_sft.py)
# ---------------------------------------------------------------------------
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
DEFAULT_IMAGE_TOKEN = "<image>"


def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, flush=True)


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
        match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(path))
        if match and glob.glob(os.path.join(path, "model*.safetensors")):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return checkpoint_dir

    candidates.sort(key=lambda item: item[0], reverse=True)
    resolved = candidates[0][1]
    rank0_print(f"Auto-resolved checkpoint weights dir: {resolved}")
    return resolved


def _list_model_safetensor_files(checkpoint_dir: str) -> List[str]:
    resolved_dir = _resolve_checkpoint_dir_with_weights(checkpoint_dir)
    single = os.path.join(resolved_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    shard_glob = sorted(glob.glob(os.path.join(resolved_dir, "model-*.safetensors")))
    if shard_glob:
        return shard_glob

    fallback = sorted(glob.glob(os.path.join(resolved_dir, "model*.safetensors")))
    return [path for path in fallback if os.path.basename(path) != "adapter_model.safetensors"]


def _checkpoint_has_embedded_lora(checkpoint_dir: str) -> bool:
    index_path = os.path.join(_resolve_checkpoint_dir_with_weights(checkpoint_dir), "model.safetensors.index.json")
    try:
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        return any(".lora_A." in key or ".lora_B." in key for key in weight_map)
    except Exception:
        pass

    for file_path in _list_model_safetensor_files(checkpoint_dir)[:1]:
        state = safetensors_load_file(file_path, device="cpu")
        return any(".lora_A." in key or ".lora_B." in key for key in state)
    return False


def inject_full_checkpoint_backbone_from_safetensors(model: torch.nn.Module, checkpoint_dir: str) -> int:
    weight_files = _list_model_safetensor_files(checkpoint_dir)
    if not weight_files:
        raise FileNotFoundError(f"No model safetensors found under {checkpoint_dir}")

    prefix_mappings = (
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
                target_root: Any = None
                for prefix, root in prefix_mappings:
                    if key.startswith(prefix):
                        target_key = key[len(prefix):]
                        target_root = root
                        break

                if target_key is None:
                    if "base_model." in key:
                        skipped += 1
                        continue
                    target_key = key
                    target_root = model

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
        f"Injected {injected} non-LoRA checkpoint tensor(s) from {len(weight_files)} file(s) "
        f"(skipped={skipped})."
    )
    if injected == 0:
        raise RuntimeError(f"No checkpoint tensors were injected from {checkpoint_dir}")
    return injected


def merge_lora_deltas_from_safetensors(
    model: torch.nn.Module,
    checkpoint_dir: str,
    lora_rank: int = 64,
    lora_alpha: float = 128.0,
) -> int:
    weight_files = _list_model_safetensor_files(checkpoint_dir)
    if not weight_files:
        return 0

    full_state: Dict[str, torch.Tensor] = {}
    for file_path in weight_files:
        full_state.update(safetensors_load_file(file_path, device="cpu"))

    lora_a: Dict[str, torch.Tensor] = {}
    lora_b: Dict[str, torch.Tensor] = {}
    for key, value in full_state.items():
        if ".lora_A." in key:
            lora_a[key[: key.index(".lora_A.")]] = value
        elif ".lora_B." in key:
            lora_b[key[: key.index(".lora_B.")]] = value

    prefix_mappings = (
        ("language_model.base_model.model.model.", model.language_model.model),
        ("model.language_model.base_model.model.model.", model.language_model.model),
        ("language_model.base_model.model.output.", model.language_model.output),
        ("model.language_model.base_model.model.output.", model.language_model.output),
    )

    scaling = lora_alpha / lora_rank
    merged = 0
    with torch.no_grad():
        for stem, a_tensor in lora_a.items():
            b_tensor = lora_b.get(stem)
            if b_tensor is None:
                continue

            target_key = None
            target_root: Any = None
            for prefix, root in prefix_mappings:
                if stem.startswith(prefix):
                    target_key = stem[len(prefix):]
                    target_root = root
                    break
            if target_key is None or target_root is None:
                continue

            target_key = target_key.replace(".base_layer", "")
            weight_key = target_key + ".weight"
            try:
                parent, leaf = _resolve_submodule_attr(target_root, weight_key)
                param = getattr(parent, leaf)
                delta = (b_tensor.float() @ a_tensor.float()) * scaling
                if isinstance(param, torch.nn.Parameter):
                    param.data.add_(delta.to(device=param.device, dtype=param.dtype))
                elif torch.is_tensor(param):
                    param.add_(delta.to(device=param.device, dtype=param.dtype))
                else:
                    continue
                merged += 1
            except Exception:
                continue

    rank0_print(f"Merged {merged} LoRA delta(s) from checkpoint safetensors (scaling={scaling:.3f}).")
    return merged


# ---------------------------------------------------------------------------
# RepeatRandomSampler  (verbatim from VLM-R1 grpo_trainer.py)
# ---------------------------------------------------------------------------

class RepeatRandomSampler(Sampler):
    """Yields each index mini_repeat_count times per batch (for GRPO generations)."""

    def __init__(self, data_source: Sized, mini_repeat_count: int,
                 batch_size: int = 1, repeat_count: int = 1,
                 seed: Optional[int] = None):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# ---------------------------------------------------------------------------
# Preprocessing helpers  (mirrors train_native_sft.py)
# ---------------------------------------------------------------------------

def _preprocess_multimodal(conversations, num_image_token):
    image_placeholder = (
        IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
    )
    result = copy.deepcopy(conversations)
    for turn in result:
        role = turn.get("from", turn.get("role", ""))
        content_key = "value" if "value" in turn else "content"
        if role == "human" and DEFAULT_IMAGE_TOKEN in str(turn.get(content_key, "")):
            turn[content_key] = (
                turn[content_key].replace(DEFAULT_IMAGE_TOKEN, image_placeholder).strip()
            )
    return result


def _tokenize_prompt(conversations, tokenizer, num_image_token):
    """Return 1-D LongTensor of prompt token ids (up to generation token)."""
    conv = _preprocess_multimodal(conversations, num_image_token)
    user_content = ""
    for turn in conv:
        role = turn.get("from", turn.get("role", ""))
        if role == "human":
            user_content = turn.get("value", turn.get("content", ""))
            break

    messages = [
        {"role": "user", "content": user_content},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )[0]
    return prompt_ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InternVLGRPODataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, num_image_token,
                 max_prompt_length=512):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_image_token = num_image_token
        self.max_prompt_length = max_prompt_length

        rank0_print(f"Loading GRPO data from {data_path} ...")
        with open(data_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        rank0_print(f"Loaded {len(self.records)} records.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        has_image = bool(item.get("image"))

        if has_image:
            try:
                img = Image.open(item["image"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (448, 448), (255, 255, 255))
            pixel_values = self.image_processor.preprocess(
                [img], return_tensors="pt"
            )["pixel_values"][0]
        else:
            pixel_values = torch.zeros((3, 448, 448), dtype=torch.float32)

        convs = item.get("conversations", [])
        prompt_ids = _tokenize_prompt(convs, self.tokenizer, self.num_image_token)
        if prompt_ids.size(0) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:]

        return {
            "prompt_ids": prompt_ids,                    # 1-D LongTensor
            "pixel_values": pixel_values,                # (C, H, W) float32
            "has_image": int(has_image),
            "ground_truth_count": item.get("ground_truth_count", 0),
            "normalized_points_1000": item.get("normalized_points_1000", []),
            "solution": item.get("solution", ""),
            "prompt_text": item.get("problem", ""),      # plain text, for logging
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class GRPOCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(inst["prompt_ids"].size(0) for inst in instances)

        input_ids_list, attn_mask_list = [], []
        for inst in instances:
            ids = inst["prompt_ids"]
            pad_len = max_len - ids.size(0)
            input_ids_list.append(
                torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), ids])
            )
            attn_mask_list.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(ids.size(0), dtype=torch.long),
                ])
            )

        pixel_values = torch.stack([inst["pixel_values"] for inst in instances]).to(torch.bfloat16)
        image_flags = torch.tensor(
            [[inst["has_image"]] for inst in instances], dtype=torch.long
        )

        return {
            "input_ids":             torch.stack(input_ids_list),
            "attention_mask":        torch.stack(attn_mask_list),
            "pixel_values":          pixel_values,
            "image_flags":           image_flags,
            "ground_truth_count":    [inst["ground_truth_count"]    for inst in instances],
            "normalized_points_1000":[inst["normalized_points_1000"] for inst in instances],
            "solution":              [inst["solution"]              for inst in instances],
            "prompt_text":           [inst["prompt_text"]           for inst in instances],
        }


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _build_reward_fn():
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from grpo_rewards import scaffold_grpo_reward  # noqa: PLC0415

    def accuracy_reward(prompts, completions, **kwargs):
        gt_counts = kwargs.get("ground_truth_count", [0] * len(completions))
        gt_points = kwargs.get("normalized_points_1000", [[] for _ in completions])
        rewards = []
        for comp, count, points in zip(completions, gt_counts, gt_points):
            text = comp if isinstance(comp, str) else ""
            result = scaffold_grpo_reward(
                text, ground_truth_count=count, normalized_points_1000=points
            )
            rewards.append(float(result["total_reward"]))
        return rewards

    return accuracy_reward


# ---------------------------------------------------------------------------
# InternVL2GRPOTrainer  — subclasses HF Trainer (like VLM-R1 does)
# ---------------------------------------------------------------------------

class InternVL2GRPOTrainer(Trainer):
    """
    GRPO trainer for InternVL2-2B.

    The training data sampler yields each prompt num_generations times so
    that each forward pass generates one completion per example.  Rewards
    are grouped by prompt to compute GRPO advantages.
    """

    def __init__(self, model, ref_model, tokenizer, reward_funcs,
                 args: GRPOConfig, train_dataset, data_collator,
                 num_generations: int = 8, epsilon: float = 0.2,
                 num_iterations: int = 2):
        self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        self.ref_model = ref_model
        self.num_generations = num_generations
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.beta = args.beta
        self._step = 0
        self._buffered_inputs = [None] * max(args.gradient_accumulation_steps, 1)
        self._metrics = defaultdict(list)
        self._debug_dumped_completions = 0

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        num_procs = self.accelerator.num_processes
        global_bs = args.per_device_train_batch_size * num_procs
        if global_bs % self.num_generations != 0:
            raise ValueError(
                f"Global batch size ({global_bs}) must be divisible by "
                f"num_generations ({self.num_generations})."
            )

        set_seed(args.seed, device_specific=True)
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    # ------------------------------------------------------------------ #
    # Sampler                                                              #
    # ------------------------------------------------------------------ #

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids", "attention_mask", "pixel_values", "image_flags",
                "ground_truth_count", "normalized_points_1000", "solution", "prompt_text",
            ]

    def _get_train_sampler(self, train_dataset=None):
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    # ------------------------------------------------------------------ #
    # Log-probability computation                                          #
    # ------------------------------------------------------------------ #

    def _get_per_token_logps(self, model, input_ids, attention_mask,
                             pixel_values, image_flags):
        # FIX: Force autograd to track inputs through frozen vision layers so that
        # gradient_checkpointing inside the LM does not abort the backward pass.
        if pixel_values is not None:
            pixel_values.requires_grad_(True)
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
        )
        logits = outputs.logits[:, :-1, :]        # (B, L-1, V)
        ids_shifted = input_ids[:, 1:]             # (B, L-1)
        per_token_logps = []
        for logits_row, ids_row in zip(logits, ids_shifted):
            log_probs = logits_row.log_softmax(dim=-1)
            token_lp = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_lp)
        return torch.stack(per_token_logps)

    # ------------------------------------------------------------------ #
    # Generation + scoring                                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _sample_completion_ids(self, model, prompt_ids, prompt_mask, pixel_values, image_flags, eos_id, pad_id):
        """Sample completion tokens autoregressively with a KV cache after the image/prompt pass."""
        max_new_tokens = int(self.args.max_completion_length)
        temperature = float(getattr(self.args, "temperature", 1.0) or 1.0)

        device = prompt_ids.device
        bsz = prompt_ids.size(0)
        completion_ids = torch.full(
            (bsz, max_new_tokens),
            fill_value=pad_id,
            dtype=prompt_ids.dtype,
            device=device,
        )

        cur_input_ids = prompt_ids
        cur_attention_mask = prompt_mask
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        past_key_values = None
        last_tokens = None

        base_model = model.module if hasattr(model, "module") else model
        language_model = getattr(base_model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Expected InternVL model to expose language_model for cached decoding")
        cache_dtype = language_model.get_input_embeddings().weight.dtype

        def _cast_past_key_values(past):
            if past is None:
                return None
            return tuple(
                tuple(
                    item.to(dtype=cache_dtype) if torch.is_tensor(item) and item.is_floating_point() else item
                    for item in layer
                )
                for layer in past
            )

        cache_configs = []
        for maybe_cfg in (getattr(base_model, "config", None), getattr(language_model, "config", None)):
            if maybe_cfg is not None and hasattr(maybe_cfg, "use_cache"):
                cache_configs.append((maybe_cfg, maybe_cfg.use_cache))

        was_training = model.training
        model.eval()
        try:
            for cfg, _old_use_cache in cache_configs:
                cfg.use_cache = True

            for t in range(max_new_tokens):
                if t == 0:
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=cur_input_ids,
                        attention_mask=cur_attention_mask,
                        image_flags=image_flags,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    outputs = language_model(
                        input_ids=last_tokens.unsqueeze(1),
                        attention_mask=cur_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError("Model did not return past_key_values during cached decoding")
                past_key_values = _cast_past_key_values(past_key_values)

                next_logits = outputs.logits[:, -1, :]
                if temperature > 0:
                    next_logits = next_logits / temperature
                probs = torch.softmax(next_logits.float(), dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                row_sums = probs.sum(dim=-1, keepdim=True)
                invalid_rows = row_sums.squeeze(1) <= 0
                if invalid_rows.any():
                    fallback = torch.argmax(next_logits, dim=-1)
                    probs[invalid_rows] = 0.0
                    probs[invalid_rows, fallback[invalid_rows]] = 1.0
                    row_sums = probs.sum(dim=-1, keepdim=True)
                probs = probs / row_sums.clamp_min(1e-8)

                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_tokens = torch.where(
                    finished,
                    torch.full_like(next_tokens, pad_id),
                    next_tokens,
                )
                completion_ids[:, t] = next_tokens

                newly_finished = next_tokens == eos_id
                finished = finished | newly_finished
                if finished.all():
                    break

                last_tokens = next_tokens
                cur_input_ids = torch.cat([cur_input_ids, next_tokens.unsqueeze(1)], dim=1)
                cur_attention_mask = torch.cat(
                    [
                        cur_attention_mask,
                        torch.ones((bsz, 1), dtype=cur_attention_mask.dtype, device=device),
                    ],
                    dim=1,
                )
        finally:
            for cfg, old_use_cache in cache_configs:
                cfg.use_cache = old_use_cache
            if was_training:
                model.train()

        return completion_ids

    def _generate_and_score_completions(self, inputs, model):
        device = self.accelerator.device

        prompt_ids    = inputs["input_ids"]        # (B, P)
        prompt_mask   = inputs["attention_mask"]   # (B, P)
        pixel_values  = inputs["pixel_values"].to(device)
        image_flags   = inputs["image_flags"].to(device)

        eos_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id or eos_id

        prompt_length = prompt_ids.size(1)
        completion_ids = self._sample_completion_ids(
            model=model,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            pixel_values=pixel_values,
            image_flags=image_flags,
            eos_id=eos_id,
            pad_id=pad_id,
        )

        # completion mask: 1 up to and including first EOS
        is_eos   = completion_ids == eos_id
        B, C     = completion_ids.shape
        eos_idx  = torch.full((B,), C, dtype=torch.long, device=device)
        has_eos  = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        seq_idx  = torch.arange(C, device=device).expand(B, -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        full_ids   = torch.cat([prompt_ids,   completion_ids],   dim=1)
        full_mask  = torch.cat([prompt_mask,  completion_mask],  dim=1)

        # old & ref log-probs
        with torch.no_grad():
            old_logps = (
                self._get_per_token_logps(model, full_ids, full_mask, pixel_values, image_flags)
                [:, prompt_length - 1:]
                if self.num_iterations > 1
                else None
            )

            if self.beta == 0.0 or self.ref_model is None:
                ref_logps = None
            else:
                ref_logps = (
                    self._get_per_token_logps(
                        self.ref_model, full_ids, full_mask, pixel_values, image_flags
                    )[:, prompt_length - 1:]
                )

        # decode
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # rewards
        rewards_per_func = torch.zeros(B, len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            rw_kwargs = {
                k: v for k, v in inputs.items()
                if k not in ("input_ids", "attention_mask", "pixel_values",
                             "image_flags", "prompt_text")
            }
            prompts_text = inputs.get("prompt_text", [""] * B)
            out = reward_func(prompts=prompts_text, completions=completions, **rw_kwargs)
            rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        dump_path = os.environ.get("DEBUG_GRPO_DUMP_COMPLETIONS", "").strip()
        if dump_path and self.accelerator.is_main_process:
            dump_limit = int(os.environ.get("DEBUG_GRPO_DUMP_LIMIT", "8"))
            remaining = max(dump_limit - self._debug_dumped_completions, 0)
            if remaining:
                os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
                reward_values = rewards_per_func.detach().float().cpu().tolist()
                lengths = completion_mask.sum(1).detach().cpu().tolist()
                prompts_text = inputs.get("prompt_text", [""] * B)
                gt_counts = inputs.get("gt_count", inputs.get("ground_truth_count", [None] * B))
                schemas = inputs.get("response_schema", [""] * B)
                with open(dump_path, "a", encoding="utf-8") as handle:
                    for row_idx in range(min(B, remaining)):
                        record = {
                            "step": int(self.state.global_step),
                            "row": row_idx,
                            "prompt": prompts_text[row_idx] if row_idx < len(prompts_text) else "",
                            "gt_count": gt_counts[row_idx] if row_idx < len(gt_counts) else None,
                            "response_schema": schemas[row_idx] if row_idx < len(schemas) else "",
                            "completion_length": int(lengths[row_idx]),
                            "rewards": reward_values[row_idx],
                            "completion": completions[row_idx],
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        self._debug_dumped_completions += 1

        rewards_per_func = self.accelerator.gather(rewards_per_func)
        rewards = rewards_per_func.sum(dim=1)

        mean_r = rewards.view(-1, self.num_generations).mean(dim=1)
        std_r  = rewards.view(-1, self.num_generations).std(dim=1)
        mean_r = mean_r.repeat_interleave(self.num_generations, dim=0)
        std_r  = std_r.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        proc_slice = slice(
            self.accelerator.process_index * B,
            (self.accelerator.process_index + 1) * B,
        )
        advantages = advantages[proc_slice]

        # metrics
        clen = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(clen)
        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_r).mean().item()
        )

        return {
            "prompt_ids":          prompt_ids,
            "prompt_mask":         prompt_mask,
            "completion_ids":      completion_ids,
            "completion_mask":     completion_mask,
            "old_per_token_logps": old_logps,
            "ref_per_token_logps": ref_logps,
            "advantages":          advantages,
            "pixel_values":        pixel_values,
            "image_flags":         image_flags,
        }

    # ------------------------------------------------------------------ #
    # GRPO loss                                                            #
    # ------------------------------------------------------------------ #

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("InternVL2GRPOTrainer does not support return_outputs=True")

        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        pixel_values    = inputs["pixel_values"]
        image_flags     = inputs["image_flags"]
        advantages      = inputs["advantages"]

        input_ids        = torch.cat([prompt_ids,  completion_ids],  dim=1)
        attention_mask   = torch.cat([prompt_mask, completion_mask], dim=1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, pixel_values, image_flags
        )
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        old_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )

        coef_1 = torch.exp(per_token_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss = -torch.min(
            coef_1 * advantages.unsqueeze(1),
            coef_2 * advantages.unsqueeze(1),
        )

        if self.beta > 0 and inputs["ref_per_token_logps"] is not None:
            ref_logps     = inputs["ref_per_token_logps"]
            per_token_kl  = (
                torch.exp(ref_logps - per_token_logps)
                - (ref_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (
                (per_token_kl * completion_mask).sum(1)
                / completion_mask.sum(1).clamp(min=1e-5)  # Prevent divide by zero
            ).mean()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        loss = (
            (per_token_loss * completion_mask).sum(1)
            / completion_mask.sum(1).clamp(min=1e-5)
        ).mean()

        is_clipped = (
            coef_1 * advantages.unsqueeze(1) < coef_2 * advantages.unsqueeze(1)
        ).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1e-5)
        self._metrics["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def log(self, logs, start_time=None):
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items()}
        logs = {**logs, **metrics}
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_path, processor_path=None, attn_impl="eager"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load image processor from processor_path (base model) if the checkpoint
    # does not contain a preprocessor_config.json
    proc_src = processor_path or model_path
    try:
        processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)
        image_processor = getattr(processor, "image_processor", None)
    except Exception:
        image_processor = None
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(
            proc_src, trust_remote_code=True
        )

    # If Stage-1 checkpoint exposes a PEFT adapter, load base model from proc_src
    # and merge adapter weights before GRPO. Loading model_path directly can leave
    # most language weights randomly initialized due key mismatches.
    adapter_dir = None
    root_adapter = pathlib.Path(model_path) / "native_peft_adapter"
    if root_adapter.is_dir():
        adapter_dir = root_adapter
    else:
        ckpt_adapters = sorted(
            pathlib.Path(model_path).glob("checkpoint-*/native_peft_adapter"),
            key=lambda p: int(p.parent.name.split("-")[-1]),
        )
        if ckpt_adapters:
            adapter_dir = ckpt_adapters[-1]

    embedded_lora = adapter_dir is None and _checkpoint_has_embedded_lora(model_path)
    model_load_path = proc_src if (adapter_dir is not None or embedded_lora) else model_path
    if embedded_lora:
        rank0_print(
            "Detected embedded LoRA/full-checkpoint safetensors; "
            f"initializing base model from {model_load_path} and merging weights from {model_path}"
        )

    try:
        model = AutoModel.from_pretrained(
            model_load_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except Exception as exc:
        rank0_print(f"attn_impl={attn_impl} load failed ({exc!r}); fallback to eager")
        model = AutoModel.from_pretrained(
            model_load_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="eager",
        )

    if embedded_lora:
        injected = inject_full_checkpoint_backbone_from_safetensors(model, model_path)
        if injected < 100:
            raise RuntimeError(
                f"Embedded checkpoint load injected only {injected} tensors from {model_path}; "
                "refusing to train from a partially initialized model."
            )
        merged = merge_lora_deltas_from_safetensors(
            model, model_path, lora_rank=64, lora_alpha=128.0
        )
        if merged == 0:
            raise RuntimeError(f"Embedded checkpoint load found no LoRA deltas in {model_path}")

    if adapter_dir is not None:
        rank0_print(f"Merging Stage-1 adapter from: {adapter_dir}")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            str(adapter_dir),
            is_trainable=False,
        )
        model.language_model = model.language_model.merge_and_unload()
        if hasattr(model.language_model, "peft_config"):
            delattr(model.language_model, "peft_config")

    model.config.use_cache = False
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if model.img_context_token_id is None or model.img_context_token_id < 0:
        raise RuntimeError(f"Could not resolve token id for {IMG_CONTEXT_TOKEN}")

    num_image_token = int(getattr(model, "num_image_token", 256))
    rank0_print(f"num_image_token={num_image_token}, "
                f"img_context_token_id={model.img_context_token_id}")
    return model, tokenizer, image_processor, num_image_token


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--processor_name_or_path", default=None,
                   help="Path to base model with preprocessor_config.json "
                        "(defaults to model_name_or_path)")
    p.add_argument("--data_path",          required=True)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--learning_rate",      type=float, default=2e-6)
    p.add_argument("--beta",               type=float, default=0.05)
    p.add_argument("--num_generations",    type=int,   default=8)
    p.add_argument("--max_prompt_length",  type=int,   default=512)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs",   type=int,   default=5)
    p.add_argument("--max_steps",          type=int,   default=-1)
    p.add_argument("--num_iterations",     type=int,   default=2)
    p.add_argument("--warmup_ratio",       type=float, default=0.05)
    p.add_argument("--max_grad_norm",      type=float, default=0.5)
    p.add_argument("--logging_steps",      type=int,   default=10)
    p.add_argument("--save_steps",         type=int,   default=200)
    p.add_argument("--save_total_limit",   type=int,   default=3)
    p.add_argument("--lora_rank",          type=int,   default=64)
    p.add_argument("--lora_alpha",         type=int,   default=128)
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--bf16",              action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rank0_print("=== InternVL2 GRPO Stage-2 ===")
    rank0_print(f"model : {args.model_name_or_path}")
    rank0_print(f"data  : {args.data_path}")
    rank0_print(f"output: {args.output_dir}")

    model, tokenizer, image_processor, num_image_token = _load_model_and_tokenizer(
        args.model_name_or_path,
        processor_path=args.processor_name_or_path,
        attn_impl=args.attn_implementation,
    )

    # Freeze vision encoder
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # Unfreeze projector to physically bridge gradients from visual features
    # into the language model inputs.
    for param in model.mlp1.parameters():
        param.requires_grad = True

    rank0_print("Loading Stage 1 SFT LoRA weights...")
    adapter_path = os.path.join(args.model_name_or_path, "native_peft_adapter")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}. Run audit script first to materialize it.")

    # Attach the Stage 1 weights and keep them trainable for Stage 2
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        adapter_path,
        is_trainable=True
    )
    rank0_print("Successfully loaded SFT adapter for RL tuning.")

    # THE GRADIENT ANCHOR: Unfreeze text embeddings for text-only inputs.
    model.language_model.get_input_embeddings().weight.requires_grad_(True)
    rank0_print("Unfroze language_model token embeddings for gradient anchoring.")

    # Reference model (frozen copy of original policy before PEFT)
    ref_model = create_reference_model(model)

    train_dataset = InternVLGRPODataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_image_token=num_image_token,
        max_prompt_length=args.max_prompt_length,
    )

    reward_fn = _build_reward_fn()

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_iterations=args.num_iterations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "preserve_rng_state": True,
        },
    )

    trainer = InternVL2GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        data_collator=GRPOCollator(tokenizer=tokenizer),
        num_generations=args.num_generations,
        epsilon=0.2,
        num_iterations=args.num_iterations,
    )

    rank0_print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    rank0_print("Done.")


if __name__ == "__main__":
    main()
