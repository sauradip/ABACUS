#!/usr/bin/env python3
"""Stage 3.2.1 HF-native multi-image UniLIP GRPO trainer."""

import argparse
import copy
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from torch import nn
from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import (
    broadcast_object_list,
    gather,
    gather_object,
    is_conversational,
    maybe_apply_chat_template,
    pad,
    profiling_context,
    selective_log_softmax,
    unwrap_model_for_generation,
)

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    IGNORE_INDEX,
    embed_tokens,
    expand_unilip_image_context,
    load_model,
    load_pil_images,
    load_processor,
    load_unilip_constants,
    module_dtype,
)


def precision_counting_reward(prompts, completions, gt_count, **kwargs):
    """
    Advanced Reward Function.
    Features: Relaxed regex, Quadratic Error Penalty, and Anti-Bucket laziness suppression.
    """
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

        base_accuracy = max(0.0, 1.0 - (error_ratio ** 2))

        precision_bonus = 0.0
        if gt >= 40:
            if error_ratio == 0:
                precision_bonus = 1.0
            elif error_ratio <= 0.05:
                precision_bonus = 0.5
            elif error_ratio <= 0.10:
                precision_bonus = 0.2

        bucket_penalty = 0.0
        if pred_count in lazy_buckets and error_ratio > 0.10:
            bucket_penalty = -0.5

        if gt < 40 and error > 2:
            base_accuracy -= 0.1

        format_bonus = 0.1
        rewards.append(base_accuracy + precision_bonus + bucket_penalty + format_bonus)

    return rewards


class GRPOProcessor:
    """Tokenizer facade with the image processor attached for UniLIP GRPO."""

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


def validate_grpo_row(row: Dict[str, Any]) -> None:
    if "prompt" not in row or "gt_count" not in row:
        raise ValueError(f"GRPO row must contain prompt and gt_count keys: {row.keys()}")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
        raise ValueError(f"Row {row.get('id')} prompt must contain only one user message")
    content = prompt[0].get("content", [])
    images = [item for item in content if isinstance(item, dict) and item.get("type") == "image"]
    texts = [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if len(images) != 2 or len(texts) != 1:
        raise ValueError(f"Row {row.get('id')} must have exactly two image items and one text item")
    if "assistant" in json.dumps(prompt).lower():
        raise ValueError(f"Row {row.get('id')} prompt contains assistant text")


def left_pad_to_length(values: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    if values.shape[1] >= target_length:
        return values
    pad_width = target_length - values.shape[1]
    padding = torch.full(
        (values.shape[0], pad_width),
        pad_value,
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat([padding, values], dim=1)


class UniLIPMultiImageGRPOTrainer(GRPOTrainer):
    """GRPOTrainer variant that keeps the two HF image entries active."""

    def _get_train_sampler(self, *args: Any, **kwargs: Any):
        return super()._get_train_sampler()

    def _get_eval_sampler(self, *args: Any, **kwargs: Any):
        return super()._get_eval_sampler()

    def _prompt_batch(self, inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts_text = [
            expand_unilip_image_context(maybe_apply_chat_template(example, self.processing_class)["prompt"])
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
            pixel_batches.append(self.processing_class.image_processor.preprocess(images, return_tensors="pt")["pixel_values"])
        prompt_inputs["pixel_values"] = torch.cat(pixel_batches, dim=0)
        return Trainer._prepare_inputs(self, prompt_inputs)

    def _model_logits(
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
            image_embeds = model_module.model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=feature_layer,
                vision_feature_select_strategy=feature_strategy,
                image_sizes=None,
            )

            image_token_id = load_unilip_constants()["UND_IMAGE_TOKEN_IDX"]
            image_token_mask = input_ids == image_token_id
            flat_embeds = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)
            expected = int(image_token_mask.sum().item())
            if expected != int(flat_embeds.shape[0]):
                raise RuntimeError(
                    "Image token count does not match vision features: "
                    f"tokens={expected}, embeds={tuple(flat_embeds.shape)}, "
                    f"pixel_values={tuple(pixel_values.shape)}"
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
        return model_module.lm_head(outputs.last_hidden_state)

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values: Optional[torch.Tensor] = None,
    ):
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
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ if not isinstance(reward_func, nn.Module) else str(reward_func)
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

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

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
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

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            pixel_values=pixel_values,
        )

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=32)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--bf16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPL", "flash_attention_2"))
    parser.add_argument("--allow_attn_fallback", type=int, default=int(os.environ.get("ALLOW_ATTN_FALLBACK", "0")))
    parser.add_argument("--trust_remote_code", type=int, default=1)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--init_only", action="store_true")
    return parser.parse_args()


def set_local_cuda_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def main() -> None:
    args = parse_args()
    set_local_cuda_device()

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    for row in dataset.select(range(min(8, len(dataset)))):
        validate_grpo_row(row)

    processor = GRPOProcessor(load_processor(args))
    model = load_model(args)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
        model.generation_config.pad_token_id = processor.pad_token_id
        model.generation_config.eos_token_id = processor.eos_token_id

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=args.report_to,
        use_vllm=False,
    )

    trainer = UniLIPMultiImageGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[precision_counting_reward],
        args=training_args,
        train_dataset=dataset,
    )
    if args.init_only:
        print("GRPO init-only preflight passed.")
        return

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
