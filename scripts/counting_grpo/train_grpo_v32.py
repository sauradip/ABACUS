#!/usr/bin/env python3
"""Stage 3.2 GRPO trainer (v32).

Uses the existing InternVL2 GRPO trainer core and plugs in:
- curriculum schema fields (response_schema)
- v32 reward script loading (grpo_reward_v32.py)
- group rollouts G=8 with group-average advantage (handled by InternVL2GRPOTrainer)
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from trl.models import create_reference_model
from trl.trainer.grpo_config import GRPOConfig


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.train_internvl_grpo import (
    GRPOCollator,
    InternVL2GRPOTrainer,
    _load_model_and_tokenizer,
    _tokenize_prompt,
    rank0_print,
)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")


def _load_reward_function(script_path: str):
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Reward script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("grpo_reward_v32_module", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    fn = getattr(module, "reward_function", None)
    if fn is None:
        raise RuntimeError(f"reward_function(...) not found in {script_path}")
    return fn


class CurriculumGRPODataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, image_processor, num_image_token, max_prompt_length=6144):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_image_token = num_image_token
        self.max_prompt_length = max_prompt_length

        with open(data_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        rank0_print(f"Loaded curriculum rows: {len(self.records)} from {data_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]

        image_path = item.get("pca_image") or item.get("image")
        has_image = bool(image_path)
        if has_image:
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (448, 448), (255, 255, 255))
            pixel_values = self.image_processor.preprocess([img], return_tensors="pt")["pixel_values"][0]
        else:
            pixel_values = torch.zeros((3, 448, 448), dtype=torch.float32)

        convs = item.get("conversations", [])
        prompt_ids = _tokenize_prompt(convs, self.tokenizer, self.num_image_token)
        if prompt_ids.size(0) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:]

        gt_count = int(item.get("gt_count", item.get("ground_truth_count", 0)))

        return {
            "prompt_ids": prompt_ids,
            "pixel_values": pixel_values,
            "has_image": int(has_image),
            "ground_truth_count": gt_count,
            "gt_count": gt_count,
            "response_schema": item.get("response_schema", "full"),
            "target_response": item.get("target_response", ""),
            "normalized_points_1000": item.get("normalized_points_1000", []),
            "solution": item.get("solution", ""),
            "prompt_text": item.get("problem", ""),
        }


class CurriculumGRPOCollator(GRPOCollator):
    def __call__(self, instances):
        batch = super().__call__(instances)
        batch["gt_count"] = [inst["gt_count"] for inst in instances]
        batch["response_schema"] = [inst["response_schema"] for inst in instances]
        batch["target_response"] = [inst["target_response"] for inst in instances]
        return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--processor_name_or_path", default="OpenGVLab/InternVL2-2B")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--reward_script", required=True)
    p.add_argument("--output_dir", default="checkpoints/stage32_grpo_r1rex_tally")
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=int(os.getenv("MAX_PROMPT_LEN", "6144")))
    p.add_argument("--max_completion_length", type=int, default=int(os.getenv("MAX_GEN_LEN", "2048")))
    p.add_argument("--per_device_train_batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "1")))
    p.add_argument("--gradient_accumulation_steps", type=int, default=int(os.getenv("GRAD_ACCUM", "16")))
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_iterations", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--bf16", type=_str2bool, nargs="?", const=True, default=True)
    p.add_argument("--allow_attn_fallback", action="store_true")
    return p.parse_args()


def _ensure_flash_attn(model, strict=True):
    candidates = [
        getattr(model.config, "_attn_implementation", None),
        getattr(model.config, "attn_implementation", None),
    ]
    active = next((x for x in candidates if isinstance(x, str) and x), "unknown")
    rank0_print(f"Active attention backend: {active}")
    if strict and "flash" not in active.lower():
        raise RuntimeError(
            f"FlashAttention2 required for Stage 3.2 but backend is '{active}'. "
            "Install/enable flash_attn or run with --allow_attn_fallback."
        )


def main():
    args = parse_args()

    rank0_print("=== Stage 3.2 GRPO v32 ===")
    rank0_print(f"Model: {args.model_name_or_path}")
    rank0_print(f"Data : {args.dataset_name}")
    rank0_print(f"Out  : {args.output_dir}")
    rank0_print(f"G    : {args.num_generations}")

    model, tokenizer, image_processor, num_image_token = _load_model_and_tokenizer(
        args.model_name_or_path,
        processor_path=args.processor_name_or_path,
        attn_impl=args.attn_implementation,
    )
    _ensure_flash_attn(model, strict=not args.allow_attn_fallback)

    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.mlp1.parameters():
        p.requires_grad = True

    # If a trainable native adapter exists, use it. Otherwise keep LM trainable as loaded.
    adapter_path = os.path.join(args.model_name_or_path, "native_peft_adapter")
    if os.path.exists(adapter_path):
        rank0_print(f"Loading trainable adapter from {adapter_path}")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            adapter_path,
            is_trainable=True,
        )

    model.language_model.get_input_embeddings().weight.requires_grad_(True)

    ref_model = create_reference_model(model)

    reward_fn = _load_reward_function(args.reward_script)

    train_dataset = CurriculumGRPODataset(
        data_path=args.dataset_name,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_image_token=num_image_token,
        max_prompt_length=args.max_prompt_length,
    )

    cfg = GRPOConfig(
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
        gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True},
    )

    trainer = InternVL2GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=train_dataset,
        data_collator=CurriculumGRPOCollator(tokenizer=tokenizer),
        num_generations=args.num_generations,
        epsilon=0.2,
        num_iterations=args.num_iterations,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    rank0_print("Stage 3.2 GRPO complete.")


if __name__ == "__main__":
    main()
