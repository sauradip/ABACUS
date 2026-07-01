#!/usr/bin/env python3
"""Pretext Task Training: Common Object Localization Across Image Pairs.

Trains UniLIP-3B to localize common objects between two related images.
Uses unicount_pretext dataset (33K image pairs with correspondence annotations).

Data format (HuggingFace dataset):
  {
    "image_1": <image>,
    "image_2": <image>,
    "common_points": [
      [x, y], ...  # Pixel coordinates of common objects
    ],
    "count": int,  # Number of common objects
    "H": int,      # Image height
    "W": int,      # Image width
    "question": "Localize common instances shared by two images"
  }

Output: Prepares encoder for counting task with spatial reasoning capability.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNILIP_MOD_PATH = REPO_ROOT / "UniLip_mod"
if str(UNILIP_MOD_PATH) not in sys.path:
    sys.path.insert(0, str(UNILIP_MOD_PATH))

from scripts.experiment_lora_counting_sft.train_lora_counting_sft import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    rank0_print,
    smart_tokenizer_resize,
    find_base_weights,
    md5_prefix,
    preprocess_multimodal,
    preprocess_internvl,
    SFTDataCollator,
    BucketBalancedSampler,
    IGNORE_INDEX,
    IMG_CONTEXT_TOKEN_ID,
)
from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)


@dataclass
class PreTextDataArguments(DataArguments):
    """Arguments for pretext task training."""
    dataset_name: str = field(
        default="unicount_pretext",
        metadata={"help": "HuggingFace dataset name or local path"}
    )


@dataclass
class PreTextTrainingArguments(TrainingArguments):
    """Extended training arguments for WandB support."""
    project_name: str = field(
        default="pretext_dual_loss_pipeline",
        metadata={"help": "WandB project name"}
    )
    run_name: str = field(
        default="phase1_pretext",
        metadata={"help": "WandB run name"}
    )


def load_pretext_dataset(cache_dir, split="train"):
    """Load unicount_pretext dataset from cache directory."""
    rank0_print(f"Loading pretext dataset from: {cache_dir}")

    # Check if it's a parquet-based dataset (data/ subdirectory)
    data_dir = f"{cache_dir}/data"
    import os
    if os.path.exists(data_dir) and any(f.endswith('.parquet') for f in os.listdir(data_dir)):
        ds = load_dataset("parquet", data_dir=data_dir, split=split)
    else:
        # Try imagefolder as fallback
        ds = load_dataset("imagefolder", data_dir=cache_dir, split=split)

    rank0_print(f"Loaded {len(ds)} pretext examples")
    return ds


class PreTextDataset(Dataset):
    """Dataset for common object localization pretext task."""

    def __init__(self, data, tokenizer, image_processor, model_args, data_args):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_args = model_args
        self.data_args = data_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load both images (handle PIL Image, bytes, or dict with bytes)
        image_1 = item["image_1"]
        image_2 = item["image_2"]

        # Check if it's a dict with "bytes" key (from older format)
        if isinstance(image_1, dict) and "bytes" in image_1:
            image_1 = image_1["bytes"]

        if isinstance(image_2, dict) and "bytes" in image_2:
            image_2 = image_2["bytes"]

        # Convert bytes to PIL Image if needed
        if isinstance(image_1, bytes):
            image_1 = Image.open(io.BytesIO(image_1)).convert("RGB")
        if isinstance(image_2, bytes):
            image_2 = Image.open(io.BytesIO(image_2)).convert("RGB")

        # If still not a PIL Image, assume it's already one
        if not isinstance(image_1, Image.Image):
            image_1 = image_1.convert("RGB") if hasattr(image_1, "convert") else image_1

        if not isinstance(image_2, Image.Image):
            image_2 = image_2.convert("RGB") if hasattr(image_2, "convert") else image_2

        # Process first image
        image_tensor_1 = self.image_processor(
            image_1, return_tensors="pt", do_resize=True, size={"height": 448, "width": 448}
        )["pixel_values"].squeeze(0)

        # Process second image (stacked as multi-image input)
        image_tensor_2 = self.image_processor(
            image_2, return_tensors="pt", do_resize=True, size={"height": 448, "width": 448}
        )["pixel_values"].squeeze(0)

        # Use only image_1 as visual input; conversation has a single <image> token
        pixel_values = image_tensor_1

        # Create conversation with question
        question = item.get("question", "Localize common instances shared by the two images. Return [x, y] locations.")
        count = item.get("count", 0)

        # Format response with common points
        common_points = item.get("common_points", [])
        response_str = f"The two images share {count} common objects at these locations: {common_points}"

        conv_sources = [
            {"from": "system", "value": "You are an object localization assistant."},
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": response_str},
        ]

        # Tokenize
        has_image = True
        prep = preprocess_internvl(
            [copy.deepcopy(conv_sources)],
            self.tokenizer,
            has_image=has_image,
        )

        return dict(
            input_ids=prep["input_ids"][0],
            labels=prep["labels"][0],
            pixel_values=pixel_values,
        )


class PreTextTrainer(Trainer):
    """Trainer for pretext task (no attention regularization)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Standard CE loss only (no AR loss in pretext phase)."""
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            und_image=inputs.get("pixel_values"),
        )

        logits = outputs.logits
        labels = inputs.get("labels")

        # Compute CE loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="mean",
        )

        return (ce_loss, outputs) if return_outputs else ce_loss


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, PreTextDataArguments, PreTextTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup WandB if not disabled
    if training_args.report_to and "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=training_args.project_name,
            name=training_args.run_name,
            config={
                "dataset": data_args.dataset_name,
                "base_model": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "num_epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "lora_r": model_args.lora_rank,
                "lora_alpha": model_args.lora_alpha,
            }
        )

    # LoRA setup (same as dual-loss)
    from peft import get_peft_config, get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType

    model_cls = load_unilip_class()
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(module, inp, out):
                out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    tokenizer = AutoProcessor.from_pretrained(model_args.mllm_hf_path, trust_remote_code=True).tokenizer
    image_processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path, trust_remote_code=True).image_processor

    # Load dataset
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(training_args.cache_dir)
    except:
        dataset = load_pretext_dataset(training_args.cache_dir, split="train")

    # Create PyTorch dataset
    train_dataset = PreTextDataset(
        dataset, tokenizer, image_processor, model_args, data_args
    )

    # Data collator
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    # Trainer
    trainer = PreTextTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save final checkpoint
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))

    # Close WandB run
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.finish()


if __name__ == "__main__":
    main()
