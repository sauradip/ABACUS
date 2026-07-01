"""
Native Stage-1 SFT trainer for InternVL2-2B without UniLIP wrappers.

This script intentionally avoids custom compute_loss logic and lets the native
InternVLChatModel forward() return the causal LM loss.
"""

import copy
import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer, Trainer

from peft import LoraConfig, TaskType, get_peft_model

IGNORE_INDEX = -100
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
DEFAULT_IMAGE_TOKEN = "<image>"
TARGET_CONTEXT_LENGTH = 8192

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="OpenGVLab/InternVL2-2B")
    attn_implementation: str = field(default="sdpa")
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    data_path: str = field(default="")
    image_processor: Optional[object] = field(default=None)
    num_image_token: int = field(default=256)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=TARGET_CONTEXT_LENGTH)
    remove_unused_columns: bool = field(default=False)


def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)


def _load_records(data_path):
    with open(data_path) as handle:
        first_char = handle.read(1)
        handle.seek(0)
        if first_char == "[":
            return json.load(handle)
        records = []
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def preprocess_multimodal(sources, num_image_token):
    image_placeholder = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}"
    for source in sources:
        for sentence in source:
            if sentence["from"] == "human" and DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, image_placeholder).strip()
            elif sentence["from"] == "gpt" and DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
    return sources


def preprocess_internvl(sources, tokenizer):
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\n' }}"
        "{% elif content['type'] == 'text' %}{{ content['text'] }}"
        "{% endif %}{% endfor %}{% endif %}"
        "{{'<|im_end|>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []
    for source in sources:
        if roles.get(source[0]["from"]) != roles["human"] and source[0]["from"] != "system":
            source = source[1:]

        if source[0].get("from", source[0].get("role")) != "system":
            source = [{
                "from": "system",
                "value": (
                    "You are a grounded counting assistant. "
                    "Respond using the exact Thought -> Scaffold -> Count format."
                ),
            }] + list(source)

        sample_ids, sample_labels = [], []
        for conv in source:
            role = conv.get("role", conv.get("from"))
            content = conv.get("content", conv.get("value"))
            role = roles.get(role, role)
            encoded = tokenizer.apply_chat_template([{"role": role, "content": content}])
            sample_ids += encoded
            if role in ["user", "system"]:
                sample_labels += [IGNORE_INDEX] * len(encoded)
            else:
                sample_labels += encoded

        max_len = getattr(tokenizer, "model_max_length", None)
        if isinstance(max_len, int) and max_len > 0 and len(sample_ids) > max_len:
            sample_ids = sample_ids[:max_len]
            sample_labels = sample_labels[:max_len]

        input_ids.append(torch.tensor(sample_ids, dtype=torch.long))
        targets.append(torch.tensor(sample_labels, dtype=torch.long))

    return {"input_ids": input_ids, "labels": targets}


class NativeSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print(f"Loading native SFT data from {data_path}...")
        self.data = _load_records(data_path)
        rank0_print(f"Loaded {len(self.data)} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        has_image = "image" in item and item["image"] is not None

        if has_image:
            try:
                img = Image.open(item["image"]).convert("RGB")
            except Exception as exc:
                rank0_print(f"Error opening {item['image']}: {exc}")
                img = Image.new("RGB", (448, 448), (255, 255, 255))

            processor = self.data_args.image_processor
            pixel_values = processor.preprocess([img], return_tensors="pt")["pixel_values"][0]
            conv_sources = preprocess_multimodal(copy.deepcopy([item["conversations"]]), self.data_args.num_image_token)
        else:
            pixel_values = torch.zeros((3, 448, 448), dtype=torch.float32)
            conv_sources = copy.deepcopy([item["conversations"]])

        prep = preprocess_internvl(conv_sources, self.tokenizer)

        return {
            "input_ids": prep["input_ids"][0],
            "labels": prep["labels"][0],
            "pixel_values": pixel_values,
            "has_image": 1 if has_image else 0,
        }


@dataclass
class NativeDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        max_len = self.tokenizer.model_max_length
        input_ids = [inst["input_ids"][:max_len] for inst in instances]
        labels = [inst["labels"][:max_len] for inst in instances]

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

        pixel_values = torch.stack([inst["pixel_values"] for inst in instances]).to(torch.bfloat16)
        image_flags = torch.tensor([[inst["has_image"]] for inst in instances], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }


def _resolve_targets_for_lora(language_model):
    suffixes = {
        name.split(".")[-1]
        for name, module in language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    wanted = ["wqkv", "wo", "w1", "w2", "w3"]
    resolved = [name for name in wanted if name in suffixes]
    if len(resolved) < len(wanted):
        rank0_print(f"WARN: partial LoRA target match. Resolved={resolved}")
    if not resolved:
        raise RuntimeError(f"Could not resolve InternLM2 LoRA targets from suffix set sample={sorted(list(suffixes))[:30]}")
    return resolved


def _load_native_model(model_args, training_args):
    # First try requested implementation; if unsupported by architecture, fall back to eager.
    attn_impl = model_args.attn_implementation
    try:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            cache_dir=training_args.cache_dir,
        )
        return model, attn_impl
    except ValueError as exc:
        if "does not support" in str(exc) and attn_impl == "sdpa":
            rank0_print("SDPA unsupported by native model class; retrying with eager attention.")
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
                cache_dir=training_args.cache_dir,
            )
            return model, "eager"
        raise


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.bf16:
        raise RuntimeError("Native SFT requires --bf16 True")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right",
        model_max_length=max(training_args.model_max_length, TARGET_CONTEXT_LENGTH),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )

    model, effective_attn = _load_native_model(model_args, training_args)
    rank0_print(f"Loaded native model with attention implementation: {effective_attn}")

    # Freeze vision branch and mm projector before LoRA.
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.mlp1.parameters():
        param.requires_grad = False

    lora_targets = _resolve_targets_for_lora(model.language_model)
    lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)

    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()

    model.config.use_cache = False
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if model.img_context_token_id is None or model.img_context_token_id < 0:
        raise RuntimeError(f"Could not resolve required token id for {IMG_CONTEXT_TOKEN}")

    data_args.image_processor = image_processor
    data_args.num_image_token = int(getattr(model, "num_image_token", data_args.num_image_token))

    rank0_print(f"num_image_token={data_args.num_image_token}, img_context_token_id={model.img_context_token_id}")

    train_dataset = NativeSFTDataset(data_args.data_path, tokenizer, data_args)
    data_collator = NativeDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_valid_checkpoint = any((checkpoint_dir / "trainer_state.json").exists() for checkpoint_dir in checkpoint_dirs)
    trainer.train(resume_from_checkpoint=True if has_valid_checkpoint else None)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
