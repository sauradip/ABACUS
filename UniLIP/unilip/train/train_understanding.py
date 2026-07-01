"""
Standalone SFT training script for UniLIP understanding (VQA counting).

This script bypasses the TI2I-centric forward() path in unilip_internvl.py
and instead directly:
  1. Loads images, tokenizes conversations
  2. Embeds tokens + vision features using the model internals
  3. Computes standard cross-entropy loss on the language model output
"""
import os
import sys
import json
import copy
import random
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor, Trainer
from torch.utils.data import Dataset
from PIL import Image

# UniLIP imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
from unilip.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    UND_IMAGE_TOKEN_IDX, IMAGE_TOKEN_IDX,
)
from unilip import conversation as conversation_lib

# These are defined inline in train_stage3.py, not in constants.py
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

logger = logging.getLogger(__name__)


def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)


# ─── Arguments ────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")
    mllm_path: str = field(default="")
    mllm_hf_path: str = field(default="")
    version: str = field(default="internvl")
    vision_tower: str = field(default="")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_projector_type: str = field(default="linear")
    unilip_factor: float = field(default=10.6)
    n_query: int = field(default=256)
    n_und_query: int = field(default=256)
    fix_vit: bool = field(default=True)
    fix_llm: bool = field(default=False)
    fix_dit: bool = field(default=True)
    fix_connect: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default="")
    data_type: str = field(default="mix")
    image_aspect_ratio: str = field(default="pad")
    is_multimodal: bool = field(default=True)
    image_processor: Optional[object] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=1024)
    mm_projector_lr: Optional[float] = field(default=None)
    freeze_mm_mlp_adapter: bool = field(default=False)
    pretrain_path: str = field(default="none")
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    remove_unused_columns: bool = field(default=False)


# ─── Tokenization ────────────────────────────────────────────────────────────

def preprocess_multimodal(sources, data_args):
    """Replace <image> tokens with the actual image context placeholder."""
    und_placeholder = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
    inst_type = None
    for source in sources:
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                inst_type = "gen"
    return sources, inst_type


def preprocess_internvl(sources, tokenizer, has_image=False):
    """Tokenize using InternVL chat template and mask human turns."""
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

        # Inject default English system prompt
        if source[0].get("from", source[0].get("role")) != "system":
            source = [{"from": "system", "value": "You are a helpful counting assistant. Answer with only a number."}] + list(source)

        input_id, target = [], []
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            role = roles.get(role, role)
            conv_msg = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv_msg)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target)
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    # Kept all tokens in V2 to ensure EOS detection
    return dict(input_ids=input_ids, labels=targets)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class UnderstandingSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print(f"Loading data from {data_path}...")
        with open(data_path) as f:
            self.data = json.load(f)
        rank0_print(f"Loaded {len(self.data)} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        has_image = "image" in item and item["image"] is not None

        if has_image:
            try:
                img = Image.open(item["image"]).convert("RGB")
            except Exception as e:
                print(f"Error opening {item['image']}: {e}")
                img = Image.new("RGB", (448, 448), (255, 255, 255))

            processor = self.data_args.image_processor
            pixel_values = processor.preprocess([img], return_tensors="pt")["pixel_values"][0]
            conv_sources, _ = preprocess_multimodal(
                copy.deepcopy([item["conversations"]]), self.data_args
            )
        else:
            pixel_values = None
            conv_sources = copy.deepcopy([item["conversations"]])

        prep = preprocess_internvl(conv_sources, self.tokenizer, has_image=has_image)
        return dict(
            input_ids=prep["input_ids"][0],
            labels=prep["labels"][0],
            pixel_values=pixel_values,
        )


# ─── Collator ─────────────────────────────────────────────────────────────────

@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"][:self.tokenizer.model_max_length] for inst in instances]
        labels = [inst["labels"][:self.tokenizer.model_max_length] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        pixel_list = [inst["pixel_values"] for inst in instances if inst["pixel_values"] is not None]
        if pixel_list:
            batch["pixel_values"] = torch.stack(pixel_list)
        else:
            batch["pixel_values"] = None

        return batch


# ─── Trainer ──────────────────────────────────────────────────────────────────

class UnderstandingTrainer(Trainer):
    """Custom trainer that computes CE loss on the understanding path only."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values", None)

        # DeepSpeedEngine has a .config dict that shadows the model's .config
        model_module = model.module if hasattr(model, "module") else model

        # Get token embeddings
        text_embeds = model_module.get_model().language_model.embed_tokens(input_ids)

        # If we have images, replace <IMG_CONTEXT> placeholders with vision features
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=model_module.vision_tower.dtype, device=model_module.device)
            vision_feature_layer = model_module.config.vision_feature_layer
            vision_feature_select_strategy = model_module.config.vision_feature_select_strategy
            with torch.no_grad():
                image_embeds = model_module.model.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_sizes=None,
                )
            # Replace UND_IMAGE_TOKEN_IDX positions with vision features
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            if und_image_idx.any():
                text_embeds = text_embeds.clone()
                text_embeds[und_image_idx] = image_embeds.to(text_embeds.device).flatten(0, 1)

        position_ids = torch.cumsum(attention_mask.int(), dim=1) - 1
        position_ids[position_ids < 0] = 0

        outputs = model_module.get_model().language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )

        # language_model returns hidden states, not CausalLM logits.
        logits = model_module.lm_head(outputs.last_hidden_state)

        # Standard causal LM cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        return (loss, outputs) if return_outputs else loss


# ─── Model setup helpers ─────────────────────────────────────────────────────

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


# ─── Main ─────────────────────────────────────────────────────────────────────

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    rank0_print(f"model_args: {model_args}")
    rank0_print(f"data_args: {data_args}")

    # Load model
    model = UniLIP_InternVLForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze LLM if requested
    if not model_args.fix_llm:
        for p in model.get_model().language_model.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True

    # Unfreeze ViT if requested
    if not model_args.fix_vit:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True
        for p in model.get_model().multi_modal_projector.parameters():
            p.requires_grad = True

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Tokenizer
    tokenizer = AutoProcessor.from_pretrained(model_args.mllm_hf_path, trust_remote_code=True).tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            dict(pad_token="<pad>", additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer, model,
        )
    elif "<image>" not in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer, model,
        )

    # Conversation format
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    rank0_print(f"Conversation format: {conversation_lib.default_conversation.version}")

    # Image processor
    data_args.image_processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path, trust_remote_code=True).image_processor
    # NOTE: Skip initialize_vision_modules() and initialize_vision_tokenizer() —
    # The UniLIP-1B checkpoint already has vision_tower, multi_modal_projector,
    # and all submodules embedded. Those init methods are for stage-by-stage
    # training from scratch and require args (unilip_path, vae_path, dit_path)
    # that don't apply to SFT.

    if training_args.pretrain_path != 'none':
        msg = model.load_state_dict(torch.load(training_args.pretrain_path), strict=False)
        rank0_print(f"Loaded pretrain: {training_args.pretrain_path}\n{msg}")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total params: {total:,}  Trainable: {trainable:,}")

    # Dataset
    train_dataset = UnderstandingSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    # Train
    trainer = UnderstandingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Only resume if a valid checkpoint with trainer state exists
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_valid_checkpoint = False
    if checkpoint_dirs:
        # Check if the latest checkpoint has trainer_state.json
        latest_checkpoint = sorted(checkpoint_dirs)[-1]
        if (latest_checkpoint / "trainer_state.json").exists():
            has_valid_checkpoint = True
    
    if has_valid_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # Save final model
    if trainer.args.should_save:
        state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        torch.save(state_dict, os.path.join(training_args.output_dir, "pytorch_model.bin"))
        model.config.save_pretrained(training_args.output_dir)
        rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
