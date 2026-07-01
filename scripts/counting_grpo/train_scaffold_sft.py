"""
Scaffold-R1 LoRA SFT on the UniLIP understanding branch.

This Stage 1 trainer enforces a clean-slate setup:
  - start from the raw pretrained InternVL base only
  - freeze the vision tower, latent queries, connector, and lm_head
  - apply LoRA only to Qwen-style language attention and MLP blocks
  - supervise the full assistant response: thought + scaffold + count

The training data is expected to come from fsc_to_scaffold.py JSONL output.
"""

import hashlib
import os
import sys
import json
import copy
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

import transformers
import transformers.utils.import_utils as _tuu
import transformers.training_args as _tta
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoProcessor, Trainer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from torch.utils.data import Dataset
from PIL import Image


_tuu.is_torch_bf16_gpu_available = lambda: True
_tta.is_torch_bf16_gpu_available = lambda: True


def _resolve_unilip_dir():
    candidates = [
        os.environ.get("UNILIP_DIR"),
        str(pathlib.Path(__file__).resolve().parents[2] / "UniLIP"),
        "/projects/u6bl/myprojects/UniLIP",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return candidates[-1]


UNILIP_DIR = _resolve_unilip_dir()
sys.path.insert(0, UNILIP_DIR)

from unilip.model.language_model.unilip_internvl import UniLIP_InternVLConfig, UniLIP_InternVLForCausalLM
from unilip.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    UND_IMAGE_TOKEN_IDX,
)
from unilip import conversation as conversation_lib


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
RUNTIME_UND_IMAGE_TOKEN_IDX = UND_IMAGE_TOKEN_IDX

logger = logging.getLogger(__name__)
LLM_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
INTERNLM2_TARGET_MODULES = [
    "wqkv",
    "wo",
    "w1",
    "w2",
    "w3",
]
FORBIDDEN_TRAINABLE_TOKENS = ["lm_head", "latent_queries", "q_former", "gen_head"]
TARGET_CONTEXT_LENGTH = 8192
LABEL_PROBE_ENABLED = os.environ.get("LABEL_PROBE", "0") == "1"
ISOLATION_BLIND_LLM = os.environ.get("ISOLATION_BLIND_LLM", "0") == "1"
ISOLATION_NATIVE_LLM = os.environ.get("ISOLATION_NATIVE_LLM", "0") == "1"
ISOLATION_DISABLE_IMAGE_TOKENS = (
    ISOLATION_BLIND_LLM
    or ISOLATION_NATIVE_LLM
    or os.environ.get("ISOLATION_DISABLE_IMAGE_TOKENS", "0") == "1"
)
ISOLATION_BYPASS_PIXELS = (
    ISOLATION_BLIND_LLM
    or ISOLATION_NATIVE_LLM
    or os.environ.get("ISOLATION_BYPASS_PIXELS", "0") == "1"
)


def resolve_und_image_token_idx(tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if token_id is None or token_id < 0:
        rank0_print(
            f"WARN: Tokenizer does not expose {IMG_CONTEXT_TOKEN}; "
            f"falling back to constant UND_IMAGE_TOKEN_IDX={UND_IMAGE_TOKEN_IDX}."
        )
        return UND_IMAGE_TOKEN_IDX
    return token_id


def rank0_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")
    mllm_hf_path: str = field(default="")
    version: str = field(default="internvl")
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    attn_implementation: str = field(default="sdpa")
    base_model_md5: str = field(default="")
    skip_md5_check: bool = field(default=False)
    fail_if_path_looks_finetuned: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default="")
    image_aspect_ratio: str = field(default="pad")
    is_multimodal: bool = field(default=True)
    image_processor: Optional[object] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=TARGET_CONTEXT_LENGTH)
    mm_projector_lr: Optional[float] = field(default=None)
    freeze_mm_mlp_adapter: bool = field(default=False)
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    remove_unused_columns: bool = field(default=False)


def preprocess_multimodal(sources, data_args):
    del data_args
    und_placeholder = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}"
    inst_type = None
    for source in sources:
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                if ISOLATION_DISABLE_IMAGE_TOKENS:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                else:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                inst_type = "gen"
    return sources, inst_type


def preprocess_internvl(sources, tokenizer, has_image=False):
    del has_image
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

        input_id, target = [], []
        for conv in source:
            role = conv.get("role", conv.get("from"))
            content = conv.get("content", conv.get("value"))
            role = roles.get(role, role)
            encode_id = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        max_len = getattr(tokenizer, "model_max_length", None)
        if isinstance(max_len, int) and max_len > 0 and len(input_id) > max_len:
            input_id = input_id[:max_len]
            target = target[:max_len]

        assert len(input_id) == len(target)
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return {"input_ids": input_ids, "labels": targets}


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


class ScaffoldSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        rank0_print(f"Loading scaffold data from {data_path}...")
        self.data = _load_records(data_path)
        rank0_print(f"Loaded {len(self.data)} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        has_image = "image" in item and item["image"] is not None

        if has_image and not ISOLATION_BYPASS_PIXELS:
            try:
                img = Image.open(item["image"]).convert("RGB")
            except Exception as exc:
                print(f"Error opening {item['image']}: {exc}")
                img = Image.new("RGB", (448, 448), (255, 255, 255))

            processor = self.data_args.image_processor
            pixel_values = processor.preprocess([img], return_tensors="pt")["pixel_values"][0]
        else:
            pixel_values = None

        if has_image:
            conv_sources, _ = preprocess_multimodal(copy.deepcopy([item["conversations"]]), self.data_args)
        else:
            conv_sources = copy.deepcopy([item["conversations"]])

        prep = preprocess_internvl(conv_sources, self.tokenizer, has_image=has_image)
        return {
            "input_ids": prep["input_ids"][0],
            "labels": prep["labels"][0],
            "pixel_values": pixel_values,
        }


@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        if LABEL_PROBE_ENABLED:
            for idx, instance in enumerate(instances):
                valid_tokens = int((instance["labels"] != IGNORE_INDEX).sum().item())
                rank0_print(
                    f"[DEBUG] sample={idx} Total tokens: {len(instance['labels'])}, "
                    f"Valid target tokens: {valid_tokens}"
                )
                if valid_tokens == 0:
                    rank0_print("[FATAL] All tokens are masked! Check string matching logic.")
                    rank0_print(self.tokenizer.decode(instance["input_ids"], skip_special_tokens=False))

        for instance in instances:
            if instance["pixel_values"] is None:
                continue
            # Vision features can only be injected when UND_IMAGE_TOKEN_IDX is present.
            if not torch.any(instance["input_ids"] == RUNTIME_UND_IMAGE_TOKEN_IDX):
                raise RuntimeError(
                    "Missing UND_IMAGE_TOKEN_IDX in input_ids for an image sample "
                    f"(expected token id {RUNTIME_UND_IMAGE_TOKEN_IDX}). "
                    "Ensure prompts contain '<image>' and preprocessing preserves image placeholders."
                )

        input_ids = [inst["input_ids"][:self.tokenizer.model_max_length] for inst in instances]
        labels = [inst["labels"][:self.tokenizer.model_max_length] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        pixel_list = [inst["pixel_values"] for inst in instances if inst["pixel_values"] is not None]
        batch["pixel_values"] = torch.stack(pixel_list) if pixel_list else None
        if batch["pixel_values"] is not None:
            # Keep vision tensors in bf16 to avoid mixed-precision drift with FA2 kernels.
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

        if ISOLATION_BYPASS_PIXELS and "pixel_values" in batch:
            # TEMP BYPASS: Do not pass images to the model.
            del batch["pixel_values"]
        return batch


class ScaffoldTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values", None)

        model_module = model.module if hasattr(model, "module") else model

        if ISOLATION_NATIVE_LLM or not hasattr(model_module, "get_model"):
            outputs = model_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            valid_targets = shift_labels.ne(IGNORE_INDEX)
            if not torch.any(valid_targets):
                raise RuntimeError("No supervised tokens found in native-LLM isolation batch.")

            valid_logits = shift_logits[valid_targets]
            if not torch.isfinite(valid_logits).all():
                raise RuntimeError(
                    "Non-finite logits detected in native-LLM isolation path."
                )

            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected in native-LLM isolation path.")
            return (loss, outputs) if return_outputs else loss

        text_embeds = model_module.get_model().language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=model_module.vision_tower.dtype, device=model_module.device)
            with torch.no_grad():
                image_embeds = model_module.model.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=model_module.config.vision_feature_layer,
                    vision_feature_select_strategy=model_module.config.vision_feature_select_strategy,
                    image_sizes=None,
                )
            und_image_idx = input_ids == RUNTIME_UND_IMAGE_TOKEN_IDX
            if und_image_idx.any():
                text_embeds = text_embeds.clone()
                text_embeds[und_image_idx] = image_embeds.to(text_embeds.device).flatten(0, 1)

        position_ids = torch.cumsum(attention_mask.int(), dim=1) - 1
        position_ids[position_ids < 0] = 0

        outputs = model_module.get_model().language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("Language model output did not include hidden states.")

        logits = model_module.lm_head(last_hidden)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_targets = shift_labels.ne(IGNORE_INDEX)
        if not torch.any(valid_targets):
            raise RuntimeError(
                "No supervised tokens found in batch after shifting labels. "
                "Check chat-template masking and assistant targets."
            )

        valid_logits = shift_logits[valid_targets]
        if not torch.isfinite(valid_logits).all():
            raise RuntimeError(
                "Non-finite logits detected on supervised positions. "
                "Training is numerically unstable; aborting instead of logging filtered zero loss."
            )

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected. "
                "Check multimodal embedding injection and optimizer precision settings."
            )
        return (loss, outputs) if return_outputs else loss


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def resolve_weight_files(model_dir):
    single_file = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single_file):
        return [single_file]

    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        with open(index_file) as handle:
            weight_map = json.load(handle).get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_paths = [os.path.join(model_dir, shard_name) for shard_name in shard_names]
        missing = [path for path in shard_paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Missing sharded base model files: {missing}")
        return shard_paths

    pytorch_bin = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(pytorch_bin):
        return [pytorch_bin]

    raise FileNotFoundError(
        f"No supported weight files found in {model_dir}. "
        "Expected model.safetensors, model.safetensors.index.json, or pytorch_model.bin."
    )


def md5sum_many(paths):
    digest = hashlib.md5()
    for path in paths:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _assert_clean_slate_path(model_path):
    suspicious_tokens = [
        "checkpoint",
        "adapter",
        "merged",
        "olympus",
        "omega",
        "slurm_runs",
        "lora",
    ]
    lowered = model_path.lower()
    if any(token in lowered for token in suspicious_tokens):
        raise RuntimeError(
            "Model path looks like a fine-tuned checkpoint, not a raw pretrained base: "
            f"{model_path}"
        )


def _freeze_non_lora_components(model):
    model_module = model.get_model()

    if hasattr(model_module, "latent_queries") and model_module.latent_queries is not None:
        model_module.latent_queries.requires_grad = False

    if hasattr(model_module, "llm_connector"):
        for param in model_module.llm_connector.parameters():
            param.requires_grad = False

    if hasattr(model_module, "vision_tower"):
        for param in model_module.vision_tower.parameters():
            param.requires_grad = False

    if hasattr(model_module, "multi_modal_projector"):
        for param in model_module.multi_modal_projector.parameters():
            param.requires_grad = False

    for param in model.lm_head.parameters():
        param.requires_grad = False


def _build_unilip_compatible_config(model_name_or_path: str, cache_dir: Optional[str]):
    """
    Convert InternVL2 chat configs (llm_config) into InternVL-style text_config
    expected by UniLIP_InternVLForCausalLM.
    """
    base_config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    config_dict = base_config.to_dict()

    if config_dict.get("model_type") == "internvl_chat" and "llm_config" in config_dict:
        converted = dict(config_dict)
        converted["text_config"] = converted.pop("llm_config")
        converted["model_type"] = "unilip_internvl"
        converted.pop("auto_map", None)

        # Avoid nested trust_remote_code prompts during AutoModel.from_config.
        text_auto_map = converted["text_config"].pop("auto_map", None)

        llm_model_type = converted["text_config"].get("model_type")
        if llm_model_type:
            llm_config_obj = getattr(base_config, "llm_config", None)
            llm_config_cls = llm_config_obj.__class__ if llm_config_obj is not None else None
            try:
                _ = CONFIG_MAPPING[llm_model_type]
            except KeyError:
                if llm_config_cls is not None:
                    try:
                        AutoConfig.register(llm_model_type, llm_config_cls, exist_ok=True)
                    except TypeError:
                        # Backward compatibility for transformers versions without exist_ok.
                        AutoConfig.register(llm_model_type, llm_config_cls)
                    rank0_print(
                        f"Registered remote llm config model_type={llm_model_type} "
                        f"with class {llm_config_cls.__name__}."
                    )

            if llm_config_cls is not None and isinstance(text_auto_map, dict):
                auto_model_ref = text_auto_map.get("AutoModel") or text_auto_map.get("AutoModelForCausalLM")
                if auto_model_ref:
                    try:
                        llm_model_cls = get_class_from_dynamic_module(
                            auto_model_ref,
                            model_name_or_path,
                            cache_dir=cache_dir,
                        )
                        try:
                            AutoModel.register(llm_config_cls, llm_model_cls, exist_ok=True)
                        except TypeError:
                            # Backward compatibility for transformers versions without exist_ok.
                            AutoModel.register(llm_config_cls, llm_model_cls)
                        rank0_print(
                            f"Registered AutoModel mapping for {llm_config_cls.__name__} -> "
                            f"{llm_model_cls.__name__}."
                        )
                    except Exception as exc:
                        rank0_print(f"WARNING: Failed to register AutoModel mapping: {exc}")

        rank0_print(
            "Detected internvl_chat base config; remapping llm_config -> text_config "
            "for UniLIP-compatible loading."
        )
        return UniLIP_InternVLConfig.from_dict(converted)

    if config_dict.get("model_type") != "unilip_internvl":
        rank0_print(
            f"Loading non-chat config model_type={config_dict.get('model_type')} "
            "with UniLIP class."
        )

    return UniLIP_InternVLConfig.from_dict(config_dict)


def _print_and_assert_trainable_parameters(model):
    rank0_print("--- SCAFFOLD-R1 TRAINABLE PARAMETERS ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"TRAINING: {name}")
            assert not any(token in name for token in FORBIDDEN_TRAINABLE_TOKENS), (
                f"SECURITY ALERT: {name} should be frozen!"
            )


def _resolve_lora_target_modules(llm):
    linear_suffixes = {
        name.split(".")[-1]
        for name, module in llm.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    # Prefer Qwen-style projections when present, else fall back to InternLM2 naming.
    if set(LLM_TARGET_MODULES).issubset(linear_suffixes):
        return list(LLM_TARGET_MODULES)
    if set(INTERNLM2_TARGET_MODULES).issubset(linear_suffixes):
        return list(INTERNLM2_TARGET_MODULES)

    resolved = [name for name in (LLM_TARGET_MODULES + INTERNLM2_TARGET_MODULES) if name in linear_suffixes]
    if not resolved:
        sample = sorted(linear_suffixes)[:30]
        raise RuntimeError(
            "Could not resolve LoRA target modules from language model. "
            f"Available linear suffixes sample: {sample}"
        )
    return resolved


def train():
    global RUNTIME_UND_IMAGE_TOKEN_IDX
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.fail_if_path_looks_finetuned:
        _assert_clean_slate_path(model_args.model_name_or_path)

    rank0_print(f"Using UniLIP imports from: {UNILIP_DIR}")
    base_weight_files = None
    is_local_model_path = os.path.isdir(model_args.model_name_or_path)
    if is_local_model_path:
        base_weight_files = resolve_weight_files(model_args.model_name_or_path)
        rank0_print(f"Base weight files: {base_weight_files}")
    else:
        rank0_print(f"Base model appears to be a hub id: {model_args.model_name_or_path}")

    preflight_md5 = None
    if model_args.skip_md5_check or not model_args.base_model_md5 or not is_local_model_path:
        if not is_local_model_path and model_args.base_model_md5:
            rank0_print("=== PRE-FLIGHT: base_model_md5 ignored for hub model id ===")
        rank0_print("=== PRE-FLIGHT: skipping base model md5 verification ===")
    else:
        preflight_md5 = md5sum_many(base_weight_files)
        rank0_print(f"=== PRE-FLIGHT: base model md5: {preflight_md5} ===")
        rank0_print(f"Expected: {model_args.base_model_md5}")
        if preflight_md5 != model_args.base_model_md5:
            raise RuntimeError(
                f"BASE MODEL MD5 MISMATCH — not starting from original base! Got {preflight_md5}"
            )

    if not training_args.bf16:
        raise RuntimeError("This training recipe requires --bf16 True for dtype stability.")
    if getattr(training_args, "fp16", False):
        raise RuntimeError("Use --fp16 False with this recipe to avoid dtype collisions.")
    torch_dtype = torch.bfloat16

    desired_attn = "eager" if ISOLATION_NATIVE_LLM else "sdpa"
    if model_args.attn_implementation != desired_attn:
        rank0_print(
            f"Overriding attn_implementation={model_args.attn_implementation} to {desired_attn}."
        )
    model_args.attn_implementation = desired_attn

    load_kwargs = {
        "cache_dir": training_args.cache_dir,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "attn_implementation": desired_attn,
    }

    from peft import LoraConfig, TaskType, get_peft_model

    if ISOLATION_NATIVE_LLM:
        rank0_print("Isolation mode enabled: loading native AutoModelForCausalLM.")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=desired_attn,
        )
        model.config.use_cache = False
        for param in model.parameters():
            param.requires_grad = False

        resolved_targets = _resolve_lora_target_modules(model)
        rank0_print(f"Resolved native LoRA target modules: {resolved_targets}")

        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=resolved_targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        model_config = _build_unilip_compatible_config(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        model = UniLIP_InternVLForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            **load_kwargs,
        )
        model.config.use_cache = False

        for param in model.parameters():
            param.requires_grad = False

        # Safety lock: keep vision-side modules fully frozen before LoRA wrapping.
        model_module = model.get_model()
        if hasattr(model_module, "vision_tower"):
            for param in model_module.vision_tower.parameters():
                param.requires_grad = False
        if hasattr(model_module, "multi_modal_projector"):
            for param in model_module.multi_modal_projector.parameters():
                param.requires_grad = False

        llm = model.get_model().language_model
        resolved_targets = _resolve_lora_target_modules(llm)
        rank0_print(f"Resolved LoRA target modules: {resolved_targets}")

        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=resolved_targets,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        peft_llm = get_peft_model(llm, lora_config)
        model.get_model().language_model = peft_llm
        _freeze_non_lora_components(model)

    trainable_cast_params = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            trainable_cast_params += param.numel()
    if trainable_cast_params:
        rank0_print(f"Cast trainable params to fp32: {trainable_cast_params/1e6:.1f}M")

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    rank0_print(
        f"Total params: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M  "
        f"({100 * trainable / total:.2f}%)"
    )
    _print_and_assert_trainable_parameters(model)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            rank0_print("Enabled input grads on base model for gradient checkpointing.")
        if not ISOLATION_NATIVE_LLM:
            llm_for_gc = model.get_model().language_model
            if hasattr(llm_for_gc, "enable_input_require_grads"):
                llm_for_gc.enable_input_require_grads()
                rank0_print("Enabled input grads on language model for gradient checkpointing.")
            else:
                def make_inputs_require_grad(module, _input, output):
                    del module
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=TARGET_CONTEXT_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    if ISOLATION_BYPASS_PIXELS:
        rank0_print("Isolation mode: bypassing vision pixel tensors.")
        image_processor = None
    else:
        processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path, trust_remote_code=True)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
            )

    tokenizer.model_max_length = max(training_args.model_max_length, TARGET_CONTEXT_LENGTH)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            {
                "pad_token": "<pad>",
                "additional_special_tokens": ["[IMG]", "[/IMG]", "<image>"],
            },
            tokenizer,
            model,
        )
    elif "<image>" not in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            {"additional_special_tokens": ["[IMG]", "[/IMG]", "<image>"]},
            tokenizer,
            model,
        )

    RUNTIME_UND_IMAGE_TOKEN_IDX = resolve_und_image_token_idx(tokenizer)
    rank0_print(
        f"Resolved UND_IMAGE_TOKEN_IDX for runtime: {RUNTIME_UND_IMAGE_TOKEN_IDX} "
        f"from token {IMG_CONTEXT_TOKEN}."
    )

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]

    data_args.image_processor = image_processor
    train_dataset = ScaffoldSFTDataset(data_args.data_path, tokenizer, data_args)
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    trainer = ScaffoldTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    has_valid_checkpoint = any((checkpoint_dir / "trainer_state.json").exists() for checkpoint_dir in checkpoint_dirs)
    trainer.train(resume_from_checkpoint=True if has_valid_checkpoint else None)
    trainer.save_state()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and not ISOLATION_NATIVE_LLM:
        rank0_print("\n=== Saving LoRA adapter and merged model ===")
        adapter_dir = os.path.join(training_args.output_dir, "adapter")
        merged_dir = os.path.join(training_args.output_dir, "merged")
        os.makedirs(adapter_dir, exist_ok=True)
        os.makedirs(merged_dir, exist_ok=True)

        peft_model = model.get_model().language_model
        peft_model.save_pretrained(adapter_dir)
        rank0_print(f"LoRA adapter saved -> {adapter_dir}")

        merged_llm = peft_model.merge_and_unload()
        model.get_model().language_model = merged_llm

        from safetensors.torch import save_file

        state_dict = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        save_file(state_dict, os.path.join(merged_dir, "model.safetensors"))
        model.config.save_pretrained(merged_dir)
        rank0_print(f"Merged model saved -> {merged_dir}")

        if preflight_md5 is not None and base_weight_files is not None:
            postflight_md5 = md5sum_many(base_weight_files)
            assert postflight_md5 == preflight_md5, (
                f"BASE MODEL MODIFIED! {preflight_md5} -> {postflight_md5}"
            )
            rank0_print(f"=== POST-FLIGHT: base model intact ({postflight_md5}) ===")

        rank0_print("\nDone.")


if __name__ == "__main__":
    train()