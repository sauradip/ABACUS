#!/usr/bin/env python3
"""Clean HF-style multi-image SFT trainer for adaptive counting prompts.

This script is intentionally separate from the older scaffold/stage15 trainers.
It consumes rows with a Hugging Face `messages` array, expects two independent
image content items, and supervises only the assistant count-only response.
"""

import argparse
import json
import os
import sys
import contextlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoProcessor, Trainer
except ImportError:
    torch = None
    F = None
    transformers = None
    Image = None
    Dataset = object
    AutoModelForCausalLM = None
    AutoImageProcessor = None
    AutoProcessor = None
    Trainer = None


IGNORE_INDEX = -100
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
UNILIP_IMAGE_TOKENS_PER_IMAGE = 256
LEGACY_PATH_MARKERS = (
    "scaffold_rex",
    "stage15",
    "stage1.5",
    "stage16",
    "rankdpo",
    "grpo",
    "native_sft",
    "internvl",
)


def rank0_print(*args: Any) -> None:
    if int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0:
        print(*args)


def apply_transformers_compat_shims() -> None:
    if transformers is None:
        return
    # transformers>=5 can expect `all_tied_weights_keys` while older remote
    # model code only defines `_tied_weights_keys`.
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        # Keep it writable because newer transformers assigns to it in post_init().
        PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}")
            return data
        rows: List[Dict[str, Any]] = []
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            rows.append(row)
        return rows


def assert_clean_model_source(path_or_id: str) -> None:
    lowered = path_or_id.lower()
    bad = [marker for marker in LEGACY_PATH_MARKERS if marker in lowered]
    if bad:
        raise ValueError(
            f"Refusing legacy/contaminated model source '{path_or_id}'. "
            f"Matched markers: {bad}"
        )

    local_config_path = Path(path_or_id) / "config.json"
    local_model_type = None
    if local_config_path.exists():
        try:
            local_model_type = json.loads(local_config_path.read_text(encoding="utf-8")).get("model_type")
        except Exception:
            local_model_type = None

    if (
        "unilip" not in lowered
        and "kanashi6/unilip" not in lowered
        and local_model_type != "unilip_internvl"
    ):
        raise ValueError(
            f"Model source does not look like a vanilla UniLIP checkpoint: {path_or_id}"
        )
    if os.path.isdir(path_or_id) and not os.path.exists(os.path.join(path_or_id, "config.json")):
        raise FileNotFoundError(f"Local model directory is missing config.json: {path_or_id}")


def assert_clean_processor_source(path_or_id: str) -> None:
    lowered = path_or_id.lower()
    bad = [
        marker
        for marker in LEGACY_PATH_MARKERS
        if marker != "internvl" and marker in lowered
    ]
    if bad:
        raise ValueError(
            f"Refusing legacy/contaminated processor source '{path_or_id}'. "
            f"Matched markers: {bad}"
        )


def read_local_config(path_or_id: str) -> Dict[str, Any]:
    config_path = Path(path_or_id) / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def resolve_processor_source(args: argparse.Namespace) -> str:
    if args.processor_name_or_path:
        assert_clean_processor_source(args.processor_name_or_path)
        return args.processor_name_or_path

    config = read_local_config(args.model_name_or_path)
    mllm_hf_path = config.get("mllm_hf_path")
    if mllm_hf_path:
        # Vanilla UniLIP checkpoints store the HF-native processor source here.
        assert_clean_processor_source(str(mllm_hf_path))
        return str(mllm_hf_path)

    assert_clean_processor_source(args.model_name_or_path)
    return args.model_name_or_path


def content_images(messages: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for message in messages:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                path = item.get("url") or item.get("path") or item.get("image")
                if path:
                    paths.append(str(path))
    return paths


def validate_messages(row: Dict[str, Any], strict_images: bool = True) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Row {row.get('id')} must contain exactly two messages")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"Row {row.get('id')} must be user then assistant")

    image_paths = content_images(messages[:1])
    if len(image_paths) != 2:
        raise ValueError(f"Row {row.get('id')} must contain exactly two user image items")
    for image_path in image_paths:
        if strict_images and not Path(image_path).exists():
            raise FileNotFoundError(f"Missing image for row {row.get('id')}: {image_path}")

    user_text = [
        item.get("text", "")
        for item in messages[0].get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if len(user_text) != 1:
        raise ValueError(f"Row {row.get('id')} must contain exactly one user text item")
    if "<image>" in user_text[0]:
        raise ValueError(f"Row {row.get('id')} contains hardcoded <image> text")

    assistant_content = messages[1].get("content", [])
    if len(assistant_content) != 1 or assistant_content[0].get("type") != "text":
        raise ValueError(f"Row {row.get('id')} assistant content must be one text item")
    payload = json.loads(assistant_content[0]["text"])
    if sorted(payload.keys()) != ["total_count"] or not isinstance(payload["total_count"], int):
        raise ValueError(f"Row {row.get('id')} assistant target must be count-only JSON")


def squeeze_batch_dim(value: Any) -> Any:
    if torch is None:
        return value
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def load_pil_images(messages: List[Dict[str, Any]]) -> List[Any]:
    if Image is None:
        raise RuntimeError("PIL is required to load images for processor fallback")
    images: List[Any] = []
    for path in content_images(messages):
        images.append(Image.open(path).convert("RGB"))
    return images


class ProcessorBundle:
    def __init__(self, tokenizer: Any, image_processor: Any):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def expand_unilip_image_context(text: str) -> str:
    replacement = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * UNILIP_IMAGE_TOKENS_PER_IMAGE) + IMG_END_TOKEN
    return text.replace(IMG_CONTEXT_TOKEN, replacement)


class HFMultiImageCountDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: Optional[Any],
        max_seq_length: int,
        validate_only: bool = False,
        strict_images: bool = True,
    ):
        self.data_path = Path(data_path)
        self.raw_data = load_json_or_jsonl(self.data_path)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.validate_only = validate_only
        self.strict_images = strict_images
        for row in self.raw_data:
            validate_messages(row, strict_images=strict_images)
        rank0_print(f"Loaded {len(self.raw_data)} HF multi-image rows from {self.data_path}")

    def __len__(self) -> int:
        return len(self.raw_data)

    def _apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = False,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Processor is required for tokenized dataset access")

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        text = expand_unilip_image_context(text)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        if include_images:
            images = load_pil_images(messages)
            image_processor = getattr(self.processor, "image_processor", None)
            if images and image_processor is None:
                raise RuntimeError("Image processor is required for image messages")
            if images:
                encoded.update(image_processor.preprocess(images, return_tensors="pt"))

        if "input_ids" not in encoded:
            raise RuntimeError("chat template tokenization did not return input_ids")
        return {key: squeeze_batch_dim(value) for key, value in encoded.items()}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.raw_data[index]
        if self.validate_only:
            return row

        messages = row["messages"]
        full = self._apply_chat_template(messages, add_generation_prompt=False)
        prompt = self._apply_chat_template(
            messages[:1],
            add_generation_prompt=True,
            include_images=False,
        )

        input_ids = full["input_ids"]
        labels = input_ids.clone()
        prompt_len = min(int(prompt["input_ids"].shape[-1]), int(labels.shape[-1]))
        labels[..., :prompt_len] = IGNORE_INDEX

        if "attention_mask" in full:
            labels = labels.masked_fill(full["attention_mask"].eq(0), IGNORE_INDEX)
        if int(labels.ne(IGNORE_INDEX).sum().item()) == 0:
            raise RuntimeError(f"No supervised assistant tokens found for row {row.get('id')}")

        full["labels"] = labels
        return full


class HFMultiImageCollator:
    def __init__(self, processor: Any):
        if torch is None:
            raise RuntimeError("torch is required for collation")
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    def _pad_1d(self, tensors: Iterable[Any], value: int) -> Any:
        return torch.nn.utils.rnn.pad_sequence(
            [tensor.long() for tensor in tensors],
            batch_first=True,
            padding_value=value,
        )

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {
            "input_ids": self._pad_1d((item["input_ids"] for item in instances), self.pad_token_id),
            "labels": self._pad_1d((item["labels"] for item in instances), IGNORE_INDEX),
        }
        if "attention_mask" in instances[0]:
            batch["attention_mask"] = self._pad_1d((item["attention_mask"] for item in instances), 0)
        else:
            batch["attention_mask"] = batch["input_ids"].ne(self.pad_token_id)

        for key in instances[0].keys():
            if key in batch:
                continue
            values = [item[key] for item in instances if key in item and torch.is_tensor(item[key])]
            if not values:
                continue
            if key == "pixel_values" and values[0].ndim == 4:
                batch[key] = torch.cat(values, dim=0)
                continue
            try:
                batch[key] = torch.stack(values)
            except RuntimeError:
                batch[key] = torch.cat(values, dim=0)
        return batch


def embed_tokens(language_model: Any, input_ids: Any) -> Any:
    if hasattr(language_model, "embed_tokens"):
        return language_model.embed_tokens(input_ids)
    input_embeddings = language_model.get_input_embeddings()
    return input_embeddings(input_ids)


def module_device(module: Any) -> Any:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def module_dtype(module: Any) -> Any:
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.bfloat16


def compute_count_ce_loss(model: Any, inputs: Dict[str, Any], return_outputs: bool = False) -> Any:
    if torch is None or F is None:
        raise RuntimeError("torch is required for forward loss")

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
            # Avoid get_image_features() decorator kwargs drift across transformers versions.
            internvl_core = model_module.model
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
        expected = int(image_token_mask.sum().item())
        flat_embeds = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)
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
    logits = model_module.lm_head(outputs.last_hidden_state)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return (loss, outputs) if return_outputs else loss


class HFMultiImageCountTrainer(Trainer):
    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        return compute_count_ce_loss(model, inputs, return_outputs=return_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--save_only_model", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--load_best_model_at_end", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--validate_json_only", action="store_true")
    parser.add_argument("--skip_model_load", action="store_true")
    parser.add_argument("--strict_images", type=int, default=1)
    parser.add_argument("--trust_remote_code", type=int, default=1)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def load_processor(args: argparse.Namespace) -> Any:
    if AutoProcessor is None or AutoImageProcessor is None:
        raise RuntimeError("transformers is required to load the processor")
    source = resolve_processor_source(args)
    rank0_print(f"Processor source: {source}")
    tokenizer_or_processor = AutoProcessor.from_pretrained(source, trust_remote_code=bool(args.trust_remote_code))
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    image_processor = getattr(tokenizer_or_processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(source, trust_remote_code=bool(args.trust_remote_code))
    return ProcessorBundle(tokenizer=tokenizer, image_processor=image_processor)


def load_unilip_class() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    unilip_root = repo_root / "UniLIP"
    if str(unilip_root) not in sys.path:
        sys.path.insert(0, str(unilip_root))
    from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM

    return UniLIP_InternVLForCausalLM


def load_unilip_constants() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    unilip_root = repo_root / "UniLIP"
    if str(unilip_root) not in sys.path:
        sys.path.insert(0, str(unilip_root))
    from unilip.constants import UND_IMAGE_TOKEN_IDX

    return {"UND_IMAGE_TOKEN_IDX": UND_IMAGE_TOKEN_IDX}


def load_model(args: argparse.Namespace) -> Any:
    if AutoModelForCausalLM is None or torch is None:
        raise RuntimeError("torch and transformers are required to load the model")
    assert_clean_model_source(args.model_name_or_path)
    kwargs = {
        "trust_remote_code": bool(args.trust_remote_code),
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
    }
    model_config = read_local_config(args.model_name_or_path)
    if model_config.get("model_type") == "unilip_internvl":
        # UniLIP internally calls AutoModel.from_pretrained while building the model.
        # low_cpu_mem_usage=True would activate meta-init and breaks that nested load.
        kwargs["low_cpu_mem_usage"] = False
        model_cls = load_unilip_class()
        try:
            kwargs["attn_implementation"] = args.attn_implementation
            model = model_cls.from_pretrained(args.model_name_or_path, **kwargs)
            return place_model(model)
        except Exception:
            if not args.allow_attn_fallback:
                raise
            kwargs.pop("attn_implementation", None)
            rank0_print("WARN: falling back to UniLIP model load without attn_implementation")
            model = model_cls.from_pretrained(args.model_name_or_path, **kwargs)
            return place_model(model)

    kwargs["low_cpu_mem_usage"] = True
    try:
        kwargs["attn_implementation"] = args.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
        return place_model(model)
    except Exception:
        if not args.allow_attn_fallback:
            raise
        kwargs.pop("attn_implementation", None)
        rank0_print("WARN: falling back to model load without attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
        return place_model(model)


def place_model(model: Any) -> Any:
    if torch is not None and torch.cuda.is_available():
        return model.to("cuda")
    return model


@contextlib.contextmanager
def force_default_cpu_device():
    if torch is None or not hasattr(torch, "set_default_device"):
        yield
        return
    try:
        torch.set_default_device("cpu")
        yield
    finally:
        # Keep default tensor device on CPU so dataloader/pin_memory remains valid.
        torch.set_default_device("cpu")


def main() -> None:
    args = parse_args()
    apply_transformers_compat_shims()
    assert_clean_model_source(args.model_name_or_path)
    if args.processor_name_or_path:
        assert_clean_processor_source(args.processor_name_or_path)

    if args.validate_json_only:
        dataset = HFMultiImageCountDataset(
            args.data_path,
            processor=None,
            max_seq_length=args.max_seq_length,
            validate_only=True,
            strict_images=bool(args.strict_images),
        )
        rank0_print(f"JSON-only validation passed for {len(dataset)} rows")
        if args.eval_data_path:
            eval_dataset = HFMultiImageCountDataset(
                args.eval_data_path,
                processor=None,
                max_seq_length=args.max_seq_length,
                validate_only=True,
                strict_images=bool(args.strict_images),
            )
            rank0_print(f"Eval JSON-only validation passed for {len(eval_dataset)} rows")
        return

    processor = load_processor(args)
    dataset = HFMultiImageCountDataset(
        args.data_path,
        processor=processor,
        max_seq_length=args.max_seq_length,
        strict_images=bool(args.strict_images),
    )
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = HFMultiImageCountDataset(
            args.eval_data_path,
            processor=processor,
            max_seq_length=args.max_seq_length,
            strict_images=bool(args.strict_images),
        )
    collator = HFMultiImageCollator(processor)

    if args.dry_run:
        sample = dataset[0]
        batch = collator([sample])
        rank0_print("Dry-run batch tensor shapes:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                rank0_print(f"  {key}: {tuple(value.shape)} {value.dtype}")
        if args.skip_model_load:
            rank0_print("Dry run stopped before model load (--skip_model_load).")
            return

    with force_default_cpu_device():
        model = load_model(args)
    model.config.use_cache = False

    if args.dry_run:
        model.eval()
        batch = collator([dataset[0]])
        device = module_device(model)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            loss = compute_count_ce_loss(model, batch)
        rank0_print(f"Dry-run forward loss={float(loss.detach().cpu())}")
        return

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
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
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_init = inspect.signature(HFMultiImageCountTrainer.__init__)
    if "tokenizer" in trainer_init.parameters:
        trainer_kwargs["tokenizer"] = getattr(processor, "tokenizer", None)
    elif "processing_class" in trainer_init.parameters:
        trainer_kwargs["processing_class"] = getattr(processor, "tokenizer", None)
    trainer = HFMultiImageCountTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    if trainer.state.best_model_checkpoint and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0:
        output_path = Path(args.output_dir)
        best_checkpoint = Path(trainer.state.best_model_checkpoint)
        (output_path / "best_checkpoint.txt").write_text(str(best_checkpoint) + "\n", encoding="utf-8")
        link_path = output_path / "best_epoch_checkpoint"
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(os.path.relpath(best_checkpoint, output_path))
        except OSError as exc:
            rank0_print(f"WARN: could not create best_epoch_checkpoint symlink: {exc}")
        rank0_print(f"Best checkpoint by {args.metric_for_best_model}: {best_checkpoint}")


if __name__ == "__main__":
    main()
