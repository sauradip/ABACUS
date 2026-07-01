"""
Zero-shot Scaffold-R1 point audit.

Runs inference on a small random subset from FSC147 test split and reports:
  - coordinate bounds validity in [0, 1000]
  - cross-image coordinate diversity
  - Chamfer distance baseline to GT normalized points

Intended to gate Stage 1 after the first 100 training steps.
"""

import argparse
import json
import os
import random
import re
import pathlib
import sys
import warnings
from typing import Any, Dict, List

import torch
from PIL import Image
from peft import PeftModel
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers.dynamic_module_utils as hf_dynamic_module_utils
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_MODULES_CACHE", str(REPO_ROOT / ".cache" / "native_sft_stage1" / "hf_modules"))
hf_dynamic_module_utils.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]

from scripts.counting_grpo.grpo_rewards import chamfer_distance


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_CONTEXT_LENGTH = 8192
SYSTEM_PROMPT = (
    "You are a grounded counting assistant. "
    "Respond using the exact Thought -> Scaffold -> Count format."
)

PROMPT_TEMPLATE = (
    "<image>\n"
    "How many {category} are in this image? "
    "Reason through the spatial arrangement and provide coordinates for every instance."
)


def resolve_local_model_path(model_id_or_path: str) -> str:
    candidate = pathlib.Path(model_id_or_path)
    if candidate.exists():
        return str(candidate)

    if model_id_or_path != "OpenGVLab/InternVL2-2B":
        return model_id_or_path

    snapshot_roots = [
        REPO_ROOT / ".cache" / "native_sft_stage1" / "hf_home" / "hub" / "models--OpenGVLab--InternVL2-2B" / "snapshots",
        REPO_ROOT / ".cache" / "scaffold_sft_stage1" / "hf_home" / "hub" / "models--OpenGVLab--InternVL2-2B" / "snapshots",
        REPO_ROOT / ".cache" / "hf_home" / "hub" / "models--OpenGVLab--InternVL2-2B" / "snapshots",
    ]
    for root in snapshot_roots:
        if not root.exists():
            continue
        for match in sorted(root.iterdir()):
            if (match / "config.json").exists() and (match / "model.safetensors").exists():
                return str(match)

    return model_id_or_path


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def _normalize_point(point, width, height):
    x_coord, y_coord = point
    normalized_x = int(round((float(x_coord) / float(width)) * 1000.0))
    normalized_y = int(round((float(y_coord) / float(height)) * 1000.0))
    normalized_x = max(0, min(1000, normalized_x))
    normalized_y = max(0, min(1000, normalized_y))
    return [normalized_x, normalized_y]


def get_model_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if model_device is not None:
        return torch.device(model_device)

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vision_dtype(model, default_dtype: torch.dtype) -> torch.dtype:
    try:
        return model.get_model().vision_tower.dtype
    except Exception:
        return default_dtype


def _register_remote_llm_mappings(model_name_or_path: str) -> None:
    """
    Register remote InternLM2 config/model classes so transformers can build
    text_config from model_type='internlm2' in checkpoint configs.
    """
    try:
        base_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[WARN] Remote config preload failed for {model_name_or_path}: {exc}")
        return

    cfg_dict = base_config.to_dict()
    if cfg_dict.get("model_type") != "internvl_chat" or "llm_config" not in cfg_dict:
        return

    llm_cfg = dict(cfg_dict["llm_config"])
    llm_model_type = llm_cfg.get("model_type")
    llm_config_obj = getattr(base_config, "llm_config", None)
    llm_config_cls = llm_config_obj.__class__ if llm_config_obj is not None else None

    if not llm_model_type or llm_config_cls is None:
        return

    try:
        _ = CONFIG_MAPPING[llm_model_type]
    except KeyError:
        try:
            AutoConfig.register(llm_model_type, llm_config_cls, exist_ok=True)
        except TypeError:
            AutoConfig.register(llm_model_type, llm_config_cls)
        print(
            f"[INFO] Registered remote llm config model_type={llm_model_type} "
            f"with class {llm_config_cls.__name__}."
        )

    text_auto_map = llm_cfg.get("auto_map")
    if isinstance(text_auto_map, dict):
        auto_model_ref = text_auto_map.get("AutoModel") or text_auto_map.get("AutoModelForCausalLM")
        if auto_model_ref:
            try:
                llm_model_cls = get_class_from_dynamic_module(auto_model_ref, model_name_or_path)
                try:
                    AutoModel.register(llm_config_cls, llm_model_cls, exist_ok=True)
                except TypeError:
                    AutoModel.register(llm_config_cls, llm_model_cls)
                print(
                    f"[INFO] Registered AutoModel mapping for {llm_config_cls.__name__} -> "
                    f"{llm_model_cls.__name__}."
                )
            except Exception as exc:
                print(f"[WARN] Failed to register AutoModel mapping: {exc}")


def load_model(checkpoint_path, tokenizer_path=None, torch_dtype_name="float16", attn_implementation="eager"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok_path = resolve_local_model_path(tokenizer_path if tokenizer_path else checkpoint_path)
    _register_remote_llm_mappings(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=True,
        model_max_length=TARGET_CONTEXT_LENGTH,
        use_fast=False,
        local_files_only=True,
    )

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
        "torch_dtype": parse_torch_dtype(torch_dtype_name),
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    def _load_with_kwargs(path: str, kwargs: Dict[str, Any]):
        try:
            return AutoModel.from_pretrained(path, local_files_only=True, **kwargs)
        except ValueError as exc:
            if kwargs.get("attn_implementation") == "sdpa" and "does not support" in str(exc):
                print("[WARN] SDPA unsupported in audit model load; retrying with eager attention.")
                retry = dict(kwargs)
                retry["attn_implementation"] = "eager"
                return AutoModel.from_pretrained(path, local_files_only=True, **retry)
            raise

    def _is_peft_full_checkpoint(ckpt_path: str) -> bool:
        sf = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.isfile(sf):
            return False
        try:
            import safetensors
            with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                sample_keys = list(f.keys())[:80]
            return any(".lora_A." in k or ".base_layer." in k for k in sample_keys)
        except Exception:
            return False

    def _materialize_native_adapter_from_full_checkpoint(
        ckpt_path: str,
        base_model_name_or_path: str,
    ) -> str:
        """Convert Trainer full-state checkpoint into a minimal native PEFT adapter directory."""
        source_path = os.path.join(ckpt_path, "model.safetensors")
        full_state = safetensors_load_file(source_path, device="cpu")

        adapter_state: Dict[str, torch.Tensor] = {}
        for key, value in full_state.items():
            if not key.startswith("language_model."):
                continue
            stripped = key[len("language_model."):]
            if ".lora_" not in stripped:
                continue
            normalized = stripped.replace(".lora_A.default.weight", ".lora_A.weight")
            normalized = normalized.replace(".lora_B.default.weight", ".lora_B.weight")
            adapter_state[normalized] = value

        if not adapter_state:
            raise RuntimeError(f"No LoRA keys found while materializing adapter from {source_path}")

        rank = None
        target_modules = set()
        for key, value in adapter_state.items():
            parts = key.split(".")
            if "lora_A" in parts and value.ndim == 2 and rank is None:
                rank = int(value.shape[0])
            if "lora_A" in parts or "lora_B" in parts:
                marker = "lora_A" if "lora_A" in parts else "lora_B"
                marker_index = parts.index(marker)
                if marker_index > 0:
                    target_modules.add(parts[marker_index - 1])

        if rank is None:
            raise RuntimeError("Could not infer LoRA rank from full checkpoint")

        adapter_dir = os.path.join(ckpt_path, "native_peft_adapter")
        os.makedirs(adapter_dir, exist_ok=True)

        adapter_weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        safetensors_save_file(adapter_state, adapter_weights_path)

        adapter_config = {
            "base_model_name_or_path": base_model_name_or_path,
            "bias": "none",
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "peft_type": "LORA",
            "r": rank,
            "target_modules": sorted(target_modules),
            "task_type": "CAUSAL_LM",
        }
        with open(adapter_config_path, "w") as handle:
            json.dump(adapter_config, handle, indent=2)

        print(
            f"[INFO] Materialized native PEFT adapter at {adapter_dir} "
            f"(rank={rank}, targets={sorted(target_modules)}, keys={len(adapter_state)})"
        )
        return adapter_dir

    def _ensure_generate(lm):
        """Transformers >= 4.50 removed generate() from PreTrainedModel base.
        Patch GenerationMixin back onto the language model class for inference."""
        from transformers import GenerationMixin
        if not hasattr(lm, "generate"):
            cls = lm.__class__
            if GenerationMixin not in cls.__mro__:
                lm.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
                print(f"[INFO] Patched GenerationMixin onto {cls.__name__} for inference.")
        return lm

    def _patch_prepare_inputs_for_generation(lm):
        """Drop incompatible cache payloads before legacy InternLM2 code inspects them."""
        if not hasattr(lm, "prepare_inputs_for_generation"):
            return lm

        orig_prepare = lm.prepare_inputs_for_generation

        def patched_prepare(input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
            if past_key_values is not None:
                try:
                    _ = past_key_values[0][0].shape
                except (AttributeError, TypeError, IndexError):
                    past_key_values = None

            return orig_prepare(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        lm.prepare_inputs_for_generation = patched_prepare
        print("[INFO] Patched prepare_inputs_for_generation for cache compatibility.")
        return lm

    adapter_weights = os.path.join(checkpoint_path, "adapter_model.safetensors")
    base_path = resolve_local_model_path(tokenizer_path if tokenizer_path else "OpenGVLab/InternVL2-2B")

    if os.path.isfile(adapter_weights) or _is_peft_full_checkpoint(checkpoint_path):
        model = _load_with_kwargs(base_path, load_kwargs)
        from peft import PeftConfig  # Imported for explicit native-PEFT path diagnostics.

        print("\n[INFO] Initializing Native PEFT Loader...")
        peft_source_path = checkpoint_path
        if not os.path.isfile(adapter_weights) and _is_peft_full_checkpoint(checkpoint_path):
            peft_source_path = _materialize_native_adapter_from_full_checkpoint(
                checkpoint_path,
                base_path,
            )
        try:
            _ = PeftConfig.from_pretrained(peft_source_path)
        except Exception:
            pass

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model.language_model = PeftModel.from_pretrained(
                    model.language_model,
                    peft_source_path,
                    is_trainable=False,
                    local_files_only=True,
                )
            missing_key_warnings = [
                str(w.message)
                for w in caught_warnings
                if "missing adapter keys" in str(w.message).lower()
            ]
            if missing_key_warnings:
                raise RuntimeError(missing_key_warnings[0])
            print("[SUCCESS] Attached LoRA adapter to language_model via PeftModel.from_pretrained")
            model.language_model = model.language_model.merge_and_unload()
            print("[SUCCESS] Merged LoRA weights into base language_model")
        except Exception as exc:
            print(f"\n[FATAL PEFT ERROR] Failed to attach LoRA adapter: {exc}")
            print("This means the checkpoint structure is incompatible with PeftModel.")
            raise SystemExit(1)

        model.language_model = _ensure_generate(model.language_model)
        model.language_model = _patch_prepare_inputs_for_generation(model.language_model)
    else:
        model = _load_with_kwargs(checkpoint_path, load_kwargs)
        model.language_model = _ensure_generate(model.language_model)
        model.language_model = _patch_prepare_inputs_for_generation(model.language_model)

    # Always set img_context_token_id required by InternVL generate().
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if img_ctx_id is None or img_ctx_id < 0:
        print(f"[WARN] Could not resolve {IMG_CONTEXT_TOKEN} from tokenizer; generate() may fail.")
    else:
        model.img_context_token_id = img_ctx_id

    # Put the model into inference mode and disable cache to bypass legacy InternLM2 KV assumptions.
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()
    model.config.use_cache = False
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False
    if hasattr(model, "language_model") and getattr(model.language_model, "generation_config", None) is not None:
        model.language_model.generation_config.use_cache = False

    return model, tokenizer


def preprocess_image(image_path, model, target_dtype: torch.dtype):
    from torchvision import transforms

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    device = get_model_device(model)
    pixel_values = transform(img).unsqueeze(0).to(device=device, dtype=target_dtype)
    return pixel_values


def run_inference(model, tokenizer, image_path, prompt, max_new_tokens=768, target_dtype=torch.float16):
    device = get_model_device(model)
    pixel_values = preprocess_image(image_path, model, target_dtype)
    num_image_token = model.num_image_token if hasattr(model, "num_image_token") else 256

    img_tokens = "<IMG_CONTEXT>" * num_image_token
    img_tag = f"<img>{img_tokens}</img>"
    # Strip any raw "<image>" placeholder from the prompt template before appending
    # the real image tag — otherwise the model sees two competing image references.
    clean_prompt = prompt.replace("<image>", "").strip()
    full_prompt = f"{clean_prompt}\n{img_tag}"

    messages = [
        {"role": "user", "content": full_prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        from transformers import GenerationConfig
        # HF v4.50+ Safety Patch: Ensure generation_config exists on the inner LLM
        if not hasattr(model.language_model, "generation_config") or model.language_model.generation_config is None:
            model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)
        
        # Fix 1: Override max_length constraint on BOTH configs (wrapper and inner LLM)
        model.language_model.generation_config.max_length = None
        model.language_model.generation_config.max_new_tokens = max_new_tokens
        if hasattr(model, "generation_config"):
            model.generation_config.max_length = None
            model.generation_config.max_new_tokens = max_new_tokens
        
        # Fix 2: Ensure img_context_token_id is set for InternVL2 generate() override
        if not hasattr(model, "img_context_token_id") or model.img_context_token_id is None:
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            model.img_context_token_id = img_ctx_id
        
        model.language_model.generation_config.use_cache = False
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "language_model") and hasattr(model.language_model, "gradient_checkpointing_disable"):
            model.language_model.gradient_checkpointing_disable()
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        
        print(f"[DEBUG] Before generate:")
        print(f"  model.generation_config.max_length: {getattr(model.generation_config, 'max_length', 'N/A')}")
        print(f"  model.language_model.generation_config.max_length: {model.language_model.generation_config.max_length}")
        print(f"  model.language_model.generation_config.max_new_tokens: {model.language_model.generation_config.max_new_tokens}")
        
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        print(f"[DEBUG] After generate:")
        print(f"  output_ids shape: {output_ids.shape}")
        print(f"  output_ids[:50]: {output_ids[0, :50].tolist()}")
        
        decoded_full = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        decoded_skip = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"[DEBUG] Decoded (no skip): {repr(decoded_full[:100])}")
        print(f"[DEBUG] Decoded (skip): {repr(decoded_skip[:100])}")
        
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def parse_scaffold_points(response_text: str) -> List[List[int]]:
    scaffold_match = re.search(r"<\|scaffold\|>\s*(\[.*?\])\s*(?:<\|count\|>|$)", response_text, re.DOTALL)
    if scaffold_match:
        candidate = scaffold_match.group(1)
        try:
            parsed = json.loads(candidate)
            points = []
            for point in parsed:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    points.append([int(round(float(point[0]))), int(round(float(point[1])))])
            return points
        except Exception:
            pass

    matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", response_text)
    return [[int(round(float(x_val))), int(round(float(y_val)))] for x_val, y_val in matches]


def points_within_bounds(points: List[List[int]]) -> bool:
    if not points:
        return False
    for x_coord, y_coord in points:
        if x_coord < 0 or x_coord > 1000 or y_coord < 0 or y_coord > 1000:
            return False
    return True


def detect_corner_collapse(points_per_image: List[List[List[int]]]) -> bool:
    flattened = [point for points in points_per_image for point in points]
    if not flattened:
        return True
    return all(point in ([0, 0], [1000, 1000]) for point in flattened)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--fsc_root", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--fail_on_bounds", action="store_true")
    parser.add_argument("--fail_on_diversity", action="store_true")
    args = parser.parse_args()

    ann_path = os.path.join(args.fsc_root, "annotation_FSC147_384.json")
    cls_path = os.path.join(args.fsc_root, "ImageClasses_FSC147.txt")
    splits_path = os.path.join(args.fsc_root, "Train_Test_Val_FSC_147.json")
    img_dir = os.path.join(args.fsc_root, "images_384_VarV2")

    with open(ann_path) as handle:
        annotations = json.load(handle)
    with open(splits_path) as handle:
        splits = json.load(handle)

    image_classes = {}
    with open(cls_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                image_classes[parts[0].strip()] = parts[1].strip().lower()

    random.seed(args.seed)
    candidates = [img_name for img_name in splits["test"] if img_name in annotations]
    sample_size = min(args.num_samples, len(candidates))
    sampled_images = random.sample(candidates, sample_size)

    model, tokenizer = load_model(
        args.model_path,
        tokenizer_path=args.tokenizer_path,
        torch_dtype_name=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    target_dtype = get_vision_dtype(model, parse_torch_dtype(args.torch_dtype))

    rows = []
    predicted_sets = []
    all_bounds_pass = True

    for img_name in sampled_images:
        ann = annotations[img_name]
        category = image_classes.get(img_name, "objects")
        prompt = PROMPT_TEMPLATE.format(category=category)
        image_path = os.path.join(img_dir, img_name)
        response = run_inference(
            model,
            tokenizer,
            image_path,
            prompt,
            max_new_tokens=args.max_new_tokens,
            target_dtype=target_dtype,
        )

        pred_points = parse_scaffold_points(response)
        gt_points = [_normalize_point(point, ann["W"], ann["H"]) for point in ann.get("points", [])]
        bounds_ok = points_within_bounds(pred_points)
        all_bounds_pass = all_bounds_pass and bounds_ok
        predicted_sets.append(pred_points)

        chamfer = chamfer_distance(pred_points, gt_points)
        row = {
            "image": img_name,
            "prompt": prompt,
            "prediction_text": response,
            "pred_points": pred_points,
            "pred_count": len(pred_points),
            "gt_count": len(gt_points),
            "bounds_pass": bounds_ok,
            "chamfer": chamfer,
        }
        rows.append(row)

    unique_sets = {json.dumps(points, separators=(",", ":")) for points in predicted_sets}
    diversity_pass = len(unique_sets) > 1 and not detect_corner_collapse(predicted_sets)
    chamfer_values = [row["chamfer"] for row in rows]
    chamfer_mean = sum(chamfer_values) / len(chamfer_values) if chamfer_values else None

    summary = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path if args.tokenizer_path else args.model_path,
        "num_samples": sample_size,
        "bounds_pass": all_bounds_pass,
        "diversity_pass": diversity_pass,
        "unique_prediction_sets": len(unique_sets),
        "chamfer_mean": chamfer_mean,
        "rows": rows,
    }

    print("=== Zero-Shot Point Audit ===")
    print(f"Samples: {sample_size}")
    print(f"Coordinate bounds: {'PASS' if all_bounds_pass else 'FAIL'}")
    print(f"Token diversity:   {'PASS' if diversity_pass else 'FAIL'} (unique sets={len(unique_sets)})")
    if chamfer_mean is not None:
        print(f"Chamfer mean:      {chamfer_mean:.3f}")
    for row in rows:
        print(
            f"- {row['image']}: bounds={row['bounds_pass']} "
            f"pred={row['pred_count']} gt={row['gt_count']} chamfer={row['chamfer']:.3f}"
        )

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w") as handle:
            json.dump(summary, handle, indent=2)

    if args.fail_on_bounds and not all_bounds_pass:
        raise SystemExit(2)
    if args.fail_on_diversity and not diversity_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()