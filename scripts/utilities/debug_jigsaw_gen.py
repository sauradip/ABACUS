#!/usr/bin/env python3
"""Debug script to test generation on a single sample."""
import os, sys, json, argparse, torch
from pathlib import Path
from PIL import Image
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Patch transformers BEFORE importing model class
import transformers.modeling_utils as tu
_orig_finalize = tu.PreTrainedModel._finalize_model_loading
def _skip_finalize(self, load_config, loading_info):
    try:
        return _orig_finalize(self, load_config, loading_info)
    except AttributeError as e:
        if "all_tied_weights_keys" in str(e):
            return
        raise
tu.PreTrainedModel._finalize_model_loading = _skip_finalize

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    load_processor,
    load_unilip_class,
)
from scripts.experiment_jigsaw.train_jigsaw_sft import resolve_processor_path

# Load model
adapter_path = "/data/amondal/unicount_runs/jigsaw_sft_final_20260430_150418/checkpoint-174"
base_model_path = "/data/amondal/model_cache/UniLIP-3B"

adapter_cfg = json.load(open(os.path.join(adapter_path, "adapter_config.json")))
base_model_path = adapter_cfg.get("base_model_name_or_path", base_model_path)

print(f"[DEBUG] Loading base model from: {base_model_path}")
print(f"[DEBUG] Loading adapter from: {adapter_path}")

# Processor
proc_source = resolve_processor_path(base_model_path)
proc_args = argparse.Namespace(
    processor_name_or_path=proc_source,
    model_name_or_path=base_model_path,
    trust_remote_code=1,
)
from scripts.counting_grpo.train_hf_multi_image_count_sft import load_processor
processor = load_processor(proc_args)

# Base model
model_cls = load_unilip_class()
base_model = model_cls.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager",
)

# Apply LoRA
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
model.eval()

# Move to GPU 0
device = torch.device("cuda:0")
model = model.to(device)
print(f"[DEBUG] Model moved to {device}")

tokenizer = getattr(processor, "tokenizer", processor)

# Test sample
sample_img_path = "/data/amondal/FSC147_hf/images_384_VarV2/190.jpg"
print(f"[DEBUG] Loading image: {sample_img_path}")

image = Image.open(sample_img_path).convert("RGB")
print(f"[DEBUG] Image shape: {image.size}")

# Prompt
prompt_text = (
    "Count the objects in this image. "
    "Answer with a single JSON object: {\"count\": <integer>}"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": sample_img_path},
            {"type": "text", "text": prompt_text},
        ],
    },
]

# Prepare inputs
print("[DEBUG] Applying chat template...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(f"[DEBUG] Chat template applied. Input text length: {len(inputs)}")

from scripts.counting_grpo.train_hf_multi_image_count_sft import expand_unilip_image_context
inputs = expand_unilip_image_context(inputs)

print("[DEBUG] Encoding inputs...")
encoded = tokenizer(
    inputs,
    return_tensors="pt",
    padding=False,
    truncation=True,
    max_length=1024,
)

print(f"[DEBUG] input_ids shape: {encoded['input_ids'].shape}")
print(f"[DEBUG] attention_mask shape: {encoded['attention_mask'].shape}")

ip = getattr(processor, "image_processor", None)
if ip is not None:
    img_proc = ip.preprocess([image], return_tensors="pt")
    encoded.update(img_proc)
    print(f"[DEBUG] pixel_values shape: {encoded['pixel_values'].shape}")

# Move to device
for k in encoded:
    if torch.is_tensor(encoded[k]):
        encoded[k] = encoded[k].to(device)

print("[DEBUG] Starting generation...")
try:
    with torch.inference_mode():
        gen_inputs = {
            "input_ids": encoded.get("input_ids"),
            "attention_mask": encoded.get("attention_mask"),
            "pixel_values": encoded.get("pixel_values"),
        }
        gen_inputs = {k: v for k, v in gen_inputs.items() if v is not None}
        
        print(f"[DEBUG] gen_inputs keys: {gen_inputs.keys()}")
        print(f"[DEBUG] input_ids device: {gen_inputs['input_ids'].device}")
        print(f"[DEBUG] pixel_values device: {gen_inputs['pixel_values'].device}")
        
        out = model.generate(
            **gen_inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    print(f"[DEBUG] Generation complete. Output shape: {out.shape}")
    print(f"[DEBUG] Full output tokens: {out[0]}")
    print(f"[DEBUG] Full output decoded: {tokenizer.decode(out[0], skip_special_tokens=False)}")
    
    input_len = gen_inputs["input_ids"].shape[1]
    print(f"[DEBUG] Input length: {input_len}")
    
    # If output is same as input, model didn't generate anything
    if out.shape[1] == input_len:
        print("[ERROR] Model returned same length as input - no new tokens generated!")
        # Try decoding the whole thing anyway
        pred_text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    else:
        new_tokens = out[0, input_len:]
        print(f"[DEBUG] New tokens count: {len(new_tokens)}")
        print(f"[DEBUG] New tokens: {new_tokens}")
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    print(f"[DEBUG] Final decoded text: '{pred_text}'")
    
except Exception as e:
    print(f"[ERROR] Generation failed: {e}")
    import traceback
    traceback.print_exc()
