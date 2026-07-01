#!/usr/bin/env python3
"""Simple test: Load model and run one inference."""
import sys
from pathlib import Path
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)
from transformers import AutoProcessor

device = torch.device("cuda:0")

print("Loading model...")
apply_transformers_compat_shims()
model_cls = load_unilip_class()

model = model_cls.from_pretrained(
    "/data/amondal/model_cache/UniLIP-3B",
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = model.to(device).eval()

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    "/data/amondal/UniCount/.hf_cache/hub/models--OpenGVLab--InternVL3-2B-hf/snapshots/cb57a075cb75a2e6d1b668b128d48bb00ae321d2",
    trust_remote_code=True
)

print("Testing with test image from SHA-B...")
test_img_path = "/data/amondal/ShanghaiTech/part_B/test_data/images/IMG_1.jpg"

if not Path(test_img_path).exists():
    print(f"Image not found: {test_img_path}")
    sys.exit(1)

img = Image.open(test_img_path).convert("RGB")
print(f"Using {test_img_path}")

pil_values = processor.image_processor.preprocess([img], return_tensors="pt")["pixel_values"]
pil_values = pil_values.to(device, dtype=torch.float16)
print(f"Pixel values shape: {pil_values.shape}")

input_ids = torch.tensor([[1, 2, 3]], device=device, dtype=torch.long)
print(f"Input IDs shape: {input_ids.shape}")

print("Running generate()...")
with torch.no_grad():
    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pil_values,
            max_new_tokens=8,
            do_sample=False,
        )
        print(f"Success! Output shape: {output.shape}")
        text = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Decoded: {text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
