#!/usr/bin/env python3
"""Inspect model structure to find embed_tokens."""
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_hf_multi_image_count_sft import (
    apply_transformers_compat_shims,
    load_unilip_class,
)

apply_transformers_compat_shims()
model_cls = load_unilip_class()

model = model_cls.from_pretrained(
    "/data/amondal/model_cache/UniLIP-3B",
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = model.eval()

print("Model class:", type(model))
print("\nModel attributes:", [a for a in dir(model) if not a.startswith('_')])

inner = model.get_model()
print("\nInner model class:", type(inner))
print("Inner model attributes:", [a for a in dir(inner) if not a.startswith('_') and 'embed' in a.lower()])

if hasattr(inner, 'language_model'):
    llm = inner.language_model
    print("\nLLM class:", type(llm))
    print("LLM has embed_tokens:", hasattr(llm, 'embed_tokens'))
    if hasattr(llm, 'model'):
        print("LLM.model has embed_tokens:", hasattr(llm.model, 'embed_tokens'))
        print("LLM.model type:", type(llm.model))

if hasattr(model, 'language_model'):
    print("\nmodel.language_model exists:", type(model.language_model))
