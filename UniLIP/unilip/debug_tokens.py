import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils._TORCH_LOAD_IS_SAFE = True
def _bypass(): pass
transformers.utils.import_utils.check_torch_load_is_safe = _bypass
transformers.modeling_utils.check_torch_load_is_safe = _bypass

import torch
from transformers import AutoProcessor, AutoTokenizer
from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
import os, sys
from PIL import Image

MODEL_PATH = "work_dirs/1b_fsc147_understanding_sft"
MLLM_HF_PATH = "/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171"

model = UniLIP_InternVLForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(MLLM_HF_PATH, trust_remote_code=True, use_fast=False)
tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]", "[/IMG]", "<image>", "<IMG_CONTEXT>"]})
processor = AutoProcessor.from_pretrained(MLLM_HF_PATH, trust_remote_code=True)

# Test on one image from FSC-147
image_path = "/projects/u6bl/myprojects/Datasets/FSC-147/images_384_VarV2/2.jpg" # GT=8
question = "How many sea shells are present in this image? Answer with only a number."

image = Image.open(image_path).convert("RGB")
pixel_values = processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to("cuda:0", dtype=torch.bfloat16)

with torch.no_grad():
    image_embeds = model.model.get_image_features(pixel_values, model.config.vision_feature_layer, model.config.vision_feature_select_strategy)

prompt = f"<|im_start|>user\n{question} <img>{'<IMG_CONTEXT>'*256}</img><|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")

ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
text_embeds = model.get_model().language_model.embed_tokens(input_ids)
mask = input_ids == ctx_id
text_embeds[mask] = image_embeds.flatten(0, 1)

attention_mask = torch.ones((1, text_embeds.shape[1]), device="cuda:0", dtype=torch.long)
language_model = model.get_model().language_model

print(f"PROMPT: {prompt}")
print("GENERATING TOKENS:")
with torch.no_grad():
    for _ in range(15):
        pos = torch.cumsum(attention_mask, dim=1) - 1
        out = language_model(inputs_embeds=text_embeds, attention_mask=attention_mask.bool(), position_ids=pos, return_dict=True)
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        token = torch.argmax(logits, dim=-1, keepdim=True)
        token_str = tokenizer.decode(token[0])
        print(f"Token: {token.item()} -> '{token_str}'")
        
        text_embeds = torch.cat([text_embeds, language_model.embed_tokens(token)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device="cuda:0", dtype=torch.long)], dim=1)
