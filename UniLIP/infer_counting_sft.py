"""
Batch counting inference on the SFT-finetuned UniLIP model.

MATCHES TRAINING TEMPLATE:
- Simple user/assistant ChatML-style (NO system prompt to avoid Chinese bias)
- Stop very early
"""
import os, sys, re, json, argparse, random
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils._TORCH_LOAD_IS_SAFE = True
def _bypass(): pass
transformers.utils.import_utils.check_torch_load_is_safe = _bypass
transformers.modeling_utils.check_torch_load_is_safe = _bypass

from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="work_dirs/1b_fsc147_understanding_sft")
    p.add_argument("--mllm-hf-path", default="/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171")
    p.add_argument("--data-path", dest="data_json", default="/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_understanding_sft.json")
    p.add_argument("--num-samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=5) 
    p.add_argument("--output-path", type=str, default="fsc147_inference_results.json")
    return p.parse_args()

def build_prompt(question: str, n_image_tokens: int) -> str:
    und_placeholder = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * n_image_tokens}{IMG_END_TOKEN}'
    # Clean ChatML WITHOUT system prompt (matches what internvl does internally when not provided)
    prompt = (
        f"<|im_start|>user\n{question} {und_placeholder}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def run_inference(model, tokenizer, processor, image_path, question, max_new_tokens=5):
    image = Image.open(image_path).convert("RGB")
    image_processor = processor
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=model.device, dtype=model.model.vision_tower.dtype)

    with torch.no_grad():
        image_embeds = model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            image_sizes=None,
        )

    prompt = build_prompt(question, 256)
    img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    text_embeds = model.get_model().language_model.embed_tokens(input_ids)
    image_token_mask = input_ids == img_context_id
    text_embeds = text_embeds.clone()
    text_embeds[image_token_mask] = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)

    attention_mask = torch.ones((1, text_embeds.shape[1]), device=model.device, dtype=torch.long)
    language_model = model.get_model().language_model

    target_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            outputs = language_model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask.bool(),
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            next_logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            if next_token.item() in target_tokens:
                break
            
            generated.append(next_token)
            next_embed = language_model.embed_tokens(next_token)
            text_embeds = torch.cat([text_embeds, next_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)

    if not generated:
        return ""
    generated_ids = torch.cat(generated, dim=1)[0]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer

def extract_number(text: str):
    # Only take the FIRST continuous block of digits
    m = re.search(r"\d+", text)
    return float(m.group()) if m else None

def main():
    args = parse_args()
    random.seed(args.seed)

    model = UniLIP_InternVLForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.mllm_hf_path, trust_remote_code=True, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]", "[/IMG]", "<image>", "<IMG_CONTEXT>"]})
    processor = AutoImageProcessor.from_pretrained(args.mllm_hf_path, trust_remote_code=True)

    with open(args.data_json) as f:
        data = json.load(f)

    if args.num_samples > 0:
        data = random.sample(data, min(args.num_samples, len(data)))

    print(f"Running inference on {len(data)} samples...")
    results = []
    for i, item in enumerate(data):
        image_path = item["image"]
        question = item["conversations"][0]["value"].replace("<image>", "").strip()
        gt_text = item["conversations"][1]["value"].strip()
        gt = extract_number(gt_text)

        pred_text = run_inference(model, tokenizer, processor, image_path, question, args.max_new_tokens)
        pred = extract_number(pred_text)
        
        match = "✓" if pred is not None and gt is not None and abs(pred - gt) < 1.5 else "✗"
        print(f"[{i+1}/{len(data)}] {match} GT={gt_text} PRED='{pred_text}'")
        results.append({"id": item["id"], "gt": gt, "pred": pred})

    valid = [(r["gt"], r["pred"]) for r in results if r["gt"] is not None and r["pred"] is not None]
    if valid:
        mae = sum(abs(g - p) for g, p in valid) / len(valid)
        print(f"\nMAE: {mae:.3f} (on {len(valid)} samples)")
        
        final_stats = {
            "mae": mae,
            "num_samples": len(valid),
            "results": results
        }
        with open(args.output_path, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    main()
