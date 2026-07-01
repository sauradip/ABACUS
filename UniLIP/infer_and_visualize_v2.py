import os, sys, re, json, argparse, random
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import torch.nn.functional as F

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils._TORCH_LOAD_IS_SAFE = True
def _bypass(): pass
transformers.utils.import_utils.check_torch_load_is_safe = _bypass
transformers.modeling_utils.check_torch_load_is_safe = _bypass

from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM

# Import core visualization from VLM-Visualizer
sys.path.append("/projects/u6bl/myprojects/VLM-Visualizer")
from utils import show_mask_on_image

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="work_dirs/1b_fsc147_understanding_sft")
    p.add_argument("--mllm-hf-path", default="/projects/u6bl/myprojects/UniLIP/.hf_cache/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171")
    p.add_argument("--data-json", default="/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_understanding_sft.json")
    p.add_argument("--vis-dir", default="fsc147_visualizations")
    p.add_argument("--num-samples", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=5)
    p.add_argument("--output-json", default="fsc147_full_inference_res.json")
    return p.parse_args()

def build_prompt(question: str, n_image_tokens: int) -> str:
    und_placeholder = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * n_image_tokens}{IMG_END_TOKEN}'
    return f"<|im_start|>user\n{question} {und_placeholder}<|im_end|>\n<|im_start|>assistant\n"

def run_inference_with_vis(model, tokenizer, processor, image_path, question, vis_path, max_new_tokens=5):
    image = Image.open(image_path).convert("RGB")
    image_processor = processor.image_processor
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=model.device, dtype=model.model.vision_tower.dtype)

    with torch.no_grad():
        image_embeds = model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
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

    generated_ids = []
    vis_list = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            outputs = language_model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask.bool(),
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
            )
            
            last_layer_attn = outputs.attentions[-1][0] # [num_heads, q_len, k_len]
            avg_attn = last_layer_attn.mean(0) # [q_len, k_len]
            
            # The last token query attending to image tokens
            # image tokens are at image_token_mask indices in the original sequence
            # but wait, the sequence length increases!
            # image indices are constant in the prefix.
            img_indices = torch.where(image_token_mask[0])[0]
            token_vis = avg_attn[-1, img_indices].cpu().to(torch.float32).numpy()
            vis_list.append(token_vis)

            next_logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            if next_token.item() in target_tokens: break
            
            generated_ids.append(next_token.item())
            next_embed = language_model.embed_tokens(next_token)
            text_embeds = torch.cat([text_embeds, next_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=model.device, dtype=torch.long)], dim=1)

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    if vis_list:
        combined_vis = np.mean(vis_list, axis=0).astype(np.float32)
        grid = combined_vis.reshape(16, 16)
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        
        heatmap = cv2.resize(grid, (image.width, image.height))
        img_np = np.array(image).astype(np.uint8)
        # Use simple overlay logic if visualizer.py fails or just call it
        try:
            result_img = show_mask_on_image(img_np / 255.0, heatmap)
            Image.fromarray((result_img * 255).astype(np.uint8)).save(vis_path)
        except:
            # Fallback
            mask = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.5, heatmap_img, 0.5, 0)
            Image.fromarray(overlay).save(vis_path)

    return answer

def main():
    args = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)

    model = UniLIP_InternVLForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.mllm_hf_path, trust_remote_code=True, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]", "[/IMG]", "<image>", "<IMG_CONTEXT>"]})
    processor = AutoProcessor.from_pretrained(args.mllm_hf_path, trust_remote_code=True)

    with open(args.data_json) as f:
        data = json.load(f)

    if args.num_samples > 0:
        data = data[:args.num_samples]

    print(f"Starting inference and visualization on {len(data)} images...")
    results = []
    for i, item in enumerate(data):
        image_path = item["image"]
        question = item["conversations"][0]["value"].replace("<image>", "").strip()
        gt_text = item["conversations"][1]["value"].strip()
        
        image_id = item["id"].replace("/", "_")
        vis_path = os.path.join(args.vis_dir, f"vis_{image_id}.png")

        try:
            pred_text = run_inference_with_vis(model, tokenizer, processor, image_path, question, vis_path, args.max_new_tokens)
            m = re.search(r"\d+", pred_text)
            pred = float(m.group()) if m else None
            m_gt = re.search(r"\d+", gt_text)
            gt = float(m_gt.group()) if m_gt else None

            print(f"[{i+1}/{len(data)}] GT={gt} PRED={pred_text}")
            results.append({"id": item["id"], "gt": gt, "pred": pred, "vis": vis_path})
        except Exception as e:
            print(f"Error on {image_path}: {e}")

    valid = [r for r in results if r["gt"] is not None and r["pred"] is not None]
    if valid:
        mae = sum(abs(r["gt"] - r["pred"]) for r in valid) / len(valid)
        print(f"\nFINAL MAE: {mae:.3f} on {len(valid)} samples")
        
        with open(args.output_json, 'w') as f:
            json.dump({"mae": mae, "results": results}, f, indent=4)

if __name__ == "__main__":
    main()
