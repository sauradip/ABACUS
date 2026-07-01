import argparse
import os
import sys

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils._TORCH_LOAD_IS_SAFE = True
transformers.utils.import_utils.check_torch_load = lambda: None
def bypass_check(): pass
transformers.utils.import_utils.check_torch_load_is_safe = bypass_check
transformers.modeling_utils.check_torch_load_is_safe = bypass_check
def bypass_check(): pass
transformers.utils.import_utils.check_torch_load_is_safe = bypass_check

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/projects/u6bl/myprojects/VLM-Visualizer")
from utils import show_mask_on_image

from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer

from unilip.constants import IMAGE_TOKEN_IDX
from unilip.mm_utils import tokenizer_image_token
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser(description="UniLIP attention visualization demo")
    parser.add_argument("--model-path", default="/projects/u6bl/myprojects/UniLIP/work_dirs/1b_stage3_fsc147_ti2i_understanding_v1")
    parser.add_argument("--image-path", default="/projects/u6bl/myprojects/UniLIP/An image with 30 apples on the table.png")
    parser.add_argument("--question", default="How many apples are present in this image? Answer with only a number.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-prefix", default="attn_vis")
    return parser.parse_args()


def build_prompt_with_image_tokens(question: str, n_image_tokens: int) -> str:
    image_block = "<image>" * n_image_tokens
    return (
        "<|im_start|>user\n"
        f"{image_block}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_image_tensor(image: Image.Image, model) -> torch.Tensor:
    processor = None
    for source in (model.config.mllm_path, model.config.mllm_hf_path):
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
            break
        except Exception:
            continue

    image_processor = getattr(processor, "image_processor", processor)
    if image_processor is not None and hasattr(image_processor, "preprocess"):
        pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        return pixel_values

    image_size = model.config.vision_config.image_size
    if isinstance(image_size, (list, tuple)):
        height, width = image_size
    else:
        height = width = int(image_size)

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def main():
    args = parse_args()
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    image_path = os.path.expanduser(args.image_path)

    # Load model manually to bypass builder.py tokenizer issues
    from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
    multi_model = UniLIP_InternVLForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")
    multi_model.eval()

    # Get MLLM path from config
    mllm_hf_path = getattr(multi_model.config, "mllm_hf_path", "OpenGVLab/InternVL3-1B-hf")
    
    tokenizer = AutoTokenizer.from_pretrained(mllm_hf_path, trust_remote_code=True, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG]", "[/IMG]", "<image>", "<IMG_CONTEXT>"]})
    
    # Load processor for image processing
    processor = AutoProcessor.from_pretrained(mllm_hf_path, trust_remote_code=True)
    
    image = Image.open(image_path).convert("RGB")
    image_processor = processor.image_processor
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=multi_model.device, dtype=multi_model.model.vision_tower.dtype)

    vision_feature_layer = multi_model.config.vision_feature_layer
    vision_feature_select_strategy = multi_model.config.vision_feature_select_strategy
    
    with torch.no_grad():
        image_embeds = multi_model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=None,
        )

    n_image_tokens = 256
    img_start, img_end, img_ctx = "<img>", "</img>", "<IMG_CONTEXT>"
    und_placeholder = f"{img_start}{img_ctx * n_image_tokens}{img_end}"
    
    # Matching SFT template
    prompt = (
        f"<|im_start|>user\n{args.question} {und_placeholder}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(multi_model.device)
    attention_mask = torch.ones_like(input_ids, device=multi_model.device)

    text_embeds = multi_model.get_model().language_model.embed_tokens(input_ids)
    img_context_id = tokenizer.convert_tokens_to_ids(img_ctx)
    image_token_mask = input_ids == img_context_id

    text_embeds = text_embeds.clone()
    text_embeds[image_token_mask] = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)

    # We need to find the START index of the vision tokens in the sequence
    vision_token_start = image_token_mask[0].nonzero()[0].item()
    vision_token_end = vision_token_start + n_image_tokens
    
    grid_size = int(np.sqrt(n_image_tokens))
    print(f"Vision tokens span from {vision_token_start} to {vision_token_end-1}, grid size: {grid_size}x{grid_size}")

    language_model = multi_model.get_model().language_model
    attention_mask = torch.ones(
        (text_embeds.shape[0], text_embeds.shape[1]),
        device=text_embeds.device,
        dtype=torch.long,
    )

    generated = []
    generated_tokens_attentions = []
    
    with torch.no_grad():
        for i in range(args.max_new_tokens):
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids[position_ids < 0] = 0
            
            outputs = language_model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask.bool(),
                position_ids=position_ids,
                output_hidden_states=False,
                output_attentions=True,  # WE NEED ATTENTIONS
                return_dict=True,
                use_cache=False,
            )
            
            # Extract attention of the LAST token across all layers
            # outputs.attentions is a tuple of (layer_1_attn, layer_2_attn, ...)
            # Each is [batch, num_heads, seq_len, seq_len]
            # We average across heads and layers
            last_token_attn = []
            for layer_attn in outputs.attentions:
                # layer_attn shape: [1, num_heads, seq_len, seq_len]
                avg_heads = layer_attn[0].mean(dim=0) # [seq_len, seq_len]
                last_token_attn.append(avg_heads[-1]) # attention of the last token to all previous tokens [seq_len]
            
            # average across layers
            avg_layer_attn = torch.stack(last_token_attn).mean(dim=0) # [seq_len]
            generated_tokens_attentions.append(avg_layer_attn.cpu())
            
            next_token_logits = multi_model.lm_head(outputs.last_hidden_state[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

            next_embed = language_model.embed_tokens(next_token)
            text_embeds = torch.cat([text_embeds, next_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=torch.long)], dim=1)

    generated_ids = torch.cat(generated, dim=1)[0]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print("Answer:", answer)

    # Now visualize for each generated token
    gen_tokens_str = [tokenizer.decode([idx], skip_special_tokens=False).strip() for idx in generated_ids]
    
    # Identify the first numeric sequence to combine their attention
    numeric_indices = []
    found_number = False
    for i, token in enumerate(gen_tokens_str):
        if any(char.isdigit() for char in token):
            numeric_indices.append(i)
            found_number = True
        elif found_number:
            break
            
    combined_attn = None
    if numeric_indices:
        # Slice the vision part first since sequence length increases each step
        vis_attns = [generated_tokens_attentions[i][vision_token_start:vision_token_end] for i in numeric_indices]
        combined_attn_vis = torch.stack(vis_attns).mean(dim=0)
        combined_text = "".join([gen_tokens_str[i] for i in numeric_indices])
        print(f"Combining tokens {numeric_indices} for result '{combined_text}'")

    np_img = np.array(image)[:, :, ::-1]
    
    num_to_show = len(gen_tokens_str)
    if combined_attn is not None:
        num_to_show += 1
        
    cols = 8
    rows = max(1, (num_to_show + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), dpi=150)
    axes = axes.flatten() if num_to_show > 1 else [axes]
        
    plot_idx = 0
    # First show combined if available
    if combined_attn_vis is not None:
        ax = axes[plot_idx]
        vis_attn = combined_attn_vis
        vis_attn = vis_attn / (vis_attn.max() + 1e-8)
        vis_attn = vis_attn.reshape(grid_size, grid_size)
        attn_over_image = F.interpolate(vis_attn.unsqueeze(0).unsqueeze(0), size=image.size[::-1], mode='bilinear').squeeze()
        img_with_attn, _ = show_mask_on_image(np_img, attn_over_image.float().numpy())
        ax.imshow(img_with_attn)
        ax.set_title(f"COUNT: {combined_text}", fontsize=12, fontweight='bold', color='red')
        ax.axis("off")
        plot_idx += 1

    for i, (token, attn) in enumerate(zip(gen_tokens_str, generated_tokens_attentions)):
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]
        vis_attn = attn[vision_token_start:vision_token_end]
        vis_attn = vis_attn / (vis_attn.max() + 1e-8)
        vis_attn = vis_attn.reshape(grid_size, grid_size)
        attn_over_image = F.interpolate(vis_attn.unsqueeze(0).unsqueeze(0), size=image.size[::-1], mode='nearest').squeeze()
        img_with_attn, _ = show_mask_on_image(np_img, attn_over_image.float().numpy())
        ax.imshow(img_with_attn)
        ax.set_title(f"Token: {token}", fontsize=10)
        ax.axis("off")
        plot_idx += 1
        
    for j in range(plot_idx, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_visualization.png")
    print(f"Saved visualization to {args.output_prefix}_visualization.png")

if __name__ == "__main__":
    main()
