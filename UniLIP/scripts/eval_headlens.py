import torch
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
from unilip.model.language_model.headlens import HeadLens
from unilip.model.builder import load_pretrained_model_general
from PIL import Image
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image as PILImage

def parse_args():
    parser = argparse.ArgumentParser(description="HeadLens evaluation script")
    parser.add_argument("--model-path", default="/data/amondal/unicount_runs/v3s_merged_base", help="Path to model checkpoint")
    parser.add_argument("--image-path", default="apple.jpg", help="Path to image for evaluation")
    parser.add_argument("--output-dir", default="./headlens_results", help="Output directory for results")
    parser.add_argument("--avg-layers", type=int, nargs='+', default=[1, 11], help="Layer indices to average for final HeadLens visualization (default: 1 11)")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path

    if not os.path.exists(model_path):
        print(f"Please ensure {model_path} exists and checkpoints are downloaded.")
        return

    print("Loading model and tokenizer...")
    try:
        model = UniLIP_InternVLForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model.config.mllm_hf_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    image_processor = AutoProcessor.from_pretrained(model.config.mllm_hf_path).image_processor

    model.eval()
    print("Model loaded successfully.")

    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return

    image = Image.open(image_path).convert('RGB')

    print(f"Loaded image: {image_path}")

    img_resized = image.resize((448, 448))
    pixel_values = image_processor(images=img_resized, return_tensors="pt").pixel_values.to(model.device).to(model.dtype)

    n_query = getattr(model.config, "n_query", 256)
    print(f"Using n_query = {n_query}")

    from unilip.constants import IMAGE_TOKEN_IDX

    prompt = "Detect and Count the number of birds in the image."
    prefix = tokenizer("<|im_start|>user\n", return_tensors="pt").input_ids[0]
    suffix = tokenizer(f"\n{prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").input_ids[0]

    if prefix[0] == tokenizer.bos_token_id:
        prefix = prefix[1:]
    if suffix[0] == tokenizer.bos_token_id:
        suffix = suffix[1:]

    img_token_tensor = torch.tensor([IMAGE_TOKEN_IDX] * n_query)

    input_ids = torch.cat([prefix, img_token_tensor, suffix]).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)

    print(f"Running forward pass with input_ids shape {input_ids.shape}...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        try:
            # Single forward pass: no labels to avoid loss computation that causes NaN
            print("Forward pass...")
            model.feature_extractor.clear()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,  # Use prepared labels (all -100 = no loss)
                und_image=pixel_values,
                output_attentions=True
            )

            is_img_token = (input_ids[0] == IMAGE_TOKEN_IDX).cpu().numpy()
            img_token_indices = np.where(is_img_token)[0]
            grid_size = int(np.sqrt(n_query))

            # 1. Visualize ALL Attention Heatmaps (DISABLED)
            # if outputs.attentions is not None:
            #     ... attention visualization code ...

            # 2. Visualize ALL HeadLens Magnitudes (using hook-extracted raw heads)
            if hasattr(model, "feature_extractor"):
                raw_heads = model.feature_extractor.raw_head_outputs
                num_layers_heads = len(raw_heads)
                print(f"Extracted raw heads from {num_layers_heads} layers")

                # Debug: print which layers have values and their stats
                print("Layer statistics:")
                for layer_idx in sorted(raw_heads.keys()):
                    raw = raw_heads[layer_idx]
                    print(f"  Layer {layer_idx}: shape={raw.shape}, min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}")

                if num_layers_heads > 0:
                    rows = (num_layers_heads + 3) // 4
                    cols = 4
                    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
                    axes = axes.flatten()
                    llm = model.model.language_model
                    if hasattr(llm, "model"):
                        llm = llm.model

                    target_layers_1_11 = args.avg_layers
                    mags_1_11 = []

                    sorted_layers = sorted(raw_heads.keys())
                    for i, layer_idx in enumerate(sorted_layers):
                        raw_head_out = raw_heads[layer_idx]
                        bsz, seq_len, num_heads, head_dim = raw_head_out.shape
                        hidden_size = num_heads * head_dim
                        headlens = HeadLens(num_heads, head_dim, hidden_size).to(model.device).to(model.dtype)

                        layer = llm.layers[layer_idx]
                        W_O = layer.self_attn.o_proj

                        translated_out = headlens(raw_head_out, W_O, head_idx=0)
                        mags = torch.norm(translated_out[0], dim=-1).float().cpu().numpy()
                        img_mags = mags[img_token_indices]

                        if layer_idx in target_layers_1_11:
                            mags_1_11.append(img_mags)

                        if grid_size * grid_size == n_query:
                            mag_heatmap = img_mags.reshape(grid_size, grid_size)
                            mag_heatmap = np.nan_to_num(mag_heatmap, nan=0.0, posinf=0.0, neginf=0.0)
                            mag_heatmap_min = np.min(mag_heatmap)
                            mag_heatmap_max = np.max(mag_heatmap)
                            if mag_heatmap_max > mag_heatmap_min:
                                mag_heatmap = (mag_heatmap - mag_heatmap_min) / (mag_heatmap_max - mag_heatmap_min + 1e-8)
                            else:
                                mag_heatmap = np.zeros_like(mag_heatmap)
                            mag_heatmap = np.clip(mag_heatmap, 0, 1)
                            mag_heatmap_img = PILImage.fromarray((mag_heatmap * 255).astype(np.uint8)).resize((448, 448), PILImage.BILINEAR)
                            axes[i].imshow(img_resized)
                            axes[i].imshow(np.array(mag_heatmap_img), alpha=0.6, cmap='hot')
                            axes[i].set_title(f"Layer {layer_idx} HeadLens")
                        axes[i].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, "all_layers_headlens.png"))
                    plt.close()
                    print(f"Saved {os.path.join(args.output_dir, 'all_layers_headlens.png')}")

                    # 3. Average for layers 1 and 11
                    if len(mags_1_11) > 0:
                        avg_mags = np.mean(mags_1_11, axis=0)
                        avg_mags = np.nan_to_num(avg_mags, nan=0.0, posinf=0.0, neginf=0.0)
                        mag_heatmap = avg_mags.reshape(grid_size, grid_size)
                        mag_heatmap_min = np.min(mag_heatmap)
                        mag_heatmap_max = np.max(mag_heatmap)
                        if mag_heatmap_max > mag_heatmap_min:
                            mag_heatmap = (mag_heatmap - mag_heatmap_min) / (mag_heatmap_max - mag_heatmap_min + 1e-8)
                        else:
                            mag_heatmap = np.zeros_like(mag_heatmap)
                        mag_heatmap = np.clip(mag_heatmap, 0, 1)
                        mag_heatmap_img = PILImage.fromarray((mag_heatmap * 255).astype(np.uint8)).resize((448, 448), PILImage.BILINEAR)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(img_resized)
                        plt.imshow(np.array(mag_heatmap_img), alpha=0.6, cmap='hot')
                        plt.title(f"Avg HeadLens (Layers {', '.join(map(str, args.avg_layers))})")
                        plt.axis('off')
                        plt.savefig(os.path.join(args.output_dir, "avg_layers_1_11_headlens.png"))
                        plt.close()
                        print(f"Saved {os.path.join(args.output_dir, 'avg_layers_1_11_headlens.png')}")
            else:
                print("Warning: feature_extractor not found on model")

        except Exception as e:
            print("Error during forward or visualization.")
            import traceback
            traceback.print_exc()
            return

if __name__ == "__main__":
    main()
