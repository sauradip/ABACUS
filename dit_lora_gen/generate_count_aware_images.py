#!/usr/bin/env python3
"""
Generate count-aware images using trained DiT LoRA checkpoints
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from safetensors.torch import safe_open

sys.path.insert(0, '/data/amondal/UniCount/UniLIP')
sys.path.insert(0, '/data/amondal/UniCount')

# Patch attribute access for newer InternVL versions
import unilip.model.unilip_internvl as unilip_module
_original_init = unilip_module.UniLIP_InternVL_MetaModel.__init__

def _patched_init(self, config):
    # Call original but catch attribute errors
    try:
        _original_init(self, config)
    except AttributeError as e:
        if "vision_model" in str(e) and hasattr(self, 'model'):
            # Newer InternVL has vision_tower, not vision_model
            if hasattr(self.model, 'vision_tower'):
                self.vision_tower = self.model.vision_tower
            if hasattr(self.model, 'mlp1'):
                self.multi_modal_projector = self.model.mlp1
        else:
            raise

unilip_module.UniLIP_InternVL_MetaModel.__init__ = _patched_init

from unilip.model.language_model.unilip_internvl import UniLIP_InternVLForCausalLM
from transformers import AutoProcessor

# Patch fp32 loading (same as train_dit_lora_fp32.py)
_original_from_pretrained = UniLIP_InternVLForCausalLM.from_pretrained

@classmethod
def _from_pretrained_fp32(cls, *args, **kwargs):
    kwargs['torch_dtype'] = torch.float32
    model = _original_from_pretrained(*args, **kwargs)
    return model.to(torch.float32)

UniLIP_InternVLForCausalLM.from_pretrained = _from_pretrained_fp32


class CountAwareGenerator:
    """Generate images with specific object counts using trained model"""

    def __init__(self, checkpoint_path, device="cuda"):
        print(f"[GEN] Loading checkpoint: {checkpoint_path}")

        # Manual DiT loading approach - create model and load weights
        print("[GEN] Initializing DiT transformer...")
        try:
            from diffusers import SanaTransformer2DModel
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            # Create Dit with standard Sana config that matches checkpoint structure
            self.dit = SanaTransformer2DModel(
                in_channels=4,
                out_channels=4,
                num_attention_heads=16,
                attention_head_dim=72,
                num_layers=28,
                num_cross_attention_heads=16,
                cross_attention_head_dim=72,
                cross_attention_dim=1536,
                caption_channels=1536,
                mlp_ratio=4.0,
                sample_size=64,
                patch_size=1
            )
            # Create noise scheduler
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        except Exception as e:
            print(f"[GEN] Warning: Could not init Dit: {e}")
            self.dit = None
            self.noise_scheduler = None

        # Load model from checkpoint
        self.model = UniLIP_InternVLForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        )

        # Manually assign DiT and scheduler if created
        if self.dit is not None:
            self.model.model.dit = self.dit
        if self.noise_scheduler is not None:
            self.model.model.noise_scheduler = self.noise_scheduler

        # Create/setup VAE decoder stub for inference
        try:
            from unilip.model.vae_modules import DCAE_Decoder
            from omegaconf import OmegaConf
            vae_config = OmegaConf.create({'model': {'dc_ae_path': ''}})
            self.model.model.vae = DCAE_Decoder(vae_config, 1536)
        except Exception as e:
            print(f"[GEN] Note: Could not create VAE decoder: {e}")

        # Try to load DiT weights if they exist in checkpoint
        print("[GEN] Loading DiT weights from checkpoint...")
        checkpoint_dir = Path(checkpoint_path)
        model_index_file = checkpoint_dir / "model.safetensors.index.json"
        if model_index_file.exists() and self.dit is not None:
            with open(model_index_file) as f:
                index = json.load(f)

            dit_keys = [k for k in index['weight_map'].keys() if 'model.dit.' in k]
            if dit_keys:
                print(f"  ✓ Found {len(dit_keys)} DiT weights")
                # Load checkpoint weights
                state_dict = {}
                for file_prefix in ("model-00001", "model-00002", "model-00003", "model-00004"):
                    file_path = checkpoint_dir / f"{file_prefix}-of-00004.safetensors"
                    if file_path.exists():
                        with safe_open(str(file_path), framework="pt") as f:
                            for key in f.keys():
                                if 'model.dit.' in key:
                                    state_dict[key] = f.get_tensor(key).to(device)

                if state_dict:
                    # Transform keys and load into dit
                    dit_state = {}
                    for k, v in state_dict.items():
                        new_k = k.replace('model.dit.base_model.model.', '')
                        dit_state[new_k] = v

                    try:
                        missing, unexpected = self.dit.load_state_dict(dit_state, strict=False)
                        print(f"  ✓ Loaded {len(dit_state)} DiT weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
                    except Exception as e:
                        print(f"  ✗ Failed to load DiT weights: {e}")

        self.model.to(device)
        self.model.eval()
        self.device = device

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            "OpenGVLab/InternVL3-2B-hf"
        )
        print(f"✓ Model loaded successfully")

    @torch.no_grad()
    def _process_text_to_embeddings(self, text):
        """Process text prompt through LLM to get latent embeddings"""
        inputs = self.processor(
            text=text,
            images=None,
            return_tensors="pt"
        ).to(self.device)

        model = self.model.model

        # Encode text through language model
        input_ids = inputs['input_ids']
        text_embeds = model.language_model(input_ids).last_hidden_state

        # Mean pool to get single representation
        caption_embed = text_embeds.mean(dim=1)  # (B, hidden_size)

        # Project to DiT caption channels
        if hasattr(model, 'projector'):
            caption_embed = model.projector(caption_embed)  # (B, caption_channels)

        return caption_embed

    @torch.no_grad()
    def _generate_image_from_caption(self, caption_embed, num_inference_steps=50):
        """Use DiT to generate image latents from caption embedding"""
        model = self.model.model
        dit = model.dit
        noise_scheduler = model.noise_scheduler

        batch_size = caption_embed.shape[0]
        height = width = 64  # Sana default
        latent_channels = 4

        # Initialize random noise
        latents = torch.randn(
            batch_size, latent_channels, height, width,
            device=caption_embed.device,
            dtype=caption_embed.dtype
        )

        # Set timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=caption_embed.device)
        timesteps = noise_scheduler.timesteps

        # Diffusion loop
        for t_idx, t in enumerate(timesteps):
            # Handle timestep properly  - must be scalar tensor or 0D
            if t.dim() > 0:
                t = t.squeeze()

            # Prepare latent model input
            latent_model_input = latents

            # Predict noise residual
            try:
                noise_pred = dit(
                    latent_model_input,
                    timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                    encoder_hidden_states=caption_embed
                ).sample
            except Exception as e:
                print(f"DiT forward failed: {e}")
                break

            # Update latents using scheduler
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    @torch.no_grad()
    def _decode_latents_to_image(self, latents):
        """Decode VAE latents to pixel image"""
        model = self.model.model
        vae = model.vae

        # Scale latents
        latents = latents / vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else latents / 0.18215

        # Decode through VAE
        image = vae.decode(latents).sample

        # Normalize to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)

        # Convert to PIL Image
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype('uint8')

        return [Image.fromarray(img) for img in image]

    @torch.no_grad()
    def generate(self, prompts, num_samples=10, output_dir=None, num_inference_steps=50):
        """
        Generate images from text prompts using trained DiT model

        Args:
            prompts: List of text prompts
            num_samples: Number of samples per prompt
            output_dir: Directory to save generated images
            num_inference_steps: Diffusion steps (higher=better quality but slower)

        Returns:
            List of generated image paths
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        generated_paths = []

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
            for sample_idx in range(num_samples):
                try:
                    # Step 1: Process text to embeddings
                    caption_embed = self._process_text_to_embeddings(prompt)

                    # Step 2: Generate image latents via diffusion
                    latents = self._generate_image_from_caption(
                        caption_embed,
                        num_inference_steps=num_inference_steps
                    )

                    # Step 3: Decode latents to pixel images
                    images = self._decode_latents_to_image(latents)

                    # Step 4: Save images
                    if output_dir and images:
                        parts = prompt.split()
                        try:
                            count = int(parts[4])
                            class_name = parts[5] if len(parts) > 5 else "object"
                            img_name = f"{count}__{class_name}_{sample_idx}.png"
                            img_path = output_dir / img_name

                            images[0].save(img_path)
                            generated_paths.append(str(img_path))
                            print(f"  ✓ Saved: {img_name}")
                        except (ValueError, IndexError) as e:
                            print(f"  ✗ Failed to parse prompt: {prompt}")

                except Exception as e:
                    print(f"✗ Generation failed for prompt '{prompt}' sample {sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()

        return generated_paths


def create_test_prompts():
    """Create count-aware test prompts"""
    categories = ["car", "person", "chair", "cup", "dog", "cat", "bottle", "book"]
    counts = [2, 3, 5, 7, 10]

    prompts = []
    for count in counts:
        for category in categories:
            prompt = f"Generate an image: A photo of {count} {category}"
            prompts.append(prompt)

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="runs/dit_lora_r128",
                       help="Base checkpoint directory")
    parser.add_argument("--checkpoints", nargs="+",
                       default=["checkpoint-200", "checkpoint-400", "checkpoint-600", "checkpoint-800", "checkpoint-1000"],
                       help="Checkpoints to use for generation")
    parser.add_argument("--output_base_dir", default="outputs/dit_lora_gen/generated_images",
                       help="Base output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples per prompt")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Count-Aware Image Generation - Config A (r=128)")
    print("="*80)

    # Create test prompts
    test_prompts = create_test_prompts()
    print(f"\n✓ Created {len(test_prompts)} test prompts")
    print(f"  Categories: car, person, chair, cup, dog, cat, bottle, book")
    print(f"  Counts: 2, 3, 5, 7, 10")

    # Generate images for each checkpoint
    for ckpt_name in args.checkpoints:
        ckpt_path = Path(args.checkpoint_dir) / ckpt_name

        if not ckpt_path.exists():
            print(f"\n✗ Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[{ckpt_name}] Generating images...")

        try:
            generator = CountAwareGenerator(str(ckpt_path))

            output_dir = Path(args.output_base_dir) / ckpt_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate
            generated_paths = generator.generate(
                test_prompts,
                num_samples=args.num_samples,
                output_dir=output_dir
            )

            print(f"✓ Generated {len(generated_paths)} images")
            print(f"  Output directory: {output_dir}")

            # Save metadata
            metadata = {
                "checkpoint": ckpt_name,
                "num_prompts": len(test_prompts),
                "num_samples": args.num_samples,
                "total_images": len(generated_paths),
                "output_directory": str(output_dir)
            }
            with open(output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"✗ Generation failed for {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("✓ Generation complete")
    print(f"✓ Images saved to: {args.output_base_dir}")
    print("\nNext step: Run YOLOv9 evaluation")
    print("  python eval_generation_yolo.py --gen_dir outputs/dit_lora_gen/generated_images")


if __name__ == "__main__":
    main()
