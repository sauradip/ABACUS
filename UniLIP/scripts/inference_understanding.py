import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor

from unilip.constants import IMAGE_TOKEN_IDX
from unilip.mm_utils import tokenizer_image_token
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser(description="UniLIP image understanding demo")
    parser.add_argument("model_path", help="Path to a UniLIP checkpoint directory")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("question", nargs="?", default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-4bit", action="store_true", help="Load the checkpoint in 4-bit mode")
    parser.add_argument("--load-8bit", action="store_true", help="Load the checkpoint in 8-bit mode")
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"<image>\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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

    tokenizer, multi_model, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM",
        model_path,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    prompt = build_prompt(args.question)
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(multi_model.device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = build_image_tensor(image, multi_model).to(device=multi_model.device, dtype=multi_model.dtype)

    try:
        output_ids = multi_model.generate(
            inputs=input_ids,
            images=pixel_values,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][input_ids.shape[1]:]
    except Exception as exc:
        # Some UniLIP checkpoints fail in generate() because a helper is missing
        # or because the multimodal path expects one token per image embedding.
        should_fallback = (
            isinstance(exc, AttributeError)
            and "prepare_inputs_labels_for_understanding" in str(exc)
        ) or (
            isinstance(exc, RuntimeError)
            and "shape mismatch" in str(exc)
        )
        if not should_fallback:
            raise

        vision_feature_layer = multi_model.config.vision_feature_layer
        vision_feature_select_strategy = multi_model.config.vision_feature_select_strategy
        image_embeds = multi_model.model.get_image_features(
            pixel_values=pixel_values.type(multi_model.model.vision_tower.dtype),
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=None,
        )

        n_image_tokens = image_embeds.shape[1]
        prompt = build_prompt_with_image_tokens(args.question, n_image_tokens)
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(multi_model.device)
        attention_mask = torch.ones_like(input_ids, device=multi_model.device)

        text_embeds = multi_model.get_model().language_model.embed_tokens(input_ids)
        image_token_mask = input_ids == IMAGE_TOKEN_IDX
        if int(image_token_mask.sum().item()) != n_image_tokens:
            raise RuntimeError("Image token count does not match image embedding length in fallback path.")

        text_embeds = text_embeds.clone()
        text_embeds[image_token_mask] = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0, 1)

        original_forward = multi_model.forward

        def forward_without_logits_to_keep(*f_args, **f_kwargs):
            f_kwargs.pop("logits_to_keep", None)
            f_kwargs.pop("pixel_values", None)
            return original_forward(*f_args, **f_kwargs)

        multi_model.forward = forward_without_logits_to_keep
        try:
            language_model = multi_model.get_model().language_model
            attention_mask = torch.ones(
                (text_embeds.shape[0], text_embeds.shape[1]),
                device=text_embeds.device,
                dtype=torch.long,
            )

            generated = []
            for _ in range(args.max_new_tokens):
                position_ids = torch.cumsum(attention_mask, dim=1) - 1
                position_ids[position_ids < 0] = 0
                outputs = language_model(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask.bool(),
                    position_ids=position_ids,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                next_token_logits = multi_model.lm_head(outputs.last_hidden_state[:, -1, :])
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated.append(next_token)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                next_embed = language_model.embed_tokens(next_token)
                text_embeds = torch.cat([text_embeds, next_embed], dim=1)
                next_attention = torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, next_attention], dim=1)

            if generated:
                generated_ids = torch.cat(generated, dim=1)[0]
            else:
                generated_ids = torch.empty((0,), device=text_embeds.device, dtype=torch.long)
        finally:
            multi_model.forward = original_forward

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(answer)


if __name__ == "__main__":
    main()
