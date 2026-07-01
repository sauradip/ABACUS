from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from unilip.model.builder import load_pretrained_model_general
from unilip.pipeline_gen import CustomGenPipeline
from unilip.utils import disable_torch_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate images with UniLIP from FSC-147 prompts JSON.")
    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=Path("/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_filename_class_count_prompt_qwen3vl.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/projects/u6bl/myprojects/UniLIP/generated_samples/fsc147"),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/projects/u6bl/myprojects/UniLIP/.resolved_models/UniLIP-3B",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def add_template(positive_prompt: str, cfg_prompt: str) -> list[str]:
    instruction = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<img>"
    return [instruction.format(input=positive_prompt), instruction.format(input=cfg_prompt)]


def load_records(prompts_json: Path) -> list[tuple[str, dict]]:
    raw = json.loads(prompts_json.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Expected filename-keyed prompt JSON.")
    return sorted(raw.items(), key=lambda item: int(Path(item[0]).stem))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    tokenizer, multi_model, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM",
        str(Path(args.model_path).expanduser()),
    )
    pipe = CustomGenPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer)

    records = load_records(args.prompts_json)
    records = records[args.start_index :]
    if args.limit > 0:
        records = records[: args.limit]

    generated = 0
    for idx, (image_name, record) in enumerate(tqdm(records, desc="UniLIP generating"), start=1):
        stem = Path(image_name).stem
        output_path = args.output_dir / f"{stem}_UniLIP.jpg"
        if args.skip_existing and output_path.exists():
            continue

        prompt = str(record["prompt"]).strip()
        model_prompts = add_template(
            f"Generate an image: {prompt}",
            "Generate an image.",
        )
        generator = torch.Generator(device=multi_model.device).manual_seed(args.seed + idx)
        gen_img = pipe(model_prompts, guidance_scale=args.guidance_scale, generator=generator)
        gen_img.save(output_path, format="JPEG", quality=95)
        generated += 1

        if idx % 25 == 0:
            print(f"[UniLIP] processed {idx}/{len(records)}, generated={generated}", flush=True)

    print(f"[UniLIP] finished. generated={generated} total_seen={len(records)} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
