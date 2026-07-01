import argparse
import json
from pathlib import Path

import webdataset as wds
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 10_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FSC-147 prompt+image pairs into WebDataset shards for UniLIP TI2I "
            "(input-image + output-image) fine-tuning."
        )
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=Path(
            "/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_filename_class_count_prompt_qwen3vl.json"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/projects/u6bl/myprojects/Datasets/FSC-147/images_384_VarV2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/projects/u6bl/myprojects/Datasets/FSC-147/edit_sft_fsc147"),
    )
    parser.add_argument("--maxcount", type=int, default=10000, help="Samples per tar shard.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing images instead of skipping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.prompts_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, dict):
        raise ValueError("Expected filename-keyed JSON object.")

    ordered_items = sorted(records.items(), key=lambda kv: int(Path(kv[0]).stem))

    out_pattern = str(args.output_dir / "%06d.tar")
    writer = wds.ShardWriter(out_pattern, maxcount=args.maxcount)

    kept = 0
    skipped = 0

    for _, (file_name, row) in enumerate(tqdm(ordered_items, desc="Converting FSC-147 TI2I")):
        image_path = args.image_dir / file_name
        prompt = str(row.get("prompt", "")).strip()

        if not image_path.exists():
            if args.strict:
                raise FileNotFoundError(f"Missing image: {image_path}")
            skipped += 1
            continue

        if not prompt:
            skipped += 1
            continue

        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            # TI2I format requires both input and output images; here we supervise
            # understanding/alignment by using the same FSC image as source and target.
            writer.write(
                {
                    "__key__": f"{kept:08d}",
                    "input.jpg": rgb,
                    "output.jpg": rgb,
                    "txt": prompt,
                }
            )
        kept += 1

    writer.close()

    print(
        f"[convert_fsc147_ti2i] done. kept={kept} skipped={skipped} "
        f"output_dir={args.output_dir} shards={len(list(args.output_dir.glob('*.tar')))}"
    )


if __name__ == "__main__":
    main()
