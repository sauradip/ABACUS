import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.counting_grpo.train_native_sft import DataArguments, NativeDataCollator, NativeSFTDataset


def shorten(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n... [truncated {len(text) - max_chars} chars] ...\n\n{tail}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exact SFT tensors for one sample.")
    parser.add_argument("--model", default="OpenGVLab/InternVL2-2B")
    parser.add_argument(
        "--data",
        default="outputs/fsc147_scaffold_full/train.jsonl",
        help="Path to train jsonl/json relative to repo root or absolute",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=6144)
    parser.add_argument("--dump-chars", type=int, default=6000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right",
        model_max_length=args.model_max_length,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
        )

    data_args = DataArguments(data_path=str(data_path), image_processor=image_processor, num_image_token=256)
    dataset = NativeSFTDataset(str(data_path), tokenizer, data_args)
    collator = NativeDataCollator(tokenizer=tokenizer)

    sample = [dataset[args.sample_index]]
    batch = collator(sample)

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    seq_len = int(input_ids.shape[0])
    valid_label_mask = labels.ne(-100)
    valid_label_ids = labels[valid_label_mask]

    input_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    label_text = tokenizer.decode(valid_label_ids.tolist(), skip_special_tokens=False)

    has_scaffold = "<|scaffold|>" in label_text
    has_count = "<|count|>" in label_text
    starts_with_prompt = "How many" in label_text[:200]
    truncated_at_cap = seq_len >= args.model_max_length

    print(f"Data path: {data_path}")
    print(f"Sample index: {args.sample_index}")
    print(f"Total Sequence Length: {seq_len}")
    print(f"Model max length: {args.model_max_length}")
    print(f"Truncated at cap: {truncated_at_cap}")
    print(f"Unmasked label token count: {int(valid_label_ids.shape[0])}")
    print(f"Label contains <|scaffold|>: {has_scaffold}")
    print(f"Label contains <|count|>: {has_count}")
    print(f"Label starts like prompt text: {starts_with_prompt}")

    print("\n=== INPUT_IDS DECODE (trimmed) ===")
    print(shorten(input_text, args.dump_chars))

    print("\n=== UNMASKED LABELS DECODE (exact supervised target, trimmed) ===")
    print(shorten(label_text, args.dump_chars))

    # Optional machine-readable summary for quick automation/debugging.
    summary = {
        "data_path": str(data_path),
        "sample_index": args.sample_index,
        "total_sequence_length": seq_len,
        "model_max_length": args.model_max_length,
        "truncated_at_cap": truncated_at_cap,
        "unmasked_label_token_count": int(valid_label_ids.shape[0]),
        "label_has_scaffold": has_scaffold,
        "label_has_count": has_count,
        "label_starts_with_prompt": starts_with_prompt,
    }
    print("\n=== SUMMARY_JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
