"""
Generate counting GRPO training data from FSC-147 in VLM-R1 JSONL format.

Uses the SAME prompt format as the counting SFT training to avoid
any distribution shift. The model sees the same prompts it was trained on.

The 'solution' field contains the GT count as a string.
VLM-R1 passes this to the reward function for scoring.
"""

import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsc_root", required=True,
                        help="Path to FSC-147 root (FSC147_384_V2/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for JSONL files")
    args = parser.parse_args()

    # Load FSC-147 metadata
    ann_path = os.path.join(args.fsc_root, "annotation_FSC147_384.json")
    cls_path = os.path.join(args.fsc_root, "ImageClasses_FSC147.txt")
    splits_path = os.path.join(args.fsc_root, "Train_Test_Val_FSC_147.json")
    img_dir = os.path.join(args.fsc_root, "images_384_VarV2")

    with open(ann_path) as f:
        annotations = json.load(f)

    # Load class names
    image_classes = {}
    with open(cls_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                image_classes[parts[0].strip()] = parts[1].strip().lower()

    with open(splits_path) as f:
        splits = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # CRITICAL: Use the EXACT same prompt format as the counting SFT.
    # The SFT used: "How many {class_name} are present in this image? Answer with only a number."
    # DO NOT change this wording. Any change = distribution shift = lower starting reward.
    PROMPT_TEMPLATE = "How many {category} are present in this image? Answer with only a number."

    for split_name in ["train", "val", "test"]:
        records = []
        skipped = 0

        for img_name in splits[split_name]:
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            if img_name not in annotations:
                skipped += 1
                continue

            # Count = number of dot annotations
            count = len(annotations[img_name]["points"])
            category = image_classes.get(img_name, "objects")

            prompt = PROMPT_TEMPLATE.format(category=category)

            records.append({
                "image": os.path.abspath(img_path),
                "problem": prompt,
                "solution": str(count),
            })

        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"{split_name}: {len(records)} samples, {skipped} skipped → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
