"""
Convert FSC147 point annotations into scaffolded grounded-counting JSONL.

The output is designed to be easy to load with the datasets library while also
remaining compatible with UniCount's existing understanding SFT format.
Each JSONL record includes:
  - image path
  - split/category/count metadata
  - normalized point coordinates in [0, 1000]
  - a scaffolded target string with <|thought|>, <|scaffold|>, and <|count|>
  - a two-turn conversations payload for SFT
"""

import argparse
import json
import os


USER_TEMPLATE = (
    "<image>\n"
    "How many {category} are in this image? "
    "Reason through the spatial arrangement and provide coordinates for every instance."
)


def resolve_fsc_root(root_arg):
    """Support both /.../FSC-147 and /.../FSC-147/FSC147_384_V2 layouts."""
    required = [
        "images_384_VarV2",
        "annotation_FSC147_384.json",
        "ImageClasses_FSC147.txt",
        "Train_Test_Val_FSC_147.json",
    ]

    def has_required(path):
        return all(os.path.exists(os.path.join(path, entry)) for entry in required)

    if has_required(root_arg):
        return root_arg

    nested = os.path.join(root_arg, "FSC147_384_V2")
    if has_required(nested):
        return nested

    raise FileNotFoundError(
        f"Could not resolve FSC root from '{root_arg}'. Expected required files in root or root/FSC147_384_V2"
    )


def load_image_classes(class_path):
    image_classes = {}
    with open(class_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                image_classes[parts[0].strip()] = parts[1].strip().lower()
    return image_classes


def normalize_point(point, width, height):
    x_coord, y_coord = point
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size ({width}, {height})")

    normalized_x = int(round((float(x_coord) / float(width)) * 1000.0))
    normalized_y = int(round((float(y_coord) / float(height)) * 1000.0))

    normalized_x = max(0, min(1000, normalized_x))
    normalized_y = max(0, min(1000, normalized_y))
    return [normalized_x, normalized_y]


def build_thought(category, count):
    noun = category if count == 1 else category
    return (
        f"The image shows {noun}. "
        "I will count from top-left to bottom-right and place one coordinate on each instance before giving the final total."
    )


def build_target_text(category, normalized_points):
    thought = build_thought(category, len(normalized_points))
    scaffold = json.dumps(normalized_points, separators=(",", ": "))
    return (
        f"<|thought|>\n{thought}\n"
        f"<|answer|>\n<|scaffold|> {scaffold}\n<|count|> {len(normalized_points)}"
    )


def build_record(image_name, split_name, image_path, category, annotation):
    width = annotation["W"]
    height = annotation["H"]
    points = annotation.get("points", [])
    normalized_points = [normalize_point(point, width, height) for point in points]
    prompt = USER_TEMPLATE.format(category=category)
    target_text = build_target_text(category, normalized_points)

    return {
        "id": image_name,
        "image": os.path.abspath(image_path),
        "split": split_name,
        "category": category,
        "ground_truth_count": len(points),
        "image_size": {"width": width, "height": height},
        "points_xy": points,
        "normalized_points_1000": normalized_points,
        "problem": prompt,
        "solution": target_text,
        "conversations": [
            {
                "from": "human",
                "value": prompt,
            },
            {
                "from": "gpt",
                "value": target_text,
            },
        ],
    }


def write_jsonl(records, output_path):
    with open(output_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fsc_root",
        required=True,
        help="Path to FSC147 root or its parent directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where train/val/test JSONL files will be written",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=0,
        help="Optional cap per split for smoke generation; 0 means no cap",
    )
    args = parser.parse_args()

    fsc_root = resolve_fsc_root(args.fsc_root)
    ann_path = os.path.join(fsc_root, "annotation_FSC147_384.json")
    cls_path = os.path.join(fsc_root, "ImageClasses_FSC147.txt")
    splits_path = os.path.join(fsc_root, "Train_Test_Val_FSC_147.json")
    image_dir = os.path.join(fsc_root, "images_384_VarV2")

    with open(ann_path) as handle:
        annotations = json.load(handle)
    with open(splits_path) as handle:
        splits = json.load(handle)
    image_classes = load_image_classes(cls_path)

    os.makedirs(args.output_dir, exist_ok=True)

    all_records = []
    total_skipped = 0
    for split_name in ["train", "val", "test"]:
        records = []
        skipped = 0

        for image_name in splits[split_name]:
            image_path = os.path.join(image_dir, image_name)
            annotation = annotations.get(image_name)

            if annotation is None or not os.path.exists(image_path):
                skipped += 1
                continue

            category = image_classes.get(image_name, "objects")
            record = build_record(image_name, split_name, image_path, category, annotation)
            records.append(record)

            if args.max_samples_per_split > 0 and len(records) >= args.max_samples_per_split:
                break

        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(records, output_path)
        all_records.extend(records)
        total_skipped += skipped
        print(f"{split_name}: wrote {len(records)} records to {output_path} ({skipped} skipped)")

    combined_path = os.path.join(args.output_dir, "all.jsonl")
    write_jsonl(all_records, combined_path)
    print(f"all: wrote {len(all_records)} combined records to {combined_path}")
    print(f"Done. Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()