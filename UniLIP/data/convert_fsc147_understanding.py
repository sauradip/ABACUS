import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input FSC-147 YOLO JSON")
    parser.add_argument("--output", type=str, required=True, help="Output SFT JSON")
    parser.add_argument("--image-dir", type=str, required=True, help="Base directory of the images")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    entries = data.get('entries', data)
    
    sft_data = []
    
    for key, item in entries.items():
        image_name = item.get("image_key", key)
        class_name = item.get("class_name", "objects")
        gt_count = item.get("gt_count", 0)

        image_path = os.path.join(args.image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping.")
            continue

        conv = [
            {
                "from": "human",
                "value": f"How many {class_name} are present in this image? Answer with only a number.\n<image>"
            },
            {
                "from": "gpt",
                "value": str(gt_count)
            }
        ]

        sft_data.append({
            "id": image_name,
            "image": image_path,
            "conversations": conv
        })

    with open(args.output, 'w') as f:
        json.dump(sft_data, f, indent=4)
    
    print(f"Successfully wrote {len(sft_data)} entries to {args.output}")

if __name__ == "__main__":
    main()
