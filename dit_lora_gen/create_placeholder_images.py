#!/usr/bin/env python3
"""Create placeholder images for YOLOv9 evaluation testing"""

from pathlib import Path
from PIL import Image
import json

categories = ["car", "person", "chair", "cup", "dog", "cat", "bottle", "book"]
counts = [2, 3, 5, 7, 10]
checkpoints = ["checkpoint-200", "checkpoint-400", "checkpoint-600", "checkpoint-800", "checkpoint-1000"]

base_dir = Path("outputs/dit_lora_gen/generated_images")

for ckpt in checkpoints:
    ckpt_dir = base_dir / ckpt
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder images
    for count in counts:
        for category in categories:
            for sample_idx in range(10):
                img_name = f"{count}__{category}_{sample_idx}.png"
                img_path = ckpt_dir / img_name
                
                # Create simple colored placeholder
                color = (73, 109, 137)
                img = Image.new('RGB', (512, 512), color=color)
                img.save(img_path)
    
    # Save metadata
    metadata = {
        "checkpoint": ckpt,
        "num_prompts": len(categories) * len(counts),
        "num_samples": 10,
        "total_images": len(categories) * len(counts) * 10,
        "output_directory": str(ckpt_dir)
    }
    with open(ckpt_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {metadata['total_images']} placeholder images for {ckpt}")

print("\n✓ All placeholder images created")
print(f"Location: {base_dir}/")
