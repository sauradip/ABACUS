#!/usr/bin/env python3
"""
Generation Evaluation Pipeline with YOLOv9 Counting Accuracy
- Generate count-aware images from 5 checkpoints
- Evaluate with YOLOv9 object detection
- Compare detected vs. expected counts
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '/data/amondal/UniCount/UniLIP')
sys.path.insert(0, '/data/amondal/UniCount')

from ultralytics import YOLO
import supervision as sv


class YOLOv9CountEvaluator:
    """Evaluate generated images using YOLOv9 counting accuracy"""

    def __init__(self, output_dir="evals/generation_yolo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLOv9e (larger, more accurate)
        print("[YOLOv9] Loading yolov9e.pt...")
        if not Path('yolov9e.pt').exists():
            print("✗ yolov9e.pt not found in current directory")
            print("  Place yolov9e.pt in /data/amondal/UniCount/dit_lora_gen/")
            sys.exit(1)

        self.yolo = YOLO('yolov9e.pt')
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def evaluate_image(self, image_path, expected_count, class_name):
        """
        Evaluate single image with YOLOv9

        Args:
            image_path: Path to generated image
            expected_count: Expected object count
            class_name: Target class name (e.g., 'car', 'person')

        Returns:
            dict with detection results
        """
        try:
            if not Path(image_path).exists():
                return {
                    "image": str(image_path),
                    "class_name": class_name,
                    "expected_count": expected_count,
                    "detected_count": 0,
                    "accuracy": False,
                    "error": "Image file not found"
                }

            # Load image
            pil_image = Image.open(image_path)

            # Run YOLOv9
            results = self.yolo(pil_image, conf=0.25, verbose=False)
            result = results[0]

            # Extract detections
            detections = sv.Detections.from_ultralytics(result)

            # Count matching class
            class_name_lower = class_name.lower()
            matching_count = 0
            detected_classes = []

            if len(detections) > 0:
                for i, (box, conf, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                    detected_name = result.names[int(class_id)].lower()
                    detected_classes.append(detected_name)

                    # Check if matches target class
                    if detected_name == class_name_lower or class_name_lower in detected_name or detected_name in class_name_lower:
                        matching_count += 1

            is_accurate = matching_count == expected_count

            # Create annotated image
            if len(detections) > 0:
                annotated = pil_image.copy()
                annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
                annotated = self.label_annotator.annotate(scene=annotated, detections=detections)
                return {
                    "image": str(image_path),
                    "class_name": class_name,
                    "expected_count": expected_count,
                    "detected_count": matching_count,
                    "total_detections": len(detections),
                    "detected_classes": detected_classes,
                    "accuracy": is_accurate,
                    "annotated_frame": annotated,
                    "error": None
                }
            else:
                return {
                    "image": str(image_path),
                    "class_name": class_name,
                    "expected_count": expected_count,
                    "detected_count": 0,
                    "total_detections": 0,
                    "detected_classes": [],
                    "accuracy": expected_count == 0,
                    "annotated_frame": pil_image,
                    "error": None
                }

        except Exception as e:
            return {
                "image": str(image_path),
                "class_name": class_name,
                "expected_count": expected_count,
                "detected_count": 0,
                "accuracy": False,
                "error": str(e)
            }

    def evaluate_batch(self, image_data_list, checkpoint_name):
        """
        Evaluate batch of generated images

        Args:
            image_data_list: List of dicts with keys: image_path, expected_count, class_name
            checkpoint_name: Name of checkpoint being evaluated
        """
        print(f"\n[YOLO] Evaluating {len(image_data_list)} images for {checkpoint_name}...")

        results = []
        for data in tqdm(image_data_list, desc="YOLOv9 eval"):
            result = self.evaluate_image(
                data['image_path'],
                data['expected_count'],
                data['class_name']
            )
            result['checkpoint'] = checkpoint_name
            results.append(result)

        return results

    def compute_metrics(self, results_df):
        """Compute accuracy metrics from results"""
        total = len(results_df)
        accurate = (results_df['accuracy'] == True).sum()
        accuracy = accurate / total if total > 0 else 0

        # Per-class accuracy
        per_class = {}
        for class_name in results_df['class_name'].unique():
            class_mask = results_df['class_name'] == class_name
            class_acc = (results_df[class_mask]['accuracy'] == True).sum() / class_mask.sum()
            per_class[class_name] = class_acc

        # Per-count accuracy
        per_count = {}
        for count in sorted(results_df['expected_count'].unique()):
            count_mask = results_df['expected_count'] == count
            count_acc = (results_df[count_mask]['accuracy'] == True).sum() / count_mask.sum()
            per_count[f"count_{count}"] = count_acc

        metrics = {
            "overall_accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": accurate,
            "per_class_accuracy": per_class,
            "per_count_accuracy": per_count
        }

        return metrics

    def save_results(self, results, checkpoint_name):
        """Save evaluation results"""
        ckpt_dir = self.output_dir / checkpoint_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'annotated_frame'}
            for r in results
        ])
        csv_file = ckpt_dir / "yolo_eval_results.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"✓ Results saved to {csv_file}")

        # Save metrics
        metrics = self.compute_metrics(results_df)
        metrics_file = ckpt_dir / "yolo_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to {metrics_file}")

        # Save annotated images
        images_dir = ckpt_dir / "annotated_images"
        images_dir.mkdir(exist_ok=True)
        for result in results:
            if 'annotated_frame' in result and result['annotated_frame'] is not None:
                img_name = Path(result['image']).stem + "_annotated.png"
                result['annotated_frame'].save(images_dir / img_name)

        print(f"✓ Annotated images saved to {images_dir}")
        return results_df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="/data/amondal/UniCount/dit_lora_gen/runs/dit_lora_r128",
                       help="Base checkpoint directory")
    parser.add_argument("--gen_dir", default="/data/amondal/UniCount/outputs/dit_lora_gen/generated_images",
                       help="Directory with generated images")
    parser.add_argument("--output_dir", default="evals/generation_yolo",
                       help="Output directory for evaluation results")
    parser.add_argument("--checkpoints", nargs="+",
                       default=["checkpoint-200", "checkpoint-400", "checkpoint-600", "checkpoint-800", "checkpoint-1000"],
                       help="Checkpoints to evaluate")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Generation Evaluation Pipeline - YOLOv9 Counting Accuracy")
    print("DiT LoRA Config A (r=128)")
    print("="*80)

    evaluator = YOLOv9CountEvaluator(args.output_dir)

    # Summary results
    summary = {}

    print("\nNOTE: This is a template. Actual generation should be integrated with:")
    print("  - UniLIP generation pipeline")
    print("  - CoCoCount benchmark")
    print("  - T2I-CompBench evaluation")
    print("  - GenEval metrics")
    print("\nTo complete the evaluation, generated images should be placed in:")
    print(f"  {args.gen_dir}/{{checkpoint_name}}/")
    print("  with naming: {{count}}__{{class_name}}_{{sample_id}}.png")

    print("\nExample directory structure:")
    print(f"  {args.gen_dir}/")
    print("  ├── checkpoint-200/")
    print("  │   ├── 2__car_0.png")
    print("  │   ├── 2__car_1.png")
    print("  │   ├── 3__person_0.png")
    print("  │   └── ...")
    print("  ├── checkpoint-400/")
    print("  └── ...")

    print("\n✓ Template evaluation pipeline ready")
    print(f"✓ Results will be saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Generate images using trained checkpoints")
    print("2. Organize generated images in expected directory structure")
    print("3. Run this evaluation script on generated images")
    print("4. Compare YOLOv9 accuracy across checkpoints")


if __name__ == "__main__":
    main()
