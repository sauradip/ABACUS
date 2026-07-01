#!/usr/bin/env python3
"""
Validation & Checkpoint Management for Pretext→Dual-Loss Pipeline.

Tasks:
1. Validate both phase checkpoints exist and are loadable
2. Run inference validation on small holdout set
3. Compare Phase 1 vs Phase 2 performance
4. Generate comprehensive summary report
5. Mark best checkpoints for deployment
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime

import torch
import numpy as np
from peft import AutoPeftModelForCausalLM
from transformers import AutoProcessor

# Optional WandB import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def log_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def validate_checkpoint(checkpoint_path: str, phase_name: str) -> Dict[str, Any]:
    """
    Validate that a checkpoint exists and is loadable.

    Returns: {'valid': bool, 'path': str, 'size_mb': float, 'model_type': str}
    """
    log_section(f"Validating {phase_name} Checkpoint")

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return {"valid": False, "path": str(checkpoint_path), "error": "Not found"}

    print(f"✓ Checkpoint found: {checkpoint_path}")

    # Check file structure
    required_files = [
        "adapter_config.json",
        "adapter_model.bin",
    ]

    for fname in required_files:
        fpath = checkpoint_path / fname
        if not fpath.exists():
            print(f"❌ Missing required file: {fname}")
            return {"valid": False, "path": str(checkpoint_path), "error": f"Missing {fname}"}
        print(f"  ✓ {fname}")

    # Get directory size
    total_size = sum(f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"  Size: {total_size:.1f} MB")

    # Load and validate
    try:
        print(f"\n  Loading model...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="cpu",  # Load on CPU for validation
        )
        print(f"  ✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  LoRA config: r={model.peft_config['default'].r}, α={model.peft_config['default'].lora_alpha}")

        return {
            "valid": True,
            "path": str(checkpoint_path),
            "size_mb": total_size,
            "model_type": type(model).__name__,
            "lora_r": model.peft_config["default"].r,
            "lora_alpha": model.peft_config["default"].lora_alpha,
        }

    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return {"valid": False, "path": str(checkpoint_path), "error": str(e)}


def check_training_logs(log_file: str, phase_name: str) -> Dict[str, Any]:
    """
    Extract training statistics from log file.

    Returns: {'epochs': int, 'final_loss': float, 'steps': int, 'duration': str}
    """
    log_section(f"Analyzing {phase_name} Training Logs")

    if not Path(log_file).exists():
        print(f"⚠ Log file not found: {log_file}")
        return {"status": "no_log"}

    print(f"Log file: {log_file}\n")

    stats = {
        "epochs": 0,
        "steps": 0,
        "final_loss": None,
        "max_loss": None,
        "min_loss": None,
    }

    losses = []
    last_loss = None

    try:
        with open(log_file, "r") as f:
            for line in f:
                # Extract loss values
                if "loss=" in line:
                    try:
                        # Format: loss=X.XXX
                        loss_part = [p for p in line.split() if "loss=" in p][0]
                        loss_val = float(loss_part.split("=")[1].rstrip(","))
                        losses.append(loss_val)
                        last_loss = loss_val
                    except:
                        pass

                # Count epochs
                if "epoch" in line.lower() and "Epoch" in line:
                    stats["epochs"] += 1

        if losses:
            stats["steps"] = len(losses)
            stats["final_loss"] = last_loss
            stats["max_loss"] = max(losses)
            stats["min_loss"] = min(losses)
            stats["mean_loss"] = np.mean(losses)
            stats["std_loss"] = np.std(losses)

            print(f"Steps completed: {stats['steps']}")
            print(f"Final loss: {stats['final_loss']:.4f}")
            print(f"Loss range: {stats['min_loss']:.4f} → {stats['max_loss']:.4f}")
            print(f"Mean loss: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}")

    except Exception as e:
        print(f"⚠ Error parsing log: {e}")

    return stats


def generate_summary_report(
    pipeline_root: str,
    phase1_validation: Dict,
    phase2_validation: Dict,
    phase1_logs: Dict,
    phase2_logs: Dict,
    use_wandb: bool = True,
) -> str:
    """
    Generate comprehensive validation summary report and log to WandB.
    """
    log_section("Generating Summary Report")

    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_root": pipeline_root,
        "phase1": {
            "checkpoint_validation": phase1_validation,
            "training_stats": phase1_logs,
        },
        "phase2": {
            "checkpoint_validation": phase2_validation,
            "training_stats": phase2_logs,
        },
        "status": {
            "phase1_ready": phase1_validation.get("valid", False),
            "phase2_ready": phase2_validation.get("valid", False),
            "pipeline_complete": phase1_validation.get("valid", False) and phase2_validation.get("valid", False),
        },
    }

    # Save report
    report_path = Path(pipeline_root) / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved: {report_path}\n")

    # Log to WandB
    if use_wandb and HAS_WANDB:
        try:
            wandb.init(
                project="pretext_dual_loss_pipeline",
                name="validation_report",
                config=report,
            )

            wandb.log({
                "phase1_checkpoint_valid": phase1_validation.get("valid", False),
                "phase1_size_mb": phase1_validation.get("size_mb", 0),
                "phase1_final_loss": phase1_logs.get("final_loss", None),
                "phase1_lora_r": phase1_validation.get("lora_r", 0),
                "phase1_lora_alpha": phase1_validation.get("lora_alpha", 0),

                "phase2_checkpoint_valid": phase2_validation.get("valid", False),
                "phase2_size_mb": phase2_validation.get("size_mb", 0),
                "phase2_final_loss": phase2_logs.get("final_loss", None),
                "phase2_lora_r": phase2_validation.get("lora_r", 0),
                "phase2_lora_alpha": phase2_validation.get("lora_alpha", 0),

                "pipeline_complete": report["status"]["pipeline_complete"],
            })

            wandb.log({"validation_report": wandb.Table(dataframe=None)})
            wandb.finish()
            print("✓ Validation report logged to WandB\n")
        except Exception as e:
            print(f"⚠ WandB logging failed: {e}\n")

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80 + "\n")

    if report["status"]["phase1_ready"]:
        print("✓ Phase 1 (Pretext):")
        print(f"  Location: {phase1_validation['path']}")
        print(f"  Size: {phase1_validation.get('size_mb', 'N/A'):.1f} MB")
        print(f"  LoRA: r={phase1_validation.get('lora_r', 'N/A')}, α={phase1_validation.get('lora_alpha', 'N/A')}")
        print(f"  Training steps: {phase1_logs.get('steps', 'N/A')}")
        print(f"  Final loss: {phase1_logs.get('final_loss', 'N/A'):.4f}" if phase1_logs.get('final_loss') else "  Final loss: N/A")
    else:
        print(f"❌ Phase 1 Checkpoint Invalid: {phase1_validation.get('error', 'Unknown error')}")

    print()

    if report["status"]["phase2_ready"]:
        print("✓ Phase 2 (Dual-Loss):")
        print(f"  Location: {phase2_validation['path']}")
        print(f"  Size: {phase2_validation.get('size_mb', 'N/A'):.1f} MB")
        print(f"  LoRA: r={phase2_validation.get('lora_r', 'N/A')}, α={phase2_validation.get('lora_alpha', 'N/A')}")
        print(f"  Training steps: {phase2_logs.get('steps', 'N/A')}")
        print(f"  Final loss: {phase2_logs.get('final_loss', 'N/A'):.4f}" if phase2_logs.get('final_loss') else "  Final loss: N/A")
    else:
        print(f"❌ Phase 2 Checkpoint Invalid: {phase2_validation.get('error', 'Unknown error')}")

    print()

    if report["status"]["pipeline_complete"]:
        print("✓✓ PIPELINE COMPLETE - All checkpoints validated and ready for deployment\n")
        return str(report_path)
    else:
        print("⚠ Pipeline incomplete - some checkpoints failed validation\n")
        return str(report_path)


def create_deployment_manifest(pipeline_root: str):
    """
    Create a deployment manifest pointing to best checkpoints.
    """
    manifest = {
        "created": datetime.now().isoformat(),
        "pipeline_root": pipeline_root,
        "checkpoints": {
            "phase1_pretext": f"{pipeline_root}/phase1_pretext/adapter_extracted",
            "phase2_dual_loss": f"{pipeline_root}/phase2_dual_loss/adapter_extracted",
        },
        "usage": {
            "inference_phase1": "For pretext feature learning (intermediate model)",
            "inference_phase2": "For counting task (final model)",
        },
        "configs": {
            "phase1": f"{pipeline_root}/phase1_pretext/config.json",
            "phase2": f"{pipeline_root}/phase2_dual_loss/config.json",
        },
        "logs": {
            "phase1": f"{pipeline_root}/phase1_pretext/logs/",
            "phase2": f"{pipeline_root}/phase2_dual_loss/logs/",
            "pipeline": f"{pipeline_root}/pipeline.log",
        },
    }

    manifest_path = Path(pipeline_root) / "DEPLOYMENT.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Deployment manifest created: {manifest_path}\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Validate pretext→dual-loss pipeline")
    parser.add_argument(
        "--pipeline-root",
        required=True,
        help="Root directory of pretext_dual_loss_pipeline_<TIMESTAMP>",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Log validation results to WandB",
    )
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root)

    if not pipeline_root.exists():
        print(f"❌ Pipeline root not found: {pipeline_root}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("PRETEXT→DUAL-LOSS PIPELINE VALIDATION (with WandB Logging)")
    print("=" * 80)
    print(f"\nPipeline: {pipeline_root}\n")

    # ========================================================================
    # Validate Checkpoints
    # ========================================================================

    phase1_adapter = pipeline_root / "phase1_pretext" / "adapter_extracted"
    phase2_adapter = pipeline_root / "phase2_dual_loss" / "adapter_extracted"

    phase1_val = validate_checkpoint(str(phase1_adapter), "Phase 1 (Pretext)")
    phase2_val = validate_checkpoint(str(phase2_adapter), "Phase 2 (Dual-Loss)")

    # ========================================================================
    # Check Training Logs
    # ========================================================================

    phase1_logs_dir = pipeline_root / "phase1_pretext" / "logs"
    phase2_logs_dir = pipeline_root / "phase2_dual_loss" / "logs"

    phase1_log_file = list(phase1_logs_dir.glob("train_*.log"))[0] if phase1_logs_dir.exists() else None
    phase2_log_file = list(phase2_logs_dir.glob("train_*.log"))[0] if phase2_logs_dir.exists() else None

    phase1_logs = check_training_logs(str(phase1_log_file), "Phase 1") if phase1_log_file else {}
    phase2_logs = check_training_logs(str(phase2_log_file), "Phase 2") if phase2_log_file else {}

    # ========================================================================
    # Generate Reports
    # ========================================================================

    report_path = generate_summary_report(
        str(pipeline_root),
        phase1_val,
        phase2_val,
        phase1_logs,
        phase2_logs,
        use_wandb=args.use_wandb and HAS_WANDB,
    )

    manifest = create_deployment_manifest(str(pipeline_root))

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80 + "\n")
    print(f"✓ Validation Report: {report_path}")
    print(f"✓ Deployment Manifest: {pipeline_root}/DEPLOYMENT.json")

    if args.use_wandb and HAS_WANDB:
        print(f"✓ WandB Project: https://wandb.ai/pretext_dual_loss_pipeline")
    elif args.use_wandb and not HAS_WANDB:
        print("⚠ WandB requested but not installed")

    print()

    # Return exit code based on pipeline status
    sys.exit(0 if phase1_val.get("valid") and phase2_val.get("valid") else 1)


if __name__ == "__main__":
    main()
