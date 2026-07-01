"""
Scaffold-R1 GRPO launcher and local reward-audit utilities.

This script does two jobs:
  1. Validate scaffold GRPO data and expose the reward wrapper shape expected by GRPO.
  2. Launch the external VLM-R1 GRPO trainer once that dependency is available.

It does not reimplement PPO/GRPO internally; instead, it prepares the exact launch
arguments and local reward plumbing so Stage 2 can start immediately after Stage 1.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.counting_grpo.grpo_rewards import compute_scaffold_reward, group_normalize_rewards


DEFAULT_VLMR1_DIR = "/projects/u6bl/myprojects/VLM-R1/src/open-r1-multimodal"
DEFAULT_STAGE1_CHECKPOINT = "checkpoints/native_sft_stage1_r64_lr2e4/checkpoint-1145"


def load_jsonl_records(data_path):
    records = []
    with open(data_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def validate_scaffold_record(record):
    required_fields = ["image", "problem", "solution", "ground_truth_count", "normalized_points_1000"]
    missing = [field_name for field_name in required_fields if field_name not in record]
    if missing:
        raise ValueError(f"Missing required scaffold fields: {missing}")

    if not isinstance(record["normalized_points_1000"], list):
        raise ValueError("normalized_points_1000 must be a list")


def reward_function(prompts, completions, **kwargs):
    del prompts
    gt_points = kwargs.get("gt_points", kwargs.get("normalized_points_1000", []))
    gt_counts = kwargs.get("gt_count", kwargs.get("ground_truth_count", []))

    rewards = []
    for completion, target_points, target_count in zip(completions, gt_points, gt_counts):
        rewards.append(compute_scaffold_reward(completion, target_points, target_count))
    return rewards


def audit_rewards(records, limit, normalize_within_group):
    subset = records[:limit]
    if not subset:
        return []

    completions = [record["solution"] for record in subset]
    prompts = [record["problem"] for record in subset]
    gt_points = [record["normalized_points_1000"] for record in subset]
    gt_count = [record["ground_truth_count"] for record in subset]
    raw_rewards = reward_function(prompts, completions, gt_points=gt_points, gt_count=gt_count)
    normalized_rewards = group_normalize_rewards(raw_rewards) if normalize_within_group else None

    audit_rows = []
    for index, record in enumerate(subset):
        audit_rows.append({
            "id": record.get("id", str(index)),
            "count": record["ground_truth_count"],
            "num_points": len(record["normalized_points_1000"]),
            "raw_reward": raw_rewards[index],
            "normalized_reward": None if normalized_rewards is None else normalized_rewards[index],
        })
    return audit_rows


def _normalize_lora_key_namespace(raw_key):
    key = raw_key

    # Stage-1 checkpoints may be wrapped under InternVL container paths.
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    if not (".lora_A." in key or ".lora_B." in key):
        return None

    # PEFT's safetensor loader expects no adapter-name segment in key suffix.
    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")

    # If keys resolve to plain model.layers.*, re-anchor to PEFT base namespace.
    if key.startswith("model.layers."):
        key = f"base_model.model.{key}"

    return key


def _materialize_native_adapter_from_full_checkpoint(model_path):
    model_dir = Path(model_path)
    adapter_config = model_dir / "adapter_config.json"
    adapter_weights = model_dir / "adapter_model.safetensors"

    if adapter_config.exists() and adapter_weights.exists():
        return str(model_dir)

    full_weights = model_dir / "model.safetensors"
    if not full_weights.exists():
        return str(model_dir)

    full_state = safetensors_load_file(str(full_weights), device="cpu")
    adapter_state = {}
    rank = None
    targets = set()

    for key, tensor in full_state.items():
        normalized = _normalize_lora_key_namespace(key)
        if normalized is None:
            continue
        adapter_state[normalized] = tensor

        parts = normalized.split(".")
        if "lora_A" in parts and tensor.ndim == 2 and rank is None:
            rank = int(tensor.shape[0])
        if "lora_A" in parts or "lora_B" in parts:
            marker = "lora_A" if "lora_A" in parts else "lora_B"
            marker_index = parts.index(marker)
            if marker_index > 0:
                targets.add(parts[marker_index - 1])

    if not adapter_state:
        return str(model_dir)

    if rank is None:
        raise RuntimeError(f"Could not infer LoRA rank from {full_weights}")

    adapter_dir = model_dir / "native_peft_adapter_stage2"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    safetensors_save_file(adapter_state, str(adapter_dir / "adapter_model.safetensors"))
    with open(adapter_dir / "adapter_config.json", "w") as handle:
        json.dump(
            {
                "base_model_name_or_path": "OpenGVLab/InternVL2-2B",
                "bias": "none",
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "r": rank,
                "target_modules": sorted(targets),
                "task_type": "CAUSAL_LM",
            },
            handle,
            indent=2,
        )

    print(
        "[INFO] Materialized Stage-2 native adapter "
        f"at {adapter_dir} (rank={rank}, targets={sorted(targets)}, keys={len(adapter_state)})"
    )
    return str(adapter_dir)


def build_launch_command(args):
    vlmr1_dir = Path(args.vlmr1_dir)
    grpo_entry = vlmr1_dir / "src" / "open_r1" / "grpo_jsonl.py"
    if not grpo_entry.exists():
        raise FileNotFoundError(
            f"VLM-R1 trainer not found at {grpo_entry}. Set --vlmr1-dir to the open-r1-multimodal root."
        )

    model_name_or_path = _materialize_native_adapter_from_full_checkpoint(args.model_name_or_path)

    command = [
        args.python_executable,
        "-m",
        "accelerate.commands.launch",
        "--dynamo_backend",
        "no",
        "--num_processes",
        str(args.num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16" if args.bf16 else "fp16",
        str(grpo_entry),
        "--dataset_name",
        "this_is_not_used",
        "--use_vllm",
        "False",
        "--model_name_or_path",
        model_name_or_path,
        "--data_file_paths",
        args.data_path,
        "--image_folders",
        "/",
        "--output_dir",
        args.output_dir,
        "--reward_funcs",
        args.reward_name,
        "--num_generations",
        str(args.num_generations),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_completion_length",
        str(args.max_completion_length),
        "--learning_rate",
        str(args.learning_rate),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--beta",
        str(args.beta),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--num_iterations",
        str(args.num_iterations),
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--attn_implementation",
        args.attn_implementation,
        "--gradient_checkpointing",
        "True",
        "--gradient_checkpointing_kwargs",
        '{"use_reentrant": false}',
        "--report_to",
        "none",
        "--trust_remote_code",
        "True",
        "--ddp_find_unused_parameters",
        "True",
    ]
    if args.bf16:
        command.append("--bf16")
    return command, grpo_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="outputs/fsc147_scaffold_full/train.jsonl")
    parser.add_argument(
        "--model_name_or_path",
        default=DEFAULT_STAGE1_CHECKPOINT,
        help="Stage 1 SFT checkpoint to initialize GRPO",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vlmr1_dir", default=DEFAULT_VLMR1_DIR, help="Path to the open-r1-multimodal root")
    parser.add_argument("--python_executable", default="python")
    parser.add_argument("--reward_name", default="scaffold_r1", help="Registry key to use after reward registration in VLM-R1")
    parser.add_argument("--task_type", default="counting")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--audit_only", action="store_true", help="Only validate data and print reward audit samples")
    parser.add_argument("--audit_limit", type=int, default=8)
    parser.add_argument("--print_command", action="store_true")
    parser.add_argument("--run_launch", action="store_true", help="Actually launch VLM-R1 after validation")
    args = parser.parse_args()

    records = load_jsonl_records(args.data_path)
    if not records:
        raise RuntimeError(f"No records found in {args.data_path}")

    for record in records[: min(8, len(records))]:
        validate_scaffold_record(record)

    audit_rows = audit_rewards(records, args.audit_limit, normalize_within_group=True)
    print("=== Scaffold-R1 Reward Audit ===")
    for row in audit_rows:
        print(
            f"{row['id']}: count={row['count']} points={row['num_points']} "
            f"raw={row['raw_reward']:.4f} norm={row['normalized_reward']:.4f}"
        )

    if args.audit_only and not args.run_launch and not args.print_command:
        return

    try:
        command, grpo_entry = build_launch_command(args)
    except FileNotFoundError as exc:
        if args.print_command and not args.run_launch:
            print("")
            print(str(exc))
            print("Reward registration required in VLM-R1: scaffold_r1_reward_func -> reward_funcs_registry['%s']" % args.reward_name)
            return
        raise

    print("")
    print(f"VLM-R1 entry: {grpo_entry}")
    print("Reward registration required in VLM-R1: scaffold_r1_reward_func -> reward_funcs_registry['%s']" % args.reward_name)
    print("Command:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.print_command and not args.run_launch:
        return

    if args.run_launch:
        subprocess.run(command, cwd=args.vlmr1_dir, check=True)


if __name__ == "__main__":
    main()