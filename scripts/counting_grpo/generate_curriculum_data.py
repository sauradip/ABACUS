#!/usr/bin/env python3
"""
generate_curriculum_data.py

Curriculum of Complexity — Stage 1.6b & Stage 3.2 data generation.

Outputs:
  curriculum_sft_low_30.jsonl   — High-fidelity tally traces for gt_count <= 30
  curriculum_grpo_full_v32.jsonl — Full schema (<=30) + summary schema (>30) for GRPO
"""
import argparse
import json
from pathlib import Path

SOURCE_JSONL = "outputs/scaffold_rex_5k_pca/all.jsonl"
SFT_OUT = "outputs/scaffold_rex_5k_pca/curriculum_sft_low_30.jsonl"
GRPO_OUT = "outputs/scaffold_rex_5k_pca/curriculum_grpo_full_v32.jsonl"
SFT_THRESHOLD = 30  # only images with gt_count <= this go into SFT


def generate_tally_trace(clusters: list, category: str) -> str:
    """Build a per-anchor Thought block with running subtotals."""
    lines = ["Thought:"]
    subtotal = 0
    for c in clusters:
        ax, ay = c["anchor"]
        n = c.get("count", 0)
        subtotal += n
        lines.append(f"Anchor ({ax},{ay}) contains {n} {category}. Subtotal: {subtotal}.")
    lines.append(f"Final Tally: {subtotal}.")
    return "\n".join(lines)


def build_sft_conversations(row: dict) -> list:
    """Return conversations list in the same format as train_stage15_sft.py expects."""
    category = row["category"]
    clusters = row["clusters"]
    gt = row["ground_truth_count"]
    thought = generate_tally_trace(clusters, category)

    # The human turn with the Assistant: Thought: prefill suffix
    human_val = row["conversations"][0]["value"]
    if not human_val.rstrip().endswith("Assistant:"):
        # Strip any old prefill suffix, then add fresh one
        human_val = human_val.rstrip()
        if human_val.endswith("Assist"):
            human_val = human_val[:-6].rstrip()
        human_val += "\nAssistant: Thought:"

    # The gpt turn: tally trace + compact JSON answer
    json_answer = json.dumps({
        "clusters": clusters,
        "total_count": gt,
    }, ensure_ascii=False)
    gpt_val = f"{thought}\n{json_answer}"

    return [
        {"from": "human", "value": human_val},
        {"from": "gpt", "value": gpt_val},
    ]


def detect_quadrant_density(clusters: list) -> str:
    """Identify which 6x6 quadrants (TL/TR/BL/BR) have the most density."""
    quad = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for c in clusters:
        ax, ay = c["anchor"]
        q = ("T" if ax <= 3 else "B") + ("L" if ay <= 3 else "R")
        quad[q] += c.get("count", 0)
    hot = [k for k, v in sorted(quad.items(), key=lambda x: -x[1]) if v > 0]
    if len(hot) >= 2:
        return f"High density detected in quadrants {hot[0]} and {hot[1]}."
    elif hot:
        return f"High density detected in quadrant {hot[0]}."
    return "Distributed density across all quadrants."


def build_grpo_row_high_density(row: dict) -> dict:
    """For gt_count > 30: strip conversations to summary schema."""
    gt = row["ground_truth_count"]
    clusters = row["clusters"]
    density_msg = detect_quadrant_density(clusters)

    target = json.dumps({
        "density_summary": density_msg,
        "total_count": gt,
    }, ensure_ascii=False)

    out = {k: v for k, v in row.items() if k != "conversations"}
    out["response_schema"] = "summary"
    out["target_response"] = target
    out["stage"] = "stage32_grpo_summary"

    # Minimal conversations kept for prompt construction in GRPO trainer
    human_val = row["conversations"][0]["value"]
    out["conversations"] = [
        {"from": "human", "value": human_val},
        {"from": "gpt", "value": target},
    ]
    return out


def build_data(source: str, sft_out: str, grpo_out: str, threshold: int) -> None:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    rows = [json.loads(line) for line in src.open()]

    sft_rows = []
    grpo_rows = []
    sft_mismatch = 0

    for row in rows:
        gt = row.get("ground_truth_count", 0)
        clusters = row.get("clusters", [])

        # Verify cluster sum matches gt before emitting
        cluster_sum = sum(c.get("count", 0) for c in clusters)

        if gt <= threshold:
            # Validate arithmetic
            if cluster_sum != gt:
                sft_mismatch += 1
                # Still emit for GRPO but skip for SFT
            else:
                sft_row = dict(row)
                sft_row["conversations"] = build_sft_conversations(row)
                sft_row["stage"] = "stage16b_curriculum_sft"
                sft_rows.append(sft_row)
            # GRPO: full schema
            grpo_row = dict(row)
            grpo_row["response_schema"] = "full"
            grpo_row["stage"] = "stage32_grpo_full"
            grpo_rows.append(grpo_row)
        else:
            # GRPO: summary schema for high-density images
            grpo_rows.append(build_grpo_row_high_density(row))

    # Write outputs
    Path(sft_out).parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(grpo_out, "w") as f:
        for r in grpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Source rows        : {len(rows)}")
    print(f"SFT rows (<=30)    : {len(sft_rows)}  (skipped {sft_mismatch} cluster-sum mismatches)")
    print(f"GRPO rows (full)   : {sum(1 for r in grpo_rows if r.get('response_schema')=='full')}")
    print(f"GRPO rows (summary): {sum(1 for r in grpo_rows if r.get('response_schema')=='summary')}")
    print(f"GRPO total         : {len(grpo_rows)}")
    print(f"SFT  → {sft_out}")
    print(f"GRPO → {grpo_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_JSONL)
    parser.add_argument("--sft_out", default=SFT_OUT)
    parser.add_argument("--grpo_out", default=GRPO_OUT)
    parser.add_argument("--threshold", type=int, default=SFT_THRESHOLD)
    args = parser.parse_args()
    build_data(args.source, args.sft_out, args.grpo_out, args.threshold)
