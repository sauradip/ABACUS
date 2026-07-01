# Specification: Stage 3.2 GRPO (Recursive-Rex Protocol)

## 1. Executive Summary
This task implements **Stage 3.2 GRPO** initialized from the **Stage 1.6b (Low-Count SFT)** checkpoint. The goal is to enforce mathematical consistency and high-density accuracy while avoiding the "Lazy 20" mode collapse. We use a **Curriculum Schema** to handle high-density images without exceeding the 8,192 token context limit.



## 2. Environment & Checkpoint Isolation
**CRITICAL**: The coding agent must use these exact paths. Do NOT mix weights from Stage 2.5 or 3.1.
- **Base Model (Initialization)**: `/projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_pca_4371501` (Stage 1.6b weights).
- **Training Source**: `outputs/scaffold_rex_5k_pca/cross_density_5k.jsonl`.
- **Output Directory**: `checkpoints/stage32_grpo_r1rex_tally/`.

## 3. Task 1: Data Preparation (Curriculum Schema)
The agent must generate `outputs/scaffold_rex_5k_pca/curriculum_grpo_v32.jsonl`.

### Logic:
- **IF `gt_count` <= 30**: Set `response_schema = "full"`. The model must output the full thought-tally and cluster list.
- **IF `gt_count` > 30**: Set `response_schema = "summary"`. The model must output a density summary and final total to save tokens.

### Target Format (High Density):
```json
{
  "thought": "High density observed across all quadrants. Concentration peak in TL.",
  "total_count": 126
}
```

## 4. Task 2: The GRPO Reward Function
Implement `scripts/counting_grpo/grpo_reward_v32.py`. This script handles the **"Missing Brace" syndrome** seen in Stage 1.6b.



### Reward Components:
1.  **Structural Recovery ($R_{format}$)**:
    - If output starts with `"clusters"`, prepend `{` and append `}`.
    - `+1.0` for perfect JSON, `+0.8` for recovered JSON, `-1.0` for fatal failure.
2.  **Schema Compliance ($R_{schema}$)**:
    - `-1.0` if `gt > 30` but model generated a long `clusters` list (token overflow prevention).
3.  **Arithmetic Consistency ($R_{math}$)**:
    - `+1.0` if `sum(clusters) == total_count`.
4.  **Numerical Accuracy ($R_{acc}$)**:
    - Score: $1.0 - (|GT - Pred| / GT)$.
    - **Crucial**: This provides the precision gradient missing from Stage 3.1.

## 5. Task 3: Training Script (GRPO Logic)
The agent must update `scripts/counting_grpo/train_grpo_v32.py` to support **CrowdVLM-R1 Group Rollouts**.

- **Group Size ($G$)**: 8 (Generate 8 completions per prompt).
- **Loss**: Compare the reward of each completion against the group average to compute the advantage.
- **Flash Attention 2**: Must be enabled to handle the 16,384 sequence length potential.

## 6. Task 4: SLURM Launch Configuration
Create `scripts/counting_grpo/launch_stage32_grpo.slurm` with these GH200-specific limits:

```bash
#!/bin/bash
#SBATCH --job-name=stage32_grpo
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --mem=120G

export MAX_PROMPT_LEN=6144
export MAX_GEN_LEN=2048
export BATCH_SIZE=1
export GRAD_ACCUM=16

accelerate launch --num_processes 4 \
    scripts/counting_grpo/train_grpo_v32.py \
    --model_name_or_path "/projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_pca_4371501" \
    --dataset_name "outputs/scaffold_rex_5k_pca/curriculum_grpo_v32.jsonl" \
    --num_generations 8 \
    --reward_script "scripts/counting_grpo/grpo_reward_v32.py" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 5e-6 \
    --save_steps 50 \
    --bf16 True
```

## 7. Success KPIs for the Coding Agent
1.  **Parse rate > 95%** (Structural discipline).
2.  **Math Consistency > 90%** (Arithmetic grounding).
3.  **Regime C MAE < 20.0** (High-density breakthrough).

---

**Does this look ready to hand off?** If so, you can just paste this into your coding agent's prompt to begin the 4,945-row processing.
