# Stage 2.5 -> Stage 3.0 Planning Log (Machine B)

Date: 2026-04-26
Workspace: UniCount

## Executive Summary

- Stage 2.5 structural run completed successfully and produced a usable checkpoint.
- Final post-DPO audit improved parse behavior vs earlier baseline but still showed high-density counting regression.
- Stage 3.0 first trial failed due to CUDA OOM.
- Stage 3.0 memory-safe rerun completed successfully with better training stability and artifact save.

## Run Timeline

| Job ID | Name | State | Elapsed | Exit | Notes |
|---|---|---|---:|---:|---|
| 4365411 | rankdpo_stage25 | COMPLETED | 00:09:01 | 0:0 | Stage 2.5 structural-negative training (22 pairs) |
| 4366074 | scaffold_audit_v2 | COMPLETED | 00:04:03 | 0:0 | Final post-DPO 15-image audit |
| 4366075 | eval_final_postdpo | FAILED | 00:00:02 | 1:0 | Inline evaluator script bug |
| 4366789 | rankdpo_stage30_r1rex | FAILED | 00:09:02 | 1:0 | Stage 3.0 trial failed with GPU OOM |
| 4366907 | rankdpo_stage30_r1rex_memfix | COMPLETED | 00:00:52 | 0:0 | Stage 3.0 memory-safe rerun success |

## Key Datasets

- Stage 2.5 structural pairs: outputs/rankdpo_rex/cross_density_structural_pairs.jsonl
- Stage 3.0 recursive pairs: outputs/rankdpo_rex/preference_pairs_v3.jsonl

## Training Metrics Snapshot

### Stage 2.5 Structural (Job 4365411)
Source log: logs/rankdpo_stage25_4365411.log

- Loaded rows: 22
- train_runtime: 28.6907
- train_samples_per_second: 1.534
- train_steps_per_second: 0.209
- train_loss: 0.6927527586619059
- epoch: 2.0

### Stage 3.0 Trial (Job 4366789)
Source log: logs/rankdpo_stage25_4366789.log

- Loaded rows: 29
- Failed before final train metrics due to OOM.

### Stage 3.0 Memfix (Job 4366907)
Source log: logs/rankdpo_stage25_4366907.log

- Loaded rows: 29
- train_runtime: 21.7755
- train_samples_per_second: 2.664
- train_steps_per_second: 0.276
- train_loss: 0.6497761011123657
- epoch: 1.55

## Final Post-DPO Audit Metrics

Source: checkpoints/rankdpo_stage25_structural_20260426_142916/final_post_dpo_audit_metrics.json

- rows_total: 15
- json_parse_rate: 0.8666666666666667 (13/15)
- prose_drift_rate: 0.0
- math_consistency_rate: 0.5
- max_pred_count: 20
- regime_C_nonzero_rate: 0.2
- image_31:
  - parse_ok: true
  - pred_count: 20
  - gt_count: 126

Interpretation:
- Syntax discipline improved.
- High-density counting remains undercounted ("Lazy 20" style collapse persists in Regime C).

## Failure Notes

### 1) Post-DPO eval helper failure (Job 4366075)
Source: logs/eval_final_postdpo_4366075.err

- Error: NameError: name 'total_count' is not defined
- Resolution: Manual corrected evaluator run generated final_post_dpo_audit_metrics.json.

### 2) Stage 3.0 OOM (Job 4366789)
Source: logs/rankdpo_stage25_4366789.err

- Error line: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.78 GiB...
- Context line included:
  - GPU 0 total: 95.00 GiB
  - Free at failure: 3.11 GiB
  - In use: 91.88 GiB

Resolution applied in rerun job 4366907:
- PER_DEVICE_TRAIN_BATCH_SIZE=1
- GRADIENT_ACCUMULATION_STEPS=8
- MAX_LENGTH=8192
- MAX_PROMPT_LENGTH=6144
- MAX_COMPLETION_LENGTH=2048
- VERIFY_MAX_NEW_TOKENS=64
- SKIP_MERGE_VERIFY=1

## Artifact Paths

- Stage 2.5 checkpoint dir: checkpoints/rankdpo_stage25_structural_20260426_142916
- Stage 2.5 final audit json: checkpoints/rankdpo_stage25_structural_20260426_142916/final_post_dpo_audit.json
- Stage 2.5 final audit metrics: checkpoints/rankdpo_stage25_structural_20260426_142916/final_post_dpo_audit_metrics.json
- Stage 3.0 trial output dir (failed run): checkpoints/rankdpo_stage30_r1rex_20260426_150539
- Stage 3.0 memfix output dir (successful): checkpoints/rankdpo_stage30_r1rex_memfix_20260426_153257

## Planning Checkpoints (Next)

1. Run 15-image audit against Stage 3.0 memfix checkpoint to compare directly against Stage 2.5 metrics.
2. Track especially:
   - parse rate target > 95%
   - regime_C_nonzero_rate target close to 1.0
   - max_pred_count should exceed 20 and approach high-density GT range
   - math_consistency_rate target > 0.9
3. If Regime C remains low, increase weighting toward accuracy/consistency in v3 generation and raise minimum reward margin for C rows.
4. Keep memory-safe training defaults for future Stage 3.x runs unless sequence lengths are strictly needed.
