# Stage 3.1 Recursive Tally Results

## Summary
- Generated at (UTC): 2026-04-26T16:49:48Z
- Training job: 4367185
- Output checkpoint: /projects/u6fb/myprojects/UniCount/checkpoints/rankdpo_stage31_r1rex_tally_20260426_161514
- Init checkpoint: /projects/u6fb/myprojects/UniCount/checkpoints/rankdpo_stage30_r1rex_memfix_20260426_153257

## Stage 3.1 Training Metrics
- train_runtime: 20.5863
- train_samples_per_second: 2.915
- train_steps_per_second: 0.291
- train_loss: 0.6559539635976156
- epoch: 1.53

## Stage 3.1 Dataset Diagnostics
- Dataset path: outputs/rankdpo_rex/preference_pairs_v3_1_tally.jsonl
- Total rows: 30
- pair_type audit_raw_repetition: 15
- pair_type recursive_rex_weighted: 15
- Recursive rows: 15
- Chosen consistency: 15/15 (100.0%)
- Anchor coverage (min/avg/max): 1.000 / 1.000 / 1.000
- Missing anchors (min/avg/max): 0.000 / 0.000 / 0.000
- Reward margin (min/avg/max): 1.511765 / 2.017675 / 2.432857

## Latest Available Validation Audit (Stage 3.0 Baseline)
- Audit JSON: checkpoints/rankdpo_stage30_r1rex_memfix_20260426_153257/stage30_validation_audit.json
- Rows: 15
- Parse rate: 100.0%
- Prose drift rate: 0.0%
- Math consistency rate: 0.0%
- Max pred count: 20
- Regime C nonzero rate: 16.7% (6 rows)

## Image 31 Snapshot
- image: 31.jpg
- gt_count: 126
- pred_count: 20
- prediction_text_preview: {"total_count": 20, "anchors_summary": { "anchors": [ 0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28, 32, 32, 36, 36, 40, 40, 44, 44, 48, 48, 52, 52, 56, 56, 60, 60, 64, 64, 68, 68, 72, 72, 76, 76, 80, 80, 84,

## Artifact Paths
- Diagnostics JSON: outputs/rankdpo_rex/stage31_output_diagnostics.json
- Training log: logs/rankdpo_stage31_4367185.log
- Training stderr: logs/rankdpo_stage31_4367185.err
