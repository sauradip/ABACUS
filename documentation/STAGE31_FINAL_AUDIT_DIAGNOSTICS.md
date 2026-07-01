# Stage 3.1 Final Audit Diagnostics

- Generated at (UTC): 2026-04-26T17:02:31Z
- Audit JSON: checkpoints/rankdpo_stage31_r1rex_tally_20260426_161514/stage31_final_audit.json
- Rows: 15

## KPI Summary
- Parse rate: 15/15 = 100.0%
- Prose drift: 0/15 = 0.0%
- Math consistency: 0/15 = 0.0%
- Max pred_count: 20
- Zero-count rows: 8/15 = 53.3%
- Regime C nonzero: 1/6 = 16.7%

## Pattern Diagnostics
- Sequence-like anchors_summary pattern rows: 1/15
- Detected repeated index-sequence style outputs (e.g., `0,0,1,1,...,20`) in prediction text.

## Math-Inconsistent Rows (Top 10)
| image | gt_count | pred_count | pred_points_len |
| :--- | ---: | ---: | ---: |
| 7.jpg | 13 | 0 | 0 |
| 9.jpg | 8 | 4 | 0 |
| 19.jpg | 9 | 0 | 0 |
| 20.jpg | 16 | 10 | 0 |
| 21.jpg | 8 | 0 | 0 |
| 33.jpg | 47 | 20 | 0 |
| 40.jpg | 34 | 10 | 0 |
| 48.jpg | 75 | 0 | 0 |
| 22.jpg | 30 | 12 | 0 |
| 34.jpg | 49 | 12 | 0 |

## Image 31 Snapshot
- image: 31.jpg
- gt_count: 126
- pred_count: 20
- prediction preview: {"total_count": 20, "anchors_summary": { "anchors": [ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20 ] } }
