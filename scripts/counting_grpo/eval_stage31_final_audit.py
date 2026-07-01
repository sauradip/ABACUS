#!/usr/bin/env python3
import json
from pathlib import Path

p = Path("checkpoints/rankdpo_stage31_r1rex_tally_20260426_161514/stage31_final_audit.json")
data = json.loads(p.read_text(encoding="utf-8"))
rows = data.get("rows", [])
n = len(rows)

parse_ok = sum(1 for r in rows if r.get("pred_count") is not None)
prose = sum(1 for r in rows if r.get("prose_drift", False))
math_ok = 0
max_pred = max((r.get("pred_count") or 0) for r in rows) if rows else 0
reg_c = [r for r in rows if (r.get("gt_count") or 0) > 50]
reg_c_nonzero = sum(1 for r in reg_c if (r.get("pred_count") or 0) > 0)

for r in rows:
    pc = r.get("pred_count")
    pts = r.get("pred_points") or []
    if pc is not None and isinstance(pts, list) and pts:
        if abs(len(pts) - pc) <= max(1, round(pc * 0.05)):
            math_ok += 1

img31 = next((r for r in rows if str(r.get("image", "")).endswith("31.jpg")), None)

print("=== Stage 3.1 Final Audit ===")
print(f"Rows: {n}")
print(f"Parse rate: {parse_ok}/{n} = {parse_ok / max(1, n):.1%}")
print(f"Prose drift: {prose}/{n}")
print(f"Math consistency: {math_ok}/{n} = {math_ok / max(1, n):.1%}")
print(f"Max pred_count: {max_pred}")
print(f"Regime C nonzero: {reg_c_nonzero}/{len(reg_c)} = {reg_c_nonzero / max(1, len(reg_c)):.1%}")

if img31:
    print(f"Image31 gt={img31.get('gt_count')} pred={img31.get('pred_count')}")
    txt = str(img31.get("prediction_text", ""))
    print("Image31 prediction preview:", txt[:280].replace("\n", " "))
