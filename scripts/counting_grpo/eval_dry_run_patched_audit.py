#!/usr/bin/env python3
import json
import re
import statistics
from pathlib import Path

IN_PATH = Path("checkpoints/scaffold_rex_stage15_pca_4360912/dry_run_patched_audit.json")
OUT_PATH = Path("checkpoints/scaffold_rex_stage15_pca_4360912/dry_run_patched_audit_metrics.json")


def robust_parse(text: str):
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        if '"total_count"' in text:
            m = re.search(r"(\{.*\})", "{" + text, re.DOTALL)
        if not m:
            return None
    s = m.group(1)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        try:
            obj = json.loads(s + "]}")
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def regime(gt: int) -> str:
    if gt < 30:
        return "A"
    if gt < 100:
        return "B"
    if gt < 200:
        return "C"
    return "X"


def cluster_sum(payload):
    if not isinstance(payload, dict):
        return None
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return None
    s = 0
    ok = False
    for c in clusters:
        if isinstance(c, dict) and isinstance(c.get("count"), int):
            s += int(c["count"])
            ok = True
    return s if ok else None


def main():
    data = json.loads(IN_PATH.read_text())
    rows = data.get("rows", [])

    parse_ok = 0
    prose_drift = 0
    math_consistent = 0
    math_checked = 0
    reg = {k: {"n": 0, "nonzero": 0, "preds": []} for k in ["A", "B", "C"]}
    max_pred = 0

    details = []
    for r in rows:
        img = r.get("image") or r.get("id")
        txt = str(r.get("prediction_text", ""))
        payload = robust_parse(txt)
        ok = isinstance(payload, dict)
        parse_ok += int(ok)

        starts_clean = txt.lstrip().startswith("{")
        prose_drift += int(not starts_clean)

        pred = int(r.get("pred_count") or 0)
        gt = int(r.get("gt_count") or 0)
        max_pred = max(max_pred, pred)

        rg = regime(gt)
        if rg in reg:
            reg[rg]["n"] += 1
            reg[rg]["nonzero"] += int(pred > 0)
            reg[rg]["preds"].append(pred)

        tcount = payload.get("total_count") if isinstance(payload, dict) else None
        csum = cluster_sum(payload)
        if isinstance(tcount, int) and isinstance(csum, int):
            math_checked += 1
            if tcount == csum:
                math_consistent += 1

        details.append(
            {
                "image": img,
                "gt": gt,
                "pred": pred,
                "parse_ok": ok,
                "starts_with_brace": starts_clean,
                "total_count": tcount,
                "cluster_sum": csum,
            }
        )

    out = {
        "rows_total": len(rows),
        "json_parse_ok": parse_ok,
        "json_parse_rate": parse_ok / len(rows) if rows else 0.0,
        "prose_drift_rows": prose_drift,
        "prose_drift_rate": prose_drift / len(rows) if rows else 0.0,
        "math_checked_rows": math_checked,
        "math_consistent_rows": math_consistent,
        "math_consistency_rate": math_consistent / math_checked if math_checked else 0.0,
        "max_pred_count": max_pred,
        "regimes": {},
        "details": details,
    }

    for k, v in reg.items():
        preds = v["preds"]
        out["regimes"][k] = {
            "n": v["n"],
            "nonzero_pred": v["nonzero"],
            "nonzero_rate": (v["nonzero"] / v["n"] if v["n"] else 0.0),
            "pred_variance": (statistics.pvariance(preds) if len(preds) >= 2 else 0.0),
            "pred_min": (min(preds) if preds else None),
            "pred_max": (max(preds) if preds else None),
        }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}")
    print(
        f"parse_rate={out['json_parse_ok']}/{out['rows_total']} ({100*out['json_parse_rate']:.1f}%), "
        f"prose_drift_rate={100*out['prose_drift_rate']:.1f}%, "
        f"math_consistency={out['math_consistent_rows']}/{out['math_checked_rows']}"
    )


if __name__ == "__main__":
    main()
