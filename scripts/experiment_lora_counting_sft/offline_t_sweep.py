#!/usr/bin/env python
"""
Offline T-sweep + direct-count analysis on existing recursive eval JSONs.

Inputs are the per-row results saved by eval_ctap_nrt_fsc147.py
(which already include both `global_count` and the final recursive `pred`).

We compute, for each T in a sweep:
    pred_T = global_count   if global_count <  T
             pred           otherwise
and report MAE / RMSE / per-bucket MAE.

We also compute the "global-only" counterfactual (T=∞, i.e. always use
global_count) which IS Experiment 3 — no re-eval needed.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

BUCKETS = [(0,20),(21,50),(51,100),(101,200),(201,500),(501,10**9)]
def bkey(c):
    for lo,hi in BUCKETS:
        if lo<=c<=hi:
            return f"{lo}-{'+'if hi>10**8 else hi}"
    return "?"

def metrics(rows, pred_fn):
    errs = []
    by_bucket = defaultdict(list)
    for r in rows:
        gt = r["gt"]; p = pred_fn(r)
        e = p - gt
        errs.append(e)
        by_bucket[bkey(gt)].append(e)
    n = len(errs)
    mae = sum(abs(e) for e in errs)/n
    rmse = math.sqrt(sum(e*e for e in errs)/n)
    bmae = {b: (sum(abs(e) for e in v)/len(v), len(v)) for b,v in by_bucket.items()}
    return mae, rmse, bmae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="path(s) to *_recursive_*.json")
    ap.add_argument("--ts", default="100,150,200,300,500,1000,999999",
                    help="T thresholds to sweep")
    args = ap.parse_args()

    ts = [int(x) for x in args.ts.split(",")]

    bucket_order = [f"{lo}-{'+'if hi>10**8 else hi}" for lo,hi in BUCKETS]

    for inp in args.inputs:
        d = json.load(open(inp))
        rows = d["rows"]
        baseline_T = d.get("T", "?")
        print(f"\n========== {Path(inp).name}  (orig T={baseline_T}) ==========")
        print(f" n={len(rows)}  recorded MAE={d['MAE']:.2f}  RMSE={d['RMSE']:.2f}")

        # Header
        hdr = f"{'T':>8s} {'MAE':>7s} {'RMSE':>8s} {'%global':>7s}  " + \
              " ".join(f"{b:>10s}" for b in bucket_order)
        print(hdr)

        for T in ts:
            def pf(r, T=T):
                # recurse only if global_count >= T
                return r["pred"] if r["global_count"] >= T else r["global_count"]
            mae, rmse, bmae = metrics(rows, pf)
            pct_global = 100.0 * sum(1 for r in rows if r["global_count"] < T)/len(rows)
            cells = []
            for b in bucket_order:
                if b in bmae:
                    m, n = bmae[b]
                    cells.append(f"{m:>5.1f}({n:>3d})")
                else:
                    cells.append(f"{'-':>10s}")
            tag = f"{T:>8d}" if T < 10**6 else f"{'∞(direct)':>8s}"
            print(f"{tag} {mae:>7.2f} {rmse:>8.2f} {pct_global:>6.1f}%  " + " ".join(cells))

if __name__ == "__main__":
    main()
