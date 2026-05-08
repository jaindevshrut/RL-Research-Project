"""
Side-by-side comparison: baseline (4-dim) vs extended (8-dim).

Reports the metrics most relevant to the user's claim that the
extended appraisal vector improves robustness:

  * RL learning curve (mean episode return over time)
  * Final greedy-policy evaluation return
  * Effective rank of the appraisal subspace
  * Emotion-classification accuracy
"""
from __future__ import annotations
import argparse, os, json
import numpy as np


def load_run(run):
    base = os.path.join("runs", run)
    out = {"run": run}
    if os.path.exists(os.path.join(base, "returns.npy")):
        out["returns"] = np.load(os.path.join(base, "returns.npy"))
    if os.path.exists(os.path.join(base, "eval.json")):
        out["eval"] = json.load(open(os.path.join(base, "eval.json")))
    rep = os.path.join(base, "analysis", "report.json")
    if os.path.exists(rep):
        out["decorr"] = json.load(open(rep))
    expl = os.path.join(base, "analysis", "explainability.json")
    if os.path.exists(expl):
        out["expl"] = json.load(open(expl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline_4dim", "extended_8dim"])
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        rows.append(load_run(r))

    print("\n=== RL performance ===")
    for r in rows:
        ret = r.get("returns")
        if ret is None or len(ret) == 0:
            print(f"  {r['run']:<20s}: (no returns)")
            continue
        last = ret[-min(20, len(ret)):].mean()
        print(f"  {r['run']:<20s}: episodes={len(ret):4d}  "
              f"last20_mean_return={last:+.3f}  best={ret.max():+.3f}")

    print("\n=== Greedy evaluation ===")
    for r in rows:
        ev = r.get("eval", [])
        if ev:
            last = ev[-1]
            print(f"  {r['run']:<20s}: frame={last[0]} mean_return={last[1]:+.3f}")

    print("\n=== Decorrelation (effective rank / max) ===")
    for r in rows:
        d = r.get("decorr")
        if d:
            print(f"  {r['run']:<20s}: rank={d['effective_rank']:.2f} / {d['max_rank']}")

    print("\n=== Emotion-classification test accuracy ===")
    for r in rows:
        e = r.get("expl")
        if e:
            print(f"  {r['run']:<20s}: acc={e['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
