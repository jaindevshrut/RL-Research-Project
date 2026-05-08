"""
Compute RMSE and R^2 for the appraisal-derived emotion predictions,
in the same metric family the paper reports in Table V.

The paper's R^2 / RMSE compare model-predicted emotion *intensities*
against *human ratings* on a 0..1 scale.

We don't have human vignette ratings here, but we have the closest
analogue available without re-running the human study:
  - ground truth y_ic : 1 if event i has class c, else 0 (one-hot)
  - prediction p_ic   : classifier's predicted probability for class c

This puts both quantities on a 0..1 scale, exactly like the paper's
intensity-vs-rating comparison. RMSE and R^2 are then computed on the
flattened (n_samples * n_classes) target.

For consistency, the classifier is the same multinomial logistic
regression used in analysis/explainability.py.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np

from analysis.explainability import LogReg, class_balance


def metrics_one_hot(P, Y_onehot):
    """RMSE and R^2 over flattened (samples * classes) targets."""
    diff = (P - Y_onehot).ravel()
    rmse = float(np.sqrt((diff ** 2).mean()))
    y = Y_onehot.ravel(); p = P.ravel()
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2


def evaluate_run(run: str):
    z = np.load(os.path.join("runs", run, "appraisals.npz"), allow_pickle=True)
    X = z["appraisals"].astype(np.float64)
    y = z["labels"].astype(int)
    dims = list(z["dims"])

    # standardise
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd

    # balanced split, same recipe as explainability.py
    Xb, yb = class_balance(Xs, y)
    n_train = int(0.8 * len(Xb))
    Xtr, Xte = Xb[:n_train], Xb[n_train:]
    ytr, yte = yb[:n_train], yb[n_train:]

    K = int(y.max() + 1)
    clf = LogReg(Xtr.shape[1], K).fit(Xtr, ytr, epochs=400)

    P_te = clf.predict_proba(Xte)
    Y_oh = np.eye(K)[yte]

    rmse, r2 = metrics_one_hot(P_te, Y_oh)
    acc = float((P_te.argmax(axis=-1) == yte).mean())
    return {"run": run, "n_dims": Xs.shape[1], "dims": dims,
            "test_accuracy": acc, "rmse": rmse, "r2": r2,
            "n_test": int(len(yte)), "n_classes": K}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["baseline_4dim", "extended_8dim", "qrdqn_8dim"])
    args = ap.parse_args()

    paper = {
        "Paper Exp 1 (free)":   {"r2": 0.65, "rmse": 0.09},
        "Paper Exp 2 (forced)": {"r2": 0.92, "rmse": 0.09},
        "Paper Exp 3 (free)":   {"r2": 0.29, "rmse": 0.04},
        "Paper Exp 3 (forced)": {"r2": 0.62, "rmse": 0.16},
    }

    rows = []
    for r in args.runs:
        try:
            rows.append(evaluate_run(r))
        except FileNotFoundError:
            print(f"  [skip] {r}: appraisals.npz not present yet")

    print("\n=== Comparison: paper's reported fit vs. our runs ===\n")
    print(f"{'Method':<32s}  {'R^2':>7s}  {'RMSE':>7s}")
    print("-" * 50)
    for name, m in paper.items():
        print(f"{name:<32s}  {m['r2']:7.3f}  {m['rmse']:7.3f}")
    print()
    for r in rows:
        label = f"Ours ({r['run']}, {r['n_dims']} dims)"
        print(f"{label:<32s}  {r['r2']:7.3f}  {r['rmse']:7.3f}   "
              f"(acc={r['test_accuracy']:.3f})")

    # Write a JSON summary
    out = {"paper": paper, "ours": rows}
    out_path = os.path.join("runs", "rmse_r2_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
