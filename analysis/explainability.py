"""
Explainability module.

Two questions we need to answer for the professor:

  Q1. Does each appraisal dimension carry information that distinguishes
      different emotion-relevant *events* in the task? (i.e. is the
      extra dim earning its keep?)
  Q2. WHICH dim drives the prediction of which event/emotion?

We answer both with simple, transparent tools (no black-box ML) so the
explanation chain is auditable end-to-end.

  (a) Train a small logistic-regression classifier on:
            X = appraisal vector,  y = event label
      Labels in our run: 0 neutral, 1 picked-up-key, 2 reached-goal, 3 lava-death.
      These map onto canonical emotions per Scherer's appraisal-emotion table:
            1 -> "joy"          (positive surprise + conducive)
            2 -> "satisfaction" (goal achieved + low urgency)
            3 -> "fear/despair" (high suddenness + obstructive + low power)
            0 -> "neutral"
  (b) Permutation importance: shuffle one dim at a time and see how much
      the classifier accuracy drops. Drop = dim's marginal contribution.
  (c) Per-class mean-appraisal heatmap: lets you literally point at
      "look, when the agent is about to die in lava, conduciveness is
      strongly negative AND power is low — that matches the appraisal
      pattern Scherer associates with despair."

Usage:
    python -m analysis.explainability --run extended_8dim
"""
from __future__ import annotations
import argparse, os, json
import numpy as np


# Scherer-style mapping (illustrative — for the report's narrative).
LABEL_NAMES = {
    0: "neutral",
    1: "joy (key picked up)",
    2: "satisfaction (goal reached)",
    3: "despair (died in lava)",
}


# ------------------------------------------------------------------------- #
# Tiny logistic regression (multinomial), pure NumPy.
# ------------------------------------------------------------------------- #
def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LogReg:
    def __init__(self, n_features, n_classes, lr=0.5, l2=1e-3):
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = lr; self.l2 = l2

    def fit(self, X, y, epochs=300):
        n, d = X.shape
        K = self.W.shape[1]
        Y = np.eye(K)[y]
        for _ in range(epochs):
            P = softmax(X @ self.W + self.b)
            grad_W = X.T @ (P - Y) / n + self.l2 * self.W
            grad_b = (P - Y).mean(axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=-1)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


def class_balance(X, y, n_per_class=None):
    """Resample to equal class counts so the classifier doesn't trivially
    score 99% by predicting 'neutral'."""
    classes = np.unique(y)
    if n_per_class is None:
        n_per_class = max(1, min((y == c).sum() for c in classes))
    idx_all = []
    rng = np.random.default_rng(0)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0: continue
        pick = rng.choice(idx, size=n_per_class, replace=len(idx) < n_per_class)
        idx_all.append(pick)
    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="extended_8dim")
    args = ap.parse_args()

    path = os.path.join("runs", args.run, "appraisals.npz")
    z = np.load(path, allow_pickle=True)
    X = z["appraisals"].astype(np.float64)
    y = z["labels"].astype(int)
    dims = list(z["dims"])
    n, d = X.shape
    print(f"{n} samples, {d} dims, classes = {np.unique(y, return_counts=True)}")

    # standardise
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd

    Xb, yb = class_balance(Xs, y)
    n_train = int(0.8 * len(Xb))
    Xtr, Xte = Xb[:n_train], Xb[n_train:]
    ytr, yte = yb[:n_train], yb[n_train:]

    K = int(y.max() + 1)
    clf = LogReg(d, K).fit(Xtr, ytr, epochs=400)
    base_acc = clf.score(Xte, yte)
    print(f"\nClassifier test accuracy (base): {base_acc:.3f}")

    # Permutation importance
    rng = np.random.default_rng(0)
    importances = []
    for j in range(d):
        accs = []
        for _ in range(5):
            Xperm = Xte.copy()
            rng.shuffle(Xperm[:, j])
            accs.append(clf.score(Xperm, yte))
        drop = base_acc - np.mean(accs)
        importances.append(drop)
        print(f"  importance[{dims[j]:>14s}] = {drop:+.3f}")

    # Per-class mean appraisal (raw, not standardised — easier to read)
    means = np.zeros((K, d))
    for c in range(K):
        m = (y == c)
        if m.any():
            means[c] = X[m].mean(0)

    # Print human-readable table
    print("\nMean appraisal vector per event class:")
    print("  class                         " + "  ".join(f"{n:>14s}" for n in dims))
    for c in range(K):
        name = LABEL_NAMES.get(c, f"class_{c}")
        cells = "  ".join(f"{means[c, j]:+14.3f}" for j in range(d))
        print(f"  {name:<28s} {cells}")

    # Save
    out = os.path.join("runs", args.run, "analysis")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "explainability.json"), "w") as f:
        json.dump({
            "dims": dims,
            "test_accuracy": base_acc,
            "permutation_importance": dict(zip(dims, [float(x) for x in importances])),
            "class_means": {LABEL_NAMES.get(c, str(c)): means[c].tolist() for c in range(K)},
        }, f, indent=2)
    print(f"\nSaved to {out}/explainability.json")


if __name__ == "__main__":
    main()
