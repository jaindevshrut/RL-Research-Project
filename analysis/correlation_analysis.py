"""
Decorrelation analysis.

Goal: prove (or refute) that our 8-dim appraisal vector carries more
INDEPENDENT information than the paper's 4-dim version.

Methods:
  (1) Pearson correlation matrix of the dims (heatmap).
      For each pair we want |r| close to 0.
  (2) Variance-Inflation-Factor (VIF) per dim.
      VIF > 5 (rule of thumb) -> redundant.
  (3) Effective rank: exp(entropy(eigenvalues / sum)).
      Tells us "how many independent dims we effectively have".
  (4) Gram-Schmidt residualisation: re-express the appraisal vector
      as a sequence of residuals -- each new dim is orthogonalised w.r.t.
      the previously-kept ones. Reports how much *new* variance every
      dim contributes.

Run:
    python -m analysis.correlation_analysis --run extended_8dim
"""
from __future__ import annotations
import argparse, os, json
import numpy as np


def pearson(X):
    return np.corrcoef(X, rowvar=False)


def vif(X):
    """VIF_i = 1 / (1 - R_i^2), where R_i^2 is from regressing dim i
    on all others. High VIF == this dim is well-predicted by the rest."""
    n, d = X.shape
    out = np.zeros(d)
    for i in range(d):
        Y = X[:, i]
        Xo = np.delete(X, i, axis=1)
        # closed-form OLS without intercept-handling for simplicity
        Xc = np.hstack([Xo, np.ones((n, 1))])
        beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
        Y_hat = Xc @ beta
        ss_res = ((Y - Y_hat) ** 2).sum()
        ss_tot = ((Y - Y.mean()) ** 2).sum() + 1e-12
        r2 = 1 - ss_res / ss_tot
        out[i] = 1.0 / max(1e-6, 1.0 - r2)
    return out


def effective_rank(X):
    """exp(H(p)) where p = eigvals / sum(eigvals).
    Equals d when all dims are uncorrelated and equal-variance,
    1 when they collapse to a single direction."""
    cov = np.cov(X, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    p = eig / (eig.sum() + 1e-12)
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def gram_schmidt_residual_variance(X):
    """For each dim in order, regress it against the residualised previous
    dims, return the residual variance share (1.0 = fully novel)."""
    n, d = X.shape
    R = X - X.mean(axis=0, keepdims=True)
    R = R / (R.std(axis=0, keepdims=True) + 1e-8)
    kept = []
    shares = []
    for i in range(d):
        v = R[:, i].copy()
        for u in kept:
            v = v - (v @ u) / (u @ u + 1e-12) * u
        var_keep = (v ** 2).sum()
        var_full = (R[:, i] ** 2).sum() + 1e-12
        shares.append(float(var_keep / var_full))
        kept.append(v)
    return np.array(shares)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="extended_8dim")
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    path = os.path.join("runs", args.run, "appraisals.npz")
    z = np.load(path, allow_pickle=True)
    X = z["appraisals"].astype(np.float64)
    dims = list(z["dims"])
    n, d = X.shape
    print(f"Loaded {n} appraisal vectors, {d} dims: {dims}")

    # Drop constant columns to avoid div-by-zero (e.g. early-training)
    std = X.std(axis=0)
    keep = std > 1e-6
    if not keep.all():
        print(f"  warning: dropping constant dims {[dims[i] for i in range(d) if not keep[i]]}")
        X = X[:, keep]; dims = [dims[i] for i in range(d) if keep[i]]
        d = len(dims)

    corr = pearson(X)
    vifs = vif(X)
    er = effective_rank(X)
    shares = gram_schmidt_residual_variance(X)

    # Report
    print("\nPearson correlation matrix:")
    header = "          " + " ".join(f"{n:>8s}" for n in dims)
    print(header)
    for i, name in enumerate(dims):
        row = " ".join(f"{corr[i,j]:+8.2f}" for j in range(d))
        print(f"{name:>10s} {row}")

    print("\nVIF per dim (>5 = redundant):")
    for n_, v in zip(dims, vifs):
        flag = "  <-- redundant" if v > 5 else ""
        print(f"  {n_:>14s}: {v:7.2f}{flag}")

    print(f"\nEffective rank: {er:.2f}  (max possible = {d})")

    print("\nGram-Schmidt residual variance share (order = config order):")
    for n_, s in zip(dims, shares):
        print(f"  {n_:>14s}: {s:.3f}")

    # Save numeric report
    save_dir = args.save_dir or os.path.join("runs", args.run, "analysis")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "corr.npy"), corr)
    with open(os.path.join(save_dir, "report.json"), "w") as f:
        json.dump({
            "dims": dims,
            "pearson": corr.tolist(),
            "vif": vifs.tolist(),
            "effective_rank": er,
            "max_rank": d,
            "gs_residual_share": shares.tolist(),
        }, f, indent=2)
    print(f"\nSaved analysis to {save_dir}")


if __name__ == "__main__":
    main()
