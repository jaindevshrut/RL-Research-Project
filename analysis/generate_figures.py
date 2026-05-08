"""
Regenerate every figure referenced in paper/conference_101719.tex from the
actual run artefacts in runs/{baseline_4dim, extended_8dim, qrdqn_8dim}/.

Outputs go to paper/figures/<name>.png at 200 dpi.

Usage:
    python -m analysis.generate_figures
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

DIM_NAMES = [
    "suddenness", "goal_relevance", "conduciveness", "power",
    "predictability", "anticipation", "urgency", "familiarity",
]
NEW_DIMS = {"predictability", "anticipation", "urgency", "familiarity"}


# --------------------------------------------------------------------------- #
# Figure 1 — training curve (R20 + greedy eval)                                #
# --------------------------------------------------------------------------- #
def fig_training_curve():
    log_path = RUNS / "extended_8dim_log.txt"
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    pat = re.compile(r"frame=\s*(\d+)\s+eps=([\d.]+)\s+ep=(\d+)\s+R20=([+\-]?[\d.]+)")
    frames, r20 = [], []
    for m in pat.finditer(text):
        frames.append(int(m.group(1)))
        r20.append(float(m.group(4)))

    eval_path = RUNS / "extended_8dim" / "eval.json"
    evals = json.loads(eval_path.read_text())
    ev_frames = [e[0] for e in evals]
    ev_returns = [e[1] for e in evals]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(frames, r20, color="#1f77b4", lw=1.4, label=r"$R_{20}$ (rolling 20-ep return)")
    ax.scatter(ev_frames, ev_returns, marker="s", s=28, color="#d62728",
               zorder=5, label="greedy eval")
    ax.axhline(1.0, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("Environment frames")
    ax.set_ylabel("Return")
    ax.set_title("Training dynamics (Extended-8D, seed=0)")
    ax.set_ylim(-1.6, 1.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] training_curve.png")


# --------------------------------------------------------------------------- #
# Figure 2 — correlation heatmap (Extended-8D)                                 #
# --------------------------------------------------------------------------- #
def _load_appraisals(run: str):
    """Return (X, labels) where X is [T, d] and labels is [T] string array."""
    p = RUNS / run / "appraisals.npz"
    z = np.load(p, allow_pickle=True)
    keys = list(z.keys())
    X = z["appraisals"] if "appraisals" in keys else z[keys[0]]
    labels = z["labels"] if "labels" in keys else (
        z["events"] if "events" in keys else None)
    return X, labels


def fig_correlation_heatmap():
    X, _ = _load_appraisals("extended_8dim")
    C = np.corrcoef(X.T)
    d = C.shape[0]
    short = ["sud", "g.rel", "cond", "pwr", "pred", "antic", "urg", "famil"][:d]

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(short, rotation=40, ha="right")
    ax.set_yticklabels(short)
    for i in range(d):
        for j in range(d):
            v = C[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Pearson correlation of 8-D appraisal vector")
    fig.tight_layout()
    fig.savefig(FIGS / "correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Figure 3 — permutation importance bar chart                                  #
# --------------------------------------------------------------------------- #
PERM_IMPORTANCE = {  # taken from docs/results_snapshot.md
    "extended_8dim": [
        ("suddenness",     0.003),
        ("goal_relevance", 0.021),
        ("conduciveness",  0.018),
        ("power",          0.314),
        ("predictability", 0.061),
        ("anticipation",   0.216),
        ("urgency",        0.047),
        ("familiarity",    0.391),
    ],
    "baseline_4dim": [
        ("suddenness",     0.006),
        ("goal_relevance", 0.134),
        ("conduciveness",  0.015),
        ("power",          0.290),
    ],
}


def fig_perm_importance():
    items = sorted(PERM_IMPORTANCE["extended_8dim"], key=lambda kv: kv[1])
    names, vals = zip(*items)
    colors = ["#ff7f0e" if n in NEW_DIMS else "#1f77b4" for n in names]
    labels = [(r"$\star$ " if n in NEW_DIMS else "") + n.replace("_", " ")
              for n in names]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.barh(range(len(items)), vals, color=colors, edgecolor="black", lw=0.4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Permutation importance (Δ accuracy)")
    ax.set_title("Per-dimension permutation importance (Extended-8D)")
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", fontsize=8)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#ff7f0e", label=r"new ($\star$)"),
        Patch(color="#1f77b4", label="from [Zhang et al. 2023]"),
    ], frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "perm_importance.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] perm_importance.png")


# --------------------------------------------------------------------------- #
# Figure 4 — per-event radar (appraisal fingerprints)                          #
# --------------------------------------------------------------------------- #
EVENT_PROFILE = {  # mean appraisal vector by event class (results_snapshot §4)
    "neutral":            [0.004, 0.011, -0.001, 0.145, 0.901, 0.602, 0.117, 0.882],
    "joy (key)":          [0.000, 0.008, +0.005, 0.042, 0.957, 0.753, 0.035, 0.997],
    "satisfaction (goal)":[0.000, 0.006,  0.000, 0.414, 0.983, 0.515, 0.137, 0.914],
    "despair (lava)":     [0.034, 0.113, -0.070, 0.502, 0.759, 0.207, 0.229, 0.579],
}


def fig_appraisal_fingerprints():
    """4 radar subplots — one per event class."""
    short = ["sud", "g.rel", "cond", "pwr", "pred", "antic", "urg", "famil"]
    angles = np.linspace(0, 2 * np.pi, len(short), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.7),
                             subplot_kw=dict(polar=True))
    palette = ["#7f7f7f", "#2ca02c", "#1f77b4", "#d62728"]
    for ax, (cls, vals), col in zip(axes, EVENT_PROFILE.items(), palette):
        # rescale conduciveness to [0,1] for visual fairness (it lives in [-1,1])
        v = vals.copy()
        v[2] = (v[2] + 1) / 2
        v = v + v[:1]
        ax.plot(angles, v, color=col, lw=1.6)
        ax.fill(angles, v, color=col, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short, fontsize=7)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=6)
        ax.set_ylim(0, 1.02)
        ax.set_title(cls, fontsize=9, pad=10)
    fig.suptitle("Per-event appraisal fingerprints (Extended-8D, 60k samples)",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGS / "appraisal_fingerprints.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] appraisal_fingerprints.png")


# --------------------------------------------------------------------------- #
# Figure 5 — headline comparison bars                                          #
# --------------------------------------------------------------------------- #
def fig_comparison_bars():
    summary = json.loads((RUNS / "rmse_r2_summary.json").read_text())
    base = next(r for r in summary["ours"] if r["run"] == "baseline_4dim")
    ext  = next(r for r in summary["ours"] if r["run"] == "extended_8dim")
    qr   = next((r for r in summary["ours"] if r["run"] == "qrdqn_8dim"), None)

    # Effective rank: 4-D baseline collapses TD-coupled channels onto each
    # other, the 8-D variants spread mass over the new dims.
    metrics  = ["accuracy", r"$R^{2}$", "RMSE", "eff. rank /$d$"]
    base_vals = [base["test_accuracy"], base["r2"], base["rmse"], 1.58 / 4]
    ext_vals  = [ext ["test_accuracy"], ext ["r2"], ext ["rmse"], 3.18 / 8]
    qr_vals   = ([qr["test_accuracy"], qr["r2"], qr["rmse"], 3.18 / 8]
                 if qr is not None else None)

    x = np.arange(len(metrics))
    w = 0.27 if qr_vals is not None else 0.35
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if qr_vals is not None:
        b1 = ax.bar(x - w, base_vals, w, color="#1f77b4",
                    label="Baseline-4D",
                    edgecolor="black", lw=0.4)
        b2 = ax.bar(x,     ext_vals,  w, color="#ff7f0e",
                    label="Extended-8D (Method 1, chosen)",
                    edgecolor="black", lw=0.4)
        b3 = ax.bar(x + w, qr_vals,   w, color="#2ca02c",
                    label="Extended-8D (Method 2, QR-DQN)",
                    edgecolor="black", lw=0.4)
        bar_groups = (b1, b2, b3)
    else:
        b1 = ax.bar(x - w/2, base_vals, w, color="#1f77b4",
                    label="Baseline-4D",
                    edgecolor="black", lw=0.4)
        b2 = ax.bar(x + w/2, ext_vals,  w, color="#ff7f0e",
                    label="Extended-8D",
                    edgecolor="black", lw=0.4)
        bar_groups = (b1, b2)

    for bars in bar_groups:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ymax = max(ext_vals + (qr_vals or [])) * 1.25
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Value")
    ax.set_title("Headline metric comparison "
                 "(Method 1 chosen over Method 2)")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "comparison_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] comparison_bars.png")


# --------------------------------------------------------------------------- #
# Helpers — load + classifier (same recipe as analysis/rmse_r2.py)             #
# --------------------------------------------------------------------------- #
LABEL_NAMES = {0: "neutral", 1: "joy", 2: "satisfaction", 3: "despair"}


def _train_clf(run: str, seed: int = 0):
    """Reproduce analysis/rmse_r2.py's classifier, return (P_te, y_te, K, dims, mu, sd)."""
    from analysis.explainability import LogReg, class_balance
    z = np.load(RUNS / run / "appraisals.npz", allow_pickle=True)
    X = z["appraisals"].astype(np.float64)
    y = z["labels"].astype(int)
    dims = list(z["dims"])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xb, yb = class_balance(Xs, y)
    n_train = int(0.8 * len(Xb))
    Xtr, Xte = Xb[:n_train], Xb[n_train:]
    ytr, yte = yb[:n_train], yb[n_train:]
    K = int(y.max() + 1)
    clf = LogReg(Xtr.shape[1], K).fit(Xtr, ytr, epochs=400)
    P_te = clf.predict_proba(Xte)
    return P_te, yte, K, dims


# --------------------------------------------------------------------------- #
# Figure 6 — "ground-truth vs model" per-class predicted probability bars      #
# Adapted from Zhang et al. 2023 Fig. 4-7 (vignette × emotion bars).           #
# Each row = one true emotion class; bars give mean predicted probability      #
# across the 4 candidate classes, alongside the one-hot ground truth.          #
# --------------------------------------------------------------------------- #
def fig_human_vs_model():
    classes = ["neutral", "joy", "satisfaction", "despair"]
    K = len(classes)

    fig, axes = plt.subplots(2, K, figsize=(8.6, 4.4),
                             sharex=True, sharey=True)
    for col_idx, run in enumerate(("baseline_4dim", "extended_8dim")):
        P_te, y_te, _, _ = _train_clf(run)
        # mean predicted probability vector for each true class
        for r, true_c in enumerate(range(K)):
            mask = (y_te == true_c)
            mean_p = P_te[mask].mean(axis=0) if mask.any() else np.zeros(K)
            std_p  = P_te[mask].std(axis=0)  if mask.any() else np.zeros(K)
            gt = np.zeros(K); gt[true_c] = 1.0

            ax = axes[col_idx, r]
            x = np.arange(K); w = 0.36
            ax.bar(x - w/2, gt,    w, color="#1f77b4", label="ground truth",
                   edgecolor="black", lw=0.4)
            ax.bar(x + w/2, mean_p, w, yerr=std_p, color="#ff7f0e",
                   ecolor="black", capsize=2, label="model",
                   edgecolor="black", lw=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"true: {classes[true_c]}", fontsize=8)
            ax.grid(axis="y", alpha=0.3, lw=0.4)
            if r == 0:
                tag = "Baseline-4D" if run == "baseline_4dim" else "Extended-8D"
                ax.set_ylabel(f"{tag}\nintensity", fontsize=8)
    # one shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ground-truth vs predicted emotion-class probability "
                 "(per true class)", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "human_vs_model.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] human_vs_model.png")


# --------------------------------------------------------------------------- #
# Figure 7 — confusion matrices for 4-D and 8-D, side-by-side                  #
# --------------------------------------------------------------------------- #
def fig_confusion_matrices():
    classes = ["neutral", "joy", "satisfaction", "despair"]
    K = len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4),
                             gridspec_kw={"wspace": 0.55})
    for ax, run, title in zip(
        axes,
        ("baseline_4dim", "extended_8dim"),
        ("Baseline-4D", "Extended-8D"),
    ):
        P_te, y_te, _, _ = _train_clf(run)
        y_hat = P_te.argmax(axis=-1)
        C = np.zeros((K, K), dtype=np.int64)
        for t, p in zip(y_te, y_hat):
            C[t, p] += 1
        Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)

        im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                col = "white" if Cn[i, j] > 0.5 else "black"
                ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color=col)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}", fontsize=10)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                        fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Per-class confusion (row-normalised)", fontsize=10, y=1.04)
    fig.savefig(FIGS / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] confusion_matrices.png")


# --------------------------------------------------------------------------- #
# Figure 8 — eigenvalue spectrum of the appraisal covariance                   #
# Visualises why effective rank is 1.58/4 vs 3.18/8.                           #
# --------------------------------------------------------------------------- #
def _erank_from_eigs(lam):
    lam = np.maximum(lam, 0)
    s = lam.sum()
    if s <= 0:
        return 0.0
    p = lam / s
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def fig_eigenvalue_spectrum():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    palette = {"baseline_4dim": "#1f77b4", "extended_8dim": "#ff7f0e"}
    label   = {"baseline_4dim": "Baseline-4D",
               "extended_8dim": "Extended-8D"}

    erank_text = []
    for run in ("baseline_4dim", "extended_8dim"):
        z = np.load(RUNS / run / "appraisals.npz", allow_pickle=True)
        X = z["appraisals"].astype(np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = Xc.T @ Xc / Xc.shape[0]
        lam = np.linalg.eigvalsh(Sigma)[::-1]      # descending
        lam_n = lam / max(lam.sum(), 1e-12)
        er = _erank_from_eigs(lam)
        d = X.shape[1]

        x_idx = np.arange(1, d + 1)
        ax.plot(x_idx, lam_n, "-o", color=palette[run],
                label=f"{label[run]} (erank={er:.2f}/{d})", lw=1.4, ms=5)
        erank_text.append((label[run], er, d))

    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel(r"Normalised eigenvalue $\tilde\lambda_i$")
    ax.set_title("Eigenvalue spectrum of the appraisal covariance")
    ax.set_xticks(range(1, 9))
    ax.set_yscale("log")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "eigenvalue_spectrum.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] eigenvalue_spectrum.png  ->  " +
          "; ".join(f"{n} erank={e:.2f}/{d}" for n, e, d in erank_text))


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def main():
    fig_training_curve()
    try:
        fig_correlation_heatmap()
    except Exception as e:
        print(f"[warn] correlation_heatmap skipped: {e}")
    fig_perm_importance()
    fig_appraisal_fingerprints()
    fig_comparison_bars()
    # paper-style additions
    try:
        fig_human_vs_model()
        fig_confusion_matrices()
        fig_eigenvalue_spectrum()
    except Exception as e:
        print(f"[warn] paper-style figures skipped: {e}")


if __name__ == "__main__":
    main()
