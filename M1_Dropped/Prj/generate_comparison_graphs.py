import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)


RESULTS_PATH = Path("improved/results/comparison_summary.csv")


def get_metric(df: pd.DataFrame, metric: str, column: str) -> float:
    row = df[df["Metric"] == metric]
    if row.empty:
        raise ValueError(f"Missing metric: {metric}")
    value = row.iloc[0][column]
    return float(value)


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    experiments = ["Exp12", "Exp3"]
    r2_original = [
        get_metric(df, "Exp12_Appraisal_R2", "Original"),
        get_metric(df, "Exp3_Appraisal_R2", "Original"),
    ]
    r2_improved = [
        get_metric(df, "Exp12_Appraisal_R2", "Improved"),
        get_metric(df, "Exp3_Appraisal_R2", "Improved"),
    ]

    rmse_original = [
        get_metric(df, "Exp12_Appraisal_RMSE", "Original"),
        get_metric(df, "Exp3_Appraisal_RMSE", "Original"),
    ]
    rmse_improved = [
        get_metric(df, "Exp12_Appraisal_RMSE", "Improved"),
        get_metric(df, "Exp3_Appraisal_RMSE", "Improved"),
    ]

    x = range(len(experiments))
    width = 0.25

    # Figure 1: Appraisal R2 (higher is better).
    fig, ax = plt.subplots()
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        r2_original,
        width,
        label="Original",
        color="#9AA5B1",
        edgecolor="black",
        linewidth=1.0,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        r2_improved,
        width,
        label="Improved (Appraisal)",
        color="#0D7C8A",
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_title("Appraisal Method: R2 Original vs Improved", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experiment", fontsize=11, fontweight="bold")
    ax.set_ylabel("R2", fontsize=11, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(experiments)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("appraisal_r2_original_vs_improved.png", dpi=300, bbox_inches="tight")
    print("Saved: appraisal_r2_original_vs_improved.png")
    plt.close()

    # Figure 2: Appraisal RMSE (lower is better).
    fig, ax = plt.subplots()
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        rmse_original,
        width,
        label="Original",
        color="#C9D6DF",
        edgecolor="black",
        linewidth=1.0,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        rmse_improved,
        width,
        label="Improved (Appraisal)",
        color="#F5A623",
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_title("Appraisal Method: RMSE Original vs Improved", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experiment", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=11, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(experiments)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("appraisal_rmse_original_vs_improved.png", dpi=300, bbox_inches="tight")
    print("Saved: appraisal_rmse_original_vs_improved.png")
    plt.close()

    # Figure 3: single panel showing only improved appraisal performance.
    fig, ax = plt.subplots()
    improved_only = r2_improved
    bars = ax.bar(experiments, improved_only, color=["#0D7C8A", "#16325C"], edgecolor="black", linewidth=1.0)
    ax.set_title("Improved Appraisal Method R2 (Only)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experiment", fontsize=11, fontweight="bold")
    ax.set_ylabel("R2", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("appraisal_improved_only_r2.png", dpi=300, bbox_inches="tight")
    print("Saved: appraisal_improved_only_r2.png")
    plt.close()

    print("\nAppraisal metrics loaded from improved/results/comparison_summary.csv")
    print(f"R2 Original: {r2_original}")
    print(f"R2 Improved: {r2_improved}")
    print(f"RMSE Original: {rmse_original}")
    print(f"RMSE Improved: {rmse_improved}")


if __name__ == "__main__":
    main()
