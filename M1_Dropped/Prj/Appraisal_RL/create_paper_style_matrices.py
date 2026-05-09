from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


PAPER_METRICS = {
    "Exp1_Free": {"cmean": 0.0032, "cvar": 0.0002, "R2": 0.65, "RMSE": 0.09},
    "Exp2_Limit": {"cmean": 0.0140, "cvar": 0.0056, "R2": 0.92, "RMSE": 0.09},
    "Exp3_Free": {"cmean": 0.0013, "cvar": 0.0001, "R2": 0.29, "RMSE": 0.04},
    "Exp3_Limit": {"cmean": 0.0034, "cvar": 0.0010, "R2": 0.62, "RMSE": 0.16},
}


PAPER_APPRAISALS = {
    "Exp1_2": {
        "Happiness": {"Suddenness": 0.00, "Goal_relevance": 0.67, "Conduciveness": 0.83, "Power": 0.95},
        "Joy": {"Suddenness": 0.80, "Goal_relevance": 1.00, "Conduciveness": 1.00, "Power": 0.00},
        "Pride": {"Suddenness": 0.50, "Goal_relevance": 1.00, "Conduciveness": 1.00, "Power": 0.10},
        "Boredom": {"Suddenness": 0.00, "Goal_relevance": 0.00, "Conduciveness": 0.50, "Power": 0.60},
        "Fear": {"Suddenness": 0.80, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.00},
        "Sadness": {"Suddenness": 0.20, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.00},
        "Shame": {"Suddenness": 0.79, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.50},
    },
    "Exp3": {
        "Anxiety": {"Suddenness": 0.20, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.00},
        "Despair": {"Suddenness": 0.81, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.00},
        "Irritation": {"Suddenness": 0.20, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.53},
        "Rage": {"Suddenness": 0.80, "Goal_relevance": 1.00, "Conduciveness": 0.00, "Power": 0.60},
    },
}


EXPERIMENTS = [
    {
        "name": "Exp1_Free",
        "group": "Exp1_2",
        "human_file": ROOT / "Exp1_2" / "data" / "human_free.csv",
        "human_sep": ";",
        "human_prefix": "Emo.",
        "model_file": ROOT / "Exp1_2" / "data" / "svm_free_0.0032_var.csv",
        "mlp_file": ROOT / "Exp1_2" / "data" / "mlp_ensemble_predictions.csv",
        "stories": ["Happiness", "Joy", "Pride", "Boredom", "Fear", "Sadness", "Shame"],
        "emotions": ["Happiness", "Joy", "Pride", "Boredom", "Fear", "Sadness", "Shame"],
    },
    {
        "name": "Exp2_Limit",
        "group": "Exp1_2",
        "human_file": ROOT / "Exp1_2" / "data" / "human_limit.csv",
        "human_sep": ";",
        "human_prefix": "Em.",
        "model_file": ROOT / "Exp1_2" / "data" / "svm_limit_0.014_var.csv",
        "mlp_file": ROOT / "Exp1_2" / "data" / "mlp_ensemble_predictions.csv",
        "stories": ["Happiness", "Joy", "Pride", "Boredom", "Fear", "Sadness", "Shame"],
        "emotions": ["Happiness", "Joy", "Pride", "Boredom", "Fear", "Sadness", "Shame"],
    },
    {
        "name": "Exp3_Free",
        "group": "Exp3",
        "human_file": ROOT / "Exp3" / "data" / "human_free_limit.csv",
        "human_sep": ";",
        "human_prefix": "Emo.",
        "model_file": ROOT / "Exp3" / "data" / "svm_free_0.0013_var.csv",
        "mlp_file": ROOT / "Exp3" / "data" / "mlp_ensemble_predictions.csv",
        "stories": ["Anxiety", "Despair", "Irritation", "Rage"],
        "emotions": ["Anxiety", "Despair", "Irritation", "Rage"],
    },
    {
        "name": "Exp3_Limit",
        "group": "Exp3",
        "human_file": ROOT / "Exp3" / "data" / "human_free_limit.csv",
        "human_sep": ";",
        "human_prefix": "mc.",
        "model_file": ROOT / "Exp3" / "data" / "svm_limit_0.0034_var.csv",
        "mlp_file": ROOT / "Exp3" / "data" / "mlp_ensemble_predictions.csv",
        "stories": ["Anxiety", "Despair", "Irritation", "Rage"],
        "emotions": ["Anxiety", "Despair", "Irritation", "Rage"],
    },
]


def human_matrix(config):
    data = pd.read_csv(config["human_file"], sep=config["human_sep"])
    emotion_cols = [f"{config['human_prefix']}{emotion}" for emotion in config["emotions"]]
    long_df = data.melt(
        id_vars=["Story"],
        value_vars=emotion_cols,
        var_name="Emotion",
        value_name="Val",
    )
    long_df["Emotion"] = long_df["Emotion"].str.removeprefix(config["human_prefix"])
    grouped = (
        long_df.groupby(["Story", "Emotion"], as_index=False)["Val"]
        .mean()
        .sort_values(["Story", "Emotion"])
    )
    return grouped


def model_matrix_from_svm(model_file):
    data = pd.read_csv(model_file)
    grouped = data.groupby(["Story", "Emotion"], as_index=False)["Val"].mean()
    return grouped.sort_values(["Story", "Emotion"])


def model_matrix_from_mlp(model_file):
    data = pd.read_csv(model_file)
    grouped = data.groupby(["Story", "Emotion"], as_index=False)["Val"].mean()
    return grouped.sort_values(["Story", "Emotion"])


def normalize_within_story(df):
    normalized = df.copy()
    normalized["Val"] = normalized.groupby("Story")["Val"].transform(lambda values: values / values.sum())
    return normalized


def compute_metrics(human_df, model_df):
    merged = human_df.merge(model_df, on=["Story", "Emotion"], suffixes=("_human", "_model"))
    X = merged[["Val_model"]].to_numpy()
    y = merged["Val_human"].to_numpy()

    regression = LinearRegression()
    regression.fit(X, y)
    r2 = regression.score(X, y)
    rmse = float(np.sqrt(np.mean((y - X[:, 0]) ** 2)))
    return r2, rmse, merged


def top1_accuracy(model_df):
    grouped = []
    for story, frame in model_df.groupby("Story"):
        best = frame.sort_values("Val", ascending=False).iloc[0]["Emotion"]
        grouped.append({"Story": story, "Predicted": best, "Correct": best == story})
    result = pd.DataFrame(grouped).sort_values("Story")
    return float(result["Correct"].mean()), result


def pivot_and_save(df, output_file, stories, emotions):
    pivot = (
        df.pivot(index="Story", columns="Emotion", values="Val")
        .reindex(index=stories, columns=emotions)
        .round(6)
    )
    pivot.to_csv(output_file)
    return pivot


def build_appraisal_comparison():
    rows = []
    for group_name, paper_values in PAPER_APPRAISALS.items():
        model_file = ROOT / group_name / "data" / "model_result.csv"
        current = pd.read_csv(model_file).set_index("Emotion")
        for emotion, dims in paper_values.items():
            for dim_name, paper_value in dims.items():
                current_value = float(current.loc[emotion, dim_name])
                rows.append(
                    {
                        "Experiment": group_name,
                        "Emotion": emotion,
                        "Dimension": dim_name,
                        "Paper": paper_value,
                        "Current": current_value,
                        "Delta": current_value - paper_value,
                    }
                )
    appraisal_df = pd.DataFrame(rows)
    appraisal_df.to_csv(ANALYSIS_DIR / "appraisal_profile_comparison.csv", index=False)
    return appraisal_df


def build_performance_outputs():
    summary_rows = []

    for config in EXPERIMENTS:
        human_raw = human_matrix(config)
        human_norm = normalize_within_story(human_raw)

        svm_raw = model_matrix_from_svm(config["model_file"])
        svm_norm = normalize_within_story(svm_raw)
        svm_r2, svm_rmse, _ = compute_metrics(human_norm, svm_norm)
        svm_top1, _ = top1_accuracy(svm_norm)

        mlp_raw = model_matrix_from_mlp(config["mlp_file"])
        mlp_norm = normalize_within_story(mlp_raw)
        mlp_r2, mlp_rmse, _ = compute_metrics(human_norm, mlp_norm)
        mlp_top1, _ = top1_accuracy(mlp_norm)

        base_name = config["name"].lower()
        pivot_and_save(human_raw, ANALYSIS_DIR / f"{base_name}_human_raw_matrix.csv", config["stories"], config["emotions"])
        pivot_and_save(human_norm, ANALYSIS_DIR / f"{base_name}_human_normalized_matrix.csv", config["stories"], config["emotions"])
        pivot_and_save(svm_norm, ANALYSIS_DIR / f"{base_name}_svm_matrix.csv", config["stories"], config["emotions"])
        pivot_and_save(mlp_norm, ANALYSIS_DIR / f"{base_name}_mlp_matrix.csv", config["stories"], config["emotions"])

        summary_rows.append(
            {
                "Experiment": config["name"],
                "Paper_cmean": PAPER_METRICS[config["name"]]["cmean"],
                "Paper_cvar": PAPER_METRICS[config["name"]]["cvar"],
                "Paper_R2": PAPER_METRICS[config["name"]]["R2"],
                "Paper_RMSE": PAPER_METRICS[config["name"]]["RMSE"],
                "Current_SVM_R2": svm_r2,
                "Current_SVM_RMSE": svm_rmse,
                "Current_SVM_Top1_Accuracy": svm_top1,
                "Current_MLP_R2": mlp_r2,
                "Current_MLP_RMSE": mlp_rmse,
                "Current_MLP_Top1_Accuracy": mlp_top1,
            }
        )

    summary_df = pd.DataFrame(summary_rows).round(6)
    summary_df.to_csv(ANALYSIS_DIR / "performance_matrix.csv", index=False)
    (ANALYSIS_DIR / "performance_matrix.md").write_text(_to_markdown(summary_df), encoding="utf-8")
    return summary_df


def _to_markdown(df):
    headers = list(df.columns)
    rows = [headers] + df.astype(str).values.tolist()
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(headers))]

    def format_row(row):
        cells = [str(value).ljust(widths[index]) for index, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    for row in df.astype(str).values.tolist():
        lines.append(format_row(row))
    return "\n".join(lines) + "\n"


def main():
    summary_df = build_performance_outputs()
    build_appraisal_comparison()
    print("Wrote:", ANALYSIS_DIR / "performance_matrix.csv")
    print("Wrote:", ANALYSIS_DIR / "performance_matrix.md")
    print("Wrote:", ANALYSIS_DIR / "appraisal_profile_comparison.csv")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
