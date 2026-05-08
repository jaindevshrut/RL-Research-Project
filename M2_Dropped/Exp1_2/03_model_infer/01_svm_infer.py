"""Run SVM inference for cached model outputs and report fit metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm

CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CURRENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from config import MODEL_TYPE


DATA_DIR = EXPERIMENT_DIR / "data"
FEATURE_COLUMNS = ["Suddenness", "Goal_relevance", "Conduciveness", "Power"]
EMOTION_ORDER = [
    "Happiness",
    "Joy",
    "Pride",
    "Boredom",
    "Fear",
    "Sadness",
    "Shame",
]
STORY_ORDER = EMOTION_ORDER
SETTINGS = {
    "free": {
        "human_path": DATA_DIR / "human_free.csv",
        "human_prefix": "Emo.",
        "legacy_output": DATA_DIR / "svm_free_0.0032_var.csv",
        "aggregate_output_template": DATA_DIR / "svm_free_{model_type}.csv",
        "seed_output_template": DATA_DIR / "svm_free_{model_type}_seed{seed}.csv",
        "summary_output_template": DATA_DIR
        / "emotion_probabilities_free_{model_type}.csv",
        "sample_size": 42,
        "sample_loc": 32,
        "sample_scale": np.sqrt(2),
    },
    "limit": {
        "human_path": DATA_DIR / "human_limit.csv",
        "human_prefix": "Em.",
        "legacy_output": DATA_DIR / "svm_limit_0.014_var.csv",
        "aggregate_output_template": DATA_DIR / "svm_limit_{model_type}.csv",
        "seed_output_template": DATA_DIR / "svm_limit_{model_type}_seed{seed}.csv",
        "summary_output_template": DATA_DIR
        / "emotion_probabilities_limit_{model_type}.csv",
        "sample_size": 30,
        "sample_loc": 140,
        "sample_scale": np.sqrt(56),
    },
}


def read_appraisal_data(data_file: Path):
    data = pd.read_csv(data_file)
    x_values = data[FEATURE_COLUMNS].values.astype(float)
    y_values = data["Emotion"].values
    return x_values, y_values


def build_feature_normalizer(x_training: np.ndarray):
    train_min = x_training.min(axis=0)
    train_span = x_training.max(axis=0) - train_min
    train_span[train_span == 0] = 1.0

    def normalize(values: np.ndarray) -> np.ndarray:
        scaled = (values - train_min) / train_span
        return np.clip(scaled, 0.0, 1.0)

    return normalize


def adapt_testing_features(x_values: np.ndarray, model_name: str) -> np.ndarray:
    adjusted = x_values.astype(float).copy()
    if model_name != "v2":
        return adjusted

    # V2 appraisals are derived from return distributions rather than bounded
    # scalar TD statistics, so we squash them into the classifier's [0, 1]
    # feature regime before applying the training-set normalizer.
    adjusted[:, 0] = np.clip(adjusted[:, 0], 0.0, 1.0)
    adjusted[:, 1] = np.maximum(adjusted[:, 1], 0.0)
    adjusted[:, 1] = adjusted[:, 1] / (1.0 + adjusted[:, 1])
    adjusted[:, 2] = 1.0 / (
        1.0 + np.exp(-np.clip(adjusted[:, 2], -50.0, 50.0) / 5.0)
    )
    adjusted[:, 3] = np.maximum(adjusted[:, 3], 0.0)
    adjusted[:, 3] = adjusted[:, 3] / (1.0 + adjusted[:, 3])
    return adjusted


def generate_prediction_result(
    sample: np.ndarray,
    filename: Path,
    x_training: np.ndarray,
    y_training: np.ndarray,
    x_testing: np.ndarray,
    story_names,
):
    rows = []
    for c_value in sample:
        svc = svm.SVC(kernel="linear", C=float(c_value), probability=True).fit(
            x_training,
            y_training,
        )
        classes = svc.classes_
        for row_index, story_name in enumerate(story_names):
            probabilities = svc.predict_proba(x_testing[row_index].reshape(1, -1))[0]
            for emotion_name, probability in zip(classes, probabilities):
                rows.append([float(c_value), story_name, emotion_name, float(probability)])

    result = pd.DataFrame(rows, columns=["C", "Story", "Emotion", "Val"])
    result.to_csv(filename, index=False)
    return result


def load_human_distribution(setting_name: str) -> pd.DataFrame:
    setting = SETTINGS[setting_name]
    human_data = pd.read_csv(setting["human_path"], sep=";")
    emotion_columns = [
        column_name
        for column_name in human_data.columns
        if column_name.startswith(setting["human_prefix"])
    ]
    long_data = human_data.melt(
        id_vars=["Story"],
        value_vars=emotion_columns,
        var_name="Emotion",
        value_name="Val",
    )
    long_data["Emotion"] = long_data["Emotion"].str.replace(
        setting["human_prefix"],
        "",
        regex=False,
    )
    grouped = long_data.groupby(["Story", "Emotion"], as_index=False)["Val"].mean()
    grouped["Val"] = grouped.groupby("Story")["Val"].transform(
        lambda values: values / values.sum() if values.sum() else values
    )
    return grouped


def summarize_probabilities(dataframe: pd.DataFrame) -> pd.DataFrame:
    grouped = dataframe.groupby(["Story", "Emotion"], as_index=False)["Val"].mean()
    grouped["Val"] = grouped.groupby("Story")["Val"].transform(
        lambda values: values / values.sum() if values.sum() else values
    )
    grouped["Story"] = pd.Categorical(grouped["Story"], STORY_ORDER, ordered=True)
    grouped["Emotion"] = pd.Categorical(grouped["Emotion"], EMOTION_ORDER, ordered=True)
    return grouped.sort_values(["Story", "Emotion"]).reset_index(drop=True)


def compute_metrics(human_distribution: pd.DataFrame, model_distribution: pd.DataFrame):
    merged = human_distribution.merge(
        model_distribution,
        on=["Story", "Emotion"],
        how="left",
        suffixes=("_human", "_model"),
    ).fillna(0.0)
    merged["Story"] = pd.Categorical(merged["Story"], STORY_ORDER, ordered=True)
    merged["Emotion"] = pd.Categorical(merged["Emotion"], EMOTION_ORDER, ordered=True)
    merged = merged.sort_values(["Story", "Emotion"]).reset_index(drop=True)

    y_true = merged["Val_human"].to_numpy(dtype=float)
    y_pred = merged["Val_model"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst == 0:
        r2 = 1.0 if np.allclose(y_true, y_pred) else 0.0
    else:
        r2 = 1.0 - (sse / sst)
    return rmse, r2, merged


def extract_seed(seed_file: Path) -> int:
    return int(seed_file.stem.split("seed")[-1])


def print_summary_table(metrics_frame: pd.DataFrame) -> None:
    if metrics_frame.empty:
        print("No cached model_result_<model>_seed*.csv files were found.")
        return

    summary = metrics_frame.groupby("Model", as_index=False).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
    )

    print("---------------------------------")
    print("Model      RMSE      R2")
    print("---------------------------------")
    for model_name in ["baseline", "v2"]:
        model_rows = summary[summary["Model"] == model_name]
        if model_rows.empty:
            continue
        row = model_rows.iloc[0]
        print(
            f"{model_name.capitalize():<10}"
            f"{row['RMSE_mean']:.4f} +/- {row['RMSE_std']:.4f}   "
            f"{row['R2_mean']:.4f} +/- {row['R2_std']:.4f}"
        )
    print("---------------------------------")


def main() -> None:
    x_training_raw, y_training = read_appraisal_data(DATA_DIR / "classifier_train.csv")
    normalize_features = build_feature_normalizer(x_training_raw)
    x_training = normalize_features(x_training_raw)
    human_distributions = {
        setting_name: load_human_distribution(setting_name)
        for setting_name in SETTINGS
    }

    results_baseline = []
    results_v2 = []
    metrics_rows = []

    for model_name in ["baseline", "v2"]:
        seed_files = sorted(
            DATA_DIR.glob(f"model_result_{model_name}_seed*.csv"),
            key=extract_seed,
        )
        if not seed_files:
            continue

        aggregate_predictions = {setting_name: [] for setting_name in SETTINGS}

        for seed_file in seed_files:
            seed = extract_seed(seed_file)
            x_testing_raw, y_testing = read_appraisal_data(seed_file)
            x_testing = normalize_features(
                adapt_testing_features(x_testing_raw, model_name)
            )
            story_names = list(y_testing)

            overall_true = []
            overall_pred = []
            detailed_row = {"Model": model_name, "Seed": seed}

            for setting_name, setting in SETTINGS.items():
                rng = np.random.default_rng(seed)
                samples = rng.normal(
                    loc=setting["sample_loc"],
                    scale=setting["sample_scale"],
                    size=setting["sample_size"],
                ) / 10000
                prediction_path = Path(
                    str(setting["seed_output_template"]).format(
                        model_type=model_name,
                        seed=seed,
                    )
                )
                prediction_frame = generate_prediction_result(
                    samples,
                    prediction_path,
                    x_training,
                    y_training,
                    x_testing,
                    story_names,
                )
                aggregate_predictions[setting_name].append(prediction_frame)

                model_distribution = summarize_probabilities(prediction_frame)
                rmse, r2, merged = compute_metrics(
                    human_distributions[setting_name],
                    model_distribution,
                )
                detailed_row[f"{setting_name}_RMSE"] = rmse
                detailed_row[f"{setting_name}_R2"] = r2
                overall_true.append(merged["Val_human"].to_numpy(dtype=float))
                overall_pred.append(merged["Val_model"].to_numpy(dtype=float))

            y_true = np.concatenate(overall_true)
            y_pred = np.concatenate(overall_pred)
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            sse = float(np.sum((y_true - y_pred) ** 2))
            sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = 1.0 - (sse / sst) if sst else 0.0

            detailed_row["RMSE"] = rmse
            detailed_row["R2"] = r2
            metrics_rows.append(detailed_row)
            if model_name == "baseline":
                results_baseline.append(detailed_row)
            else:
                results_v2.append(detailed_row)

        for setting_name, frames in aggregate_predictions.items():
            combined_predictions = pd.concat(frames, ignore_index=True)
            aggregate_output = Path(
                str(SETTINGS[setting_name]["aggregate_output_template"]).format(
                    model_type=model_name,
                )
            )
            combined_predictions.to_csv(aggregate_output, index=False)

            summarized = summarize_probabilities(combined_predictions)
            summary_output = Path(
                str(SETTINGS[setting_name]["summary_output_template"]).format(
                    model_type=model_name,
                )
            )
            summarized.to_csv(summary_output, index=False)

            if model_name == MODEL_TYPE:
                combined_predictions.to_csv(
                    SETTINGS[setting_name]["legacy_output"],
                    index=False,
                )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(DATA_DIR / "model_metrics_detailed.csv", index=False)

    if not metrics_frame.empty:
        comparison = metrics_frame.groupby("Model", as_index=False).agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        comparison.to_csv(DATA_DIR / "model_metrics_summary.csv", index=False)

    print_summary_table(metrics_frame)

    if results_baseline:
        print(f"Cached baseline seeds: {len(results_baseline)}")
    if results_v2:
        print(f"Cached v2 seeds: {len(results_v2)}")
    print("Emotion probability summaries were written to data/emotion_probabilities_*.")


if __name__ == "__main__":
    main()
