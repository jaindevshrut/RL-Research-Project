import argparse
import csv
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .appraisal import compute_appraisal_vector
from .classifiers import (
    FEATURES_4D,
    MLPEnsembleClassifier,
    SVMBaseline,
    load_dataset,
    load_model_results,
    write_mlp_probability_table,
    write_svm_probability_table,
)
from .dqn_agent import DQNAgent
from .environments import EXP12_SCENARIOS, EXP3_SCENARIOS


EXPERIMENT_CONFIG = {
    "exp12": {
        "scenarios": EXP12_SCENARIOS,
        "free_mean": 32,
        "free_var": 2,
        "free_n": 42,
        "free_output": "svm_free_0.0032_var.csv",
        "limit_mean": 140,
        "limit_var": 56,
        "limit_n": 30,
        "limit_output": "svm_limit_0.014_var.csv",
    },
    "exp3": {
        "scenarios": EXP3_SCENARIOS,
        "free_mean": 13,
        "free_var": 1,
        "free_n": 34,
        "free_output": "svm_free_0.0013_var.csv",
        "limit_mean": 34,
        "limit_var": 10,
        "limit_n": 34,
        "limit_output": "svm_limit_0.0034_var.csv",
    },
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_scenario(config, run_index, seed):
    seed_everything(seed + run_index)

    mdp = config["cls"]()
    agent = DQNAgent(mdp, **config.get("agent_kwargs", {}))
    agent.train(episodes=config["train_episodes"], i_change=config["i_change"])
    sim_result = agent.simulate_episode(terminate=config["terminate"])
    if sim_result is None:
        return None
    return compute_appraisal_vector(agent, sim_result, mode="6d")


def average_appraisals(appraisals):
    keys = appraisals[0].keys()
    averaged = {}
    for key in keys:
        averaged[key] = float(np.median([row[key] for row in appraisals]))
    return averaged


def write_model_results(output_dir, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    four_d_file = output_dir / "model_result.csv"
    six_d_file = output_dir / "model_result_6d.csv"

    with open(four_d_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Emotion"] + FEATURES_4D)
        for emotion, appraisal in results.items():
            writer.writerow([emotion] + [appraisal[key] for key in FEATURES_4D])

    with open(six_d_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Emotion",
                "Suddenness",
                "Goal_relevance",
                "Conduciveness",
                "Power",
                "Intrinsic_unpredictability",
                "Normative_significance",
            ]
        )
        for emotion, appraisal in results.items():
            writer.writerow(
                [
                    emotion,
                    appraisal["Suddenness"],
                    appraisal["Goal_relevance"],
                    appraisal["Conduciveness"],
                    appraisal["Power"],
                    appraisal["Intrinsic_unpredictability"],
                    appraisal["Normative_significance"],
                ]
            )

    return four_d_file, six_d_file


def run_experiment(experiment_name, output_dir, runs=5, seed=7, episode_scale=1.0):
    config = EXPERIMENT_CONFIG[experiment_name]
    results = OrderedDict()

    for emotion, scenario in config["scenarios"].items():
        scaled_scenario = dict(scenario)
        scaled_scenario["train_episodes"] = max(
            1, int(round(scaled_scenario["train_episodes"] * episode_scale))
        )
        scaled_scenario["i_change"] = int(round(scaled_scenario["i_change"] * episode_scale))

        appraisals = []
        for run_index in range(runs):
            appraisal = run_scenario(scaled_scenario, run_index, seed + len(results) * 101)
            if appraisal is not None:
                appraisals.append(appraisal)

        if not appraisals:
            appraisals = [
                {
                    "Suddenness": 0.0,
                    "Goal_relevance": 0.0,
                    "Conduciveness": 0.5,
                    "Power": 0.0,
                    "Intrinsic_unpredictability": 0.0,
                    "Normative_significance": 0.0,
                }
            ]

        results[emotion] = average_appraisals(appraisals)

    return write_model_results(Path(output_dir), results), results


def _sample_c_values(mean_base, variance_base, sample_count, seed):
    np.random.seed(seed)
    return np.random.normal(mean_base, np.sqrt(variance_base), sample_count) / 10000.0


def run_classifier_suite(experiment_name, data_dir, seed=7):
    config = EXPERIMENT_CONFIG[experiment_name]
    data_dir = Path(data_dir)
    seed_everything(seed)

    X_train, y_train = load_dataset(data_dir / "classifier_train.csv")
    X_test, story_names = load_model_results(data_dir / "model_result.csv")
    class_names = list(dict.fromkeys(y_train))

    free_samples = _sample_c_values(
        config["free_mean"], config["free_var"], config["free_n"], seed
    )
    free_probabilities = []
    for c_value in free_samples:
        clf = SVMBaseline(float(c_value)).fit(X_train, y_train)
        free_probabilities.append(clf.predict_proba(X_test))
    write_svm_probability_table(
        data_dir / config["free_output"],
        free_samples,
        clf.classes_,
        story_names,
        free_probabilities,
    )

    limit_samples = _sample_c_values(
        config["limit_mean"], config["limit_var"], config["limit_n"], seed + 1
    )
    limit_probabilities = []
    for c_value in limit_samples:
        clf = SVMBaseline(float(c_value)).fit(X_train, y_train)
        limit_probabilities.append(clf.predict_proba(X_test))
    write_svm_probability_table(
        data_dir / config["limit_output"],
        limit_samples,
        clf.classes_,
        story_names,
        limit_probabilities,
    )

    mlp = MLPEnsembleClassifier(input_dim=len(FEATURES_4D), class_names=class_names)
    mlp.fit(X_train, y_train)
    mlp_probabilities = mlp.predict_proba(X_test)
    mlp_predictions = mlp.predict(X_test)
    write_mlp_probability_table(
        data_dir / "mlp_ensemble_predictions.csv",
        class_names,
        story_names,
        mlp_probabilities,
        mlp_predictions,
    )

    return {
        "svm_free": data_dir / config["free_output"],
        "svm_limit": data_dir / config["limit_output"],
        "mlp": data_dir / "mlp_ensemble_predictions.csv",
    }


def run_full_pipeline(experiment_name, data_dir, runs=3, seed=7, episode_scale=1.0):
    result_files, _ = run_experiment(
        experiment_name, data_dir, runs=runs, seed=seed, episode_scale=episode_scale
    )
    classifier_files = run_classifier_suite(experiment_name, data_dir, seed=seed)
    return result_files, classifier_files


def main():
    parser = argparse.ArgumentParser(description="Run the improved baseline-compatible pipeline.")
    parser.add_argument("--experiment", choices=["exp12", "exp3", "all"], default="all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-scale", type=float, default=1.0)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    experiment_dirs = {
        "exp12": args.repo_root / "Exp1_2" / "data",
        "exp3": args.repo_root / "Exp3" / "data",
    }

    selected = ["exp12", "exp3"] if args.experiment == "all" else [args.experiment]
    for experiment_name in selected:
        result_files, classifier_files = run_full_pipeline(
            experiment_name,
            experiment_dirs[experiment_name],
            runs=args.runs,
            seed=args.seed,
            episode_scale=args.episode_scale,
        )
        four_d_file, six_d_file = result_files
        print(f"{experiment_name}: wrote {four_d_file}")
        print(f"{experiment_name}: wrote {six_d_file}")
        for name, path in classifier_files.items():
            print(f"{experiment_name}: wrote {name} -> {path}")


if __name__ == "__main__":
    main()
