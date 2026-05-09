"""
improved_experiment.py - Unified Experiment Runner
====================================================
Main script that:
  1. Trains DQN agents on all 11 emotion MDPs
  2. Computes 6D appraisal vectors
  3. Trains neural classifier ensemble + SVM baseline
  4. Predicts emotions and computes metrics
  5. Prints comparison table vs original paper results

Usage:
    python improved_experiment.py

Author: Improved Appraisal-RL Model
"""

import os
import sys
import csv
import time
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments import ALL_SCENARIOS, EXP12_SCENARIOS, EXP3_SCENARIOS
from dqn_agent import DQNAgent
from appraisal import compute_appraisal_vector
from neural_classifier import (
    NeuralClassifierEnsemble, SVMBaseline,
    generate_training_data,
    EXP12_PATTERNS_4D, EXP12_PATTERNS_6D,
    EXP3_PATTERNS_4D, EXP3_PATTERNS_6D,
)


# =============================================================================
# Original paper results (from the paper & saved CSV data)
# =============================================================================

ORIGINAL_EXP12_RESULTS = {
    'Boredom':   {'Suddenness': 0.000, 'Goal_relevance': 0.000, 'Conduciveness': 0.500, 'Power': 0.600},
    'Fear':      {'Suddenness': 0.797, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.000},
    'Happiness': {'Suddenness': 0.000, 'Goal_relevance': 0.668, 'Conduciveness': 0.834, 'Power': 0.949},
    'Joy':       {'Suddenness': 0.802, 'Goal_relevance': 1.000, 'Conduciveness': 1.000, 'Power': 0.000},
    'Pride':     {'Suddenness': 0.508, 'Goal_relevance': 1.000, 'Conduciveness': 1.000, 'Power': 0.146},
    'Sadness':   {'Suddenness': 0.198, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.000},
    'Shame':     {'Suddenness': 0.797, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.435},
}

ORIGINAL_EXP3_RESULTS = {
    'Anxiety':    {'Suddenness': 0.200, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.000},
    'Despair':    {'Suddenness': 0.807, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.000},
    'Irritation': {'Suddenness': 0.196, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.532},
    'Rage':       {'Suddenness': 0.801, 'Goal_relevance': 1.000, 'Conduciveness': 0.000, 'Power': 0.774},
}

# Paper reported metrics
ORIGINAL_METRICS = {
    'Exp1_free':  {'R2': 0.65, 'RMSE': 0.09},
    'Exp2_limit': {'R2': 0.92, 'RMSE': 0.09},
    'Exp3':       {'R2': 0.62, 'RMSE': 0.09},
}


def run_scenario(emotion_name, config, n_runs=3):
    """Run a single emotion scenario multiple times and average appraisals.

    DIFF vs original: Uses DQN instead of tabular Q-learning, runs multiple
    times for statistical stability, and computes 6D appraisal vector.
    """
    all_appraisals = []

    for run in range(n_runs):
        mdp = config['cls']()
        agent = DQNAgent(mdp, lr=5e-3, gamma=0.9, alpha=0.3, epsilon=0.3,
                         buffer_size=10000, batch_size=64)

        # Train
        agent.train(i_max=config['train_episodes'], i_change=config['i_change'])

        # Simulate story episode
        sim_result = agent.simulate_episode(terminate=config['terminate'])

        if sim_result is None:
            continue

        # Compute 6D appraisal vector
        appraisal = compute_appraisal_vector(agent, sim_result, mdp, mode='6d')
        all_appraisals.append(appraisal)

    if not all_appraisals:
        return {k: 0.0 for k in ['Suddenness', 'Goal_relevance', 'Conduciveness',
                                   'Power', 'Intrinsic_unpredictability', 'Normative_significance']}

    # Average over runs
    avg_appraisal = {}
    keys = all_appraisals[0].keys()
    for k in keys:
        vals = [a[k] for a in all_appraisals]
        avg_appraisal[k] = float(np.mean(vals))

    return avg_appraisal


def compute_r2_rmse(model_probs, target_emotion_names, class_names):
    """Compute R² and RMSE metrics for emotion prediction accuracy.

    For each scenario, we compare the predicted probability for the
    correct emotion against the ideal (1.0 for correct, uniform for others).

    Returns precision-based R² and RMSE matching the paper's methodology.
    """
    precisions = []
    for i, target in enumerate(target_emotion_names):
        if target in class_names:
            idx = class_names.index(target)
            precisions.append(model_probs[i][idx])
        else:
            precisions.append(0.0)

    precisions = np.array(precisions)
    mean_prec = np.mean(precisions)

    # R² relative to mean baseline
    ss_res = np.sum((precisions - 1.0) ** 2)
    ss_tot = np.sum((precisions - mean_prec) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    rmse = np.sqrt(np.mean((precisions - 1.0) ** 2))

    return r2, rmse, precisions


def compute_appraisal_r2_rmse(model_results, original_results, dims=None):
    """Compute R² and RMSE by comparing model appraisals vs original paper appraisals.

    This matches the paper's methodology: flatten all appraisal dimensions
    across all emotions, then compute R² = 1 - SS_res/SS_tot.

    Args:
        model_results: dict {emotion: {dim: value}}
        original_results: dict {emotion: {dim: value}}
        dims: list of dimension names to compare (defaults to 4D original dims)

    Returns:
        r2, rmse
    """
    if dims is None:
        dims = ['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power']

    model_vals = []
    original_vals = []

    for emo in original_results:
        if emo in model_results:
            for d in dims:
                if d in model_results[emo] and d in original_results[emo]:
                    model_vals.append(model_results[emo][d])
                    original_vals.append(original_results[emo][d])

    model_vals = np.array(model_vals)
    original_vals = np.array(original_vals)

    if len(model_vals) == 0:
        return 0.0, 1.0

    # R² : how well model values explain original values
    ss_res = np.sum((model_vals - original_vals) ** 2)
    mean_orig = np.mean(original_vals)
    ss_tot = np.sum((original_vals - mean_orig) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    rmse = np.sqrt(np.mean((model_vals - original_vals) ** 2))

    return r2, rmse


def main():
    print("=" * 75)
    print("  IMPROVED APPRAISAL-RL MODEL")
    print("  3 Novel Improvements over the Original Paper")
    print("=" * 75)
    print()

    os.makedirs('results', exist_ok=True)
    start_time = time.time()

    # =====================================================================
    # STEP 1: Run all emotion scenarios with DQN agent
    # =====================================================================
    print("[STEP 1] Training DQN agents on all 11 emotion scenarios...")
    print("-" * 60)

    exp12_results = OrderedDict()
    exp3_results = OrderedDict()

    for name, config in EXP12_SCENARIOS.items():
        print(f"  Training {name:12s} ...", end=" ", flush=True)
        appraisal = run_scenario(name, config, n_runs=5)
        exp12_results[name] = appraisal
        print(f"OK  Sud={appraisal['Suddenness']:.3f}  "
              f"GR={appraisal['Goal_relevance']:.3f}  "
              f"Cdc={appraisal['Conduciveness']:.3f}  "
              f"Pwr={appraisal['Power']:.3f}  "
              f"IU={appraisal['Intrinsic_unpredictability']:.3f}  "
              f"NS={appraisal['Normative_significance']:.3f}")

    for name, config in EXP3_SCENARIOS.items():
        print(f"  Training {name:12s} ...", end=" ", flush=True)
        appraisal = run_scenario(name, config, n_runs=5)
        exp3_results[name] = appraisal
        print(f"OK  Sud={appraisal['Suddenness']:.3f}  "
              f"GR={appraisal['Goal_relevance']:.3f}  "
              f"Cdc={appraisal['Conduciveness']:.3f}  "
              f"Pwr={appraisal['Power']:.3f}  "
              f"IU={appraisal['Intrinsic_unpredictability']:.3f}  "
              f"NS={appraisal['Normative_significance']:.3f}")

    # Save appraisal results
    all_results = {**exp12_results, **exp3_results}
    with open('results/model_result_6d.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Emotion', 'Suddenness', 'Goal_relevance', 'Conduciveness',
                          'Power', 'Intrinsic_unpredictability', 'Normative_significance'])
        for emo, app in all_results.items():
            writer.writerow([emo, app['Suddenness'], app['Goal_relevance'],
                             app['Conduciveness'], app['Power'],
                             app['Intrinsic_unpredictability'],
                             app['Normative_significance']])

    print(f"\n  Appraisal results saved to results/model_result_6d.csv")

    # =====================================================================
    # STEP 2: Train classifiers
    # =====================================================================
    print(f"\n[STEP 2] Training emotion classifiers...")
    print("-" * 60)

    # --- Experiment 1/2: Neural Classifier (6D) ---
    print("  Loading 4D data from original dataset and augmenting to 6D for Exp1/2 (7 emotions)...")
    csv_12 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Appraisal_RL', 'Exp1_2', 'data', 'classifier_train.csv')
    X_train_12, y_train_12, feat_12 = generate_training_data(EXP12_PATTERNS_6D, csv_path=csv_12)
    class_names_12 = list(EXP12_PATTERNS_6D.keys())

    print(f"  Training neural ensemble (10 models) on {len(X_train_12)} samples...")
    nn_clf_12 = NeuralClassifierEnsemble(
        input_dim=6, class_names=class_names_12, n_models=10, epochs=200, lr=1e-3
    )
    nn_clf_12.fit(X_train_12, y_train_12)

    # --- SVM Baseline (4D) for comparison ---
    print("  Training SVM baseline (4D) for comparison using original dataset...")
    X_train_12_4d, y_train_12_4d, _ = generate_training_data(EXP12_PATTERNS_4D, csv_path=csv_12)
    svm_free_12 = SVMBaseline(C=0.0032, class_names=class_names_12)
    svm_free_12.fit(X_train_12_4d, y_train_12_4d)
    svm_limit_12 = SVMBaseline(C=0.014, class_names=class_names_12)
    svm_limit_12.fit(X_train_12_4d, y_train_12_4d)

    # --- Experiment 3: Neural Classifier (6D) ---
    print("  Loading 4D data from original dataset and augmenting to 6D for Exp3 (4 emotions)...")
    csv_3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Appraisal_RL', 'Exp3', 'data', 'classifier_train.csv')
    X_train_3, y_train_3, feat_3 = generate_training_data(EXP3_PATTERNS_6D, csv_path=csv_3)
    class_names_3 = list(EXP3_PATTERNS_6D.keys())

    print(f"  Training neural ensemble (10 models) on {len(X_train_3)} samples...")
    nn_clf_3 = NeuralClassifierEnsemble(
        input_dim=6, class_names=class_names_3, n_models=10, epochs=200, lr=1e-3
    )
    nn_clf_3.fit(X_train_3, y_train_3)

    # --- SVM Baseline for Exp3 ---
    X_train_3_4d, y_train_3_4d, _ = generate_training_data(EXP3_PATTERNS_4D, csv_path=csv_3)
    svm_clf_3 = SVMBaseline(C=0.0032, class_names=class_names_3)
    svm_clf_3.fit(X_train_3_4d, y_train_3_4d)

    print("  All classifiers trained successfully!")

    # =====================================================================
    # STEP 3: Predict emotions for model data
    # =====================================================================
    print(f"\n[STEP 3] Predicting emotions from appraisal vectors...")
    print("-" * 60)

    # --- Exp1/2: Neural (6D) predictions ---
    X_test_12_6d = np.array([
        [exp12_results[emo]['Suddenness'],
         exp12_results[emo]['Goal_relevance'],
         exp12_results[emo]['Conduciveness'],
         exp12_results[emo]['Power'],
         exp12_results[emo]['Intrinsic_unpredictability'],
         exp12_results[emo]['Normative_significance']]
        for emo in class_names_12
    ], dtype=np.float32)

    nn_probs_12 = nn_clf_12.predict_proba(X_test_12_6d)
    nn_preds_12 = nn_clf_12.predict(X_test_12_6d)

    # --- Exp1/2: SVM (4D) predictions for comparison ---
    X_test_12_4d = np.array([
        [exp12_results[emo]['Suddenness'],
         exp12_results[emo]['Goal_relevance'],
         exp12_results[emo]['Conduciveness'],
         exp12_results[emo]['Power']]
        for emo in class_names_12
    ], dtype=np.float32)

    svm_free_probs_12 = svm_free_12.predict_proba(X_test_12_4d)
    svm_limit_probs_12 = svm_limit_12.predict_proba(X_test_12_4d)

    # --- Exp3: Neural (6D) predictions ---
    X_test_3_6d = np.array([
        [exp3_results[emo]['Suddenness'],
         exp3_results[emo]['Goal_relevance'],
         exp3_results[emo]['Conduciveness'],
         exp3_results[emo]['Power'],
         exp3_results[emo]['Intrinsic_unpredictability'],
         exp3_results[emo]['Normative_significance']]
        for emo in class_names_3
    ], dtype=np.float32)

    nn_probs_3 = nn_clf_3.predict_proba(X_test_3_6d)
    nn_preds_3 = nn_clf_3.predict(X_test_3_6d)

    # --- Exp3: SVM (4D) predictions ---
    X_test_3_4d = np.array([
        [exp3_results[emo]['Suddenness'],
         exp3_results[emo]['Goal_relevance'],
         exp3_results[emo]['Conduciveness'],
         exp3_results[emo]['Power']]
        for emo in class_names_3
    ], dtype=np.float32)

    svm_probs_3 = svm_clf_3.predict_proba(X_test_3_4d)

    # =====================================================================
    # STEP 4: Compute Metrics
    # =====================================================================
    print(f"\n[STEP 4] Computing metrics...")
    print("-" * 60)

    # --- Appraisal-level R²/RMSE (paper's methodology) ---
    # Compare model appraisal vectors vs original paper appraisal vectors (4D)
    r2_appraisal_12, rmse_appraisal_12 = compute_appraisal_r2_rmse(
        exp12_results, ORIGINAL_EXP12_RESULTS)
    r2_appraisal_3, rmse_appraisal_3 = compute_appraisal_r2_rmse(
        exp3_results, ORIGINAL_EXP3_RESULTS)

    # --- Classifier-based metrics (supplementary) ---
    # Compute R² and RMSE for each experiment
    r2_nn_12, rmse_nn_12, prec_nn_12 = compute_r2_rmse(nn_probs_12, class_names_12, class_names_12)
    r2_svm_free_12, rmse_svm_free_12, prec_svm_free_12 = compute_r2_rmse(
        svm_free_probs_12, class_names_12, list(svm_free_12.model.classes_))
    r2_svm_limit_12, rmse_svm_limit_12, prec_svm_limit_12 = compute_r2_rmse(
        svm_limit_probs_12, class_names_12, list(svm_limit_12.model.classes_))

    r2_nn_3, rmse_nn_3, prec_nn_3 = compute_r2_rmse(nn_probs_3, class_names_3, class_names_3)
    r2_svm_3, rmse_svm_3, prec_svm_3 = compute_r2_rmse(
        svm_probs_3, class_names_3, list(svm_clf_3.model.classes_))

    # Top-1 accuracy
    acc_nn_12 = sum(1 for p, t in zip(nn_preds_12, class_names_12) if p == t) / len(class_names_12)
    acc_nn_3 = sum(1 for p, t in zip(nn_preds_3, class_names_3) if p == t) / len(class_names_3)

    # Ranking accuracy: is correct emotion in top-2?
    def top_k_accuracy(probs, targets, class_list, k=2):
        correct = 0
        for i, target in enumerate(targets):
            if target in class_list:
                top_k_idx = np.argsort(probs[i])[-k:]
                top_k_names = [class_list[j] for j in top_k_idx]
                if target in top_k_names:
                    correct += 1
        return correct / len(targets)

    rank_nn_12 = top_k_accuracy(nn_probs_12, class_names_12, class_names_12, k=2)
    rank_nn_3 = top_k_accuracy(nn_probs_3, class_names_3, class_names_3, k=2)

    # =====================================================================
    # STEP 5: Print Results
    # =====================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 75}")
    print("  RESULTS")
    print(f"{'=' * 75}")

    # --- Appraisal Vectors ---
    print("\n  [A] MODEL APPRAISAL VECTORS (6D) — DQN Agent")
    print("  " + "-" * 73)
    print(f"  {'Emotion':12s} | {'Sud':6s} | {'GR':6s} | {'Cdc':6s} | "
          f"{'Pwr':6s} | {'IU':6s} | {'NS':6s}")
    print("  " + "-" * 73)
    for emo, app in all_results.items():
        print(f"  {emo:12s} | {app['Suddenness']:6.3f} | {app['Goal_relevance']:6.3f} | "
              f"{app['Conduciveness']:6.3f} | {app['Power']:6.3f} | "
              f"{app['Intrinsic_unpredictability']:6.3f} | "
              f"{app['Normative_significance']:6.3f}")

    # --- Emotion Predictions ---
    print("\n  [B] EMOTION PREDICTIONS — Neural Ensemble (6D)")
    print("  " + "-" * 50)
    print(f"  {'True Emotion':12s} | {'Predicted':12s} | {'Confidence':10s}")
    print("  " + "-" * 50)
    for i, emo in enumerate(class_names_12):
        pred = nn_preds_12[i]
        conf = nn_probs_12[i][class_names_12.index(pred)]
        marker = "✓" if pred == emo else "✗"
        print(f"  {emo:12s} | {pred:12s} | {conf:8.4f}  {marker}")
    for i, emo in enumerate(class_names_3):
        pred = nn_preds_3[i]
        conf = nn_probs_3[i][class_names_3.index(pred)]
        marker = "✓" if pred == emo else "✗"
        print(f"  {emo:12s} | {pred:12s} | {conf:8.4f}  {marker}")

    # --- Comparison Table ---
    print(f"\n  {'=' * 75}")
    print("  [C] COMPARISON TABLE: Original Paper vs. Improved Model")
    print(f"  {'=' * 75}")
    print()

    # --- Appraisal-level metrics (paper's methodology) ---
    print("  >> Appraisal-Level Metrics (model vs. paper appraisal vectors, 4D)")
    print(f"  {'Metric':25s} | {'Original Paper':18s} | {'Improved (Ours)':18s} | {'Δ':8s}")
    print("  " + "-" * 75)
    print(f"  {'Exp1/2 Appraisal R²':25s} | {ORIGINAL_METRICS['Exp1_free']['R2']:18.4f} | "
          f"{r2_appraisal_12:18.4f} | {r2_appraisal_12 - ORIGINAL_METRICS['Exp1_free']['R2']:+8.4f}")
    print(f"  {'Exp1/2 Appraisal RMSE':25s} | {ORIGINAL_METRICS['Exp1_free']['RMSE']:18.4f} | "
          f"{rmse_appraisal_12:18.4f} | {rmse_appraisal_12 - ORIGINAL_METRICS['Exp1_free']['RMSE']:+8.4f}")
    print(f"  {'Exp3 Appraisal R²':25s} | {ORIGINAL_METRICS['Exp3']['R2']:18.4f} | "
          f"{r2_appraisal_3:18.4f} | {r2_appraisal_3 - ORIGINAL_METRICS['Exp3']['R2']:+8.4f}")
    print(f"  {'Exp3 Appraisal RMSE':25s} | {ORIGINAL_METRICS['Exp3']['RMSE']:18.4f} | "
          f"{rmse_appraisal_3:18.4f} | {rmse_appraisal_3 - ORIGINAL_METRICS['Exp3']['RMSE']:+8.4f}")

    print()
    print("  >> Classifier-Based Metrics (supplementary)")
    print("  " + "-" * 75)

    # Exp1 Free (SVM, for backward compat)
    print(f"  {'SVM Exp1 Free R²':25s} | {ORIGINAL_METRICS['Exp1_free']['R2']:18.4f} | "
          f"{r2_svm_free_12:18.4f} | {r2_svm_free_12 - ORIGINAL_METRICS['Exp1_free']['R2']:+8.4f}")
    print(f"  {'SVM Exp2 Limit R²':25s} | {ORIGINAL_METRICS['Exp2_limit']['R2']:18.4f} | "
          f"{r2_svm_limit_12:18.4f} | {r2_svm_limit_12 - ORIGINAL_METRICS['Exp2_limit']['R2']:+8.4f}")
    print(f"  {'SVM Exp3 R²':25s} | {ORIGINAL_METRICS['Exp3']['R2']:18.4f} | "
          f"{r2_svm_3:18.4f} | {r2_svm_3 - ORIGINAL_METRICS['Exp3']['R2']:+8.4f}")

    print("  " + "-" * 75)

    print(f"  {'NN 6D Exp1/2 R²':25s} | {'N/A':18s} | {r2_nn_12:18.4f} | {'NEW':>8s}")
    print(f"  {'NN 6D Exp3 R²':25s} | {'N/A':18s} | {r2_nn_3:18.4f} | {'NEW':>8s}")
    print(f"  {'NN 6D Exp1/2 RMSE':25s} | {'N/A':18s} | {rmse_nn_12:18.4f} | {'NEW':>8s}")
    print(f"  {'NN 6D Exp3 RMSE':25s} | {'N/A':18s} | {rmse_nn_3:18.4f} | {'NEW':>8s}")

    print("  " + "-" * 75)

    print(f"  {'NN 6D Top-1 Acc Exp1/2':25s} | {'N/A':18s} | {acc_nn_12:18.4f} | {'NEW':>8s}")
    print(f"  {'NN 6D Top-1 Acc Exp3':25s} | {'N/A':18s} | {acc_nn_3:18.4f} | {'NEW':>8s}")
    print(f"  {'NN 6D Top-2 Acc Exp1/2':25s} | {'N/A':18s} | {rank_nn_12:18.4f} | {'NEW':>8s}")
    print(f"  {'NN 6D Top-2 Acc Exp3':25s} | {'N/A':18s} | {rank_nn_3:18.4f} | {'NEW':>8s}")

    print("  " + "-" * 75)
    print(f"  {'Appraisal Dimensions':25s} | {'4':18s} | {'6':18s} | {'+2':>8s}")
    print(f"  {'RL Algorithm':25s} | {'Tabular Q-learning':18s} | {'DQN (neural)':18s} | {'NEW':>8s}")
    print(f"  {'Classifier':25s} | {'SVM (linear)':18s} | {'MLP Ensemble(10)':18s} | {'NEW':>8s}")

    # --- Novel contributions summary ---
    print(f"\n  {'=' * 75}")
    print("  [D] NOVEL CONTRIBUTIONS SUMMARY")
    print(f"  {'=' * 75}")
    print("""
  1. DQN AGENT: Replaced tabular Q-learning with a neural network-based
     Deep Q-Network using experience replay and target network. This enables
     scaling to larger environments and extracts richer gradient signals.

  2. EXTENDED 6D APPRAISAL: Added two new CPM appraisal checks:
     - Intrinsic Unpredictability (transition entropy): captures how
       inherently stochastic the environment is
     - Normative Significance (TD error z-score): captures how unusual
       the current outcome is relative to the agent's history

  3. NEURAL CLASSIFIER ENSEMBLE: Replaced single SVM with an ensemble
     of 10 MLP classifiers with dropout for variance-reduced, more
     robust emotion prediction. Supports the extended 6D feature space.
""")

    # Save detailed results
    with open('results/comparison_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Original', 'Improved', 'Delta'])
        writer.writerow(['Exp12_Appraisal_R2', ORIGINAL_METRICS['Exp1_free']['R2'], r2_appraisal_12,
                          r2_appraisal_12 - ORIGINAL_METRICS['Exp1_free']['R2']])
        writer.writerow(['Exp3_Appraisal_R2', ORIGINAL_METRICS['Exp3']['R2'], r2_appraisal_3,
                          r2_appraisal_3 - ORIGINAL_METRICS['Exp3']['R2']])
        writer.writerow(['Exp12_Appraisal_RMSE', ORIGINAL_METRICS['Exp1_free']['RMSE'], rmse_appraisal_12,
                          rmse_appraisal_12 - ORIGINAL_METRICS['Exp1_free']['RMSE']])
        writer.writerow(['Exp3_Appraisal_RMSE', ORIGINAL_METRICS['Exp3']['RMSE'], rmse_appraisal_3,
                          rmse_appraisal_3 - ORIGINAL_METRICS['Exp3']['RMSE']])
        writer.writerow(['SVM_Exp1_Free_R2', ORIGINAL_METRICS['Exp1_free']['R2'], r2_svm_free_12,
                          r2_svm_free_12 - ORIGINAL_METRICS['Exp1_free']['R2']])
        writer.writerow(['SVM_Exp2_Limit_R2', ORIGINAL_METRICS['Exp2_limit']['R2'], r2_svm_limit_12,
                          r2_svm_limit_12 - ORIGINAL_METRICS['Exp2_limit']['R2']])
        writer.writerow(['SVM_Exp3_R2', ORIGINAL_METRICS['Exp3']['R2'], r2_svm_3,
                          r2_svm_3 - ORIGINAL_METRICS['Exp3']['R2']])
        writer.writerow(['NN_6D_Exp12_R2', 'N/A', r2_nn_12, 'NEW'])
        writer.writerow(['NN_6D_Exp3_R2', 'N/A', r2_nn_3, 'NEW'])
        writer.writerow(['NN_6D_Exp12_Top1_Acc', 'N/A', acc_nn_12, 'NEW'])
        writer.writerow(['NN_6D_Exp3_Top1_Acc', 'N/A', acc_nn_3, 'NEW'])

    # Save per-emotion precision
    with open('results/emotion_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Emotion', 'Predicted', 'Confidence', 'Correct'])
        for i, emo in enumerate(class_names_12):
            pred = nn_preds_12[i]
            conf = nn_probs_12[i][class_names_12.index(pred)]
            writer.writerow(['Exp1_2', emo, pred, f'{conf:.4f}', pred == emo])
        for i, emo in enumerate(class_names_3):
            pred = nn_preds_3[i]
            conf = nn_probs_3[i][class_names_3.index(pred)]
            writer.writerow(['Exp3', emo, pred, f'{conf:.4f}', pred == emo])

    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Results saved to results/")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
