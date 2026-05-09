"""
appraisal.py - Extended 6-Dimensional Appraisal Vector Computation
===================================================================
IMPROVEMENT #2: Extends the original 4 appraisal checks to 6 by adding:
  5. Intrinsic Unpredictability — entropy of the learned transition model
  6. Normative Significance — deviation of TD error from running average
"""

import numpy as np


def compute_appraisal_vector(agent, sim_result, mdp, mode='6d'):
    """Compute appraisal vector from simulation results.

    Args:
        agent: DQNAgent instance (for appraisal methods)
        sim_result: dict returned by agent.simulate_episode()
        mdp: the MDP environment
        mode: '4d' for original paper checks, '6d' for extended

    Returns:
        dict with appraisal dimension names and values
    """
    prev_state = sim_result['prev_state']
    prev_action = sim_result['prev_action']
    current_state = sim_result['current_state']

    appraisals = {}

    # ---- 1. Suddenness (original) ----
    appraisals['Suddenness'] = agent.appraise_suddenness(prev_state, prev_action, current_state)

    # ---- 2. Goal Relevance (original) ----
    appraisals['Goal_relevance'] = agent.appraise_goal_relevance()

    # ---- 3. Conduciveness (original) ----
    appraisals['Conduciveness'] = agent.appraise_conduciveness()

    # ---- 4. Power (original) ----
    appraisals['Power'] = agent.appraise_power()

    if mode == '6d':
        # ---- 5. Intrinsic Unpredictability (NEW) ----
        appraisals['Intrinsic_unpredictability'] = _compute_intrinsic_unpredictability(
            prev_state, prev_action, sim_result['t_hat'], mdp
        )

        # ---- 6. Normative Significance (NEW) ----
        appraisals['Normative_significance'] = _compute_normative_significance(
            sim_result['td_error'], sim_result['td_error_history']
        )

    return appraisals


def _compute_intrinsic_unpredictability(prev_state, prev_action, t_hat, mdp):
    """Intrinsic Unpredictability = normalized entropy of transition distribution.

    High entropy → environment is inherently unpredictable.
    Normalized by log(n_outcomes) so result is in [0, 1].
    """
    if prev_state is None or prev_action is None:
        return 0.0

    counts = t_hat.get(prev_state, {}).get(prev_action, {})
    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Count non-zero transitions
    nonzero = {s: c for s, c in counts.items() if c > 0}
    if len(nonzero) <= 1:
        return 0.0  # Deterministic transition

    # Compute entropy
    entropy = 0.0
    for s2, c in nonzero.items():
        p = c / total
        entropy -= p * np.log(p + 1e-10)

    # Normalize by max possible entropy for observed outcomes
    max_entropy = np.log(len(nonzero))
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def _compute_normative_significance(td_error, td_error_history):
    """Normative Significance = how unusual this TD error is.

    Uses z-score: |δ - δ̄| / (σ_δ + ε), mapped to [0, 1].
    """
    if len(td_error_history) < 2:
        return 0.5

    history = np.array(td_error_history[-500:])  # Use recent history
    mean_tde = np.mean(history)
    std_tde = np.std(history) + 1e-8

    z_score = abs(td_error - mean_tde) / std_tde
    # Map z-score to [0, 1]: z=0→0, z=2→0.67, z=3→1.0
    return min(1.0, z_score / 3.0)
