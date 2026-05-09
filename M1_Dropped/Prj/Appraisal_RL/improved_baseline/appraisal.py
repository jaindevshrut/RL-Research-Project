import numpy as np


def compute_appraisal_vector(agent, sim_result, mode="6d"):
    prev_state = sim_result["prev_state"]
    prev_action = sim_result["prev_action"]
    current_state = sim_result["current_state"]

    appraisals = {
        "Suddenness": agent.appraise_suddenness(prev_state, prev_action, current_state),
        "Goal_relevance": agent.appraise_goal_relevance(),
        "Conduciveness": agent.appraise_conduciveness(),
        "Power": agent.appraise_power(),
    }

    if mode == "6d":
        appraisals["Intrinsic_unpredictability"] = _compute_intrinsic_unpredictability(
            prev_state, prev_action, sim_result["t_hat"]
        )
        appraisals["Normative_significance"] = _compute_normative_significance(
            sim_result["td_error"], sim_result["td_error_history"]
        )

    return appraisals


def _compute_intrinsic_unpredictability(prev_state, prev_action, t_hat):
    if prev_state is None or prev_action is None:
        return 0.0

    counts = t_hat.get(prev_state, {}).get(prev_action, {})
    total = sum(counts.values())
    if total == 0:
        return 0.0

    nonzero = [count for count in counts.values() if count > 0]
    if len(nonzero) <= 1:
        return 0.0

    entropy = 0.0
    for count in nonzero:
        prob = count / total
        entropy -= prob * np.log(prob + 1e-10)

    max_entropy = np.log(len(nonzero))
    if max_entropy == 0:
        return 0.0

    return float(min(1.0, entropy / max_entropy))


def _compute_normative_significance(td_error, td_error_history):
    if len(td_error_history) < 2:
        return 0.5

    history = np.array(td_error_history[-500:], dtype=float)
    std = float(np.std(history))
    if std == 0:
        return 0.0

    z_score = abs(float(td_error) - float(np.mean(history))) / (std + 1e-8)
    return float(min(1.0, z_score / 3.0))
