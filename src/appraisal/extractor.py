"""
Appraisal vector extraction from DQN learning signals.

The original paper (Zhang/Broekens/Jokinen 2023) maps 4 appraisal checks
from Scherer's CPM onto RL quantities (suddenness, goal_relevance,
conduciveness, power). Three of those four ride on the TD-error in
different ways, so they are inherently correlated.

This module:
  (a) implements those 4 checks faithfully on top of a Dueling Double DQN,
  (b) adds 4 new checks that are designed (by construction) to tap
      DIFFERENT statistical properties of the agent's experience:
          - predictability  : ensemble-disagreement (epistemic uncertainty)
          - anticipation    : V(s) from the dueling V-head (absolute, not delta)
          - urgency         : t / max_steps (pure time)
          - familiarity     : log-visit count of s (state, not transition)
  (c) keeps small running statistics so that the outputs are scale-stable
      throughout training (a TD error of 0.5 should mean "big" early in
      training even if magnitudes change later).

We DO NOT orthogonalize on-the-fly inside training (that would obscure the
raw appraisal signal). Instead we measure correlation post-hoc and apply
Gram-Schmidt/whitening as an explicit analysis step
(see analysis/correlation_analysis.py).
"""
from __future__ import annotations
from collections import defaultdict
import math
from typing import Dict, Tuple
import numpy as np
import torch


class RunningMaxAbs:
    """Tracks the running maximum |x| — used to normalise unbounded signals
    like |TD error| into [0, 1] without freezing the scale at iter 0."""

    def __init__(self, eps: float = 1e-3):
        self.value = eps

    def update(self, x: float) -> float:
        a = abs(float(x))
        if a > self.value:
            self.value = a
        return self.value


class AppraisalExtractor:
    """
    Computes an N-dim appraisal vector per timestep.

    The dimension order follows AppraisalConfig.use_dims (it lets us
    silence the 4 newer dims when reproducing the paper-style baseline).
    """

    ALL_DIMS = (
        "suddenness", "goal_relevance", "conduciveness", "power",
        "predictability", "anticipation", "urgency", "familiarity",
    )

    def __init__(self, use_dims, td_scale: float = 1.0,
                 value_scale: float = 1.0, max_steps: int = 80):
        for d in use_dims:
            assert d in self.ALL_DIMS, f"Unknown appraisal dim: {d}"
        self.dims = list(use_dims)
        self.td_scale = td_scale
        self.value_scale = value_scale
        self.max_steps = max_steps

        # Count tables (used for suddenness + familiarity)
        self.transition_counts: Dict[Tuple, Dict] = defaultdict(
            lambda: defaultdict(int))
        self.state_counts: Dict[Tuple, int] = defaultdict(int)
        self.max_state_count = 1

        # Running scalers
        self.td_max = RunningMaxAbs()
        self.q_std_max = RunningMaxAbs()

    # ------------------------------------------------------------------ #
    # Per-step update of count tables
    # ------------------------------------------------------------------ #
    def observe_transition(self, sid, action, sid_next):
        key = (sid, int(action))
        self.transition_counts[key][sid_next] += 1
        self.state_counts[sid_next] += 1
        if self.state_counts[sid_next] > self.max_state_count:
            self.max_state_count = self.state_counts[sid_next]

    # ------------------------------------------------------------------ #
    # Individual appraisal computations
    # ------------------------------------------------------------------ #
    def _suddenness(self, sid, action, sid_next) -> float:
        """1 - p(s' | s,a). Captures transition rarity.
        Note: observe_transition() is called BEFORE this method, so the
        current transition is already counted; we reverse that to get the
        a-priori probability the agent had right before observing it.
        """
        counts = self.transition_counts.get((sid, int(action)), {})
        # remove the just-observed count to recover the prior estimate
        c_prior = max(0, counts.get(sid_next, 0) - 1)
        total_prior = max(0, sum(counts.values()) - 1)
        if total_prior == 0:
            # truly novel transition -> max suddenness
            return 1.0
        p = c_prior / total_prior
        return float(np.clip(1.0 - p, 0.0, 1.0))

    def _goal_relevance(self, td_error: float) -> float:
        """|TD| / running_max  — magnitude of the value update."""
        m = self.td_max.update(td_error)
        return float(np.clip(abs(td_error) / m, 0.0, 1.0))

    def _conduciveness(self, td_error: float) -> float:
        """tanh(td / scale) -> [-1, 1].
        Sign separates good/bad; magnitude separates mild/strong.
        Correlated with goal_relevance by construction (shares |TD|).
        """
        return float(math.tanh(td_error / max(self.td_scale, 1e-6)))

    def _power(self, q_values: np.ndarray) -> float:
        """(max_a Q - mean_a Q) / (max_a |Q| + eps).
        Spread of action-effects, scale-invariant.
        Independent of the absolute value level (uses centred range).
        """
        q = np.asarray(q_values).flatten()
        spread = float(q.max() - q.mean())
        denom = float(np.abs(q).max() + 1e-3)
        return float(np.clip(spread / denom, 0.0, 1.0))

    def _predictability(self, q_std: float) -> float:
        """1 - normalised(ensemble_std).
        High when heads agree -> agent feels certain about Q.
        Independent of TD sign and of state-visit count.
        """
        m = self.q_std_max.update(q_std)
        return float(np.clip(1.0 - q_std / m, 0.0, 1.0))

    def _anticipation(self, v_state: float) -> float:
        """tanh(V(s)/scale) — the dueling V-head's prediction.
        Differs from conduciveness because conduciveness is a *delta*
        (TD error) while anticipation is an *absolute* level (V(s))."""
        return float(math.tanh(v_state / max(self.value_scale, 1e-6)))

    def _urgency(self, t: int) -> float:
        """t / max_steps. Pure time -> orthogonal to value signals."""
        return float(np.clip(t / max(1, self.max_steps), 0.0, 1.0))

    def _familiarity(self, sid_next) -> float:
        """log(N(s)+1) / log(N_max+1). State-visit, not transition.
        Distinct from suddenness, which uses transition probability."""
        n = self.state_counts.get(sid_next, 0)
        return float(math.log(n + 1.0) / math.log(self.max_state_count + 1.0))

    # ------------------------------------------------------------------ #
    # The public entry-point: one full appraisal vector per timestep
    # ------------------------------------------------------------------ #
    def compute(self,
                sid, action, sid_next,
                td_error: float,
                q_values: np.ndarray,
                v_state: float,
                q_std: float,
                t: int) -> np.ndarray:
        """Returns a numpy vector aligned with self.dims."""
        feats = {
            "suddenness":     self._suddenness(sid, action, sid_next),
            "goal_relevance": self._goal_relevance(td_error),
            "conduciveness":  self._conduciveness(td_error),
            "power":          self._power(q_values),
            "predictability": self._predictability(q_std),
            "anticipation":   self._anticipation(v_state),
            "urgency":        self._urgency(t),
            "familiarity":    self._familiarity(sid_next),
        }
        return np.array([feats[d] for d in self.dims], dtype=np.float32)
