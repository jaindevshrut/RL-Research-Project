"""
Dueling Double DQN with a bootstrap ensemble of K heads.

Why this architecture (key talking points for the report):

(A) DUELING split:
    Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )
    V(s) is exposed directly -> feeds the "anticipation" appraisal.
    A(s,a) - mean_a A(s,a)  -> action-effect range -> "power" appraisal.
    Without dueling, V(s) would be implicit and we'd have to estimate it.

(B) DOUBLE DQN:
    target = r + gamma * Q_target( s', argmax_a Q_online(s', a) )
    Reduces over-estimation -> a cleaner, less biased TD-error.
    Cleaner TD-error means our "conduciveness" and "goal_relevance"
    appraisals are not contaminated by Jensen-style bias.

(C) ENSEMBLE OF K HEADS:
    Each head is a randomly initialised copy of the value head.
    Disagreement across heads -> epistemic uncertainty estimate.
    -> directly feeds the new "predictability" appraisal,
       a dimension that did not exist in the original 4-dim setup.

The shared trunk keeps it cheap; only the small heads are duplicated.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class DuelingHead(nn.Module):
    """One V-stream and one A-stream — combined into Q via the dueling identity."""

    def __init__(self, in_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.value = _mlp(in_dim, hidden, 1)
        self.adv = _mlp(in_dim, hidden, n_actions)

    def forward(self, x: torch.Tensor):
        v = self.value(x)                           # (B, 1)
        a = self.adv(x)                             # (B, A)
        a_centered = a - a.mean(dim=-1, keepdim=True)
        q = v + a_centered                          # (B, A)
        return q, v.squeeze(-1), a_centered


class DuelingEnsembleDQN(nn.Module):
    """
    Trunk -> K dueling heads.

    Outputs:
        q_all      : (B, K, A)   - per-head Q values
        v_all      : (B, K)      - per-head V(s)
        a_all      : (B, K, A)   - per-head centered advantages
    The mean over K is used for action selection / Bellman target.
    The std over K is used for the predictability appraisal.
    """

    def __init__(self, obs_shape, n_actions: int, hidden_dim: int = 128,
                 n_ensemble_heads: int = 3):
        super().__init__()
        h, w, c = obs_shape
        self.n_actions = n_actions
        self.k = n_ensemble_heads

        # Small CNN trunk — input is (B, C, H, W)
        self.trunk = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat = 32 * h * w
        self.feature_proj = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            DuelingHead(hidden_dim, hidden_dim, n_actions)
            for _ in range(self.k)
        ])

    # ------------------------------------------------------------------ #
    # Forward variants
    # ------------------------------------------------------------------ #
    def features(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, H, W, C) uint8/float -> (B, C, H, W) float
        if obs.dtype != torch.float32:
            obs = obs.float()
        x = obs.permute(0, 3, 1, 2)
        x = self.trunk(x)
        x = self.feature_proj(x)
        return x

    def forward(self, obs: torch.Tensor):
        x = self.features(obs)
        qs, vs, as_ = [], [], []
        for head in self.heads:
            q, v, a = head(x)
            qs.append(q); vs.append(v); as_.append(a)
        q_all = torch.stack(qs, dim=1)              # (B, K, A)
        v_all = torch.stack(vs, dim=1)              # (B, K)
        a_all = torch.stack(as_, dim=1)             # (B, K, A)
        return q_all, v_all, a_all

    def q_mean(self, obs: torch.Tensor) -> torch.Tensor:
        q_all, _, _ = self.forward(obs)
        return q_all.mean(dim=1)                    # (B, A)

    def q_std(self, obs: torch.Tensor) -> torch.Tensor:
        q_all, _, _ = self.forward(obs)
        # std across heads, averaged across actions -> scalar uncertainty
        return q_all.std(dim=1).mean(dim=-1)        # (B,)
