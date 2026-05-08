"""
Quantile Regression DQN (Method 2).

Implements the algorithm of Dabney, Rowland, Bellemare and Munos
("Distributional Reinforcement Learning with Quantile Regression",
AAAI 2018). The action-value distribution Z(s, a) is represented by
N atoms placed at quantile midpoints tau_i = (i - 0.5) / N for
i = 1, ..., N, and learned via the asymmetric quantile Huber loss.

Why we tried this as a SECOND method (kept as a comparison only):
    - QR-DQN exposes a *full* return distribution rather than a
      single scalar Q(s, a). The width of that distribution is a
      candidate signal for the "predictability" appraisal that does
      not depend on a bootstrap ensemble.
    - It tests whether distributional information alone -- without
      the dueling V-head and without the K-head ensemble -- is
      enough to drive the eight-dimensional appraisal vector.

In our experiments QR-DQN underperforms the Dueling-Ensemble DQN
of Method 1 on this grid world; we report its numbers honestly in
the master conclusion table so readers can see the comparison.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class QRDQN(nn.Module):
    """
    Trunk -> head producing N quantile atoms per action.

    Output layout:
        forward(obs) -> theta of shape (B, A, N)
                        theta[b, a, i] is the i-th quantile estimate of
                        the return distribution Z(s_b, a).
        q_values(obs) -> mean over atoms, shape (B, A).
    """

    def __init__(self, obs_shape, n_actions: int, hidden_dim: int = 128,
                 n_quantiles: int = 51):
        super().__init__()
        h, w, c = obs_shape
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        # Same trunk shape as Method 1 so the comparison isolates the
        # value-head representation rather than the feature extractor.
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
        self.head = _mlp(hidden_dim, hidden_dim, n_actions * n_quantiles)

        # Quantile midpoints tau_i = (i - 0.5) / N, registered as a buffer
        # so it follows the module to its device automatically.
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("taus", taus)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype != torch.float32:
            obs = obs.float()
        x = obs.permute(0, 3, 1, 2)
        x = self.trunk(x)
        x = self.feature_proj(x)
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.features(obs)
        z = self.head(x)
        return z.view(-1, self.n_actions, self.n_quantiles)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).mean(dim=-1)
