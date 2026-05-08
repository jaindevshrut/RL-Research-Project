"""
QR-DQN agent (Method 2).

Implements the Double-DQN style action selection on top of the
quantile-regression value head, with the asymmetric quantile Huber
loss from Dabney et al. (AAAI 2018, equation 10).

Reads the same per-step appraisal vector as Method 1, with two
adjustments to the inputs:

    - q_values(s):  taken as the per-action MEAN over quantile atoms.
    - q_std(s):     the per-action STD over quantile atoms of the
                    GREEDY action's distribution. This replaces the
                    cross-head dispersion used in Method 1 with the
                    intra-distribution spread of the chosen return,
                    which is what QR-DQN exposes.
    - v_state(s):   max_a Q(s, a). QR-DQN does not have a dueling
                    V-head, so we use the standard scalar V_pi.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .networks import QRDQN
from .appraisal import AppraisalExtractor
from .replay_buffer import (
    Transition, UniformReplayBuffer, PrioritizedReplayBuffer,
)


def _to_tensor(x, device):
    return torch.as_tensor(x, device=device).unsqueeze(0)


def quantile_huber_loss(theta_pred: torch.Tensor,
                        target_atoms: torch.Tensor,
                        taus: torch.Tensor,
                        kappa: float = 1.0) -> torch.Tensor:
    """
    Asymmetric quantile Huber loss.

    Inputs
    ------
    theta_pred   : (B, N) quantile predictions for the chosen action
    target_atoms : (B, N) target atoms (no gradient)
    taus         : (N,)   quantile midpoints in (0, 1)

    The pairwise residual u_{ij} = target_atoms[:, j] - theta_pred[:, i]
    is penalised by the standard Huber loss (around |u| <= kappa) and
    weighted by |tau_i - I(u < 0)|. We then average over the j-axis
    (target atoms) and sum over the i-axis (predicted atoms), as in
    eq. (10) of Dabney et al. (2018).
    """
    # u: (B, N_pred, N_tgt)
    u = target_atoms.unsqueeze(1) - theta_pred.unsqueeze(2)
    # Huber piece
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa,
                        0.5 * u.pow(2),
                        kappa * (abs_u - 0.5 * kappa))
    # asymmetric weighting
    indicator = (u.detach() < 0).float()           # I(u < 0)
    tau_w = (taus.view(1, -1, 1) - indicator).abs()
    loss = (tau_w * huber).mean(dim=2).sum(dim=1)  # (B,)
    return loss


class QRDQNAgent:
    def __init__(self, cfg, obs_shape, n_actions: int):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.n_actions = n_actions
        self.n_quantiles = int(getattr(cfg.dqn, "n_quantiles", 51))

        self.online = QRDQN(
            obs_shape, n_actions,
            hidden_dim=cfg.dqn.hidden_dim,
            n_quantiles=self.n_quantiles,
        ).to(self.device)
        self.target = deepcopy(self.online).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.dqn.lr)

        if cfg.dqn.use_prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                cfg.dqn.buffer_size, alpha=cfg.dqn.per_alpha, beta=cfg.dqn.per_beta)
        else:
            self.buffer = UniformReplayBuffer(cfg.dqn.buffer_size)

        self.appraisal = AppraisalExtractor(
            use_dims=cfg.appraisal.use_dims,
            td_scale=cfg.appraisal.td_scale,
            value_scale=cfg.appraisal.value_scale,
            max_steps=cfg.env.max_steps,
        )

        self.frames = 0
        self.last_target_sync = 0

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def epsilon(self) -> float:
        c = self.cfg.dqn
        frac = min(1.0, self.frames / max(1, c.eps_decay_frames))
        return c.eps_start + (c.eps_end - c.eps_start) * frac

    @torch.no_grad()
    def act(self, obs):
        if np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        q = self.online.q_values(_to_tensor(obs, self.device))
        return int(q.argmax(dim=-1).item())

    # ------------------------------------------------------------------ #
    # Per-step appraisal — uses Q / V / quantile-spread from ONLINE net
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_appraisal(self, obs, sid, action, reward, next_obs,
                       next_sid, done, t) -> np.ndarray:
        s = _to_tensor(obs, self.device)
        s2 = _to_tensor(next_obs, self.device)

        z_s = self.online(s)                        # (1, A, N)
        q_s = z_s.mean(dim=-1)                      # (1, A)
        a_star_s = int(q_s.argmax(dim=-1).item())
        v_s = float(q_s.max(dim=-1).values.item())  # scalar V(s)
        # intrinsic uncertainty proxy: std over quantile atoms of the
        # greedy action's distribution
        q_std_s = float(z_s[0, a_star_s].std().item())
        q_sa = float(q_s[0, action].item())
        q_values = q_s.cpu().numpy()[0]

        # Bellman target — Double-DQN form (online picks a*, target evaluates)
        if done:
            target = float(reward)
        else:
            online_q2 = self.online.q_values(s2)
            a_star = int(online_q2.argmax(dim=-1).item())
            target_q2 = self.target.q_values(s2)
            target = float(reward) + float(self.cfg.dqn.gamma) * float(
                target_q2[0, a_star].item())

        td_error = target - q_sa

        self.appraisal.observe_transition(sid, action, next_sid)
        return self.appraisal.compute(
            sid=sid, action=action, sid_next=next_sid,
            td_error=td_error, q_values=q_values,
            v_state=v_s, q_std=q_std_s, t=t,
        )

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #
    def push_transition(self, tr: Transition):
        self.buffer.push(tr)

    def learn(self) -> dict:
        if len(self.buffer) < self.cfg.dqn.min_buffer:
            return {}

        batch, idx, weights = self.buffer.sample(self.cfg.dqn.batch_size)
        obs = torch.as_tensor(np.stack([b.obs for b in batch]), device=self.device)
        next_obs = torch.as_tensor(np.stack([b.next_obs for b in batch]),
                                   device=self.device)
        actions = torch.as_tensor([b.action for b in batch],
                                  device=self.device, dtype=torch.long)
        rewards = torch.as_tensor([b.reward for b in batch],
                                  device=self.device, dtype=torch.float32)
        dones = torch.as_tensor([float(b.done) for b in batch],
                                device=self.device, dtype=torch.float32)
        w = torch.as_tensor(weights, device=self.device)

        B = obs.size(0)
        N = self.n_quantiles
        # Predicted atoms for the executed action -> (B, N)
        z_all = self.online(obs)                            # (B, A, N)
        a_idx = actions.view(-1, 1, 1).expand(-1, 1, N)
        theta_pred = z_all.gather(1, a_idx).squeeze(1)      # (B, N)

        with torch.no_grad():
            # Double-DQN action selection on the target side
            online_q2 = self.online.q_values(next_obs)
            a_star = online_q2.argmax(dim=-1)               # (B,)
            z_target_all = self.target(next_obs)            # (B, A, N)
            a_idx2 = a_star.view(-1, 1, 1).expand(-1, 1, N)
            theta_next = z_target_all.gather(1, a_idx2).squeeze(1)  # (B, N)
            target_atoms = (rewards.unsqueeze(1)
                            + self.cfg.dqn.gamma
                            * (1.0 - dones.unsqueeze(1)) * theta_next)

        per_sample_loss = quantile_huber_loss(
            theta_pred, target_atoms, self.online.taus, kappa=1.0)  # (B,)
        loss = (w * per_sample_loss).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Priorities use mean predicted-atom error against the target mean
        with torch.no_grad():
            td_for_prio = (target_atoms.mean(dim=1)
                           - theta_pred.mean(dim=1)).abs().cpu().numpy()
        self.buffer.update_priorities(idx, td_for_prio)

        if self.frames - self.last_target_sync >= self.cfg.dqn.target_update_every:
            self.target.load_state_dict(self.online.state_dict())
            self.last_target_sync = self.frames

        return {"loss": float(loss.item()),
                "td_mean": float(td_for_prio.mean())}
