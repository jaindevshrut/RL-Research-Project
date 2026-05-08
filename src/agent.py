"""
Dueling Double DQN agent that *also* emits an appraisal vector at every step.

This module is intentionally short — almost all the design decisions
live in networks/dueling_dqn.py and appraisal/extractor.py. Here we only
wire them together with the standard DQN training rules.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .networks import DuelingEnsembleDQN
from .appraisal import AppraisalExtractor
from .replay_buffer import (
    Transition, UniformReplayBuffer, PrioritizedReplayBuffer,
)


def _to_tensor(x, device):
    return torch.as_tensor(x, device=device).unsqueeze(0)


class DQNAgent:
    def __init__(self, cfg, obs_shape, n_actions: int):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.n_actions = n_actions

        self.online = DuelingEnsembleDQN(
            obs_shape, n_actions,
            hidden_dim=cfg.dqn.hidden_dim,
            n_ensemble_heads=cfg.dqn.n_ensemble_heads,
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
        q = self.online.q_mean(_to_tensor(obs, self.device))
        return int(q.argmax(dim=-1).item())

    # ------------------------------------------------------------------ #
    # Per-step appraisal — uses Q/V/std from the ONLINE net
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_appraisal(self, obs, sid, action, reward, next_obs,
                       next_sid, done, t) -> np.ndarray:
        # Compute TD error for THIS specific transition (Double-DQN form),
        # purely for the appraisal signal — not for the gradient.
        s = _to_tensor(obs, self.device)
        s2 = _to_tensor(next_obs, self.device)

        q_all_s, v_all_s, _ = self.online(s)
        q_s = q_all_s.mean(dim=1)                    # (1, A)
        v_s = v_all_s.mean(dim=1).item()             # scalar
        q_std_s = q_all_s.std(dim=1).mean().item()   # scalar
        q_sa = q_s[0, action].item()
        q_values = q_s.cpu().numpy()[0]

        # bootstrap target
        if done:
            target = float(reward)
        else:
            online_q2 = self.online.q_mean(s2)
            a_star = int(online_q2.argmax(dim=-1).item())
            target_q2 = self.target.q_mean(s2)
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

        # online Q(s, a) -- we use the per-head mean for the loss target,
        # but we keep heads separate so each head is trained.
        q_all, _, _ = self.online(obs)               # (B, K, A)
        q_taken = q_all.gather(2, actions.view(-1, 1, 1).expand(-1, q_all.size(1), 1)).squeeze(-1)
        # q_taken: (B, K)

        with torch.no_grad():
            # Double DQN: action chosen by online, evaluated by target
            online_q2 = self.online.q_mean(next_obs)
            a_star = online_q2.argmax(dim=-1)         # (B,)
            target_q2_all, _, _ = self.target(next_obs)   # (B, K, A)
            target_q2 = target_q2_all.gather(2, a_star.view(-1, 1, 1).expand(-1, target_q2_all.size(1), 1)).squeeze(-1)
            # (B, K)
            target = rewards.unsqueeze(1) + self.cfg.dqn.gamma * (1.0 - dones.unsqueeze(1)) * target_q2

        td = target - q_taken                         # (B, K)
        # Huber loss, weighted by IS-weights
        loss = (w.unsqueeze(1) * F.smooth_l1_loss(q_taken, target, reduction="none")).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # priorities use mean |TD| across heads
        td_for_prio = td.detach().abs().mean(dim=1).cpu().numpy()
        self.buffer.update_priorities(idx, td_for_prio)

        # Sync target net
        if self.frames - self.last_target_sync >= self.cfg.dqn.target_update_every:
            self.target.load_state_dict(self.online.state_dict())
            self.last_target_sync = self.frames

        return {"loss": float(loss.item()),
                "td_mean": float(td.detach().abs().mean().item())}
