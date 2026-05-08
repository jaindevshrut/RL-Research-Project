"""
Replay buffer with optional prioritisation (proportional PER).

PER is useful here because our appraisal signals depend on TD error.
Prioritising surprising transitions:
  * speeds up learning (well-known PER benefit), and
  * makes our extracted appraisals reflect the events that *actually*
    matter for the policy — which is the whole point of mapping CPM
    onto RL signals.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    sid: tuple
    next_sid: tuple
    t: int


class UniformReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self): return len(self.buf)

    def push(self, tr: Transition):
        self.buf.append(tr)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, idx, weights

    def update_priorities(self, idx, td_errors): pass


class PrioritizedReplayBuffer:
    """
    Proportional PER with sum-tree-free implementation
    (we just keep priorities in a numpy array; for buffers up to ~10^5
    transitions this is fast enough and far simpler than a SumTree).
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buf: list = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def __len__(self): return len(self.buf)

    def push(self, tr: Transition):
        if len(self.buf) < self.capacity:
            self.buf.append(tr)
        else:
            self.buf[self.pos] = tr
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.buf)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / (prios.sum() + 1e-8)
        idx = np.random.choice(n, size=batch_size, p=probs)
        batch = [self.buf[i] for i in idx]
        # importance-sampling weights (unbias the gradient)
        weights = (n * probs[idx]) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        return batch, idx, weights

    def update_priorities(self, idx, td_errors):
        for i, td in zip(idx, td_errors):
            p = float(abs(td)) + 1e-6
            self.priorities[i] = p
            if p > self.max_priority:
                self.max_priority = p
