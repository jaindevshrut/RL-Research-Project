"""
Self-contained MiniGrid-style task — no external env dependency.

Design is deliberately minimal so every appraisal signal is traceable
by hand: the agent walks a grid, picks up a key, reaches a goal, and
loses if it steps into lava. The transition function is fully observable,
which lets us compute count-based novelty cleanly.

Why a custom env (not Gym/MiniGrid):
- Zero install pain.
- Transparent transitions: makes the math behind suddenness / familiarity
  inspectable in a debugger.
- Small state space keeps Q-tables / count tables tractable for analysis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# Cell tokens (ints — observation is a small image of these tokens)
EMPTY = 0
WALL = 1
LAVA = 2
KEY = 3
GOAL = 4
AGENT = 5

# Actions
LEFT, RIGHT, UP, DOWN, PICKUP = 0, 1, 2, 3, 4
N_ACTIONS = 5


@dataclass
class StepResult:
    obs: np.ndarray            # (grid_size, grid_size) uint8
    reward: float
    done: bool
    info: dict


class GridWorld:
    """
    Layout: agent starts top-left; goal at bottom-right.
    The episode ends successfully when the agent reaches GOAL holding a key.
    Lava ends the episode with a large negative reward.

    Reward structure is deliberately sparse-ish so that TD errors are
    informative (i.e., they spike on key events, which is exactly what
    we want appraisal signals to capture).
    """

    def __init__(self, grid_size: int = 7, max_steps: int = 80,
                 n_lava: int = 3, n_keys: int = 1, seed: int = 0):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_lava = n_lava
        self.n_keys = n_keys
        self.rng = np.random.default_rng(seed)
        self.action_space_n = N_ACTIONS
        # observation = grid (HxW) + has_key flag (broadcast as channel)
        self.obs_shape = (grid_size, grid_size, 2)
        self._build_layout()

    # ------------------------------------------------------------------ #
    # Layout / reset
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        gs = self.grid_size
        self.base_grid = np.full((gs, gs), EMPTY, dtype=np.uint8)
        # walls on the border
        self.base_grid[0, :] = WALL
        self.base_grid[-1, :] = WALL
        self.base_grid[:, 0] = WALL
        self.base_grid[:, -1] = WALL

        # goal
        self.goal_pos = (gs - 2, gs - 2)
        self.base_grid[self.goal_pos] = GOAL

        # lava — random interior cells, avoid corners + goal
        interior = [(r, c) for r in range(1, gs - 1) for c in range(1, gs - 1)]
        forbidden = {self.goal_pos, (1, 1)}
        candidates = [p for p in interior if p not in forbidden]
        idx = self.rng.choice(len(candidates), size=self.n_lava, replace=False)
        self.lava_positions = [candidates[i] for i in idx]
        for p in self.lava_positions:
            self.base_grid[p] = LAVA

        # keys
        cand2 = [p for p in candidates if self.base_grid[p] == EMPTY]
        idx = self.rng.choice(len(cand2), size=self.n_keys, replace=False)
        self.key_positions = [cand2[i] for i in idx]
        for p in self.key_positions:
            self.base_grid[p] = KEY

    def reset(self) -> np.ndarray:
        self.grid = self.base_grid.copy()
        self.agent_pos = (1, 1)
        self.has_key = False
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        g = self.grid.copy()
        # render agent on top
        g[self.agent_pos] = AGENT
        key_channel = np.full_like(g, int(self.has_key), dtype=np.uint8)
        return np.stack([g, key_channel], axis=-1)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> StepResult:
        self.t += 1
        r, c = self.agent_pos
        if action == LEFT:    nr, nc = r, c - 1
        elif action == RIGHT: nr, nc = r, c + 1
        elif action == UP:    nr, nc = r - 1, c
        elif action == DOWN:  nr, nc = r + 1, c
        else:                 nr, nc = r, c   # PICKUP doesn't move

        cell = self.grid[nr, nc] if action != PICKUP else self.grid[r, c]

        reward, done = -0.01, False  # small step cost -> urgency matters

        if action == PICKUP:
            if self.grid[r, c] == KEY:
                self.has_key = True
                self.grid[r, c] = EMPTY
                reward += 0.2  # mid-size positive shock
        else:
            if cell == WALL:
                # bumping a wall: stay put, mild penalty
                reward -= 0.02
            elif cell == LAVA:
                self.agent_pos = (nr, nc)
                reward -= 1.0
                done = True
            else:
                self.agent_pos = (nr, nc)
                if cell == GOAL:
                    if self.has_key:
                        reward += 1.0
                    else:
                        reward += 0.1   # partial reward — encourages finding key
                    done = True

        if self.t >= self.max_steps:
            done = True

        return StepResult(self._obs(), float(reward), done, {
            "has_key": self.has_key,
            "agent_pos": self.agent_pos,
            "t": self.t,
        })

    # ------------------------------------------------------------------ #
    # State id — used for count-based novelty / familiarity
    # ------------------------------------------------------------------ #
    def state_id(self) -> Tuple[int, int, int]:
        r, c = self.agent_pos
        return (r, c, int(self.has_key))
