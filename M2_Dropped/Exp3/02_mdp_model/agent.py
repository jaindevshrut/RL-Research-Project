"""Agent implementations for baseline Q-learning and V2 distributional RL.

V2 uses Distributional RL:
Instead of scalar Q-values, we model a distribution Z(s, a).
Appraisals are computed from this distribution:
- Variance -> goal relevance
- Mean -> conduciveness
- Tail probability -> fear, mapped into the suddenness slot
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    HUBER_KAPPA,
    LOG_INTERVAL,
    MODEL_TYPE,
    NUM_QUANTILES,
    PER_EPSILON,
    SEED,
    USE_PER,
)


class agent():
    """Tabular baseline agent with an optional distributional RL upgrade."""

    def __init__(self, mdp):
        random.seed(SEED)
        np.random.seed(SEED)

        self.epsilon = 0.3
        self.gamma = 0.9
        self.alpha = 0.3
        self.mdp = mdp
        self.model_type = MODEL_TYPE
        self.use_distributional = self.model_type == "v2"

        self.q: Dict[str, Dict[str, float]] = {}
        self.Z: Dict[str, Dict[str, np.ndarray]] = {}
        self.t_hat: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.td_error = 0.0
        self.old_q = 0.0
        self.max_q_table = 0.0

        self.goal_app = 0.0
        self.cdc_app = 0.0
        self.power_app = 0.0
        self.sud_app = 0.0

        self.quantile_tau = (
            np.arange(NUM_QUANTILES, dtype=float) + 0.5
        ) / NUM_QUANTILES
        self.replay_buffer: List[Tuple[str, str, float, str, bool]] = []
        self.replay_priorities: List[float] = []
        self.replay_capacity = 512
        self.replay_interval = 5
        self.last_priority = PER_EPSILON

        self.episode_reward = 0.0
        self.episode_rewards: List[float] = []
        self.total_steps = 0

        for state in self.mdp.t.keys():
            self.q[state] = {}
            self.Z[state] = {}
            self.t_hat[state] = {}
            for action in self.mdp.t[state]:
                self._ensure_state_action(state, action)

    def _ensure_state_action(self, state: str, action: str) -> None:
        if state not in self.q:
            self.q[state] = {}
        if state not in self.Z:
            self.Z[state] = {}
        if state not in self.t_hat:
            self.t_hat[state] = {}

        if action not in self.q[state]:
            self.q[state][action] = 0.0
        if action not in self.Z[state]:
            self.Z[state][action] = np.zeros(NUM_QUANTILES, dtype=float)
        if action not in self.t_hat[state]:
            self.t_hat[state][action] = {
                next_state: 0 for next_state in self.mdp.t.keys()
            }

    def _available_actions(self, state: str) -> List[str]:
        return list(self.mdp.t.get(state, {}).keys())

    def get_mean_Q(self, state: str, action: str) -> float:
        if self.use_distributional:
            self._ensure_state_action(state, action)
            return float(np.mean(self.Z[state][action]))
        self._ensure_state_action(state, action)
        return float(self.q[state][action])

    def get_mean_q(self, state: str, action: str) -> float:
        return self.get_mean_Q(state, action)

    def _sync_q_value(self, state: str, action: str) -> None:
        self.q[state][action] = self.get_mean_Q(state, action)

    def _best_action(self, state: str) -> Optional[str]:
        actions = self._available_actions(state)
        if not actions:
            return None
        return max(actions, key=lambda action: self.get_mean_Q(state, action))

    def _max_next_q(self, state: str) -> float:
        best_action = self._best_action(state)
        if best_action is None:
            return 0.0
        return self.get_mean_Q(state, best_action)

    def quantile_loss(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return the QR-DQN style quantile Huber loss."""

        pred = pred.reshape(-1, 1)
        target = target.reshape(1, -1)
        delta = target - pred
        abs_delta = np.abs(delta)
        huber = np.where(
            abs_delta <= HUBER_KAPPA,
            0.5 * delta ** 2,
            HUBER_KAPPA * (abs_delta - 0.5 * HUBER_KAPPA),
        )
        weight = np.abs(
            self.quantile_tau.reshape(-1, 1) - (delta < 0).astype(float)
        )
        loss = weight * huber / HUBER_KAPPA
        return float(np.mean(loss)), delta, weight

    def _distributional_target(
        self,
        reward: float,
        next_state: str,
        terminal: bool,
    ) -> np.ndarray:
        if terminal:
            return np.full(NUM_QUANTILES, reward, dtype=float)

        next_action = self._best_action(next_state)
        if next_action is None:
            return np.full(NUM_QUANTILES, reward, dtype=float)

        self._ensure_state_action(next_state, next_action)
        z_next = self.Z[next_state][next_action]
        return reward + self.gamma * z_next

    def _apply_scalar_update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        terminal: bool,
        increment_transition: bool = True,
    ) -> None:
        self._ensure_state_action(state, action)
        previous_q = self.q[state][action]
        self.old_q = previous_q

        if terminal:
            target = reward
        else:
            target = reward + self.gamma * self._max_next_q(next_state)

        self.td_error = self.alpha * (target - previous_q)
        self.q[state][action] = previous_q + self.td_error
        self.last_priority = abs(target - previous_q) + PER_EPSILON

        if increment_transition:
            self.t_hat[state][action][next_state] += 1

    def _apply_distribution_update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        terminal: bool,
        increment_transition: bool = True,
    ) -> None:
        self._ensure_state_action(state, action)
        pred = self.Z[state][action].copy()
        target = self._distributional_target(reward, next_state, terminal)
        loss, delta, weight = self.quantile_loss(pred, target)
        clipped_delta = np.clip(delta, -HUBER_KAPPA, HUBER_KAPPA)
        update = np.mean(weight * clipped_delta, axis=1)

        self.old_q = float(np.mean(pred))
        self.td_error = self.alpha * (float(np.mean(target)) - self.old_q)
        self.Z[state][action] = pred + self.alpha * update
        self.q[state][action] = float(np.mean(self.Z[state][action]))
        self.last_priority = (
            float(np.mean(np.abs(target - self.Z[state][action]))) + PER_EPSILON
        )

        if increment_transition:
            self.t_hat[state][action][next_state] += 1

        self.quantile_loss_value = loss

    def _compute_transition_priority(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        terminal: bool,
    ) -> float:
        self._ensure_state_action(state, action)
        if self.use_distributional:
            target = self._distributional_target(reward, next_state, terminal)
            pred = self.Z[state][action]
            return float(np.mean(np.abs(target - pred)) + PER_EPSILON)

        pred = self.q[state][action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * self._max_next_q(next_state)
        return float(abs(target - pred) + PER_EPSILON)

    def _store_transition(
        self,
        state: Optional[str],
        action: Optional[str],
        reward: float,
        next_state: str,
        terminal: bool,
    ) -> None:
        if not self.use_distributional or not USE_PER:
            return
        if state is None or action is None:
            return

        priority = self._compute_transition_priority(
            state,
            action,
            reward,
            next_state,
            terminal,
        )
        if len(self.replay_buffer) >= self.replay_capacity:
            self.replay_buffer.pop(0)
            self.replay_priorities.pop(0)

        self.replay_buffer.append((state, action, reward, next_state, terminal))
        self.replay_priorities.append(priority)

    def _replay_update(self) -> None:
        if not self.use_distributional or not USE_PER or not self.replay_buffer:
            return
        if self.total_steps % self.replay_interval != 0:
            return

        priorities = np.asarray(self.replay_priorities, dtype=float)
        total_priority = float(np.sum(priorities))
        if total_priority <= 0:
            replay_index = np.random.randint(len(self.replay_buffer))
        else:
            replay_index = int(
                np.random.choice(len(self.replay_buffer), p=priorities / total_priority)
            )

        state, action, reward, next_state, terminal = self.replay_buffer[replay_index]
        self._apply_distribution_update(
            state,
            action,
            reward,
            next_state,
            terminal,
            increment_transition=False,
        )
        self.replay_priorities[replay_index] = self._compute_transition_priority(
            state,
            action,
            reward,
            next_state,
            terminal,
        )

    def update_q_learning(self):
        if self.mdp.previous_state is None or self.mdp.action is None:
            return

        if self.use_distributional:
            self._apply_distribution_update(
                self.mdp.previous_state,
                self.mdp.action,
                self.mdp.reward,
                self.mdp.state,
                terminal=False,
            )
        else:
            self._apply_scalar_update(
                self.mdp.previous_state,
                self.mdp.action,
                self.mdp.reward,
                self.mdp.state,
                terminal=False,
            )

    def get_td_error(self):
        if self.mdp.previous_state is not None:
            self.mdp.tde.append(self.td_error)

    def update_q_td(self):
        if self.mdp.previous_state is None or self.mdp.action is None:
            return

        if self.use_distributional:
            self._apply_distribution_update(
                self.mdp.previous_state,
                self.mdp.action,
                self.mdp.reward,
                self.mdp.state,
                terminal=True,
            )
        else:
            self._apply_scalar_update(
                self.mdp.previous_state,
                self.mdp.action,
                self.mdp.reward,
                self.mdp.state,
                terminal=True,
            )

    def get_max_q_table(self):
        values = [
            max(action_values.values())
            for action_values in self.q.values()
            if action_values
        ]
        max_q_table = max(values) if values else 0.0
        if max_q_table == 0:
            max_q_table += 1
        self.max_q_table = max_q_table
        return max_q_table

    def choose_action_epsilon_greedy(self):
        self.mdp.previous_action = self.mdp.action

        if (
            (self.mdp.story_m or self.mdp.model_changed)
            and self.mdp.state == self.mdp.chosen_state
        ):
            self.mdp.action = self.mdp.chosen_action
            return

        if random.random() < self.epsilon:
            actions = self._available_actions(self.mdp.state)
            self.mdp.action = random.choice(actions)
            return

        best_action = self._best_action(self.mdp.state)
        self.mdp.action = best_action

    def do_step(self):
        self.update_q_learning()
        self.get_td_error()
        self.choose_action_epsilon_greedy()
        self.mdp.transition()
        self.mdp.calculate_reward()
        self.episode_reward += self.mdp.reward

        if self.mdp.terminal:
            self.update_q_td()

        self._store_transition(
            self.mdp.previous_state,
            self.mdp.action,
            self.mdp.reward,
            self.mdp.state,
            self.mdp.terminal,
        )
        self.total_steps += 1
        self._replay_update()

    def _mean_value(self) -> float:
        values = [
            self.get_mean_Q(state, action)
            for state in self.q
            for action in self.q[state]
        ]
        return float(np.mean(values)) if values else 0.0

    def _mean_variance(self) -> float:
        if not self.use_distributional:
            return 0.0
        variances = [
            float(np.var(self.Z[state][action]))
            for state in self.Z
            for action in self.Z[state]
        ]
        return float(np.mean(variances)) if variances else 0.0

    def _log_episode(self, episode_index: int, total_episodes: int) -> None:
        if episode_index % LOG_INTERVAL != 0 and episode_index != total_episodes:
            return

        if self.use_distributional:
            print(
                f"[v2] Episode {episode_index}/{total_episodes} "
                f"reward={self.episode_reward:.3f} "
                f"mean(Z)={self._mean_value():.3f} "
                f"var(Z)={self._mean_variance():.3f}"
            )
            return

        print(
            f"[baseline] Episode {episode_index}/{total_episodes} "
            f"reward={self.episode_reward:.3f} "
            f"mean(Q)={self._mean_value():.3f}"
        )

    def train(self, i_max, i_change=0):
        i = 0
        while i < i_max:
            if i == i_max - i_change and not self.mdp.model_changed:
                self.mdp.make_transition(story_mode=False, model_changed=True)
            self.do_step()

            if self.mdp.terminal:
                i += 1
                self.episode_rewards.append(self.episode_reward)
                self._log_episode(i, i_max)
                self.episode_reward = 0.0
                self.mdp.reset()

    def _get_appraisal_context(self) -> Tuple[str, Optional[str], List[str]]:
        state = self.mdp.chosen_state or self.mdp.state
        actions = self._available_actions(state)
        if self.mdp.chosen_action in actions:
            action = self.mdp.chosen_action
        elif self.mdp.action in actions:
            action = self.mdp.action
        else:
            action = actions[0] if actions else None
        return state, action, actions

    def compute_appraisal_v2(self, state: str, action: str):
        z_values = self.Z[state][action]
        goal_relevance = float(np.var(z_values))
        conduciveness = float(np.mean(z_values))

        _, _, actions = self._get_appraisal_context()
        action_means = [
            float(np.mean(self.Z[state][candidate_action]))
            for candidate_action in actions
        ]
        power = float(np.var(action_means)) if action_means else 0.0
        fear = float(np.mean(z_values < 0))
        return goal_relevance, conduciveness, power, fear

    def appraise_power(self):
        if self.use_distributional:
            state, action, _ = self._get_appraisal_context()
            if action is None:
                self.power_app = 0.0
            else:
                _, _, self.power_app, _ = self.compute_appraisal_v2(state, action)
            return self.power_app

        state = self.mdp.chosen_state
        q_values = list(self.q[state].values())
        avg_q = sum(q_values) / len(q_values)
        min_q = min(q_values)
        max_q = max(q_values)
        if min_q == 0 and max_q == 0:
            self.power_app = 0.0
        elif abs(min_q) < max_q and max_q != 0:
            self.power_app = abs((max_q - avg_q) / max_q)
        elif min_q != 0:
            self.power_app = abs((min_q - avg_q) / min_q)
        else:
            self.power_app = 0.0
        return self.power_app

    def appraise_goal_relevance(self):
        if self.use_distributional:
            state, action, _ = self._get_appraisal_context()
            if action is None:
                self.goal_app = 0.0
            else:
                self.goal_app, _, _, _ = self.compute_appraisal_v2(state, action)
            return self.goal_app

        self.goal_app = min(1, abs(self.td_error))
        return self.goal_app

    def appraise_suddenness(self):
        if self.use_distributional:
            state, action, _ = self._get_appraisal_context()
            if action is None:
                self.sud_app = 0.0
            else:
                _, _, _, fear = self.compute_appraisal_v2(state, action)
                self.sud_app = fear
            return self.sud_app

        s = sum(self.t_hat[self.mdp.previous_state][self.mdp.previous_action].values())
        if s > 0:
            self.sud_app = (
                1
                - self.t_hat[self.mdp.previous_state][self.mdp.previous_action][
                    self.mdp.state
                ]
                / s
            )
        else:
            self.sud_app = 0
        return self.sud_app

    def appraise_conduciveness(self):
        if self.use_distributional:
            state, action, _ = self._get_appraisal_context()
            if action is None:
                self.cdc_app = 0.0
            else:
                _, self.cdc_app, _, _ = self.compute_appraisal_v2(state, action)
            return self.cdc_app

        self.cdc_app = max(-1, min(1, self.td_error)) / 2 + 0.5
        return self.cdc_app

    def _update_appraisal_cache(self) -> None:
        self.sud_app = self.appraise_suddenness()
        self.goal_app = self.appraise_goal_relevance()
        self.cdc_app = self.appraise_conduciveness()
        self.power_app = self.appraise_power()

    def simulate_episode(self, terminate=None):
        self.mdp.make_transition(story_mode=True)
        self.mdp.reset()

        while True:
            self.do_step()
            if self.mdp.terminal:
                return

            if terminate == self.mdp.state:
                self.update_q_learning()
                self.get_td_error()
                self.choose_action_epsilon_greedy()
                self.get_max_q_table()
                self._update_appraisal_cache()

                rounded_tde = [round(num, 3) for num in self.mdp.tde]
                print("TDE list:\t", rounded_tde)
                print("Manual terminate")
                print("Suddenness:\t", round(self.sud_app, 4))
                print("Goal relevance:\t", round(self.goal_app, 4))
                print("Conduciveness:\t", round(self.cdc_app, 4))
                print("Power:\t\t", round(self.power_app, 4))
                return
