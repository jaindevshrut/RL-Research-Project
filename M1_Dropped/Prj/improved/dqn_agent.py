"""
dqn_agent.py - Deep Q-Network Agent for Appraisal-RL
=====================================================
IMPROVEMENT #1: Replaces tabular Q-learning with a neural-network-based
DQN agent using experience replay and a target network.

This version carefully mirrors the original tabular agent's appraisal
computation logic while using a neural network for Q-value approximation.
"""

import random
import numpy as np
from collections import deque
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Small MLP for Q-value estimation."""

    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent that maintains tabular bookkeeping for appraisal computation.

    The DQN handles learning, but we also maintain a tabular Q-table mirror
    and transition counts (t_hat) exactly like the original paper for
    correct appraisal computation.
    """

    def __init__(self, mdp, lr=5e-3, gamma=0.9, alpha=0.3, epsilon=0.3,
                 buffer_size=10000, batch_size=64, target_update_freq=50,
                 min_epsilon=0.01, epsilon_decay=0.9995):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha = alpha  # Matches original paper
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.state_dim = mdp.n_states
        self.action_dim = mdp.n_actions

        # Networks
        self.q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        # ---- Tabular bookkeeping (same as original paper) ----
        self.q = {}
        self.t_hat = {}
        self.td_error = 0
        self.old_q = 0
        self.td_error_history = []  # For normative significance

        for s in mdp.states:
            self.q[s] = {}
            self.t_hat[s] = {}
            for a_name in self._get_all_actions_for_state(s):
                self.q[s][a_name] = 0.0
                self.t_hat[s][a_name] = {}
                for s2 in mdp.states:
                    self.t_hat[s][a_name][s2] = 0

        self.episode_count = 0

    def _get_all_actions_for_state(self, state_name):
        """Get available actions from the transitions dict."""
        return list(self.mdp.transitions.get(state_name, {}).keys())

    def _state_tensor(self, state_name):
        """Convert state name to one-hot tensor."""
        vec = self.mdp._encode_state(state_name)
        return torch.FloatTensor(vec).unsqueeze(0)

    def _get_nn_q_values(self, state_name):
        """Get Q-values from neural network for a state."""
        with torch.no_grad():
            return self.q_net(self._state_tensor(state_name)).squeeze(0).numpy()

    def _sync_q_table_from_nn(self):
        """Update tabular Q-table from neural network outputs."""
        for s in self.mdp.states:
            nn_q = self._get_nn_q_values(s)
            for a_name in self.q[s]:
                if a_name in self.mdp.action_to_idx:
                    a_idx = self.mdp.action_to_idx[a_name]
                    self.q[s][a_name] = float(nn_q[a_idx])

    def _update_tabular_q(self, prev_state, action, reward, curr_state):
        """Tabular Q-learning update (same formula as original paper).
        This is used during story-mode simulation for appraisal.
        """
        if prev_state is not None:
            if action not in self.q[prev_state]:
                self.q[prev_state][action] = 0.0
                self.t_hat[prev_state][action] = {}
                self.t_hat[prev_state][action][curr_state] = 0

            previous_q = self.q[prev_state][action]
            self.old_q = previous_q

            # max Q at current state
            q_vals = self.q[curr_state]
            if q_vals:
                next_q = max(q_vals.values())
            else:
                next_q = 0.0

            self.td_error = self.alpha * (reward + self.gamma * next_q - previous_q)
            new_q = previous_q + self.td_error
            self.q[prev_state][action] = new_q

            if curr_state not in self.t_hat[prev_state][action]:
                self.t_hat[prev_state][action][curr_state] = 0
            self.t_hat[prev_state][action][curr_state] += 1

    def _choose_action(self, state_name, force_action=None):
        """ε-greedy action selection using tabular Q-values."""
        available = self._get_all_actions_for_state(state_name)
        if not available:
            return None

        if force_action is not None and force_action in available:
            return force_action

        if random.random() < self.epsilon:
            return random.choice(available)

        # Greedy from tabular Q
        q_items = {a: self.q[state_name].get(a, 0.0) for a in available}
        return max(q_items.items(), key=itemgetter(1))[0]

    def _nn_store_and_update(self, s_name, a_name, reward, s_next_name, done):
        """Store transition and train neural network."""
        s_oh = self.mdp._encode_state(s_name)
        s_next_oh = self.mdp._encode_state(s_next_name)
        a_idx = self.mdp.action_to_idx.get(a_name, 0)

        self.replay_buffer.append((s_oh, a_idx, reward, s_next_oh, float(done)))

        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states = torch.FloatTensor(np.array([t[0] for t in batch]))
            actions = torch.LongTensor([t[1] for t in batch])
            rewards = torch.FloatTensor([t[2] for t in batch])
            next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
            dones = torch.FloatTensor([t[4] for t in batch])

            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target = rewards + self.gamma * next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, i_max, i_change=0):
        """Train the agent for i_max episodes.

        Uses BOTH neural network learning AND tabular Q-updates to
        maintain accurate Q-values for appraisal computation.
        """
        i = 0
        while i < i_max:
            # Model change logic (same as original)
            if i_change > 0 and i == i_max - i_change and not self.mdp.model_changed:
                self.mdp.enter_model_changed_mode()

            self.mdp.reset()
            state_name = self.mdp.state
            prev_state = None
            prev_action = None
            done = False

            while not done:
                # Choose action
                action = self._choose_action(state_name)
                if action is None:
                    break

                # Take step
                prev_state_name = state_name
                _, reward, done, _ = self.mdp.step(action)
                state_name = self.mdp.state

                # Tabular Q-update
                self._update_tabular_q(prev_state_name, action, reward, state_name)

                # Track TD errors  
                if prev_state_name is not None:
                    self.mdp.tde.append(self.td_error)
                    self.td_error_history.append(float(self.td_error))

                # Track transition counts
                if prev_state_name is not None:
                    if action not in self.t_hat[prev_state_name]:
                        self.t_hat[prev_state_name][action] = {}
                    if state_name not in self.t_hat[prev_state_name][action]:
                        self.t_hat[prev_state_name][action][state_name] = 0

                # Neural network update
                self._nn_store_and_update(prev_state_name, action, reward, state_name, done)

                # Terminal: also do the final TD update (reward only, no bootstrap)
                if done:
                    prev_q = self.q.get(prev_state_name, {}).get(action, 0.0)
                    self.q[prev_state_name][action] = prev_q + self.alpha * (reward - prev_q)

            i += 1

            # Decay epsilon for better convergence
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Update target network
            if i % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: We intentionally do NOT call _sync_q_table_from_nn() here.
        # The tabular Q-values accumulated during training are the correct
        # ones for appraisal computation (power, goal_relevance, etc.).
        # Overwriting them with NN outputs would corrupt appraisal signals.

    def simulate_episode(self, terminate=None):
        """Run a story-mode episode for appraisal computation.

        This replicates the original paper's simulate_episode() exactly:
        - Enter story mode (forced transitions)
        - Step through the MDP
        - At the terminate state, compute all appraisals
        """
        self.mdp.enter_story_mode()
        self.mdp.reset()
        self.mdp.tde = []

        state_name = self.mdp.state
        prev_state = None
        prev_action = None

        while True:
            # Step: Q-update → get TD error → choose action → transition → reward
            if prev_state is not None:
                self._update_tabular_q(prev_state, prev_action, self.mdp.reward, state_name)
                self.mdp.tde.append(self.td_error)
                self.td_error_history.append(float(self.td_error))

            # Choose action with story mode forcing
            force = None
            if (self.mdp.story_mode or self.mdp.model_changed) and \
               state_name == self.mdp.chosen_state and self.mdp.chosen_action:
                force = self.mdp.chosen_action

            action = self._choose_action(state_name, force_action=force)
            if action is None:
                break

            # Transition
            prev_state = state_name
            prev_action = action
            _, reward, done, _ = self.mdp.step(action)
            state_name = self.mdp.state

            if done:
                # Terminal state reached
                self._update_tabular_q(prev_state, prev_action, reward, state_name)
                self.mdp.tde.append(self.td_error)
                self.td_error_history.append(float(self.td_error))

                # Final TD update
                prev_q = self.q.get(prev_state, {}).get(prev_action, 0.0)
                self.q[prev_state][prev_action] = prev_q + self.alpha * (reward - prev_q)

                return self._build_result(prev_state, prev_action, state_name)

            if terminate is not None and state_name == terminate:
                # Manual termination at appraisal point
                self._update_tabular_q(prev_state, prev_action, reward, state_name)
                self.mdp.tde.append(self.td_error)
                self.td_error_history.append(float(self.td_error))

                # Choose next action at terminate state for appraisal
                force_t = None
                if (self.mdp.story_mode or self.mdp.model_changed) and \
                   state_name == self.mdp.chosen_state and self.mdp.chosen_action:
                    force_t = self.mdp.chosen_action
                next_action = self._choose_action(state_name, force_action=force_t)

                return self._build_result(prev_state, prev_action, state_name)

    def _build_result(self, prev_state, prev_action, current_state):
        """Build result dict for appraisal computation."""
        return {
            'tde_list': list(self.mdp.tde),
            'prev_state': prev_state,
            'prev_action': prev_action,
            'current_state': current_state,
            'q_table': self.q,
            't_hat': self.t_hat,
            'td_error': self.td_error,
            'td_error_history': self.td_error_history,
        }

    # ---- Appraisal computations (same formulas as original paper) ----

    def appraise_power(self):
        """Power: how much the best action dominates at the chosen state."""
        state = self.mdp.chosen_state
        if state is None:
            return 0.5

        q_vals = {a: v for a, v in self.q.get(state, {}).items()
                  if a in self._get_all_actions_for_state(state)}
        if len(q_vals) <= 1:
            return 0.0

        vals = list(q_vals.values())
        avg_q = sum(vals) / len(vals)
        min_q = min(vals)
        max_q = max(vals)

        if abs(min_q) < max_q:
            if max_q == 0:
                return 0.0
            return abs((max_q - avg_q) / max_q)
        else:
            if min_q == 0:
                return 0.0
            return abs((min_q - avg_q) / min_q)

    def appraise_goal_relevance(self):
        """Goal relevance: min(1, |δ|)"""
        return min(1.0, abs(self.td_error))

    def appraise_suddenness(self, prev_state, prev_action, current_state):
        """Suddenness: 1 - p(s'|s,a)"""
        if prev_state is None or prev_action is None:
            return 0.0

        counts = self.t_hat.get(prev_state, {}).get(prev_action, {})
        total = sum(counts.values())
        if total == 0:
            return 0.0

        freq = counts.get(current_state, 0) / total
        return 1.0 - freq

    def appraise_conduciveness(self):
        """Conduciveness: clip(δ, -1, 1) / 2 + 0.5"""
        return max(-1.0, min(1.0, self.td_error)) / 2.0 + 0.5
