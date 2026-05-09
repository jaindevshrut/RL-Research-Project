import random
from collections import deque
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs):
        return self.net(inputs)


class DQNAgent:
    def __init__(
        self,
        mdp,
        lr=5e-3,
        gamma=0.9,
        alpha=0.3,
        epsilon=0.3,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=50,
        min_epsilon=0.01,
        epsilon_decay=0.9995,
        replay_warmup=256,
        grad_clip=1.0,
    ):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_warmup = replay_warmup
        self.grad_clip = grad_clip

        self.q_net = QNetwork(mdp.n_states, mdp.n_actions)
        self.target_net = QNetwork(mdp.n_states, mdp.n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        self.q = {}
        self.t_hat = {}
        self.td_error = 0.0
        self.old_q = 0.0
        self.td_error_history = []

        for state in mdp.states:
            self.q[state] = {}
            self.t_hat[state] = {}
            for action_name in self._get_actions_for_state(state):
                self.q[state][action_name] = 0.0
                self.t_hat[state][action_name] = {next_state: 0 for next_state in mdp.states}

    def _get_actions_for_state(self, state_name):
        return list(self.mdp.transitions.get(state_name, {}).keys())

    def _greedy_action_from_network(self, state_name):
        available = self._get_actions_for_state(state_name)
        state_tensor = torch.FloatTensor(self.mdp._encode_state(state_name)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0)
        return max(
            available,
            key=lambda action: float(q_values[self.mdp.action_to_idx[action]].item()),
        )

    def _choose_action(self, state_name, force_action=None):
        available = self._get_actions_for_state(state_name)
        if not available:
            return None

        if force_action is not None and force_action in available:
            return force_action

        if random.random() < self.epsilon:
            return random.choice(available)

        if len(self.replay_buffer) >= self.replay_warmup:
            return self._greedy_action_from_network(state_name)

        q_items = {action: self.q[state_name].get(action, 0.0) for action in available}
        return max(q_items.items(), key=itemgetter(1))[0]

    def _update_tabular_q(self, previous_state, action, reward, current_state):
        if previous_state is None:
            return

        if action not in self.q[previous_state]:
            self.q[previous_state][action] = 0.0
            self.t_hat[previous_state][action] = {state: 0 for state in self.mdp.states}

        previous_q = self.q[previous_state][action]
        self.old_q = previous_q

        next_q_values = self.q.get(current_state, {})
        next_q = max(next_q_values.values()) if next_q_values else 0.0

        self.td_error = self.alpha * (reward + self.gamma * next_q - previous_q)
        self.q[previous_state][action] = previous_q + self.td_error
        self.t_hat[previous_state][action][current_state] = (
            self.t_hat[previous_state][action].get(current_state, 0) + 1
        )

    def _nn_store_and_update(self, state_name, action_name, reward, next_state_name, done):
        state_vector = self.mdp._encode_state(state_name)
        next_state_vector = self.mdp._encode_state(next_state_name)
        action_index = self.mdp.action_to_idx.get(action_name, 0)
        self.replay_buffer.append(
            (state_vector, action_index, reward, next_state_vector, float(done))
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor(np.array([row[0] for row in batch]))
        actions = torch.LongTensor([row[1] for row in batch])
        rewards = torch.FloatTensor([row[2] for row in batch])
        next_states = torch.FloatTensor(np.array([row[3] for row in batch]))
        dones = torch.FloatTensor([row[4] for row in batch])

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

    def train(self, episodes, i_change=0):
        episode = 0
        while episode < episodes:
            if i_change > 0 and episode == episodes - i_change and not self.mdp.model_changed:
                self.mdp.enter_model_changed_mode()

            self.mdp.reset()
            state_name = self.mdp.state
            done = False

            while not done:
                action = self._choose_action(state_name)
                if action is None:
                    break

                previous_state_name = state_name
                _, reward, done, _ = self.mdp.step(action)
                state_name = self.mdp.state

                self._update_tabular_q(previous_state_name, action, reward, state_name)
                self.mdp.tde.append(self.td_error)
                self.td_error_history.append(float(self.td_error))
                self._nn_store_and_update(previous_state_name, action, reward, state_name, done)

                if done:
                    previous_q = self.q.get(previous_state_name, {}).get(action, 0.0)
                    self.q[previous_state_name][action] = previous_q + self.alpha * (
                        reward - previous_q
                    )

            episode += 1
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

    def simulate_episode(self, terminate=None):
        self.mdp.enter_story_mode()
        self.mdp.reset()
        self.mdp.tde = []

        state_name = self.mdp.state
        previous_state = None
        previous_action = None

        while True:
            if previous_state is not None:
                self._update_tabular_q(previous_state, previous_action, self.mdp.reward, state_name)
                self.mdp.tde.append(self.td_error)
                self.td_error_history.append(float(self.td_error))

            forced_action = None
            if (
                (self.mdp.story_mode or self.mdp.model_changed)
                and state_name == self.mdp.chosen_state
                and self.mdp.chosen_action
            ):
                forced_action = self.mdp.chosen_action

            action = self._choose_action(state_name, force_action=forced_action)
            if action is None:
                return None

            previous_state = state_name
            previous_action = action
            _, reward, done, _ = self.mdp.step(action)
            state_name = self.mdp.state

            if done or (terminate is not None and state_name == terminate):
                self._update_tabular_q(previous_state, previous_action, reward, state_name)
                self.mdp.tde.append(self.td_error)
                self.td_error_history.append(float(self.td_error))

                if done:
                    previous_q = self.q.get(previous_state, {}).get(previous_action, 0.0)
                    self.q[previous_state][previous_action] = previous_q + self.alpha * (
                        reward - previous_q
                    )

                return {
                    "prev_state": previous_state,
                    "prev_action": previous_action,
                    "current_state": state_name,
                    "t_hat": self.t_hat,
                    "td_error": self.td_error,
                    "td_error_history": list(self.td_error_history),
                }

    def appraise_power(self):
        state = self.mdp.chosen_state
        if state is None:
            return 0.0

        q_values = {
            action: value
            for action, value in self.q.get(state, {}).items()
            if action in self._get_actions_for_state(state)
        }
        if len(q_values) <= 1:
            return 0.0

        values = list(q_values.values())
        average_q = sum(values) / len(values)
        min_q = min(values)
        max_q = max(values)

        if abs(min_q) < max_q:
            if max_q == 0:
                return 0.0
            return abs((max_q - average_q) / max_q)

        if min_q == 0:
            return 0.0
        return abs((min_q - average_q) / min_q)

    def appraise_goal_relevance(self):
        return min(1.0, abs(self.td_error))

    def appraise_suddenness(self, previous_state, previous_action, current_state):
        if previous_state is None or previous_action is None:
            return 0.0

        counts = self.t_hat.get(previous_state, {}).get(previous_action, {})
        total = sum(counts.values())
        if total == 0:
            return 0.0

        frequency = counts.get(current_state, 0) / total
        return 1.0 - frequency

    def appraise_conduciveness(self):
        return max(-1.0, min(1.0, self.td_error)) / 2.0 + 0.5
