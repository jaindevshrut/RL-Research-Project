import random

import numpy as np


class AppraisalMDP:
    def __init__(self):
        self.state_to_idx = {state: index for index, state in enumerate(self.states)}
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.action_to_idx = {action: index for index, action in enumerate(self.actions)}

        self.story_mode = False
        self.model_changed = False
        self.chosen_state = None
        self.chosen_action = None
        self._build_transitions()

        self.state = None
        self.previous_state = None
        self.previous_action = None
        self.action = None
        self.reward = 0
        self.terminal = False
        self.tde = []

    def reset(self):
        self.state = self.states[0]
        self.previous_state = None
        self.previous_action = None
        self.action = None
        self.reward = 0
        self.terminal = False
        self.tde = []
        return self._encode_state(self.state)

    def _encode_state(self, state_name):
        encoded = np.zeros(self.n_states, dtype=np.float32)
        encoded[self.state_to_idx[state_name]] = 1.0
        return encoded

    def step(self, action_name):
        self.previous_action = self.action
        self.previous_state = self.state
        self.action = action_name

        transition_map = self.transitions[self.state][action_name]
        next_states = list(transition_map.keys())
        probabilities = list(transition_map.values())
        self.state = random.choices(next_states, weights=probabilities, k=1)[0]

        self.reward = self._reward(self.state)
        self.terminal = self.state in self.terminal_states
        return self._encode_state(self.state), self.reward, self.terminal, {}

    def enter_story_mode(self):
        self.story_mode = True
        self._build_story_transitions()

    def enter_model_changed_mode(self):
        self.model_changed = True
        self._build_model_changed_transitions()

    def _build_model_changed_transitions(self):
        return None


class BoredomMDP(AppraisalMDP):
    states = ["S", "P", "E", "G"]
    actions = ["frwd", "a1", "a2"]
    terminal_states = ["E", "G"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"P": 1.0}},
            "P": {"a1": {"E": 1.0}, "a2": {"G": 1.0}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = "P"
        self.chosen_action = "a1"

    def _reward(self, state):
        if state == "E":
            return -1
        if state == "G":
            return 5
        return -1


class FearMDP(AppraisalMDP):
    states = ["S", "S1", "P", "E", "G"]
    actions = ["frwd"]
    terminal_states = ["E", "G"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"P": 0.2, "G": 0.8}},
            "P": {"frwd": {"E": 1.0}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"P": 1.0, "G": 0.0}
        self.chosen_state = "P"
        self.chosen_action = "frwd"

    def _reward(self, state):
        if state == "E":
            return -10
        if state == "G":
            return 10
        return -1


class HappinessMDP(AppraisalMDP):
    states = ["S", "S1", "G", "E"]
    actions = ["frwd", "a1", "a2"]
    terminal_states = ["G", "E"]

    def __init__(self):
        self.reward_boost = False
        super().__init__()

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"a1": {"G": 1.0}, "a2": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
            "E": {"frwd": {"E": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = "S1"
        self.chosen_action = "a1"
        self.reward_boost = True

    def _build_model_changed_transitions(self):
        self.reward_boost = True

    def _reward(self, state):
        if state == "G":
            return 10 if self.reward_boost else 7
        if state == "E":
            return -10
        if state == "S1" and self.reward_boost:
            return 0
        return -3


class JoyMDP(AppraisalMDP):
    states = ["S", "S1", "G", "E"]
    actions = ["frwd"]
    terminal_states = ["G", "E"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"E": 0.8, "G": 0.2}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"G": 1.0, "E": 0.0}
        self.chosen_state = "S1"
        self.chosen_action = "frwd"

    def _reward(self, state):
        if state == "E":
            return -1
        if state == "G":
            return 10
        return -1


class PrideMDP(AppraisalMDP):
    states = ["S", "S1", "S2", "G", "G_plus"]
    actions = ["frwd", "a1", "a2"]
    terminal_states = ["G", "G_plus"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"a1": {"G": 1.0}, "a2": {"S2": 1.0}},
            "S2": {"frwd": {"G_plus": 0.5, "G": 0.5}},
            "G": {"frwd": {"G": 1.0}},
            "G_plus": {"frwd": {"G_plus": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = "S1"
        self.chosen_action = "a2"
        self.transitions["S2"]["frwd"] = {"G_plus": 1.0, "G": 0.0}

    def _reward(self, state):
        if state == "G":
            return 5
        if state == "S2":
            return -5
        if state == "G_plus":
            return 10
        return -1


class SadnessMDP(AppraisalMDP):
    states = ["S", "S1", "P", "E", "G"]
    actions = ["frwd"]
    terminal_states = ["E", "G"]

    def __init__(self):
        self.severe_punishment = False
        super().__init__()

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"P": 0.8, "G": 0.2}},
            "P": {"frwd": {"E": 1.0}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"P": 1.0, "G": 0.0}
        self.chosen_state = "P"
        self.chosen_action = "frwd"

    def _build_model_changed_transitions(self):
        self.severe_punishment = True

    def _reward(self, state):
        if state == "E":
            return -10 if self.severe_punishment else -1
        if state == "G":
            return 10
        return -1


class ShameMDP(AppraisalMDP):
    states = ["S", "S1", "S2", "G", "E"]
    actions = ["frwd", "a1", "a2"]
    terminal_states = ["G", "E"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"a1": {"G": 1.0}, "a2": {"S2": 1.0}},
            "S2": {"frwd": {"E": 0.2, "G": 0.8}},
            "G": {"frwd": {"G": 1.0}},
            "E": {"frwd": {"E": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = "S1"
        self.chosen_action = "a2"
        self.transitions["S2"]["frwd"] = {"E": 1.0, "G": 0.0}

    def _reward(self, state):
        if state == "E":
            return -10
        if state == "G":
            return 5
        return -1


class AnxietyMDP(AppraisalMDP):
    states = ["S", "S1", "P", "E", "G"]
    actions = ["frwd"]
    terminal_states = ["E", "G"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"P": 0.8, "G": 0.2}},
            "P": {"frwd": {"E": 1.0}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"P": 1.0, "G": 0.0}
        self.chosen_state = "P"
        self.chosen_action = None

    def _reward(self, state):
        if state == "E":
            return -10
        if state == "P":
            return -5
        if state == "G":
            return 10
        return -1


class DespairMDP(AppraisalMDP):
    states = ["S", "S1", "P", "E", "G"]
    actions = ["frwd"]
    terminal_states = ["E", "G"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"P": 0.2, "G": 0.8}},
            "P": {"frwd": {"E": 1.0}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"P": 1.0, "G": 0.0}
        self.chosen_state = "P"
        self.chosen_action = None

    def _reward(self, state):
        if state == "E":
            return -10
        if state == "G":
            return 10
        return -1


class IrritationMDP(AppraisalMDP):
    states = ["S", "S1", "P", "E", "G"]
    actions = ["frwd", "a1", "a2", "a3"]
    terminal_states = ["E", "G"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"P": 0.8, "G": 0.2}},
            "P": {"a1": {"E": 1.0}, "a2": {"S": 0.8, "G": 0.2}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"P": 1.0, "G": 0.0}
        self.chosen_state = "P"
        self.chosen_action = None

    def _reward(self, state):
        if state in ("E", "P"):
            return -10
        if state == "G":
            return 10
        return -1


class RageMDP(AppraisalMDP):
    states = ["S", "P", "S1", "E", "G"]
    actions = ["frwd", "a1", "a2"]
    terminal_states = ["E", "G"]

    def _build_transitions(self):
        self.transitions = {
            "S": {"frwd": {"S1": 1.0}},
            "S1": {"frwd": {"P": 0.2, "G": 0.8}},
            "P": {"a1": {"E": 1.0}, "a2": {"S": 1.0}},
            "E": {"frwd": {"E": 1.0}},
            "G": {"frwd": {"G": 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions["S1"]["frwd"] = {"P": 1.0, "G": 0.0}
        self.chosen_state = "P"
        self.chosen_action = None

    def _reward(self, state):
        if state in ("E", "P"):
            return -10
        if state == "G":
            return 10
        return -1


EXP12_SCENARIOS = {
    "Boredom": {"cls": BoredomMDP, "terminate": "P", "train_episodes": 20000, "i_change": 0},
    "Fear": {"cls": FearMDP, "terminate": "P", "train_episodes": 20000, "i_change": 0},
    "Happiness": {
        "cls": HappinessMDP,
        "terminate": "S1",
        "train_episodes": 30000,
        "i_change": 5,
        "agent_kwargs": {"epsilon": 0.40, "min_epsilon": 0.05, "epsilon_decay": 0.9997},
    },
    "Joy": {"cls": JoyMDP, "terminate": "G", "train_episodes": 20000, "i_change": 0},
    "Pride": {
        "cls": PrideMDP,
        "terminate": "G_plus",
        "train_episodes": 40000,
        "i_change": 0,
        "agent_kwargs": {"epsilon": 0.45, "min_epsilon": 0.05, "epsilon_decay": 0.9997},
    },
    "Sadness": {"cls": SadnessMDP, "terminate": "P", "train_episodes": 20000, "i_change": 10},
    "Shame": {"cls": ShameMDP, "terminate": "E", "train_episodes": 20000, "i_change": 0},
}

EXP3_SCENARIOS = {
    "Anxiety": {"cls": AnxietyMDP, "terminate": "P", "train_episodes": 10000, "i_change": 0},
    "Despair": {"cls": DespairMDP, "terminate": "P", "train_episodes": 30000, "i_change": 0},
    "Irritation": {
        "cls": IrritationMDP,
        "terminate": "P",
        "train_episodes": 50000,
        "i_change": 0,
        "agent_kwargs": {"epsilon": 0.45, "min_epsilon": 0.05, "epsilon_decay": 0.9997},
    },
    "Rage": {
        "cls": RageMDP,
        "terminate": "P",
        "train_episodes": 50000,
        "i_change": 0,
        "agent_kwargs": {"epsilon": 0.40, "min_epsilon": 0.05, "epsilon_decay": 0.9997},
    },
}
