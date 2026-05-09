"""
environments.py - Unified MDP environments for all 11 emotion scenarios.
==========================================================================
IMPROVEMENT: All MDPs now share a common AppraisalMDP base class with
gym-like reset()/step() interface. States are one-hot encoded for use
with the DQN agent. Original tabular transition dynamics are preserved
exactly so results are directly comparable to the paper.

Covers:
  Exp1/2: Boredom, Fear, Happiness, Joy, Pride, Sadness, Shame
  Exp3:   Anxiety, Despair, Irritation, Rage
"""

import random
import numpy as np


class AppraisalMDP:
    """Base class for all emotion-eliciting MDP environments.

    Provides a gym-like interface with one-hot state encoding.
    Subclasses must define:
        - states: list of state names
        - actions: list of action names
        - terminal_states: list of terminal state names
        - _build_transitions(): sets self.transitions dict
        - _build_story_transitions(): sets story-mode transitions
        - _reward(state): returns scalar reward
    """

    def __init__(self):
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

        # Build default transitions
        self.story_mode = False
        self.model_changed = False
        self.chosen_state = None
        self.chosen_action = None
        self._build_transitions()

        self.state = None
        self.previous_state = None
        self.action = None
        self.previous_action = None
        self.reward = 0
        self.terminal = False
        self.tde = []

    def reset(self):
        """Reset environment to start state. Returns one-hot state vector."""
        self.state = self.states[0]  # Always 'S'
        self.previous_state = None
        self.action = None
        self.previous_action = None
        self.reward = 0
        self.terminal = False
        self.tde = []
        return self._encode_state(self.state)

    def _encode_state(self, state_name):
        """One-hot encode a state name."""
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[self.state_to_idx[state_name]] = 1.0
        return vec

    def get_available_actions(self, state_name=None):
        """Return list of available action names for a state."""
        if state_name is None:
            state_name = self.state
        return list(self.transitions.get(state_name, {}).keys())

    def step(self, action_name):
        """Take an action, transition to next state.
        Returns: (next_state_onehot, reward, done, info)
        """
        self.previous_action = self.action
        self.previous_state = self.state
        self.action = action_name

        # Sample next state from transition probs
        trans = self.transitions[self.state][action_name]
        next_states = list(trans.keys())
        probs = list(trans.values())
        self.state = random.choices(next_states, weights=probs, k=1)[0]

        # Calculate reward
        self.reward = self._reward(self.state)

        # Check terminal
        if self.state in self.terminal_states:
            self.terminal = True

        return self._encode_state(self.state), self.reward, self.terminal, {}

    def enter_story_mode(self):
        """Switch to story mode (forced narrative path)."""
        self.story_mode = True
        self._build_story_transitions()

    def enter_model_changed_mode(self):
        """Switch to model-changed mode."""
        self.model_changed = True
        self._build_model_changed_transitions()

    def _build_model_changed_transitions(self):
        """Default: no change. Override in subclasses that use model_changed."""
        pass


# =============================================================================
# Experiment 1/2 Environments (7 emotions)
# =============================================================================

class BoredomMDP(AppraisalMDP):
    """Boredom: low novelty, low goal relevance, medium power."""
    states = ['S', 'P', 'E', 'G']
    actions = ['frwd', 'a1', 'a2']
    terminal_states = ['E', 'G']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'P': 1.0}},
            'P': {'a1': {'E': 1.0}, 'a2': {'G': 1.0}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = 'P'
        self.chosen_action = 'a1'

    def _reward(self, state):
        if state == 'E':
            return -1
        elif state == 'G':
            return 5
        return -1


class FearMDP(AppraisalMDP):
    """Fear: high suddenness, high goal relevance, obstructive, very low power."""
    states = ['S', 'S1', 'P', 'E', 'G']
    actions = ['frwd']
    terminal_states = ['E', 'G']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'P': 0.2, 'G': 0.8}},
            'P': {'frwd': {'E': 1.0}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'P': 1.0, 'G': 0.0}
        self.chosen_state = 'P'
        self.chosen_action = 'frwd'

    def _reward(self, state):
        if state == 'E':
            return -10
        elif state == 'G':
            return 10
        return -1


class HappinessMDP(AppraisalMDP):
    """Happiness: low suddenness, medium goal rel, high conduciveness, open power."""
    states = ['S', 'S1', 'G', 'E']
    actions = ['frwd', 'a1', 'a2']
    terminal_states = ['G', 'E']

    def __init__(self):
        super().__init__()
        self._reward_boost = False

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'a1': {'G': 1.0}, 'a2': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
            'E': {'frwd': {'E': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = 'S1'
        self.chosen_action = 'a1'
        self._reward_boost = True

    def _build_model_changed_transitions(self):
        self._reward_boost = True

    def _reward(self, state):
        if state == 'G':
            return 10 if self._reward_boost else 7
        elif state == 'E':
            return -10
        elif state == 'S1' and self._reward_boost:
            return 0
        return -3


class JoyMDP(AppraisalMDP):
    """Joy: high suddenness, high goal relevance, very high conduciveness."""
    states = ['S', 'S1', 'G', 'E']
    actions = ['frwd']
    terminal_states = ['G', 'E']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'E': 0.8, 'G': 0.2}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'G': 1.0, 'E': 0.0}
        self.chosen_state = 'S1'
        self.chosen_action = 'frwd'

    def _reward(self, state):
        if state == 'E':
            return -1
        elif state == 'G':
            return 10
        return -1


class PrideMDP(AppraisalMDP):
    """Pride: open suddenness, high goal rel, high conduciveness."""
    states = ['S', 'S1', 'S2', 'G', 'G_plus']
    actions = ['frwd', 'a1', 'a2']
    terminal_states = ['G', 'G_plus']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'a1': {'G': 1.0}, 'a2': {'S2': 1.0}},
            'S2': {'frwd': {'G_plus': 0.5, 'G': 0.5}},
            'G': {'frwd': {'G': 1.0}},
            'G_plus': {'frwd': {'G_plus': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = 'S1'
        self.chosen_action = 'a2'
        self.transitions['S2']['frwd'] = {'G_plus': 1.0, 'G': 0.0}

    def _reward(self, state):
        if state == 'G':
            return 5
        elif state == 'S2':
            return -5
        elif state == 'G_plus':
            return 10
        return -1


class SadnessMDP(AppraisalMDP):
    """Sadness: low suddenness, high goal rel, obstructive, very low power."""
    states = ['S', 'S1', 'P', 'E', 'G']
    actions = ['frwd']
    terminal_states = ['E', 'G']

    def __init__(self):
        super().__init__()
        self._use_severe_punishment = False

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'P': 0.8, 'G': 0.2}},
            'P': {'frwd': {'E': 1.0}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'P': 1.0, 'G': 0.0}
        self.chosen_state = 'P'
        self.chosen_action = 'frwd'

    def _build_model_changed_transitions(self):
        self._use_severe_punishment = True

    def _reward(self, state):
        if state == 'E':
            return -10 if self._use_severe_punishment else -1
        elif state == 'G':
            return 10
        return -1


class ShameMDP(AppraisalMDP):
    """Shame: open suddenness, high goal rel, open conduciveness, open power."""
    states = ['S', 'S1', 'S2', 'G', 'E']
    actions = ['frwd', 'a1', 'a2']
    terminal_states = ['G', 'E']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'a1': {'G': 1.0}, 'a2': {'S2': 1.0}},
            'S2': {'frwd': {'E': 0.2, 'G': 0.8}},
            'G': {'frwd': {'G': 1.0}},
            'E': {'frwd': {'E': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.chosen_state = 'S1'
        self.chosen_action = 'a2'
        self.transitions['S2']['frwd'] = {'E': 1.0, 'G': 0.0}

    def _reward(self, state):
        if state == 'E':
            return -10
        elif state == 'G':
            return 5
        return -1


# =============================================================================
# Experiment 3 Environments (4 emotions)
# =============================================================================

class AnxietyMDP(AppraisalMDP):
    """Anxiety: low suddenness, medium goal rel, obstructive, low power."""
    states = ['S', 'S1', 'P', 'E', 'G']
    actions = ['frwd']
    terminal_states = ['E', 'G']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'P': 0.8, 'G': 0.2}},
            'P': {'frwd': {'E': 1.0}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'P': 1.0, 'G': 0.0}
        self.chosen_state = 'P'
        self.chosen_action = None  # Only 'frwd' available

    def _reward(self, state):
        if state == 'E':
            return -10
        elif state == 'P':
            return -5
        elif state == 'G':
            return 10
        return -1


class DespairMDP(AppraisalMDP):
    """Despair: high suddenness, high goal rel, obstructive, very low power."""
    states = ['S', 'S1', 'P', 'E', 'G']
    actions = ['frwd']
    terminal_states = ['E', 'G']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'P': 0.2, 'G': 0.8}},
            'P': {'frwd': {'E': 1.0}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'P': 1.0, 'G': 0.0}
        self.chosen_state = 'P'
        self.chosen_action = None

    def _reward(self, state):
        if state == 'E':
            return -10
        elif state == 'G':
            return 10
        return -1


class IrritationMDP(AppraisalMDP):
    """Irritation: low suddenness, medium goal rel, obstructive, medium power."""
    states = ['S', 'S1', 'P', 'E', 'G']
    actions = ['frwd', 'a1', 'a2', 'a3']
    terminal_states = ['E', 'G']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'P': 0.8, 'G': 0.2}},
            'P': {'a1': {'E': 1.0}, 'a2': {'S': 0.8, 'G': 0.2}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'P': 1.0, 'G': 0.0}
        self.chosen_state = 'P'
        self.chosen_action = None

    def _reward(self, state):
        if state in ('E', 'P'):
            return -10
        elif state == 'G':
            return 10
        return -1


class RageMDP(AppraisalMDP):
    """Rage: high suddenness, high goal rel, obstructive, high power."""
    states = ['S', 'P', 'S1', 'E', 'G']
    actions = ['frwd', 'a1', 'a2']
    terminal_states = ['E', 'G']

    def _build_transitions(self):
        self.transitions = {
            'S': {'frwd': {'S1': 1.0}},
            'S1': {'frwd': {'P': 0.2, 'G': 0.8}},
            'P': {'a1': {'E': 1.0}, 'a2': {'S': 1.0}},
            'E': {'frwd': {'E': 1.0}},
            'G': {'frwd': {'G': 1.0}},
        }

    def _build_story_transitions(self):
        self._build_transitions()
        self.transitions['S1']['frwd'] = {'P': 1.0, 'G': 0.0}
        self.chosen_state = 'P'
        self.chosen_action = None

    def _reward(self, state):
        if state in ('E', 'P'):
            return -10
        elif state == 'G':
            return 10
        return -1


# =============================================================================
# Registry
# =============================================================================

# Exp1/2 scenarios with their terminate states and training configs
EXP12_SCENARIOS = {
    'Boredom':   {'cls': BoredomMDP,   'terminate': 'P',      'train_episodes': 20000, 'i_change': 0},
    'Fear':      {'cls': FearMDP,      'terminate': 'P',      'train_episodes': 20000, 'i_change': 0},
    'Happiness': {'cls': HappinessMDP, 'terminate': 'S1',     'train_episodes': 20000, 'i_change': 5},
    'Joy':       {'cls': JoyMDP,       'terminate': 'G',      'train_episodes': 20000, 'i_change': 0},
    'Pride':     {'cls': PrideMDP,     'terminate': 'G_plus', 'train_episodes': 20000, 'i_change': 0},
    'Sadness':   {'cls': SadnessMDP,   'terminate': 'P',      'train_episodes': 20000, 'i_change': 10},
    'Shame':     {'cls': ShameMDP,     'terminate': 'E',      'train_episodes': 20000, 'i_change': 0},
}

EXP3_SCENARIOS = {
    'Anxiety':    {'cls': AnxietyMDP,    'terminate': 'P', 'train_episodes': 20000, 'i_change': 0},
    'Despair':    {'cls': DespairMDP,    'terminate': 'P', 'train_episodes': 20000, 'i_change': 0},
    'Irritation': {'cls': IrritationMDP, 'terminate': 'P', 'train_episodes': 20000, 'i_change': 0},
    'Rage':       {'cls': RageMDP,       'terminate': 'P', 'train_episodes': 20000, 'i_change': 0},
}

ALL_SCENARIOS = {**EXP12_SCENARIOS, **EXP3_SCENARIOS}
