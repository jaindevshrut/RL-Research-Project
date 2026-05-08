"""
Central hyperparameter configuration.

All knobs in one place so the experiment is easy to reproduce.
Each section is annotated with WHY the value matters for the
appraisal-RL story (so the report can cite specific numbers).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    grid_size: int = 7
    max_steps: int = 80
    n_lava: int = 3
    n_keys: int = 1
    seed: int = 0


@dataclass
class DQNConfig:
    # Network
    hidden_dim: int = 128
    n_ensemble_heads: int = 3        # bootstrap heads -> epistemic uncertainty
    dueling: bool = True             # split V(s) and A(s,a)
    double_dqn: bool = True          # decoupled action-selection / target-eval

    # Optimization
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    target_update_every: int = 500   # frames between target-net syncs

    # Replay
    buffer_size: int = 50_000
    min_buffer: int = 1_000          # warm-up before learning starts
    use_prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frames: int = 20_000

    # QR-DQN (Method 2). Ignored by Method 1.
    n_quantiles: int = 51            # number of atoms in Z(s, a)


@dataclass
class TrainConfig:
    total_frames: int = 60_000
    log_every: int = 1_000
    eval_every: int = 5_000
    eval_episodes: int = 20
    seed: int = 0
    device: str = "cpu"


@dataclass
class AppraisalConfig:
    # Which dimensions are active. Order is fixed (see appraisal/extractor.py).
    # Index 0..3 are the four checks from the original paper.
    # Index 4..7 are our additions (designed to be orthogonal).
    use_dims: List[str] = field(default_factory=lambda: [
        "suddenness",
        "goal_relevance",
        "conduciveness",
        "power",
        "predictability",
        "anticipation",
        "urgency",
        "familiarity",
    ])
    td_scale: float = 1.0            # softening factor for tanh(TD/scale)
    value_scale: float = 1.0         # softening factor for tanh(V/scale)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    appraisal: AppraisalConfig = field(default_factory=AppraisalConfig)
    run_name: str = "extended_8dim"


def baseline_config() -> Config:
    """4-dim baseline: only the original paper's checks."""
    cfg = Config()
    cfg.appraisal.use_dims = [
        "suddenness", "goal_relevance", "conduciveness", "power",
    ]
    cfg.run_name = "baseline_4dim"
    return cfg


def extended_config() -> Config:
    """Our extension with 8 (theoretically) decorrelated dims."""
    return Config()


def qrdqn_config() -> Config:
    """Method 2 — QR-DQN backbone with the same 8-D appraisal vector.

    Reported as a comparison only: Method 1 (Dueling Double DQN with a
    bootstrap ensemble) is the headline configuration of this paper.
    """
    cfg = Config()
    cfg.run_name = "qrdqn_8dim"
    return cfg
