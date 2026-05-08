"""Reuse the shared experiment configuration from Exp1_2."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SHARED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "Exp1_2" / "config.py"
_SPEC = spec_from_file_location("_shared_exp_config", _SHARED_CONFIG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load shared config from {_SHARED_CONFIG_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MODEL_TYPE = _MODULE.MODEL_TYPE
NUM_QUANTILES = _MODULE.NUM_QUANTILES
USE_PER = _MODULE.USE_PER
SEED = _MODULE.SEED
SEEDS = _MODULE.SEEDS
LOG_INTERVAL = _MODULE.LOG_INTERVAL
HUBER_KAPPA = _MODULE.HUBER_KAPPA
PER_EPSILON = _MODULE.PER_EPSILON
