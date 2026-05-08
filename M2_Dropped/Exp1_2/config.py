"""Shared configuration for baseline and V2 experiment runs."""

from __future__ import annotations

import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_TYPE = os.getenv("MODEL_TYPE", "baseline").strip().lower()
if MODEL_TYPE not in {"baseline", "v2"}:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE!r}")

NUM_QUANTILES = int(os.getenv("NUM_QUANTILES", "64"))
USE_PER = _get_bool("USE_PER", True)
SEED = int(os.getenv("SEED", "42"))

# Shared evaluation settings.
SEEDS = [42, 123, 999, 2024, 7]
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "5000"))
HUBER_KAPPA = float(os.getenv("HUBER_KAPPA", "1.0"))
PER_EPSILON = float(os.getenv("PER_EPSILON", "1e-6"))
