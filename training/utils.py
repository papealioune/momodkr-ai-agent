"""Config loading + LR / entropy schedule helpers for SB3.

The training YAML (e.g. configs/training/v1_engine_cold.yaml) expresses
linearly-annealed learning rate and entropy coefficient. SB3 PPO accepts
either a float or a `Callable[[float], float]` where the argument is the
fraction of training remaining (1.0 at start -> 0.0 at end).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """Returns f(progress_remaining): start at progress_remaining=1.0, end at 0.0."""
    if start < end:
        raise ValueError(f"linear_schedule expects start >= end (decay), got {start} -> {end}")

    def f(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)

    return f


def resolve_schedule_or_float(spec: dict[str, Any] | float) -> float | Callable[[float], float]:
    """Accepts either a scalar or {schedule, start, end} dict from YAML."""
    if isinstance(spec, dict):
        kind = spec.get("schedule", "linear")
        if kind != "linear":
            raise ValueError(f"unsupported schedule {kind!r}; only 'linear' supported")
        return linear_schedule(float(spec["start"]), float(spec["end"]))
    return float(spec)


_ACTIVATION_MAP: dict[str, str] = {
    "tanh": "Tanh",
    "relu": "ReLU",
    "elu": "ELU",
    "leaky_relu": "LeakyReLU",
    "gelu": "GELU",
    "silu": "SiLU",
}


def resolve_policy_kwargs(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate YAML-friendly policy_kwargs into SB3-callable form.

    SB3 expects `activation_fn` as a torch.nn class, not a string. YAML can
    only express strings, so we map "tanh" -> torch.nn.Tanh here. Returns a
    fresh dict so the original config remains YAML-safe (re-loadable).
    """
    if not spec:
        return spec
    import torch.nn as nn

    out = dict(spec)
    act = out.get("activation_fn")
    if isinstance(act, str):
        key = act.lower()
        cls_name = _ACTIVATION_MAP.get(key)
        if cls_name is None:
            raise ValueError(
                f"unknown activation_fn {act!r}; supported: {sorted(_ACTIVATION_MAP)}"
            )
        out["activation_fn"] = getattr(nn, cls_name)
    return out


def categorical_entropy_normalised(probs) -> float:
    """Shannon entropy of a categorical distribution, divided by ln(n_actions).

    Returns a value in [0, 1]: 0 = fully collapsed; 1 = uniform random.
    Used by the sigma_divergence_killswitch (the Discrete-action analogue
    of moleapp's sigma > 2.0 rule on Box actions).
    """
    import math

    import numpy as np

    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr / arr.sum(axis=-1, keepdims=True)
    entropy = -(arr * np.log(arr)).sum(axis=-1)
    n = arr.shape[-1]
    return float(entropy.mean() / math.log(n))
