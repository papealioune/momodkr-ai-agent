"""Train-set z-score normalisation stats for the obs vector.

Why this exists:
  - OFI is in base-asset units (BTC, ETH), depth columns can sit at
    tens-of-thousands of base units, micro_price_log_ret is ~1e-4.
    Without normalisation, PPO's MLP has to internally re-scale six orders
    of magnitude per feature, which is a known convergence wall.
  - We z-score on the CHRONOLOGICAL train split only -- never the eval
    split (moleapp lesson on leakage). The stats are then frozen at
    inference, baked into the ONNX graph, and reused by the live Rust
    feature_builder.

Stats are persisted as a small JSON next to the episode parquets:

    data/episodes/<SYM>/<feature_version>/norm_stats.json

JSON schema:
    {
      "feature_version": "0.1.0",
      "feature_spec_checksum": "<16-hex>",
      "n_train_rows": 12345,
      "clip": 10.0,
      "stats": { "<feature>": {"mean": float, "std": float}, ... }
    }

Position features are NOT z-scored (their semantics already constrain
them: pos_signed_notional_pct in [-0.17, 0.17], pos_hold_ticks_norm in
[0, 1], pos_unrealized_pnl_pct in [-1, +inf] but practically bounded by
DD-kill and liquidation). The env clips them at the very end of the obs
build to keep the policy net's input bounded.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_NAMES,
)

DEFAULT_CLIP = 10.0
EPS = 1e-8


@dataclass
class NormStats:
    feature_version: str
    feature_spec_checksum: str
    n_train_rows: int
    clip: float
    # Stored as parallel lists in canonical MARKET_FEATURE_NAMES order
    # for fast bulk numpy ops at runtime.
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    @property
    def mean_array(self) -> np.ndarray:
        return np.asarray(self.mean, dtype=np.float32)

    @property
    def std_array(self) -> np.ndarray:
        return np.asarray(self.std, dtype=np.float32)


def compute_norm_stats(train_df: pd.DataFrame, clip: float = DEFAULT_CLIP) -> NormStats:
    """Compute per-feature mean + std on the train split.

    `train_df` must contain every column in MARKET_FEATURE_NAMES. Stats
    are computed only over those columns; sim-state columns are ignored.
    """
    missing = [c for c in MARKET_FEATURE_NAMES if c not in train_df.columns]
    if missing:
        raise KeyError(f"train_df missing market features for stats: {missing}")
    mean_arr = np.zeros(len(MARKET_FEATURE_NAMES), dtype=np.float64)
    std_arr = np.ones(len(MARKET_FEATURE_NAMES), dtype=np.float64)
    for i, c in enumerate(MARKET_FEATURE_NAMES):
        col = train_df[c].to_numpy(dtype=np.float64)
        mean_arr[i] = float(np.mean(col))
        std_arr[i] = float(np.std(col, ddof=0))
    # Guard against zero variance (constant features) -- never divide by 0.
    std_arr = np.where(std_arr < EPS, 1.0, std_arr)
    return NormStats(
        feature_version=FEATURE_VERSION,
        feature_spec_checksum=FEATURE_SPEC_CHECKSUM,
        n_train_rows=int(len(train_df)),
        clip=float(clip),
        mean=mean_arr.astype(np.float32).tolist(),
        std=std_arr.astype(np.float32).tolist(),
    )


def save_norm_stats(stats: NormStats, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(stats), indent=2))
    return path


def load_norm_stats(path: Path) -> NormStats:
    raw = json.loads(path.read_text())
    if raw.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"norm_stats feature_version {raw.get('feature_version')!r} != current {FEATURE_VERSION!r}"
        )
    if raw.get("feature_spec_checksum") != FEATURE_SPEC_CHECKSUM:
        raise ValueError(
            f"norm_stats feature_spec_checksum {raw.get('feature_spec_checksum')!r} "
            f"!= current {FEATURE_SPEC_CHECKSUM!r}; retrain or re-export."
        )
    if len(raw.get("mean", [])) != len(MARKET_FEATURE_NAMES):
        raise ValueError(
            f"norm_stats mean length {len(raw.get('mean', []))} != expected {len(MARKET_FEATURE_NAMES)}"
        )
    # Streaming Welford in episode_builder propagates NaN forever once it
    # ingests a NaN batch -- the funding columns have leading-NaN rows at
    # each per-day boundary, so the persisted mean/std for those columns
    # land as NaN. Sanitize on load: NaN mean -> 0.0, NaN/zero std -> 1.0.
    # This pairs with the env's nan_to_num on the raw features (the env
    # zero-fills the funding columns, so passing through with mean=0/std=1
    # leaves them at 0.0 = neutral "no funding pressure" -- which is what
    # the reward function already assumes).
    mean = [0.0 if (m is None or (isinstance(m, float) and not math.isfinite(m))) else float(m) for m in raw["mean"]]
    std = [
        1.0 if (s is None or (isinstance(s, float) and not math.isfinite(s)) or float(s) < EPS) else float(s)
        for s in raw["std"]
    ]
    raw["mean"] = mean
    raw["std"] = std
    return NormStats(**raw)


def apply_zscore(features: np.ndarray, stats: NormStats) -> np.ndarray:
    """Apply (x - mean) / std then clip to +/- stats.clip.

    `features` may be (n_features,) or (batch, n_features); the broadcast
    handles both. Output dtype matches input.
    """
    mean = stats.mean_array
    std = stats.std_array
    z = (features.astype(np.float32) - mean) / std
    if stats.clip > 0:
        z = np.clip(z, -stats.clip, stats.clip)
    return z.astype(features.dtype, copy=False)


def norm_stats_path_for_episodes(episodes_dir: Path) -> Path:
    return episodes_dir / "norm_stats.json"
