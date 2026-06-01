from pathlib import Path

import numpy as np
import pandas as pd

from envs.momodkr_env import EnvConfig, MomoDkrEnv
from serving.feature_version import MARKET_FEATURE_NAMES


def _episode(n: int = 4000, base_mid: float = 50_000.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 100
    log_ret = rng.standard_normal(n) * 1e-4
    mid = base_mid * np.exp(np.cumsum(log_ret))
    half_spread = base_mid * 1e-5
    df = pd.DataFrame({"ts_ms": ts})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32) * 0.1
    df["mid"] = mid.astype(np.float32)
    df["bid_px"] = (mid - half_spread).astype(np.float32)
    df["ask_px"] = (mid + half_spread).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n, 50.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    return df


def test_env_loads_multiple_parquets(tmp_path: Path) -> None:
    a = tmp_path / "btc.parquet"
    b = tmp_path / "eth.parquet"
    _episode(base_mid=50_000.0, seed=1).to_parquet(a, index=False)
    _episode(base_mid=3_000.0, seed=2).to_parquet(b, index=False)
    env = MomoDkrEnv([a, b], EnvConfig(episode_length_ticks=200, apply_obs_normalisation=False), seed=0)
    assert env.parquet_paths == [a, b]
    assert len(env._features_pool) == 2


def test_env_reset_samples_active_index_from_pool(tmp_path: Path) -> None:
    paths = []
    for i, base in enumerate([50_000.0, 3_000.0, 150.0]):
        p = tmp_path / f"sym_{i}.parquet"
        _episode(base_mid=base, seed=i).to_parquet(p, index=False)
        paths.append(p)
    env = MomoDkrEnv(paths, EnvConfig(episode_length_ticks=200, apply_obs_normalisation=False), seed=0)
    seen_indices: set[int] = set()
    for s in range(50):
        env.reset(seed=s)
        seen_indices.add(env._active_idx)
    # over 50 resets all three symbols should have been picked at least once
    assert seen_indices == {0, 1, 2}


def test_single_parquet_backward_compat(tmp_path: Path) -> None:
    p = tmp_path / "btc.parquet"
    _episode().to_parquet(p, index=False)
    env = MomoDkrEnv(p, EnvConfig(episode_length_ticks=200, apply_obs_normalisation=False), seed=0)
    assert env.parquet_paths == [p]
    obs, info = env.reset(seed=0)
    assert obs.shape[0] > 0
    assert info["active_parquet"] == str(p)


def test_env_rejects_too_small_parquet_in_pool(tmp_path: Path) -> None:
    big = tmp_path / "big.parquet"
    small = tmp_path / "small.parquet"
    _episode(n=4000).to_parquet(big, index=False)
    _episode(n=50).to_parquet(small, index=False)
    import pytest

    with pytest.raises(ValueError, match="rows"):
        MomoDkrEnv([big, small], EnvConfig(episode_length_ticks=200, apply_obs_normalisation=False), seed=0)
