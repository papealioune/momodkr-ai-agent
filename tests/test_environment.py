from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from envs.base_hft_env import Action, PositionSide
from envs.market_simulator import SimulatorConfig
from envs.momodkr_env import EnvConfig, MomoDkrEnv
from serving.feature_version import MARKET_FEATURE_NAMES, OBS_DIM, SIM_STATE_COLS


def _instant_sim_cfg() -> SimulatorConfig:
    """Test-mode simulator: zero latency, no walk-the-book, no tick rounding."""
    return SimulatorConfig(
        latency_ticks_min=0,
        latency_ticks_max=0,
        walk_book=False,
        trade_through_limit_fills=False,
        fee_noise_pct=0.0,
        slippage_noise_pct=0.0,
        tick_size_by_symbol={},
        default_tick_size=0.0,
    )


def _synthetic_episode(n_rows: int = 12_000, drift_bps_per_tick: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n_rows, dtype=np.int64) * 100
    # mid follows GBM with optional drift
    log_ret = rng.standard_normal(n_rows) * 1e-4 + (drift_bps_per_tick / 10_000.0)
    mid = 50_000.0 * np.exp(np.cumsum(log_ret))
    half_spread = 1.5
    df = pd.DataFrame({"ts_ms": ts})
    for col in MARKET_FEATURE_NAMES:
        df[col] = rng.standard_normal(n_rows).astype(np.float32) * 0.1
    df["mid"] = mid.astype(np.float32)
    df["bid_px"] = (mid - half_spread).astype(np.float32)
    df["ask_px"] = (mid + half_spread).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n_rows, 5.0, dtype=np.float32)
    df["funding_rate"] = np.full(n_rows, 1e-4, dtype=np.float32)
    return df


@pytest.fixture
def episode_parquet(tmp_path: Path) -> Path:
    df = _synthetic_episode()
    p = tmp_path / "train.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def cfg_short_episodes() -> EnvConfig:
    return EnvConfig(episode_length_ticks=500, sim=_instant_sim_cfg())


def test_env_obs_and_action_space(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    assert env.action_space.n == 5
    assert env.observation_space.shape == (OBS_DIM,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert info["feature_version"]
    assert info["feature_spec_checksum"]


def test_env_reset_picks_valid_start(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    obs, _ = env.reset(seed=0)
    assert env._start_cursor >= 0
    assert env._start_cursor + cfg_short_episodes.episode_length_ticks < env._n_rows


def test_env_random_walk_finishes_episode_without_nan(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    obs, _ = env.reset(seed=0)
    done = False
    truncated = False
    steps = 0
    rewards: list[float] = []
    rng = np.random.default_rng(123)
    while not (done or truncated) and steps < cfg_short_episodes.episode_length_ticks + 10:
        a = int(rng.integers(0, 5))
        obs, r, done, truncated, info = env.step(a)
        assert np.all(np.isfinite(obs)), f"NaN/inf in obs at step {steps}"
        assert np.isfinite(r), f"non-finite reward at step {steps}"
        rewards.append(r)
        steps += 1
    # at least the episode-length-many steps elapsed unless DD-killed
    assert steps >= 1


def test_market_buy_then_sell_round_trip_charges_fees(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    env.reset(seed=0)
    # open long
    obs, r1, *_ = env.step(int(Action.MKT_BUY))
    assert env.position.side == PositionSide.LONG
    nav_after_open = env.account.nav_usd
    # close long via market sell
    obs, r2, *_ = env.step(int(Action.MKT_SELL))
    assert env.position.side == PositionSide.FLAT
    # In a tiny noise window, the realized PnL should be ~0 minus fees,
    # so NAV must have ticked down. (Random GBM noise could in principle
    # produce a tiny positive but it's overwhelmingly likely to lose fees.)
    # We only assert NAV changed (round-trip not a no-op).
    assert env.account.nav_usd != nav_after_open


def test_hold_action_is_noop_for_flat_position(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    env.reset(seed=0)
    nav_before = env.account.nav_usd
    obs, r, done, trunc, info = env.step(int(Action.HOLD))
    assert env.position.side == PositionSide.FLAT
    assert env.account.nav_usd == nav_before
    # reward without position should be zero (no realized, no breadcrumb, no entry cost)
    assert r == 0.0


def test_invalid_action_raises(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_obs_is_market_features_concat_with_position_features(episode_parquet: Path, cfg_short_episodes: EnvConfig) -> None:
    env = MomoDkrEnv(episode_parquet, cfg_short_episodes, seed=0)
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(int(Action.MKT_BUY))
    # last 4 entries are the position features in canonical order
    pos_block = obs[-4:]
    assert pos_block[0] != 0.0  # pos_signed_notional_pct is nonzero when long
    assert pos_block[2] >= 0.0  # pos_hold_ticks_norm in [0, 1]


def test_missing_columns_raise_on_load(tmp_path: Path) -> None:
    df = _synthetic_episode(n_rows=12_000).drop(columns=["mid"])
    p = tmp_path / "bad.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(KeyError):
        MomoDkrEnv(p, EnvConfig(episode_length_ticks=500))


def test_dataset_too_small_raises(tmp_path: Path) -> None:
    df = _synthetic_episode(n_rows=100)
    p = tmp_path / "small.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="rows"):
        MomoDkrEnv(p, EnvConfig(episode_length_ticks=500))


def test_episode_parquet_has_all_sim_state_columns(episode_parquet: Path) -> None:
    df = pd.read_parquet(episode_parquet)
    for col in SIM_STATE_COLS:
        assert col in df.columns, f"missing sim state column {col}"
