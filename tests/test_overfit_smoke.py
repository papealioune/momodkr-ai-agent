"""Phase 3 gate: deterministic policy on a strongly-trending market produces
positive cumulative reward + positive realized PnL.

This is the moleapp-style overfit smoke. It does NOT use PPO -- the goal
is to prove the env's reward function is oriented correctly. If always-long
on a clearly bullish market loses money, the carrot/penalty math is wrong
and no training will fix it.

The full PPO overfit smoke (3-seed, Mann-Whitney) belongs to Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from envs.base_hft_env import Action, PositionSide
from envs.market_simulator import SimulatorConfig
from envs.momodkr_env import EnvConfig, MomoDkrEnv
from serving.feature_version import MARKET_FEATURE_NAMES


def _instant_cfg(episode_length_ticks: int) -> EnvConfig:
    sim = SimulatorConfig(
        latency_ticks_min=0,
        latency_ticks_max=0,
        walk_book=False,
        trade_through_limit_fills=False,
        fee_noise_pct=0.0,
        slippage_noise_pct=0.0,
        tick_size_by_symbol={},
        default_tick_size=0.0,
    )
    return EnvConfig(episode_length_ticks=episode_length_ticks, sim=sim)


def _trending_episode(n: int = 20_000, drift_bps_per_tick: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic episode with a deterministic positive log-drift on top of small noise.

    1 bp/tick at 100ms = ~36 bps/sec = ~2.16% / minute -> any reasonable hold is
    overwhelmingly positive even with fees and noise.
    """
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 100
    log_ret = rng.standard_normal(n) * 1e-5 + drift_bps_per_tick / 10_000.0
    mid = 50_000.0 * np.exp(np.cumsum(log_ret))
    half_spread = 1.5
    df = pd.DataFrame({"ts_ms": ts})
    for col in MARKET_FEATURE_NAMES:
        df[col] = rng.standard_normal(n).astype(np.float32) * 0.1
    df["mid"] = mid.astype(np.float32)
    df["bid_px"] = (mid - half_spread).astype(np.float32)
    df["ask_px"] = (mid + half_spread).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n, 100.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    return df


@pytest.fixture
def bullish_episode(tmp_path: Path) -> Path:
    p = tmp_path / "bullish.parquet"
    _trending_episode().to_parquet(p, index=False)
    return p


def _run_always_long(env: MomoDkrEnv) -> tuple[float, float]:
    """Open long on first step, hold until truncation, return (cum_reward, final_nav)."""
    env.reset(seed=0)
    _, r, terminated, truncated, _ = env.step(int(Action.MKT_BUY))
    cum_reward = r
    while not (terminated or truncated):
        _, r, terminated, truncated, _ = env.step(int(Action.HOLD))
        cum_reward += r
    if env.position.side != PositionSide.FLAT:
        _, r, *_ = env.step(int(Action.MKT_SELL))
        cum_reward += r
    return cum_reward, env.account.nav_usd


def test_overfit_smoke_always_long_on_bullish_drift_is_profitable(bullish_episode: Path) -> None:
    env = MomoDkrEnv(bullish_episode, _instant_cfg(5_000))
    cum_reward, final_nav = _run_always_long(env)
    assert cum_reward > 0, f"carrot reward never went positive: cum={cum_reward}"
    assert final_nav > env.config.initial_nav_usd, f"NAV did not grow: {final_nav} vs initial {env.config.initial_nav_usd}"


def test_overfit_smoke_drawdown_kill_doesnt_fire_on_bullish_drift(bullish_episode: Path) -> None:
    """Strong uptrend + always-long should never trip the 5% DD kill."""
    env = MomoDkrEnv(bullish_episode, _instant_cfg(5_000))
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(int(Action.MKT_BUY))
    while not terminated:
        _, _, terminated, truncated, info = env.step(int(Action.HOLD))
        if truncated:
            break
    assert info.get("reason") != "drawdown_kill"


def test_overfit_smoke_always_short_on_bullish_drift_loses_money(bullish_episode: Path) -> None:
    """Sanity check the other side: shorting a bullish market should bleed."""
    env = MomoDkrEnv(bullish_episode, _instant_cfg(2_000))
    env.reset(seed=0)
    _, r, terminated, truncated, _ = env.step(int(Action.MKT_SELL))
    cum_reward = r
    while not (terminated or truncated):
        _, r, terminated, truncated, _ = env.step(int(Action.HOLD))
        cum_reward += r
    # either DD-killed (terminated) or NAV down -- either way carrot is negative or floored
    assert env.account.nav_usd < env.config.initial_nav_usd or terminated
