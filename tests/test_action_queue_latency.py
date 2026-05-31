from pathlib import Path

import numpy as np
import pandas as pd

from envs.base_hft_env import Action, ActionQueue, PositionSide
from envs.market_simulator import SimulatorConfig, draw_latency_ticks
from envs.momodkr_env import EnvConfig, MomoDkrEnv
from serving.feature_version import MARKET_FEATURE_NAMES


def test_draw_latency_returns_zero_when_max_is_zero() -> None:
    rng = np.random.default_rng(0)
    cfg = SimulatorConfig(latency_ticks_min=0, latency_ticks_max=0)
    for _ in range(20):
        assert draw_latency_ticks(rng, cfg) == 0


def test_draw_latency_returns_bounded_integer() -> None:
    rng = np.random.default_rng(0)
    cfg = SimulatorConfig(latency_ticks_min=2, latency_ticks_max=5)
    for _ in range(200):
        lat = draw_latency_ticks(rng, cfg)
        assert 2 <= lat <= 5


def test_action_queue_pops_ready_in_order() -> None:
    q = ActionQueue()
    q.push(action=1, execute_at_tick=10, issued_at_tick=5)
    q.push(action=2, execute_at_tick=12, issued_at_tick=7)
    assert q.pop_ready(current_tick=9) == []
    ready = q.pop_ready(current_tick=10)
    assert len(ready) == 1
    assert ready[0].action == 1
    ready = q.pop_ready(current_tick=15)
    assert len(ready) == 1
    assert ready[0].action == 2


def test_action_queue_cancel_all_increments_counter() -> None:
    q = ActionQueue()
    q.push(action=1, execute_at_tick=5, issued_at_tick=0)
    q.push(action=2, execute_at_tick=7, issued_at_tick=1)
    assert q.cancel_all() == 2
    assert q.cancellations == 2
    assert q.n_in_flight == 0


def _episode(n: int = 1000, base_mid: float = 50_000.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 100
    mid = np.full(n, base_mid, dtype=np.float32)
    df = pd.DataFrame({"ts_ms": ts})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32) * 0.1
    df["mid"] = mid
    df["bid_px"] = (mid - 1.5).astype(np.float32)
    df["ask_px"] = (mid + 1.5).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n, 100.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    return df


def test_env_market_buy_delayed_by_latency_ticks(tmp_path: Path) -> None:
    """With latency=3, the buy executes 3 HOLD-steps after the issuing step.

    Step 0: agent sends MKT_BUY at cursor T. Action enqueued with
            execute_at_tick=T+3. Position still flat at end of step.
    Step 1: HOLD, cursor=T+1, not ready.
    Step 2: HOLD, cursor=T+2, not ready.
    Step 3: HOLD, cursor=T+3, pop_ready fires -> position opens.
    """
    p = tmp_path / "ep.parquet"
    _episode(n=1000).to_parquet(p, index=False)
    sim = SimulatorConfig(latency_ticks_min=3, latency_ticks_max=3, walk_book=False, trade_through_limit_fills=False, fee_noise_pct=0.0, slippage_noise_pct=0.0, tick_size_by_symbol={}, default_tick_size=0.0)
    env = MomoDkrEnv(p, EnvConfig(episode_length_ticks=200, sim=sim), seed=0)
    env.reset(seed=0)
    env.step(int(Action.MKT_BUY))
    assert env.position.side == PositionSide.FLAT
    env.step(int(Action.HOLD))
    assert env.position.side == PositionSide.FLAT
    env.step(int(Action.HOLD))
    assert env.position.side == PositionSide.FLAT
    env.step(int(Action.HOLD))
    assert env.position.side == PositionSide.LONG


def test_env_zero_latency_executes_buy_immediately(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _episode(n=1000).to_parquet(p, index=False)
    sim = SimulatorConfig(latency_ticks_min=0, latency_ticks_max=0, walk_book=False, trade_through_limit_fills=False, fee_noise_pct=0.0, slippage_noise_pct=0.0, tick_size_by_symbol={}, default_tick_size=0.0)
    env = MomoDkrEnv(p, EnvConfig(episode_length_ticks=200, sim=sim), seed=0)
    env.reset(seed=0)
    env.step(int(Action.MKT_BUY))
    assert env.position.side == PositionSide.LONG


def test_env_records_cancellations_in_info(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _episode(n=1000).to_parquet(p, index=False)
    sim = SimulatorConfig(latency_ticks_min=5, latency_ticks_max=5, walk_book=False, trade_through_limit_fills=False, fee_noise_pct=0.0, slippage_noise_pct=0.0, tick_size_by_symbol={}, default_tick_size=0.0)
    env = MomoDkrEnv(p, EnvConfig(episode_length_ticks=200, sim=sim), seed=0)
    env.reset(seed=0)
    env.step(int(Action.MKT_BUY))    # in-flight
    _, _, _, _, info = env.step(int(Action.MKT_SELL))  # cancels prior buy + enqueues sell
    assert info["n_cancellations_this_step"] >= 1
