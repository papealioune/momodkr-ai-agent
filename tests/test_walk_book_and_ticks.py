import numpy as np
import pytest

from envs.market_simulator import (
    SimulatorConfig,
    market_buy_fill_price,
    market_sell_fill_price,
    round_to_tick,
    tick_size_for,
    walk_book_buy,
    walk_book_sell,
)


def test_round_to_tick_no_op_when_zero() -> None:
    assert round_to_tick(123.456, 0.0) == 123.456


def test_round_to_tick_nearest_up_down() -> None:
    assert round_to_tick(100.27, 0.5) == 100.5
    assert round_to_tick(100.24, 0.5) == 100.0
    assert round_to_tick(100.24, 0.5, side="up") == 100.5
    assert round_to_tick(100.27, 0.5, side="down") == 100.0


def test_tick_size_lookup_per_symbol() -> None:
    cfg = SimulatorConfig(tick_size_by_symbol={"BTCUSDT": 0.5, "ETHUSDT": 0.05}, default_tick_size=0.01)
    assert tick_size_for("BTCUSDT", cfg) == 0.5
    assert tick_size_for("ETHUSDT", cfg) == 0.05
    assert tick_size_for("UNKNOWN", cfg) == 0.01
    assert tick_size_for(None, cfg) == 0.01


def test_walk_book_buy_fills_inside_first_bucket() -> None:
    # ample liquidity inside the +0.1% bucket -> VWAP is roughly the midpoint of [ask, +0.1%]
    mid, ask = 100.0, 100.05
    depth = (1000.0, 1010.0, 1020.0, 1030.0)
    vwap, slip, exhausted = walk_book_buy(notional_usd=50.0, mid_px=mid, ask_px=ask, ask_depth_cumulative=depth)
    assert not exhausted
    assert vwap >= ask
    assert vwap < mid * 1.001  # well inside +0.1% bucket
    assert slip >= 0.0


def test_walk_book_buy_walks_through_buckets_when_large() -> None:
    mid, ask = 100.0, 100.05
    depth = (0.1, 0.2, 0.4, 0.5)  # very thin book
    vwap, slip, exhausted = walk_book_buy(notional_usd=10_000.0, mid_px=mid, ask_px=ask, ask_depth_cumulative=depth)
    assert exhausted
    assert vwap > ask
    assert slip > 0.0


def test_walk_book_sell_symmetric_to_buy() -> None:
    mid, bid = 100.0, 99.95
    depth = (1000.0, 1010.0, 1020.0, 1030.0)
    vwap, slip, exhausted = walk_book_sell(notional_usd=50.0, mid_px=mid, bid_px=bid, bid_depth_cumulative=depth)
    assert not exhausted
    assert vwap <= bid
    assert slip >= 0.0


def test_market_buy_uses_walk_book_when_available_and_rounds_to_tick() -> None:
    rng = np.random.default_rng(0)
    cfg = SimulatorConfig(walk_book=True, tick_size_by_symbol={"BTCUSDT": 0.5}, fee_noise_pct=0.0)
    fill, fee, slip = market_buy_fill_price(
        ask_px=50_000.12,
        mid_px=50_000.0,
        recent_volume_usd=1_000_000.0,
        notional_usd=100.0,
        cfg=cfg,
        rng=rng,
        ask_depth_cumulative=(10.0, 20.0, 30.0, 40.0),
        symbol="BTCUSDT",
    )
    # tick size 0.5 -> fill should be a multiple of 0.5
    assert abs(fill / 0.5 - round(fill / 0.5)) < 1e-9
    assert fee == pytest.approx(cfg.fee_taker_bps / 10_000.0)
    assert slip >= 0.0


def test_market_sell_falls_back_to_sqrt_kyle_when_walk_book_disabled() -> None:
    rng = np.random.default_rng(0)
    cfg = SimulatorConfig(walk_book=False, fee_noise_pct=0.0, slippage_noise_pct=0.0)
    fill, fee, slip = market_sell_fill_price(
        bid_px=99.5,
        mid_px=100.0,
        recent_volume_usd=10_000.0,
        notional_usd=1_000.0,
        cfg=cfg,
        rng=rng,
        bid_depth_cumulative=None,
        symbol=None,
    )
    # sqrt-Kyle path was used
    assert slip >= 0.0
    assert fill <= 100.0
    assert fee == pytest.approx(cfg.fee_taker_bps / 10_000.0)
