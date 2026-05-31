import numpy as np
import pytest

from envs.market_simulator import (
    SimulatorConfig,
    is_liquidated,
    limit_buy_fill,
    limit_sell_fill,
    market_buy_fill_price,
    market_sell_fill_price,
    per_tick_funding_pct,
    sqrt_kyle_slippage_pct,
)


def _cfg(**overrides) -> SimulatorConfig:
    base = dict(
        fee_taker_bps=3.5,
        fee_maker_bps=1.0,
        fee_noise_pct=0.0,
        slippage_c=0.001,
        slippage_noise_pct=0.0,
        latency_ticks_min=0,
        latency_ticks_max=0,
        walk_book=False,                 # unit-test legacy path
        trade_through_limit_fills=False,  # legacy mid-move heuristic
        limit_fill_threshold_bps=1.0,
        limit_max_age_ticks=10,
        funding_interval_ticks=288_000,
        liq_maintenance_margin=0.03,
        leverage=6,
        tick_size_by_symbol={},
        default_tick_size=0.0,
    )
    base.update(overrides)
    return SimulatorConfig(**base)


def test_sqrt_kyle_slippage_is_zero_when_no_volume() -> None:
    rng = np.random.default_rng(0)
    cfg = _cfg()
    assert sqrt_kyle_slippage_pct(notional_usd=1_000.0, recent_volume_usd=0.0, cfg=cfg, rng=rng) == 0.0


def test_sqrt_kyle_slippage_scales_with_sqrt_participation() -> None:
    rng = np.random.default_rng(0)
    cfg = _cfg(slippage_noise_pct=0.0)
    s1 = sqrt_kyle_slippage_pct(100.0, 10_000.0, cfg, rng)
    s2 = sqrt_kyle_slippage_pct(400.0, 10_000.0, cfg, rng)
    # 4x participation -> 2x slippage
    assert s2 == pytest.approx(2.0 * s1, rel=1e-6)


def test_market_buy_fills_above_or_at_mid_with_taker_fee() -> None:
    rng = np.random.default_rng(0)
    cfg = _cfg()
    fill, fee, slip = market_buy_fill_price(ask_px=100.5, mid_px=100.0, recent_volume_usd=1_000_000.0, notional_usd=1_000.0, cfg=cfg, rng=rng)
    assert fill >= 100.0
    assert fee == pytest.approx(cfg.fee_taker_bps / 10_000.0)
    assert slip >= 0.0


def test_market_sell_fills_below_or_at_mid_with_taker_fee() -> None:
    rng = np.random.default_rng(0)
    cfg = _cfg()
    fill, fee, slip = market_sell_fill_price(bid_px=99.5, mid_px=100.0, recent_volume_usd=1_000_000.0, notional_usd=1_000.0, cfg=cfg, rng=rng)
    assert fill <= 100.0
    assert fee == pytest.approx(cfg.fee_taker_bps / 10_000.0)


def test_limit_buy_fills_only_when_mid_drops_past_threshold() -> None:
    cfg = _cfg(limit_fill_threshold_bps=1.0)
    # 5bps drop -> fill
    filled, fill_px, fee = limit_buy_fill(prev_mid=100.0, next_mid=99.95, post_price=99.99, cfg=cfg)
    assert filled
    assert fill_px == pytest.approx(99.99)
    assert fee == pytest.approx(cfg.fee_maker_bps / 10_000.0)
    # 0.5bps drop -> no fill
    filled, _, _ = limit_buy_fill(prev_mid=100.0, next_mid=99.995, post_price=99.99, cfg=cfg)
    assert not filled
    # rising mid -> no fill
    filled, _, _ = limit_buy_fill(prev_mid=100.0, next_mid=100.05, post_price=99.99, cfg=cfg)
    assert not filled


def test_limit_sell_fills_only_when_mid_rises_past_threshold() -> None:
    cfg = _cfg(limit_fill_threshold_bps=1.0)
    filled, fill_px, fee = limit_sell_fill(prev_mid=100.0, next_mid=100.05, post_price=100.01, cfg=cfg)
    assert filled
    assert fill_px == pytest.approx(100.01)
    assert fee == pytest.approx(cfg.fee_maker_bps / 10_000.0)
    filled, _, _ = limit_sell_fill(prev_mid=100.0, next_mid=100.005, post_price=100.01, cfg=cfg)
    assert not filled


def test_per_tick_funding_proportional_to_8h_rate() -> None:
    cfg = _cfg(funding_interval_ticks=10)
    assert per_tick_funding_pct(funding_rate_8h=0.001, cfg=cfg) == pytest.approx(0.0001)
    assert per_tick_funding_pct(funding_rate_8h=0.0, cfg=cfg) == 0.0
    assert per_tick_funding_pct(funding_rate_8h=float("nan"), cfg=cfg) == 0.0


def test_liquidation_threshold_at_leverage_inverse_minus_mm() -> None:
    cfg = _cfg(leverage=6, liq_maintenance_margin=0.03)
    threshold = -(1.0 / 6) + 0.03  # ~ -0.1367
    assert not is_liquidated(threshold + 1e-6, cfg)
    assert is_liquidated(threshold - 1e-6, cfg)
    # zero leverage -> never liquidated
    assert not is_liquidated(-1.0, _cfg(leverage=0))
