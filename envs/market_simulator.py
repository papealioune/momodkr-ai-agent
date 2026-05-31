"""Tick-level market simulator: fills, fees, sqrt-Kyle impact, latency, funding.

Used by the env to translate (action, current snapshot, position state)
into (fill_price_or_None, fee_pct, slippage_pct) plus a per-tick funding
accrual. Pure functions where possible -- the env owns state.

Pricing conventions:
  - Market buy fills at ask + slippage; market sell fills at bid - slippage.
  - Limit buy posts at best_bid; fills if mid moves down >= 1bp next tick
    (approximation -- real queue dynamics are out of scope for v1).
  - Limit sell posts at best_ask; fills if mid moves up >= 1bp next tick.
  - Maker fills get a lower fee tier.

Domain randomization (moleapp lesson 3.4):
  - Fee noise +/- 5%
  - Slippage coefficient noise +/- 20%
  - Latency uniform in [latency_bps_min, latency_bps_max]

All randomization is driven by an injected np.random.Generator so episodes
remain seedable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FUNDING_INTERVAL_TICKS_DEFAULT = 288_000  # 8h at 100ms grid


@dataclass(frozen=True)
class SimulatorConfig:
    fee_taker_bps: float = 3.5
    fee_maker_bps: float = 1.0
    slippage_c: float = 0.001            # sqrt-Kyle coefficient
    latency_bps_min: float = 5.0
    latency_bps_max: float = 20.0
    fee_noise_pct: float = 0.05
    slippage_noise_pct: float = 0.20
    funding_interval_ticks: int = FUNDING_INTERVAL_TICKS_DEFAULT
    limit_fill_threshold_bps: float = 1.0     # mid-move threshold for limit fills
    limit_max_age_ticks: int = 10              # cancel resting limits after this many ticks
    liq_maintenance_margin: float = 0.03      # 3% maintenance margin
    leverage: int = 6


def _noisy(rng: np.random.Generator, base: float, noise_pct: float) -> float:
    return base * (1.0 + noise_pct * float(rng.standard_normal()))


def _latency_bps(rng: np.random.Generator, cfg: SimulatorConfig) -> float:
    return float(rng.uniform(cfg.latency_bps_min, cfg.latency_bps_max))


def sqrt_kyle_slippage_pct(notional_usd: float, recent_volume_usd: float, cfg: SimulatorConfig, rng: np.random.Generator) -> float:
    """Permanent market impact ~ c * sqrt(participation). Returns positive pct (e.g. 0.0005 = 5bps).

    Returns 0 if recent volume is non-positive (degenerate input).
    """
    if recent_volume_usd <= 0:
        return 0.0
    part = max(notional_usd / recent_volume_usd, 0.0)
    base = cfg.slippage_c * float(np.sqrt(part))
    return max(_noisy(rng, base, cfg.slippage_noise_pct), 0.0)


def market_buy_fill_price(ask_px: float, mid_px: float, recent_volume_usd: float, notional_usd: float, cfg: SimulatorConfig, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (fill_price, fee_pct, slippage_pct). Slippage and latency both push price up against the taker."""
    slip = sqrt_kyle_slippage_pct(notional_usd, recent_volume_usd, cfg, rng)
    lat_pct = _latency_bps(rng, cfg) / 10_000.0
    fee = _noisy(rng, cfg.fee_taker_bps / 10_000.0, cfg.fee_noise_pct)
    fill = ask_px * (1.0 + slip + lat_pct)
    if fill < mid_px:
        fill = mid_px
    return float(fill), float(fee), float(slip)


def market_sell_fill_price(bid_px: float, mid_px: float, recent_volume_usd: float, notional_usd: float, cfg: SimulatorConfig, rng: np.random.Generator) -> tuple[float, float, float]:
    slip = sqrt_kyle_slippage_pct(notional_usd, recent_volume_usd, cfg, rng)
    lat_pct = _latency_bps(rng, cfg) / 10_000.0
    fee = _noisy(rng, cfg.fee_taker_bps / 10_000.0, cfg.fee_noise_pct)
    fill = bid_px * (1.0 - slip - lat_pct)
    if fill > mid_px:
        fill = mid_px
    return float(fill), float(fee), float(slip)


def limit_buy_fill(prev_mid: float, next_mid: float, post_price: float, cfg: SimulatorConfig) -> tuple[bool, float, float]:
    """A resting bid fills (we BUY) if next tick mid drops >= threshold below previous mid.

    Returns (filled, fill_price, fee_pct). When filled, fee is maker tier.
    """
    if prev_mid <= 0:
        return False, 0.0, 0.0
    drop_bps = (prev_mid - next_mid) / prev_mid * 10_000.0
    if drop_bps >= cfg.limit_fill_threshold_bps:
        return True, post_price, cfg.fee_maker_bps / 10_000.0
    return False, 0.0, 0.0


def limit_sell_fill(prev_mid: float, next_mid: float, post_price: float, cfg: SimulatorConfig) -> tuple[bool, float, float]:
    """A resting ask fills (we SELL) if next tick mid rises >= threshold above previous mid."""
    if prev_mid <= 0:
        return False, 0.0, 0.0
    rise_bps = (next_mid - prev_mid) / prev_mid * 10_000.0
    if rise_bps >= cfg.limit_fill_threshold_bps:
        return True, post_price, cfg.fee_maker_bps / 10_000.0
    return False, 0.0, 0.0


def per_tick_funding_pct(funding_rate_8h: float, cfg: SimulatorConfig) -> float:
    """Linear interpolation of the 8h funding rate over the funding interval ticks."""
    if not np.isfinite(funding_rate_8h) or cfg.funding_interval_ticks <= 0:
        return 0.0
    return float(funding_rate_8h) / float(cfg.funding_interval_ticks)


def is_liquidated(unrealized_pnl_pct_on_notional: float, cfg: SimulatorConfig) -> bool:
    """At leverage L, account is wiped when unrealized PnL on notional reaches -(1/L) + maintenance margin."""
    if cfg.leverage <= 0:
        return False
    threshold = -(1.0 / cfg.leverage) + cfg.liq_maintenance_margin
    return unrealized_pnl_pct_on_notional <= threshold
