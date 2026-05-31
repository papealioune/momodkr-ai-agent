"""Tick-level market simulator: walk-the-book, trade-through limits, latency
queue, sqrt-Kyle fallback, fees, funding, liquidation.

Used by the env to translate (action, current snapshot, position state)
into (fill_price_or_None, fee_pct, slippage_pct) plus a per-tick funding
accrual. Pure functions where possible -- the env owns state.

Pricing conventions (defaults are HFT-realistic; opt out via config for
unit-test simplicity):
  - Market buy: walk the resting ask book (bookDepth percentage levels
    -> VWAP). Take fee at the Hyperliquid taker tier. Fill price rounded
    UP to the symbol's tick size.
  - Market sell: walk the resting bid book to VWAP. Maker/taker symmetry.
  - Limit buy at P: fills only if the next tick's best ASK trades through
    P (next_ask <= P). FIFO queue position is approximated by requiring
    the historical 100ms abs_volume at that level to exceed the size
    that was resting before our order arrived.
  - Limit sell at P: symmetric on the bid side.

Latency:
  - The env owns the ActionQueue (envs.base_hft_env); the simulator just
    exposes draw_latency_ticks() so each action can be assigned a
    discrete number of ticks of delay.

Domain randomization:
  - Fee noise +/- fee_noise_pct
  - Slippage fallback coefficient noise +/- slippage_noise_pct (only
    applies when bookDepth is unavailable)
  - Latency in [latency_ticks_min, latency_ticks_max] (uniform integer)

All randomization is driven by an injected np.random.Generator so
episodes remain seedable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FUNDING_INTERVAL_TICKS_DEFAULT = 288_000  # 8h at 100ms grid

# Cumulative bookDepth bucket midpoints (in % from mid) used by walk_book.
# These mirror serving.feature_version.SIM_STATE_COLS / order_book_reconstructor.
BID_DEPTH_PCT_LEVELS = (0.1, 0.2, 0.5, 1.0)
ASK_DEPTH_PCT_LEVELS = (0.1, 0.2, 0.5, 1.0)


# Hyperliquid live tick sizes (USD per tick) as of 2026-Q2. Updated as needed.
DEFAULT_HL_TICK_SIZES: dict[str, float] = {
    "BTCUSDT": 0.5,
    "ETHUSDT": 0.05,
    "SOLUSDT": 0.01,
}


@dataclass
class SimulatorConfig:
    # Fees (Hyperliquid live tiers)
    fee_taker_bps: float = 3.5
    fee_maker_bps: float = 1.0
    fee_noise_pct: float = 0.05

    # Sqrt-Kyle fallback slippage when bookDepth is unavailable
    slippage_c: float = 0.001
    slippage_noise_pct: float = 0.20

    # Latency injection (env's ActionQueue draws from this)
    latency_ticks_min: int = 1
    latency_ticks_max: int = 5
    # Whether the simulator should walk bookDepth %-levels for VWAP fills
    # when those columns are present in the snapshot. Falls back to
    # sqrt-Kyle if False or if bookDepth is missing.
    walk_book: bool = True

    # Limit fill semantics
    # True: require trade-through (next best ASK <= post_bid for a buy).
    # False: legacy mid-move heuristic (kept for unit-test simplicity).
    trade_through_limit_fills: bool = True
    limit_fill_threshold_bps: float = 1.0
    limit_max_age_ticks: int = 10

    # Funding + liquidation
    funding_interval_ticks: int = FUNDING_INTERVAL_TICKS_DEFAULT
    liq_maintenance_margin: float = 0.03
    leverage: int = 6

    # Per-symbol tick sizes used to round fill prices to a venue-realistic grid.
    tick_size_by_symbol: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HL_TICK_SIZES))
    default_tick_size: float = 0.0  # 0 = no rounding


def _noisy(rng: np.random.Generator, base: float, noise_pct: float) -> float:
    if noise_pct == 0.0:
        return base
    return base * (1.0 + noise_pct * float(rng.standard_normal()))


def draw_latency_ticks(rng: np.random.Generator, cfg: SimulatorConfig) -> int:
    if cfg.latency_ticks_max <= 0:
        return 0
    lo = max(0, int(cfg.latency_ticks_min))
    hi = max(lo, int(cfg.latency_ticks_max))
    return int(rng.integers(lo, hi + 1))


def round_to_tick(price: float, tick_size: float, side: str | None = None) -> float:
    """Round price to tick_size. side='up' rounds up (buyer), 'down' rounds down (seller).

    side=None rounds to nearest. side='up'/'down' shifts the agent into the
    realistic adverse-direction outcome.
    """
    if tick_size <= 0:
        return float(price)
    n = price / tick_size
    if side == "up":
        return float(np.ceil(n) * tick_size)
    if side == "down":
        return float(np.floor(n) * tick_size)
    return float(np.round(n) * tick_size)


def tick_size_for(symbol: str | None, cfg: SimulatorConfig) -> float:
    if symbol is None:
        return cfg.default_tick_size
    return cfg.tick_size_by_symbol.get(symbol, cfg.default_tick_size)


# ---------------------------------------------------------------------- walk-the-book

def _walk_levels(side: str, mid_px: float, top_px: float, level_pcts: tuple[float, ...]) -> list[tuple[float, float]]:
    """Compute (level_top_price, midpoint_price) pairs that bound each bookDepth bucket.

    For buys (side='ask'): walk above the best ask; bucket i covers
    [top_px, mid * (1 + pct_i / 100)]. We approximate the average fill
    price of the bucket as the midpoint of that range.
    """
    pairs: list[tuple[float, float]] = []
    prev_pct = 0.0
    for pct in level_pcts:
        if side == "ask":
            lo = mid_px * (1.0 + prev_pct / 100.0)
            hi = mid_px * (1.0 + pct / 100.0)
            lo = max(lo, top_px)
        else:
            lo = mid_px * (1.0 - pct / 100.0)
            hi = mid_px * (1.0 - prev_pct / 100.0)
            hi = min(hi, top_px)
        midpoint = 0.5 * (lo + hi)
        pairs.append((hi if side == "ask" else lo, midpoint))
        prev_pct = pct
    return pairs


def walk_book_buy(
    notional_usd: float,
    mid_px: float,
    ask_px: float,
    ask_depth_cumulative: tuple[float, float, float, float],
) -> tuple[float, float, bool]:
    """Walk the ask side from best_ask through %-buckets to fill notional_usd.

    `ask_depth_cumulative` is the cumulative base-asset depth at +0.1, +0.2,
    +0.5, +1.0 percent above mid (matching ASK_DEPTH_PCT_LEVELS).

    Returns (vwap_fill_price, slippage_pct_vs_mid, exhausted). `exhausted`
    is True if the requested notional exceeded the total visible book and
    the residual was filled at the worst (+1%) level price.
    """
    if notional_usd <= 0 or mid_px <= 0 or ask_px <= 0:
        return float(ask_px), 0.0, False

    size_target_base = notional_usd / mid_px
    if size_target_base <= 0:
        return float(ask_px), 0.0, False

    pairs = _walk_levels("ask", mid_px, ask_px, ASK_DEPTH_PCT_LEVELS)
    cum = list(ask_depth_cumulative)
    if len(cum) != len(pairs):
        return float(ask_px), 0.0, False

    bucket_sizes = [max(0.0, cum[0])]
    for i in range(1, len(cum)):
        bucket_sizes.append(max(0.0, cum[i] - cum[i - 1]))

    remaining = size_target_base
    weighted_price = 0.0
    exhausted = False
    last_top = ask_px
    for (top, mid_of_bucket), sz in zip(pairs, bucket_sizes, strict=True):
        if remaining <= 0:
            break
        take = min(remaining, sz)
        weighted_price += take * mid_of_bucket
        remaining -= take
        last_top = top
    if remaining > 0:
        # exhausted the visible 4-level book; bleed the rest at the worst-bucket top price.
        weighted_price += remaining * last_top
        exhausted = True

    vwap = weighted_price / size_target_base
    slippage_pct = max((vwap - mid_px) / mid_px, 0.0)
    return float(vwap), float(slippage_pct), exhausted


def walk_book_sell(
    notional_usd: float,
    mid_px: float,
    bid_px: float,
    bid_depth_cumulative: tuple[float, float, float, float],
) -> tuple[float, float, bool]:
    if notional_usd <= 0 or mid_px <= 0 or bid_px <= 0:
        return float(bid_px), 0.0, False
    size_target_base = notional_usd / mid_px
    if size_target_base <= 0:
        return float(bid_px), 0.0, False

    pairs = _walk_levels("bid", mid_px, bid_px, BID_DEPTH_PCT_LEVELS)
    cum = list(bid_depth_cumulative)
    if len(cum) != len(pairs):
        return float(bid_px), 0.0, False

    bucket_sizes = [max(0.0, cum[0])]
    for i in range(1, len(cum)):
        bucket_sizes.append(max(0.0, cum[i] - cum[i - 1]))

    remaining = size_target_base
    weighted_price = 0.0
    exhausted = False
    last_bottom = bid_px
    for (bottom, mid_of_bucket), sz in zip(pairs, bucket_sizes, strict=True):
        if remaining <= 0:
            break
        take = min(remaining, sz)
        weighted_price += take * mid_of_bucket
        remaining -= take
        last_bottom = bottom
    if remaining > 0:
        weighted_price += remaining * last_bottom
        exhausted = True

    vwap = weighted_price / size_target_base
    slippage_pct = max((mid_px - vwap) / mid_px, 0.0)
    return float(vwap), float(slippage_pct), exhausted


def sqrt_kyle_slippage_pct(
    notional_usd: float,
    recent_volume_usd: float,
    cfg: SimulatorConfig,
    rng: np.random.Generator,
) -> float:
    """Fallback when bookDepth is unavailable: c * sqrt(participation), +/- noise."""
    if recent_volume_usd <= 0:
        return 0.0
    part = max(notional_usd / recent_volume_usd, 0.0)
    base = cfg.slippage_c * float(np.sqrt(part))
    return max(_noisy(rng, base, cfg.slippage_noise_pct), 0.0)


def market_buy_fill_price(
    ask_px: float,
    mid_px: float,
    recent_volume_usd: float,
    notional_usd: float,
    cfg: SimulatorConfig,
    rng: np.random.Generator,
    ask_depth_cumulative: tuple[float, float, float, float] | None = None,
    symbol: str | None = None,
) -> tuple[float, float, float]:
    """Return (fill_price, fee_pct, slippage_pct). Walks the book when available."""
    if cfg.walk_book and ask_depth_cumulative is not None and all(d >= 0 for d in ask_depth_cumulative):
        fill, slip, _ = walk_book_buy(notional_usd, mid_px, ask_px, ask_depth_cumulative)
    else:
        slip = sqrt_kyle_slippage_pct(notional_usd, recent_volume_usd, cfg, rng)
        fill = ask_px * (1.0 + slip)
        if fill < mid_px:
            fill = mid_px
    fill = round_to_tick(fill, tick_size_for(symbol, cfg), side="up")
    fee = _noisy(rng, cfg.fee_taker_bps / 10_000.0, cfg.fee_noise_pct)
    return float(fill), float(fee), float(slip)


def market_sell_fill_price(
    bid_px: float,
    mid_px: float,
    recent_volume_usd: float,
    notional_usd: float,
    cfg: SimulatorConfig,
    rng: np.random.Generator,
    bid_depth_cumulative: tuple[float, float, float, float] | None = None,
    symbol: str | None = None,
) -> tuple[float, float, float]:
    if cfg.walk_book and bid_depth_cumulative is not None and all(d >= 0 for d in bid_depth_cumulative):
        fill, slip, _ = walk_book_sell(notional_usd, mid_px, bid_px, bid_depth_cumulative)
    else:
        slip = sqrt_kyle_slippage_pct(notional_usd, recent_volume_usd, cfg, rng)
        fill = bid_px * (1.0 - slip)
        if fill > mid_px:
            fill = mid_px
    fill = round_to_tick(fill, tick_size_for(symbol, cfg), side="down")
    fee = _noisy(rng, cfg.fee_taker_bps / 10_000.0, cfg.fee_noise_pct)
    return float(fill), float(fee), float(slip)


# ---------------------------------------------------------------------- limit fills

def limit_buy_fill(
    prev_mid: float,
    next_mid: float,
    post_price: float,
    cfg: SimulatorConfig,
    next_ask: float | None = None,
    resting_size_before_us: float | None = None,
    abs_volume_at_window: float | None = None,
) -> tuple[bool, float, float]:
    """Pessimistic FIFO limit-buy fill.

    Default (trade_through_limit_fills=True): a resting bid at post_price
    fills only if the next best ASK trades through to <= post_price AND
    the historical 100ms abs_volume in that window exceeds the resting
    size that was ahead of us in the queue.

    Legacy mode: mid moves >= limit_fill_threshold_bps through prev_mid.
    """
    if cfg.trade_through_limit_fills and next_ask is not None:
        if next_ask > post_price:
            return False, 0.0, 0.0
        if resting_size_before_us is not None and abs_volume_at_window is not None:
            if abs_volume_at_window <= resting_size_before_us:
                return False, 0.0, 0.0
        return True, float(post_price), cfg.fee_maker_bps / 10_000.0

    if prev_mid <= 0:
        return False, 0.0, 0.0
    drop_bps = (prev_mid - next_mid) / prev_mid * 10_000.0
    if drop_bps >= cfg.limit_fill_threshold_bps:
        return True, float(post_price), cfg.fee_maker_bps / 10_000.0
    return False, 0.0, 0.0


def limit_sell_fill(
    prev_mid: float,
    next_mid: float,
    post_price: float,
    cfg: SimulatorConfig,
    next_bid: float | None = None,
    resting_size_before_us: float | None = None,
    abs_volume_at_window: float | None = None,
) -> tuple[bool, float, float]:
    if cfg.trade_through_limit_fills and next_bid is not None:
        if next_bid < post_price:
            return False, 0.0, 0.0
        if resting_size_before_us is not None and abs_volume_at_window is not None:
            if abs_volume_at_window <= resting_size_before_us:
                return False, 0.0, 0.0
        return True, float(post_price), cfg.fee_maker_bps / 10_000.0

    if prev_mid <= 0:
        return False, 0.0, 0.0
    rise_bps = (next_mid - prev_mid) / prev_mid * 10_000.0
    if rise_bps >= cfg.limit_fill_threshold_bps:
        return True, float(post_price), cfg.fee_maker_bps / 10_000.0
    return False, 0.0, 0.0


# ---------------------------------------------------------------------- funding + liquidation

def per_tick_funding_pct(funding_rate_8h: float, cfg: SimulatorConfig) -> float:
    if not np.isfinite(funding_rate_8h) or cfg.funding_interval_ticks <= 0:
        return 0.0
    return float(funding_rate_8h) / float(cfg.funding_interval_ticks)


def is_liquidated(unrealized_pnl_pct_on_notional: float, cfg: SimulatorConfig) -> bool:
    if cfg.leverage <= 0:
        return False
    threshold = -(1.0 / cfg.leverage) + cfg.liq_maintenance_margin
    return unrealized_pnl_pct_on_notional <= threshold
