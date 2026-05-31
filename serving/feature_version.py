"""Single source of truth for the observation feature contract.

Train and live inference MUST agree on:
  - Which features exist
  - What order they appear in the observation vector
  - What version of the feature spec they were computed with

This module is the gate against silent train/inference skew. Any change to
feature semantics requires bumping FEATURE_VERSION; the env, the ONNX
export, and the Rust feature_builder all check this constant at startup.
"""

from __future__ import annotations

import hashlib

FEATURE_VERSION = "0.1.0"


MARKET_FEATURE_NAMES: tuple[str, ...] = (
    # multi-window order flow imbalance (signed volume, base-asset units)
    "ofi_1s",
    "ofi_5s",
    "ofi_30s",
    # micro-price 1s log return
    "micro_price_log_ret",
    # snapshot-level quote shape
    "log_spread_bps",
    "top1_size_imbalance",
    # bookDepth at -0.1 / -0.2 / -0.5 / -1.0 percent levels (cumulative depth, base units)
    "bid_depth_pct_neg_0_1",
    "bid_depth_pct_neg_0_2",
    "bid_depth_pct_neg_0_5",
    "bid_depth_pct_neg_1_0",
    # bookDepth at +0.1 / +0.2 / +0.5 / +1.0 percent levels
    "ask_depth_pct_pos_0_1",
    "ask_depth_pct_pos_0_2",
    "ask_depth_pct_pos_0_5",
    "ask_depth_pct_pos_1_0",
    # realized vol of micro-price log returns over rolling windows
    "realized_vol_5s",
    "realized_vol_30s",
    "realized_vol_5min",
    # trade flow imbalance normalised by abs(volume) per window
    "trade_flow_imb_1s",
    "trade_flow_imb_5s",
    "trade_flow_imb_30s",
    # rolling 24h cumulative funding (sign of recent funding regime)
    "funding_cumulative",
    # current 8h funding rate at this snapshot
    "funding_8h_rate",
    # time-of-day / day-of-week cyclical encodings
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)

POSITION_FEATURE_NAMES: tuple[str, ...] = (
    "pos_signed_notional_pct",
    "pos_unrealized_pnl_pct",
    "pos_hold_ticks_norm",
    "pos_peak_unrealized_pct",
)

ALL_FEATURE_NAMES: tuple[str, ...] = MARKET_FEATURE_NAMES + POSITION_FEATURE_NAMES

MARKET_FEATURE_DIM = len(MARKET_FEATURE_NAMES)
POSITION_FEATURE_DIM = len(POSITION_FEATURE_NAMES)
OBS_DIM = len(ALL_FEATURE_NAMES)

assert MARKET_FEATURE_DIM == 26, f"expected 26 market features, got {MARKET_FEATURE_DIM}"
assert POSITION_FEATURE_DIM == 4, f"expected 4 position features, got {POSITION_FEATURE_DIM}"
assert OBS_DIM == 30, f"expected 30-dim observation, got {OBS_DIM}"


# Simulator state columns preserved alongside features but NOT exposed as
# observations to the policy. The env reads them to simulate fills, fees,
# slippage, and funding accruals. Keeping them in the episode parquet
# avoids a separate snapshot join at training time.
SIM_STATE_COLS: tuple[str, ...] = (
    "mid",
    "bid_px",
    "ask_px",
    "abs_volume_100ms",
    "funding_rate",
)


def feature_spec_checksum() -> str:
    """Stable hash of (version, ordered feature names) for skew detection.

    Compare this at inference time against the value baked into the ONNX
    metadata; any mismatch means the feature contract drifted and you must
    retrain or fix the live feature_builder.
    """
    payload = FEATURE_VERSION + "|" + "|".join(ALL_FEATURE_NAMES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


FEATURE_SPEC_CHECKSUM = feature_spec_checksum()
