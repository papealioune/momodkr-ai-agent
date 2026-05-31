"""Momentum + realized-volatility features over rolling 100ms-grid windows.

Backward-looking only. The micro-price return uses a 1s lookback (10 ticks)
to reduce noise vs single-tick returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICKS_PER_SECOND = 10

MICRO_PRICE_RET_WINDOW_TICKS = 1 * TICKS_PER_SECOND  # 1s log return

REALIZED_VOL_WINDOWS_TICKS = {
    "realized_vol_5s": 5 * TICKS_PER_SECOND,
    "realized_vol_30s": 30 * TICKS_PER_SECOND,
    "realized_vol_5min": 5 * 60 * TICKS_PER_SECOND,
}


def micro_price_log_return(snapshots: pd.DataFrame, window_ticks: int = MICRO_PRICE_RET_WINDOW_TICKS) -> pd.Series:
    """log(micro_price[t] / micro_price[t - window_ticks])."""
    if "micro_price" not in snapshots.columns:
        raise KeyError("snapshots must include 'micro_price'")
    p = snapshots["micro_price"].where(snapshots["micro_price"] > 0, np.nan)
    return np.log(p / p.shift(window_ticks))


def realized_volatility(snapshots: pd.DataFrame, window_ticks: int, tick_log_ret: pd.Series | None = None) -> pd.Series:
    """sqrt(sum_{i in window} r_i^2) where r_i is the per-tick micro-price log return.

    Annualisation is left to the env; this returns raw RV in the same units
    as per-tick returns.
    """
    if tick_log_ret is None:
        if "micro_price" not in snapshots.columns:
            raise KeyError("snapshots must include 'micro_price'")
        p = snapshots["micro_price"].where(snapshots["micro_price"] > 0, np.nan)
        tick_log_ret = np.log(p / p.shift(1))
    sq = tick_log_ret.pow(2)
    return np.sqrt(sq.rolling(window=window_ticks, min_periods=window_ticks).sum())


def add_momentum_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    out = snapshots.copy()
    out["micro_price_log_ret"] = micro_price_log_return(snapshots)

    p = snapshots["micro_price"].where(snapshots["micro_price"] > 0, np.nan)
    tick_log = np.log(p / p.shift(1))
    for name, w in REALIZED_VOL_WINDOWS_TICKS.items():
        out[name] = realized_volatility(snapshots, w, tick_log_ret=tick_log)
    return out
