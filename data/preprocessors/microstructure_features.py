"""Microstructure feature aggregations over rolling 100ms-grid windows.

All windows look ONLY backward (no look-ahead bias). For a snapshot at
tick i, the window of length N ticks is [i-N+1, i] inclusive. The first
N-1 rows of every rolling feature are NaN, which the caller drops or
fills before the env consumes them.

Units assume a 100ms grid (1s = 10 ticks, 5s = 50 ticks, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICKS_PER_SECOND = 10

OFI_WINDOWS_TICKS = {
    "ofi_1s": 1 * TICKS_PER_SECOND,
    "ofi_5s": 5 * TICKS_PER_SECOND,
    "ofi_30s": 30 * TICKS_PER_SECOND,
}

TRADE_FLOW_WINDOWS_TICKS = {
    "trade_flow_imb_1s": 1 * TICKS_PER_SECOND,
    "trade_flow_imb_5s": 5 * TICKS_PER_SECOND,
    "trade_flow_imb_30s": 30 * TICKS_PER_SECOND,
}


def rolling_ofi(snapshots: pd.DataFrame, window_ticks: int) -> pd.Series:
    """Sum of signed trade flow (ofi_100ms) over the last `window_ticks` 100ms buckets.

    Result is in base-asset units; positive = net buying pressure.
    """
    if "ofi_100ms" not in snapshots.columns:
        raise KeyError("snapshots must include 'ofi_100ms' (produced by the reconstructor)")
    return snapshots["ofi_100ms"].rolling(window=window_ticks, min_periods=window_ticks).sum()


def rolling_trade_flow_imbalance(snapshots: pd.DataFrame, window_ticks: int) -> pd.Series:
    """Signed volume normalised by absolute volume over the window.

    Returns a value in [-1, 1]: +1 means every trade in the window was a taker buy.
    Returns 0 when abs_volume is 0 (no trades).
    """
    for col in ("signed_volume_100ms", "abs_volume_100ms"):
        if col not in snapshots.columns:
            raise KeyError(f"snapshots must include '{col}' (produced by the reconstructor)")
    signed = snapshots["signed_volume_100ms"].rolling(window=window_ticks, min_periods=window_ticks).sum()
    abs_vol = snapshots["abs_volume_100ms"].rolling(window=window_ticks, min_periods=window_ticks).sum()
    safe = abs_vol.where(abs_vol > 0, np.nan)
    out = (signed / safe).fillna(0.0)
    out[abs_vol.isna()] = np.nan  # preserve warmup NaNs
    return out


def add_ofi_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    out = snapshots.copy()
    for name, w in OFI_WINDOWS_TICKS.items():
        out[name] = rolling_ofi(snapshots, w)
    return out


def add_trade_flow_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    out = snapshots.copy()
    for name, w in TRADE_FLOW_WINDOWS_TICKS.items():
        out[name] = rolling_trade_flow_imbalance(snapshots, w)
    return out
