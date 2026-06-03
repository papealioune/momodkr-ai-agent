"""End-to-end feature engineering: snapshots -> (T, 26) market-feature matrix.

The 4 position features (pos_*) are NOT computed here -- the env adds them
live based on the agent's state. This module emits only the 26 market
features defined in serving.feature_version.MARKET_FEATURE_NAMES, in that
exact order.

Inputs (per symbol, per day):
  data/datasets/<SYMBOL>/snapshots/<YYYY-MM-DD>.parquet
Outputs:
  data/datasets/<SYMBOL>/features/<YYYY-MM-DD>.parquet
    schema: ts_ms (int64), <26 market features as float32>, feature_version (str)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessors.microstructure_features import add_ofi_features, add_trade_flow_features
from data.preprocessors.momentum_features import TICKS_PER_SECOND, add_momentum_features
from serving.feature_version import FEATURE_VERSION, MARKET_FEATURE_NAMES, SIM_STATE_COLS

logger = logging.getLogger(__name__)

FUNDING_CUM_WINDOW_TICKS = 24 * 60 * 60 * TICKS_PER_SECOND  # rolling 24h


def _cyclical_time_features(ts_ms: pd.Series) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(ts_ms.astype("int64"), unit="ms", utc=True)
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    dow = ts.dt.dayofweek.to_numpy()  # Mon=0..Sun=6
    two_pi = 2.0 * np.pi
    return {
        "hour_of_day_sin": np.sin(two_pi * hour / 24.0),
        "hour_of_day_cos": np.cos(two_pi * hour / 24.0),
        "day_of_week_sin": np.sin(two_pi * dow / 7.0),
        "day_of_week_cos": np.cos(two_pi * dow / 7.0),
    }


def _funding_features(snapshots: pd.DataFrame) -> dict[str, pd.Series]:
    if "funding_rate" not in snapshots.columns:
        rate = pd.Series(np.nan, index=snapshots.index, name="funding_rate")
    else:
        rate = snapshots["funding_rate"].astype("float64")
    # ffill so the agent always sees the most-recent published rate; leading
    # NaNs (before the first publish) stay NaN and get dropped downstream
    rate_ff = rate.ffill()
    cum_24h = rate_ff.rolling(window=FUNDING_CUM_WINDOW_TICKS, min_periods=1).sum()
    return {
        "funding_8h_rate": rate_ff,
        "funding_cumulative": cum_24h,
    }


def build_market_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Produce a DataFrame with `ts_ms` + the 26 market features in canonical order.

    Rolling features require warmup; rows before warmup completes are dropped.
    """
    df = add_ofi_features(snapshots)
    df = add_trade_flow_features(df)
    df = add_momentum_features(df)

    for k, v in _funding_features(df).items():
        df[k] = v
    for k, v in _cyclical_time_features(df["ts_ms"]).items():
        df[k] = v

    # Days where Binance Vision didn't archive bookDepth (a small minority) end up
    # with no depth columns in the snapshot. Treat that as "no depth visible" (0.0)
    # rather than failing the whole day -- 0 is a valid "thin book" signal the
    # policy can learn from, and tolerating it loses < 0.2% of the dataset to
    # known archive gaps instead of dropping every day with any missing column.
    # Non-depth missing columns still hard-fail (they'd be a real bug).
    depth_prefixes = ("bid_depth_pct_", "ask_depth_pct_")
    missing = [c for c in MARKET_FEATURE_NAMES if c not in df.columns]
    non_depth_missing = [c for c in missing if not c.startswith(depth_prefixes)]
    if non_depth_missing:
        raise KeyError(f"feature engineer did not produce: {non_depth_missing}")
    for c in missing:
        df[c] = 0.0

    sim_cols_present = [c for c in SIM_STATE_COLS if c in df.columns]
    out = df[["ts_ms", *MARKET_FEATURE_NAMES, *sim_cols_present]].copy()
    # drop rolling-window warmup rows where any non-funding rolling feature is NaN
    rolling_cols = [
        "ofi_1s", "ofi_5s", "ofi_30s",
        "micro_price_log_ret",
        "realized_vol_5s", "realized_vol_30s", "realized_vol_5min",
        "trade_flow_imb_1s", "trade_flow_imb_5s", "trade_flow_imb_30s",
    ]
    out = out.dropna(subset=rolling_cols).reset_index(drop=True)
    for c in MARKET_FEATURE_NAMES:
        out[c] = out[c].astype("float32")
    for c in sim_cols_present:
        out[c] = out[c].astype("float32")
    return out


def build_features_for_day(
    symbol: str,
    day_iso: str,
    dataset_root: Path,
) -> Path | None:
    src = dataset_root / symbol / "snapshots" / f"{day_iso}.parquet"
    if not src.exists():
        logger.warning("missing snapshots for %s %s: %s", symbol, day_iso, src)
        return None
    snaps = pd.read_parquet(src)
    feats = build_market_features(snaps)
    if feats.empty:
        logger.warning("no features after warmup drop for %s %s", symbol, day_iso)
        return None
    feats["feature_version"] = FEATURE_VERSION
    dest = dataset_root / symbol / "features" / f"{day_iso}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(dest, index=False, compression="zstd")
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build per-day market-feature parquets")
    p.add_argument("--symbol", required=True)
    p.add_argument("--day", required=True, help="YYYY-MM-DD")
    p.add_argument("--dataset-root", default="data/datasets")
    args = p.parse_args()
    dest = build_features_for_day(args.symbol, args.day, Path(args.dataset_root))
    if dest is None:
        raise SystemExit("nothing produced")
    logger.info("wrote %s", dest)


if __name__ == "__main__":
    main()
