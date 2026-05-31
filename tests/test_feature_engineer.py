import numpy as np
import pandas as pd

from data.preprocessors.feature_engineer import build_market_features
from serving.feature_version import MARKET_FEATURE_NAMES, SIM_STATE_COLS


def _synthetic_snapshot_day(n_ticks: int = 4500, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = start_ms + np.arange(n_ticks, dtype=np.int64) * 100
    log_ret = rng.standard_normal(n_ticks) * 1e-4
    mid = 50_000.0 * np.exp(np.cumsum(log_ret))
    half_spread = 1.5
    bid_px = mid - half_spread
    ask_px = mid + half_spread
    bid_sz = rng.uniform(0.1, 2.0, n_ticks)
    ask_sz = rng.uniform(0.1, 2.0, n_ticks)
    micro = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)
    return pd.DataFrame(
        {
            "ts_ms": ts,
            "bid_px": bid_px,
            "bid_sz": bid_sz,
            "ask_px": ask_px,
            "ask_sz": ask_sz,
            "mid": mid,
            "micro_price": micro,
            "log_spread_bps": np.full(n_ticks, 3.0),
            "top1_size_imbalance": (bid_sz - ask_sz) / (bid_sz + ask_sz),
            "ofi_100ms": rng.standard_normal(n_ticks),
            "signed_volume_100ms": rng.standard_normal(n_ticks),
            "abs_volume_100ms": np.abs(rng.standard_normal(n_ticks)),
            "trade_count_100ms": rng.integers(0, 5, n_ticks),
            "bid_depth_pct_neg_0_1": rng.uniform(5, 20, n_ticks),
            "bid_depth_pct_neg_0_2": rng.uniform(10, 40, n_ticks),
            "bid_depth_pct_neg_0_5": rng.uniform(50, 200, n_ticks),
            "bid_depth_pct_neg_1_0": rng.uniform(100, 500, n_ticks),
            "ask_depth_pct_pos_0_1": rng.uniform(5, 20, n_ticks),
            "ask_depth_pct_pos_0_2": rng.uniform(10, 40, n_ticks),
            "ask_depth_pct_pos_0_5": rng.uniform(50, 200, n_ticks),
            "ask_depth_pct_pos_1_0": rng.uniform(100, 500, n_ticks),
            "funding_rate": np.full(n_ticks, 1e-4),
        }
    )


def test_build_market_features_returns_canonical_columns() -> None:
    snaps = _synthetic_snapshot_day()
    feats = build_market_features(snaps)
    # feature_engineer keeps sim-state columns (mid, bid_px, ask_px, abs_volume_100ms, funding_rate)
    # alongside the 26 obs features so the env can simulate fills without re-loading snapshots.
    expected = ["ts_ms", *MARKET_FEATURE_NAMES, *SIM_STATE_COLS]
    assert list(feats.columns) == expected


def test_build_market_features_drops_warmup_rows() -> None:
    snaps = _synthetic_snapshot_day(n_ticks=4500)
    feats = build_market_features(snaps)
    # warmup is 5min RV = 3000 ticks; need at least 3000 ticks before first valid
    assert len(feats) <= len(snaps) - 2999
    assert len(feats) > 0


def test_build_market_features_no_nan_after_warmup() -> None:
    snaps = _synthetic_snapshot_day()
    feats = build_market_features(snaps)
    for col in MARKET_FEATURE_NAMES:
        assert not feats[col].isna().any(), f"NaN in {col}"


def test_build_market_features_float32() -> None:
    snaps = _synthetic_snapshot_day()
    feats = build_market_features(snaps)
    for col in MARKET_FEATURE_NAMES:
        assert feats[col].dtype == np.float32


def test_build_market_features_ts_monotonic() -> None:
    snaps = _synthetic_snapshot_day()
    feats = build_market_features(snaps)
    assert feats["ts_ms"].is_monotonic_increasing


def test_build_market_features_time_encoding_in_unit_circle() -> None:
    snaps = _synthetic_snapshot_day()
    feats = build_market_features(snaps)
    for col in ("hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos"):
        valid = feats[col]
        assert (valid >= -1.0001).all() and (valid <= 1.0001).all()
