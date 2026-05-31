import numpy as np
import pandas as pd
import pytest

from data.preprocessors.momentum_features import (
    REALIZED_VOL_WINDOWS_TICKS,
    add_momentum_features,
    micro_price_log_return,
    realized_volatility,
)


def _snapshots(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    log_ret = rng.standard_normal(n) * 1e-4
    micro = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {
            "ts_ms": np.arange(n, dtype=np.int64) * 100,
            "micro_price": micro,
        }
    )


def test_micro_price_log_return_first_window_is_nan() -> None:
    snaps = _snapshots(n=50)
    out = micro_price_log_return(snaps, window_ticks=10)
    assert out.iloc[:10].isna().all()
    assert not out.iloc[10:].isna().any()


def test_micro_price_log_return_matches_definition() -> None:
    snaps = _snapshots(n=200)
    w = 10
    out = micro_price_log_return(snaps, window_ticks=w)
    expected = np.log(snaps["micro_price"].iloc[w:].to_numpy() / snaps["micro_price"].iloc[:-w].to_numpy())
    np.testing.assert_allclose(out.dropna().to_numpy(), expected, rtol=1e-9, atol=1e-12)


def test_realized_volatility_is_nonnegative_and_finite() -> None:
    snaps = _snapshots(n=4000)
    rv = realized_volatility(snaps, window_ticks=50)
    valid = rv.dropna()
    assert (valid >= 0).all()
    assert np.isfinite(valid).all()


def test_realized_volatility_zero_when_price_constant() -> None:
    snaps = pd.DataFrame(
        {
            "ts_ms": np.arange(100, dtype=np.int64) * 100,
            "micro_price": np.full(100, 50.0),
        }
    )
    rv = realized_volatility(snaps, window_ticks=10)
    assert (rv.dropna() == 0).all()


def test_add_momentum_features_adds_all_canonical_columns() -> None:
    snaps = _snapshots(n=4000)
    out = add_momentum_features(snaps)
    assert "micro_price_log_ret" in out.columns
    for name in REALIZED_VOL_WINDOWS_TICKS:
        assert name in out.columns


def test_realized_volatility_5min_window_warmup() -> None:
    snaps = _snapshots(n=4000)
    out = add_momentum_features(snaps)
    rv = out["realized_vol_5min"]
    warmup = 5 * 60 * 10  # 3000 ticks
    assert rv.iloc[:warmup - 1].isna().all()
    assert pytest.approx(rv.iloc[warmup - 1]) != 0  # likely nonzero after warmup
