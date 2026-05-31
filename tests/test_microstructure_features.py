import numpy as np
import pandas as pd
import pytest

from data.preprocessors.microstructure_features import (
    OFI_WINDOWS_TICKS,
    TRADE_FLOW_WINDOWS_TICKS,
    add_ofi_features,
    add_trade_flow_features,
    rolling_ofi,
    rolling_trade_flow_imbalance,
)


def _snapshots(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    qty = np.abs(rng.standard_normal(n))
    side = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return pd.DataFrame(
        {
            "ts_ms": np.arange(n, dtype=np.int64) * 100,
            "ofi_100ms": qty * side,
            "signed_volume_100ms": qty * side,
            "abs_volume_100ms": qty,
        }
    )


def test_rolling_ofi_warmup_then_sums() -> None:
    snaps = _snapshots(n=50)
    out = rolling_ofi(snaps, window_ticks=10)
    assert out.iloc[:9].isna().all()
    expected_first_valid = snaps["ofi_100ms"].iloc[:10].sum()
    assert out.iloc[9] == pytest.approx(expected_first_valid)


def test_rolling_ofi_matches_brute_force() -> None:
    snaps = _snapshots(n=100)
    w = 7
    out = rolling_ofi(snaps, window_ticks=w)
    for i in range(w - 1, len(snaps)):
        assert out.iloc[i] == pytest.approx(snaps["ofi_100ms"].iloc[i - w + 1:i + 1].sum())


def test_trade_flow_imbalance_in_unit_interval() -> None:
    snaps = _snapshots(n=200)
    out = rolling_trade_flow_imbalance(snaps, window_ticks=10)
    valid = out.dropna()
    assert (valid >= -1.0).all()
    assert (valid <= 1.0).all()


def test_trade_flow_imbalance_returns_zero_on_zero_volume() -> None:
    snaps = pd.DataFrame(
        {
            "ts_ms": np.arange(20, dtype=np.int64) * 100,
            "signed_volume_100ms": np.zeros(20),
            "abs_volume_100ms": np.zeros(20),
        }
    )
    out = rolling_trade_flow_imbalance(snaps, window_ticks=5)
    assert (out.dropna() == 0.0).all()


def test_add_ofi_features_adds_canonical_columns() -> None:
    out = add_ofi_features(_snapshots())
    for name in OFI_WINDOWS_TICKS:
        assert name in out.columns


def test_add_trade_flow_features_adds_canonical_columns() -> None:
    out = add_trade_flow_features(_snapshots())
    for name in TRADE_FLOW_WINDOWS_TICKS:
        assert name in out.columns


def test_no_look_ahead_in_ofi() -> None:
    """Reverse the second half of the input; first half's features must be unchanged."""
    snaps = _snapshots(n=100)
    feats_orig = add_ofi_features(snaps)["ofi_1s"]
    perturbed = snaps.copy()
    perturbed.loc[50:, "ofi_100ms"] = perturbed.loc[50:, "ofi_100ms"].to_numpy()[::-1]
    feats_pert = add_ofi_features(perturbed)["ofi_1s"]
    assert feats_orig.iloc[:50].equals(feats_pert.iloc[:50])
