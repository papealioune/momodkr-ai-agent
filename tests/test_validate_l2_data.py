import numpy as np
import pandas as pd

from data.validators.validate_l2_data import (
    validate_agg_trades,
    validate_book_depth,
    validate_book_ticker,
    validate_snapshots,
)


def _good_book_ticker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": np.arange(10) * 100 + 1_700_000_000_000,
            "bid_px": np.full(10, 99.5),
            "bid_sz": np.full(10, 1.0),
            "ask_px": np.full(10, 100.5),
            "ask_sz": np.full(10, 1.0),
        }
    )


def _good_snapshot(n: int = 100) -> pd.DataFrame:
    ts = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
    return pd.DataFrame(
        {
            "ts_ms": ts,
            "bid_px": np.full(n, 99.5),
            "ask_px": np.full(n, 100.5),
            "mid": np.full(n, 100.0),
            "micro_price": np.full(n, 100.0),
            "log_spread_bps": np.full(n, 1.0),
        }
    )


def test_book_ticker_passes_on_clean_data() -> None:
    res = validate_book_ticker(_good_book_ticker())
    assert res.passed, res.summary()


def test_book_ticker_flags_crossed_book() -> None:
    df = _good_book_ticker()
    df.loc[3, "bid_px"] = 200.0  # bid > ask
    res = validate_book_ticker(df)
    assert not res.passed
    assert any(i.check == "crossed_book" for i in res.errors)


def test_book_ticker_flags_non_monotonic_ts() -> None:
    df = _good_book_ticker()
    df.loc[5, "ts_ms"] = df.loc[2, "ts_ms"]
    res = validate_book_ticker(df)
    assert any(i.check == "monotonic" for i in res.errors)


def test_book_ticker_flags_nonpositive_sizes() -> None:
    df = _good_book_ticker()
    df.loc[1, "bid_sz"] = 0
    res = validate_book_ticker(df)
    assert any(i.check == "nonpositive" for i in res.errors)


def test_agg_trades_flags_nonpositive_quantity() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": [1, 2, 3],
            "price": [100.0, 100.0, 100.0],
            "quantity": [0.1, 0.0, 0.2],
            "is_buyer_maker": [False, True, False],
            "signed_qty": [0.1, 0.0, 0.2],
        }
    )
    res = validate_agg_trades(df)
    assert any(i.check == "nonpositive_qty" for i in res.errors)


def test_book_depth_flags_negative_depth() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": [1, 1, 1],
            "percentage": [-0.5, 0.0, 0.5],
            "depth": [1.0, -2.0, 1.0],
            "notional": [100.0, 100.0, 100.0],
        }
    )
    res = validate_book_depth(df)
    assert any(i.check == "negative_depth" for i in res.errors)


def test_snapshots_pass_on_clean_data() -> None:
    res = validate_snapshots(_good_snapshot())
    assert res.passed, res.summary()


def test_snapshots_flag_crossed_book() -> None:
    snap = _good_snapshot()
    snap.loc[10, "bid_px"] = 200.0
    res = validate_snapshots(snap)
    assert not res.passed
    assert any(i.check == "crossed_book" for i in res.errors)


def test_snapshots_flag_large_grid_gap() -> None:
    snap = _good_snapshot(n=20)
    # introduce a 3-second gap
    snap.loc[10:, "ts_ms"] = snap.loc[10:, "ts_ms"] + 3_000
    res = validate_snapshots(snap)
    assert any(i.check == "snapshot_gap" and i.severity == "ERROR" for i in res.issues)


def test_snapshots_mid_vs_kline_drift_pass_when_close() -> None:
    snap = _good_snapshot(n=40_000)  # 40000 * 100ms = 4000s ~ 1.1h
    kline = pd.DataFrame(
        {
            "open_time": [snap["ts_ms"].iloc[0]],
            "close": [100.0],  # exactly mid, so drift = 0
        }
    )
    res = validate_snapshots(snap, kline_1h=kline)
    assert res.passed, res.summary()


def test_snapshots_mid_vs_kline_drift_flags_large_drift() -> None:
    snap = _good_snapshot(n=40_000)
    kline = pd.DataFrame(
        {
            "open_time": [snap["ts_ms"].iloc[0]],
            "close": [100.05],  # 5bps drift, above 1bp limit
        }
    )
    res = validate_snapshots(snap, kline_1h=kline)
    assert any(i.check == "mid_vs_kline_drift" for i in res.issues)
