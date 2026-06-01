import numpy as np
import pandas as pd
import pytest

from data.reconstructors.order_book_reconstructor import (
    DEFAULT_GRID_MS,
    ReconstructionInputs,
    aggregate_trade_flow,
    align_top_of_book,
    derive_quote_features,
    reconstruct,
)


def _book_ticker(seed: int = 0, n: int = 50, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * 47 + start_ms  # irregular ~47ms cadence
    mid = 50_000 + rng.standard_normal(n).cumsum() * 5
    half_spread = 1.5
    return pd.DataFrame(
        {
            "ts_ms": ts.astype("int64"),
            "bid_px": mid - half_spread,
            "bid_sz": rng.uniform(0.1, 2.0, n),
            "ask_px": mid + half_spread,
            "ask_sz": rng.uniform(0.1, 2.0, n),
        }
    )


def _agg_trades(seed: int = 1, n: int = 30, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = np.sort(start_ms + rng.integers(0, 5_000, size=n))
    side_is_buyer_maker = rng.random(n) < 0.5  # True = taker sold
    qty = rng.uniform(0.01, 0.5, n)
    signed = np.where(side_is_buyer_maker, -qty, qty)
    return pd.DataFrame(
        {
            "ts_ms": ts.astype("int64"),
            "price": 50_000 + rng.standard_normal(n) * 5,
            "quantity": qty,
            "is_buyer_maker": side_is_buyer_maker,
            "signed_qty": signed,
        }
    )


def _book_depth(start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    levels = [-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0]
    rows = []
    for snap in range(3):
        ts = start_ms + snap * 1_000  # one snapshot per second
        for pct in levels:
            rows.append(
                {
                    "ts_ms": ts,
                    "percentage": pct,
                    "depth": 10.0 + abs(pct) * 2.0,
                    "notional": (10.0 + abs(pct) * 2.0) * 50_000,
                }
            )
    return pd.DataFrame(rows)


def test_grid_alignment_uses_uniform_step() -> None:
    bt = _book_ticker(n=10)
    grid_ts = np.arange(bt["ts_ms"].min(), bt["ts_ms"].max(), DEFAULT_GRID_MS)
    top = align_top_of_book(bt, grid_ts)
    assert len(top) == len(grid_ts)
    assert top["ts_ms"].is_monotonic_increasing
    assert not top[["bid_px", "ask_px"]].isna().any().any()


def test_derive_quote_features_micro_price_stoikov_weighting() -> None:
    # Stoikov micro-price = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)
    # Heavy bid -> ask gets lifted -> micro collapses toward ask.
    # Heavy ask -> bid gets hit -> micro collapses toward bid.
    df = pd.DataFrame(
        {
            "bid_px": [100.0, 100.0, 100.0],
            "bid_sz": [1.0, 10.0, 0.01],
            "ask_px": [101.0, 101.0, 101.0],
            "ask_sz": [1.0, 0.01, 10.0],
        }
    )
    feats = derive_quote_features(df)
    assert feats["micro_price"].iloc[0] == pytest.approx(100.5)
    assert feats["micro_price"].iloc[1] == pytest.approx(101.0, abs=0.01)
    assert feats["micro_price"].iloc[2] == pytest.approx(100.0, abs=0.01)
    assert (feats["log_spread_bps"] > 0).all()


def test_aggregate_trade_flow_signed_volume_matches_input() -> None:
    bt = _book_ticker(n=20)
    grid_ts = np.arange(bt["ts_ms"].min(), bt["ts_ms"].max(), DEFAULT_GRID_MS)
    at = _agg_trades(n=40, start_ms=int(bt["ts_ms"].min()))
    flow = aggregate_trade_flow(at, grid_ts, DEFAULT_GRID_MS)
    in_window = at[(at["ts_ms"] >= grid_ts[0]) & (at["ts_ms"] < grid_ts[-1])]
    expected_signed = in_window["signed_qty"].sum()
    assert flow["ofi_100ms"].sum() == pytest.approx(expected_signed, rel=1e-6, abs=1e-9)
    assert int(flow["trade_count_100ms"].sum()) == len(in_window)


def test_reconstruct_produces_uniform_grid_and_no_nans_in_quotes() -> None:
    inputs = ReconstructionInputs(
        book_ticker=_book_ticker(n=200),
        agg_trades=_agg_trades(n=80),
        book_depth=_book_depth(),
        funding=None,
    )
    snaps = reconstruct(inputs)
    diffs = snaps["ts_ms"].diff().dropna().unique()
    assert diffs.tolist() == [DEFAULT_GRID_MS]
    for col in ["bid_px", "ask_px", "mid", "micro_price", "log_spread_bps"]:
        assert not snaps[col].isna().any(), f"NaNs in {col}"
    assert (snaps["mid"] >= snaps["bid_px"]).all()
    assert (snaps["mid"] <= snaps["ask_px"]).all()


def test_reconstruct_no_nan_when_first_bookticker_ts_not_grid_aligned() -> None:
    """Regression: real Binance bookTicker first ts is rarely a multiple of 100ms.
    Without ceil-up grid alignment, grid_ts[0] precedes the first quote and the
    asof-align emits NaN -- the validator's NaN gate then fails the entire day.
    """
    # First ts is 1_700_000_000_037 -- NOT divisible by 100. Flooring start_ms
    # to a 100ms grid would put grid_ts[0] = 1_700_000_000_000 (37ms BEFORE the
    # first quote) and force NaN at row 0.
    bt = pd.DataFrame(
        {
            "ts_ms": (1_700_000_000_037 + np.arange(200) * 47).astype("int64"),
            "bid_px": np.full(200, 99.5),
            "bid_sz": np.full(200, 1.0),
            "ask_px": np.full(200, 100.5),
            "ask_sz": np.full(200, 1.0),
        }
    )
    inputs = ReconstructionInputs(
        book_ticker=bt,
        agg_trades=_agg_trades(n=30, start_ms=1_700_000_000_037),
        book_depth=_book_depth(start_ms=1_700_000_000_037),
        funding=None,
    )
    snaps = reconstruct(inputs)
    assert len(snaps) > 0
    # The leading-edge NaN bug would manifest here.
    for col in ("bid_px", "ask_px", "mid", "micro_price"):
        assert not snaps[col].isna().any(), f"NaN in {col} at grid edge"
    # grid_ts[0] should now be CEIL'd to the next multiple of 100ms.
    assert int(snaps["ts_ms"].iloc[0]) % 100 == 0
    assert int(snaps["ts_ms"].iloc[0]) >= int(bt["ts_ms"].iloc[0])


def test_reconstruct_empty_book_ticker_raises() -> None:
    empty = pd.DataFrame(columns=["ts_ms", "bid_px", "bid_sz", "ask_px", "ask_sz"])
    inputs = ReconstructionInputs(
        book_ticker=empty,
        agg_trades=_agg_trades(n=5),
        book_depth=_book_depth(),
    )
    with pytest.raises(ValueError):
        reconstruct(inputs)
