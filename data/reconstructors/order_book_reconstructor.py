"""Reconstruct 100ms-aligned market snapshots from Binance Vision raw data.

Inputs (per symbol, per day):
- bookTicker.parquet  ts_ms, bid_px, bid_sz, ask_px, ask_sz   (sub-second cadence)
- aggTrades.parquet   ts_ms, price, quantity, is_buyer_maker, signed_qty
- bookDepth.parquet   ts_ms, percentage, depth, notional      (per-minute snapshots)

Output (per symbol, per day):
- snapshots.parquet   ts_ms (100ms grid), bid_px, bid_sz, ask_px, ask_sz,
                      mid, micro_price, log_spread_bps, top1_size_imbalance,
                      ofi_100ms, signed_volume_100ms, trade_count_100ms,
                      bid_depth_pct_neg_0_1, bid_depth_pct_neg_0_2,
                      bid_depth_pct_neg_0_5, bid_depth_pct_neg_1_0,
                      ask_depth_pct_pos_0_1, ask_depth_pct_pos_0_2,
                      ask_depth_pct_pos_0_5, ask_depth_pct_pos_1_0,
                      funding_rate (forward-filled, NaN if not provided)

The reconstructor does NOT compute the multi-window features (1s/5s/30s
OFI etc.) that go into the final RL observation. Those are computed by
the env at episode-load time so the same snapshot can serve different
feature versions without re-running this stage.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_GRID_MS = 100

DEPTH_PCT_LEVELS_BID = [-0.1, -0.2, -0.5, -1.0]
DEPTH_PCT_LEVELS_ASK = [0.1, 0.2, 0.5, 1.0]


@dataclass
class ReconstructionInputs:
    book_ticker: pd.DataFrame
    agg_trades: pd.DataFrame
    book_depth: pd.DataFrame
    funding: pd.DataFrame | None = None


def _grid_timestamps(start_ms: int, end_ms: int, grid_ms: int) -> np.ndarray:
    # CEIL start to next grid -- ensures the FIRST grid tick has a valid as-of
    # bookTicker observation (rounding DOWN would put grid_ts[0] BEFORE the first
    # quote and asof_align would emit NaN there, failing the NaN gate downstream).
    start_aligned = ((start_ms + grid_ms - 1) // grid_ms) * grid_ms
    # FLOOR end to previous grid -- symmetric so the last tick is also asof-valid.
    end_aligned = (end_ms // grid_ms) * grid_ms
    if end_aligned < start_aligned:
        return np.array([], dtype=np.int64)
    return np.arange(start_aligned, end_aligned + 1, grid_ms, dtype=np.int64)


def _asof_align(grid_ts: np.ndarray, src: pd.DataFrame) -> pd.DataFrame:
    """Return src rows as-of the grid_ts (last known value at or before each tick)."""
    src = src.sort_values("ts_ms").reset_index(drop=True)
    idx = np.searchsorted(src["ts_ms"].to_numpy(), grid_ts, side="right") - 1
    valid = idx >= 0
    out = src.iloc[np.where(valid, idx, 0)].copy()
    out.index = pd.RangeIndex(len(grid_ts))
    out.loc[~valid, src.columns.drop("ts_ms")] = np.nan
    out["ts_ms"] = grid_ts
    return out


def align_top_of_book(book_ticker: pd.DataFrame, grid_ts: np.ndarray) -> pd.DataFrame:
    """Forward-fill bookTicker to the snapshot grid."""
    cols = ["bid_px", "bid_sz", "ask_px", "ask_sz"]
    aligned = _asof_align(grid_ts, book_ticker[["ts_ms", *cols]])
    return aligned[["ts_ms", *cols]]


def aggregate_trade_flow(
    agg_trades: pd.DataFrame,
    grid_ts: np.ndarray,
    grid_ms: int,
) -> pd.DataFrame:
    """Bucket aggTrades into 100ms windows ending at each grid tick.

    Window [grid_ts - grid_ms, grid_ts).
    """
    edges = np.concatenate([[grid_ts[0] - grid_ms], grid_ts])
    bucket = np.searchsorted(edges, agg_trades["ts_ms"].to_numpy(), side="right") - 1
    in_range = (bucket >= 0) & (bucket < len(grid_ts))
    bucket = bucket[in_range]
    signed = agg_trades["signed_qty"].to_numpy()[in_range]
    qty = agg_trades["quantity"].to_numpy()[in_range]

    n = len(grid_ts)
    ofi = np.bincount(bucket, weights=signed, minlength=n)[:n]
    vol = np.bincount(bucket, weights=qty, minlength=n)[:n]
    cnt = np.bincount(bucket, minlength=n)[:n].astype(np.int64)

    return pd.DataFrame(
        {
            "ts_ms": grid_ts,
            "ofi_100ms": ofi,
            "signed_volume_100ms": ofi,  # alias kept for clarity downstream
            "abs_volume_100ms": vol,
            "trade_count_100ms": cnt,
        }
    )


def align_book_depth(book_depth: pd.DataFrame, grid_ts: np.ndarray) -> pd.DataFrame:
    """Pivot bookDepth so each grid tick has cumulative depth at each percentage level."""
    if book_depth.empty:
        return pd.DataFrame({"ts_ms": grid_ts})

    wanted = sorted(set(DEPTH_PCT_LEVELS_BID + DEPTH_PCT_LEVELS_ASK))
    df = book_depth[book_depth["percentage"].isin(wanted)].copy()
    if df.empty:
        return pd.DataFrame({"ts_ms": grid_ts})

    wide = df.pivot_table(index="ts_ms", columns="percentage", values="depth", aggfunc="last")
    wide = wide.sort_index().reset_index()

    out = pd.DataFrame({"ts_ms": grid_ts})
    for pct in DEPTH_PCT_LEVELS_BID:
        col = f"bid_depth_pct_neg_{abs(pct):.1f}".replace(".", "_")
        if pct in wide.columns:
            sub = wide[["ts_ms", pct]].rename(columns={pct: col})
            aligned = _asof_align(grid_ts, sub)
            out[col] = aligned[col].to_numpy()
        else:
            out[col] = np.nan
    for pct in DEPTH_PCT_LEVELS_ASK:
        col = f"ask_depth_pct_pos_{pct:.1f}".replace(".", "_")
        if pct in wide.columns:
            sub = wide[["ts_ms", pct]].rename(columns={pct: col})
            aligned = _asof_align(grid_ts, sub)
            out[col] = aligned[col].to_numpy()
        else:
            out[col] = np.nan
    return out


def align_funding(funding: pd.DataFrame | None, grid_ts: np.ndarray) -> pd.Series:
    if funding is None or funding.empty:
        return pd.Series(np.nan, index=range(len(grid_ts)), name="funding_rate")
    sub = funding[["ts_ms", "funding_rate"]]
    aligned = _asof_align(grid_ts, sub)
    s = pd.Series(aligned["funding_rate"].to_numpy(), name="funding_rate")
    return s


def derive_quote_features(top: pd.DataFrame) -> pd.DataFrame:
    bid_px = top["bid_px"]
    ask_px = top["ask_px"]
    bid_sz = top["bid_sz"]
    ask_sz = top["ask_sz"]
    mid = (bid_px + ask_px) / 2.0
    spread = (ask_px - bid_px).clip(lower=0.0)
    log_spread_bps = np.where(mid > 0, np.log1p(spread / mid) * 10_000, 0.0)
    sz_sum = (bid_sz + ask_sz).replace(0, np.nan)
    top1_imb = (bid_sz - ask_sz) / sz_sum
    micro_price = np.where(sz_sum > 0, (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz), mid)
    return pd.DataFrame(
        {
            "mid": mid.to_numpy(),
            "micro_price": micro_price,
            "log_spread_bps": log_spread_bps,
            "top1_size_imbalance": top1_imb.fillna(0.0).to_numpy(),
        }
    )


def reconstruct(
    inputs: ReconstructionInputs,
    grid_ms: int = DEFAULT_GRID_MS,
) -> pd.DataFrame:
    """Produce a 100ms-aligned snapshot DataFrame for a single trading day."""
    bt = inputs.book_ticker
    at = inputs.agg_trades
    if bt.empty:
        raise ValueError("book_ticker is empty; cannot reconstruct")

    start_ms = int(bt["ts_ms"].iloc[0])
    end_ms = int(bt["ts_ms"].iloc[-1])
    if not at.empty:
        end_ms = max(end_ms, int(at["ts_ms"].iloc[-1]))
    grid_ts = _grid_timestamps(start_ms, end_ms, grid_ms)

    top = align_top_of_book(bt, grid_ts)
    quote_feats = derive_quote_features(top)
    flow = aggregate_trade_flow(at, grid_ts, grid_ms) if not at.empty else pd.DataFrame(
        {
            "ts_ms": grid_ts,
            "ofi_100ms": 0.0,
            "signed_volume_100ms": 0.0,
            "abs_volume_100ms": 0.0,
            "trade_count_100ms": 0,
        }
    )
    depth = align_book_depth(inputs.book_depth, grid_ts) if not inputs.book_depth.empty else pd.DataFrame({"ts_ms": grid_ts})
    funding = align_funding(inputs.funding, grid_ts)

    out = pd.DataFrame({"ts_ms": grid_ts})
    out = out.join(top.drop(columns="ts_ms"))
    out = out.join(quote_feats)
    out = out.join(flow.drop(columns="ts_ms"))
    out = out.join(depth.drop(columns="ts_ms"))
    out["funding_rate"] = funding.to_numpy()

    # Safety net: drop any leading/trailing rows where the asof-align couldn't
    # find a valid bookTicker quote. With the ceil/floor grid alignment above
    # this should never fire, but if a future input has internal gaps we'd
    # rather lose those ticks than poison the whole day's validation.
    out = out.dropna(subset=["bid_px", "ask_px", "mid"]).reset_index(drop=True)
    return out


def reconstruct_day(
    symbol: str,
    day_iso: str,
    dataset_root: Path,
    funding_path: Path | None = None,
    grid_ms: int = DEFAULT_GRID_MS,
) -> pd.DataFrame:
    bt_path = dataset_root / symbol / "bookTicker" / f"{day_iso}.parquet"
    at_path = dataset_root / symbol / "aggTrades" / f"{day_iso}.parquet"
    bd_path = dataset_root / symbol / "bookDepth" / f"{day_iso}.parquet"

    if not bt_path.exists():
        raise FileNotFoundError(f"missing bookTicker for {symbol} {day_iso}: {bt_path}")

    book_ticker = pd.read_parquet(bt_path)
    agg_trades = pd.read_parquet(at_path) if at_path.exists() else pd.DataFrame(
        columns=["ts_ms", "price", "quantity", "is_buyer_maker", "signed_qty"]
    )
    book_depth = pd.read_parquet(bd_path) if bd_path.exists() else pd.DataFrame(
        columns=["ts_ms", "percentage", "depth", "notional"]
    )
    funding = pd.read_parquet(funding_path) if funding_path and funding_path.exists() else None

    inputs = ReconstructionInputs(
        book_ticker=book_ticker,
        agg_trades=agg_trades,
        book_depth=book_depth,
        funding=funding,
    )
    return reconstruct(inputs, grid_ms=grid_ms)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Reconstruct 100ms snapshots from Binance Vision parquets")
    p.add_argument("--symbol", required=True)
    p.add_argument("--day", required=True, help="YYYY-MM-DD")
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--funding-parquet", default=None, help="optional funding parquet path")
    p.add_argument("--output-root", default="data/datasets")
    p.add_argument("--grid-ms", type=int, default=DEFAULT_GRID_MS)
    args = p.parse_args()

    funding_path = Path(args.funding_parquet) if args.funding_parquet else None
    df = reconstruct_day(args.symbol, args.day, Path(args.dataset_root), funding_path, args.grid_ms)
    dest = Path(args.output_root) / args.symbol / "snapshots" / f"{args.day}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False, compression="zstd")
    logger.info("wrote %d snapshots to %s", len(df), dest)


if __name__ == "__main__":
    main()
