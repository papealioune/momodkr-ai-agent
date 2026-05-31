"""Validate Binance Vision L2 data and reconstructed snapshots before training.

RL agents exploit data glitches. A single crossed book, NaN mid, or stale
funding rate will teach the agent a fake pattern. This validator catches
the common L2 failure modes BEFORE the data hits the env.

Two entry points:
- validate_raw_streams(book_ticker, agg_trades, book_depth)
- validate_snapshots(snapshots, kline_1h=None) -- the Phase 1 gate
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MAX_GAP_MS = 1_000               # >1s without a quote update is suspicious
MAX_SPREAD_BPS = 50.0            # majors typically <10bps; >50bps flags illiquid bursts
MID_DRIFT_BPS_LIMIT = 1.0        # Phase 1 gate: snapshot mid vs Binance kline mid <=1bp


@dataclass
class Issue:
    severity: str   # "ERROR" or "WARNING"
    check: str
    message: str
    ts_ms: int | None = None
    value: float | None = None


@dataclass
class ValidationResult:
    label: str
    row_count: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self, max_lines: int = 20) -> str:
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'} for {self.label}",
            f"  rows={self.row_count}  errors={len(self.errors)}  warnings={len(self.warnings)}",
        ]
        for i in self.errors[:max_lines]:
            ts = f" @ {i.ts_ms}" if i.ts_ms is not None else ""
            lines.append(f"  [E] {i.check}: {i.message}{ts}")
        for i in self.warnings[:max_lines]:
            ts = f" @ {i.ts_ms}" if i.ts_ms is not None else ""
            lines.append(f"  [W] {i.check}: {i.message}{ts}")
        return "\n".join(lines)


def _add(res: ValidationResult, sev: str, check: str, msg: str, ts: int | None = None, val: float | None = None) -> None:
    res.issues.append(Issue(severity=sev, check=check, message=msg, ts_ms=ts, value=val))


# --- raw stream checks --------------------------------------------------------

def validate_book_ticker(df: pd.DataFrame, label: str = "bookTicker") -> ValidationResult:
    res = ValidationResult(label=label, row_count=len(df))
    if df.empty:
        _add(res, "ERROR", "empty", "bookTicker DataFrame is empty")
        return res

    required = {"ts_ms", "bid_px", "bid_sz", "ask_px", "ask_sz"}
    missing = required - set(df.columns)
    if missing:
        _add(res, "ERROR", "columns", f"missing columns: {missing}")
        return res

    if not df["ts_ms"].is_monotonic_increasing:
        _add(res, "ERROR", "monotonic", "ts_ms not monotonic increasing")

    if (df[["bid_px", "ask_px", "bid_sz", "ask_sz"]].isna().any().any()):
        _add(res, "ERROR", "nan", "NaN in bid/ask price or size columns")

    crossed = df[df["bid_px"] >= df["ask_px"]]
    if not crossed.empty:
        _add(
            res,
            "ERROR",
            "crossed_book",
            f"{len(crossed)} ticks with bid_px >= ask_px",
            ts=int(crossed["ts_ms"].iloc[0]),
        )

    nonpos = df[(df[["bid_px", "ask_px", "bid_sz", "ask_sz"]] <= 0).any(axis=1)]
    if not nonpos.empty:
        _add(res, "ERROR", "nonpositive", f"{len(nonpos)} ticks with non-positive bid/ask px or sz")

    return res


def validate_agg_trades(df: pd.DataFrame, label: str = "aggTrades") -> ValidationResult:
    res = ValidationResult(label=label, row_count=len(df))
    if df.empty:
        _add(res, "WARNING", "empty", "aggTrades DataFrame is empty")
        return res

    required = {"ts_ms", "price", "quantity", "is_buyer_maker", "signed_qty"}
    missing = required - set(df.columns)
    if missing:
        _add(res, "ERROR", "columns", f"missing columns: {missing}")
        return res

    if not df["ts_ms"].is_monotonic_increasing:
        _add(res, "ERROR", "monotonic", "ts_ms not monotonic increasing")

    bad_qty = df[df["quantity"] <= 0]
    if not bad_qty.empty:
        _add(res, "ERROR", "nonpositive_qty", f"{len(bad_qty)} trades with quantity <= 0")

    bad_px = df[df["price"] <= 0]
    if not bad_px.empty:
        _add(res, "ERROR", "nonpositive_price", f"{len(bad_px)} trades with price <= 0")

    return res


def validate_book_depth(df: pd.DataFrame, label: str = "bookDepth") -> ValidationResult:
    res = ValidationResult(label=label, row_count=len(df))
    if df.empty:
        _add(res, "WARNING", "empty", "bookDepth DataFrame is empty")
        return res

    required = {"ts_ms", "percentage", "depth", "notional"}
    missing = required - set(df.columns)
    if missing:
        _add(res, "ERROR", "columns", f"missing columns: {missing}")
        return res

    if (df["depth"] < 0).any() or (df["notional"] < 0).any():
        _add(res, "ERROR", "negative_depth", "depth or notional has negative values")

    return res


def validate_raw_streams(
    book_ticker: pd.DataFrame,
    agg_trades: pd.DataFrame,
    book_depth: pd.DataFrame,
) -> dict[str, ValidationResult]:
    return {
        "bookTicker": validate_book_ticker(book_ticker),
        "aggTrades": validate_agg_trades(agg_trades),
        "bookDepth": validate_book_depth(book_depth),
    }


# --- reconstructed snapshot checks -------------------------------------------

def validate_snapshots(
    snapshots: pd.DataFrame,
    kline_1h: pd.DataFrame | None = None,
    label: str = "snapshots",
    grid_ms: int = 100,
) -> ValidationResult:
    """Phase 1 gate. Includes the mid-vs-kline drift check (<=1bp) when kline_1h provided."""
    res = ValidationResult(label=label, row_count=len(snapshots))
    if snapshots.empty:
        _add(res, "ERROR", "empty", "snapshots DataFrame is empty")
        return res

    required = {"ts_ms", "bid_px", "ask_px", "mid"}
    missing = required - set(snapshots.columns)
    if missing:
        _add(res, "ERROR", "columns", f"missing columns: {missing}")
        return res

    if not snapshots["ts_ms"].is_monotonic_increasing:
        _add(res, "ERROR", "monotonic", "ts_ms not monotonic increasing")

    gaps = snapshots["ts_ms"].diff().dropna()
    expected = grid_ms
    bad_gaps = gaps[gaps != expected]
    if not bad_gaps.empty:
        # the reconstructor produces a uniform grid; any deviation is a bug
        large = bad_gaps[bad_gaps > MAX_GAP_MS]
        if not large.empty:
            _add(
                res,
                "ERROR",
                "snapshot_gap",
                f"{len(large)} gaps > {MAX_GAP_MS}ms in 100ms grid",
                ts=int(snapshots.loc[large.index[0], "ts_ms"]),
            )
        else:
            _add(res, "WARNING", "snapshot_gap", f"{len(bad_gaps)} non-uniform grid steps")

    if snapshots[["bid_px", "ask_px", "mid"]].isna().any().any():
        _add(res, "ERROR", "nan", "NaN in bid_px / ask_px / mid")

    crossed = snapshots[snapshots["bid_px"] >= snapshots["ask_px"]]
    if not crossed.empty:
        _add(res, "ERROR", "crossed_book", f"{len(crossed)} snapshots with bid_px >= ask_px")

    if "log_spread_bps" in snapshots.columns:
        wide = snapshots[snapshots["log_spread_bps"] > MAX_SPREAD_BPS]
        if len(wide) > len(snapshots) * 0.01:  # more than 1% of ticks wide-spread
            _add(
                res,
                "WARNING",
                "wide_spread",
                f"{len(wide)} ({len(wide) / len(snapshots) * 100:.2f}%) ticks with spread > {MAX_SPREAD_BPS}bps",
            )

    if kline_1h is not None and not kline_1h.empty:
        _check_mid_vs_kline(snapshots, kline_1h, res)

    return res


def _check_mid_vs_kline(snapshots: pd.DataFrame, kline_1h: pd.DataFrame, res: ValidationResult) -> None:
    """Snapshot mid at each kline close_time should be within MID_DRIFT_BPS_LIMIT bps of the kline close."""
    klines = kline_1h.copy()
    if "open_time" in klines.columns:
        klines["ts_ms"] = pd.to_numeric(klines["open_time"], errors="coerce").astype("int64")
    elif "timestamp" in klines.columns:
        klines["ts_ms"] = pd.to_datetime(klines["timestamp"]).astype("int64") // 1_000_000
    else:
        _add(res, "WARNING", "kline_xref_skip", "kline lacks open_time/timestamp; skipping mid drift check")
        return

    klines = klines.sort_values("ts_ms").reset_index(drop=True)
    klines["close_time_ms"] = klines["ts_ms"] + 60 * 60 * 1_000 - 1
    klines["close"] = pd.to_numeric(klines["close"], errors="coerce")

    snap_ts = snapshots["ts_ms"].to_numpy()
    snap_mid = snapshots["mid"].to_numpy()
    n_checked = 0
    n_bad = 0
    worst = 0.0
    for ct, kp in zip(klines["close_time_ms"].to_numpy(), klines["close"].to_numpy(), strict=False):
        if np.isnan(kp):
            continue
        idx = np.searchsorted(snap_ts, ct, side="right") - 1
        if idx < 0 or idx >= len(snap_mid):
            continue
        mid = float(snap_mid[idx])
        if mid <= 0:
            continue
        drift_bps = abs(mid - kp) / kp * 10_000
        worst = max(worst, drift_bps)
        n_checked += 1
        if drift_bps > MID_DRIFT_BPS_LIMIT:
            n_bad += 1

    if n_checked == 0:
        _add(res, "WARNING", "kline_xref_empty", "no kline close_times overlapped snapshot range")
        return

    if n_bad > 0:
        _add(
            res,
            "ERROR" if n_bad / n_checked > 0.05 else "WARNING",
            "mid_vs_kline_drift",
            f"{n_bad}/{n_checked} kline closes had |mid-close| > {MID_DRIFT_BPS_LIMIT}bps (worst {worst:.2f}bps)",
            val=worst,
        )


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Validate MomoDkr L2 data + reconstructed snapshots")
    p.add_argument("--snapshots", required=True, help="path to reconstructed snapshots parquet")
    p.add_argument("--kline-1h", default=None, help="optional 1h kline parquet for mid-drift cross-check")
    p.add_argument("--grid-ms", type=int, default=100)
    args = p.parse_args()

    snaps = pd.read_parquet(args.snapshots)
    klines = pd.read_parquet(args.kline_1h) if args.kline_1h else None
    res = validate_snapshots(snaps, kline_1h=klines, label=Path(args.snapshots).name, grid_ms=args.grid_ms)
    print(res.summary())
    raise SystemExit(0 if res.passed else 1)


if __name__ == "__main__":
    main()
