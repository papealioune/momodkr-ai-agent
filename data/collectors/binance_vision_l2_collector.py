"""Binance Vision L2-ish data collector for MomoDkr.

Binance Vision archives the following daily ZIPs per USDS-M perpetual symbol:

- bookTicker  best bid/ask + sizes at every top-of-book update (sub-second)
- aggTrades   aggregated trade events
- bookDepth   depth at fixed percentage levels (per-minute snapshots)
- klines      OHLCV bars (1h used by validator for cross-checking mid drift)

Binance does NOT publicly archive tick-level depth-update diffs, so true
top-10 L2 reconstruction is impossible from Vision alone. The
order_book_reconstructor combines bookTicker (top-1) + bookDepth (per-
minute depth at percentage levels) + aggTrades (OFI) into a feature
vector. A v2 may upgrade to Tardis.dev / CryptoLake for true top-N L2.

URL shape:
    https://data.binance.vision/data/futures/um/daily/{stream}/{symbol}/{symbol}-{stream}-{yyyy-MM-dd}.zip
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.collectors.asset_config import BINANCE_LISTING_DATES

logger = logging.getLogger(__name__)

BINANCE_VISION_DAILY = "https://data.binance.vision/data/futures/um/daily"

STREAMS = ("bookTicker", "aggTrades", "bookDepth")


@dataclass(frozen=True)
class FetchTask:
    symbol: str
    stream: str
    day: date

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.stream}-{self.day.isoformat()}.zip"

    @property
    def url(self) -> str:
        return f"{BINANCE_VISION_DAILY}/{self.stream}/{self.symbol}/{self.filename}"

    def output_path(self, root: Path) -> Path:
        return root / self.symbol / self.stream / self.filename


def build_session(retries: int = 3, backoff: float = 1.0, timeout: int = 60) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.timeout = timeout
    return s


def day_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end {end} < start {start}")
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def filter_to_listing(symbol: str, days: Iterable[date]) -> list[date]:
    listing_str = BINANCE_LISTING_DATES.get(symbol)
    if not listing_str:
        return list(days)
    listing = datetime.strptime(listing_str, "%Y-%m-%d").date()
    return [d for d in days if d >= listing]


def build_tasks(
    symbols: Iterable[str],
    start: date,
    end: date,
    streams: Iterable[str] = STREAMS,
) -> list[FetchTask]:
    days = day_range(start, end)
    out: list[FetchTask] = []
    for sym in symbols:
        for d in filter_to_listing(sym, days):
            for s in streams:
                out.append(FetchTask(symbol=sym, stream=s, day=d))
    return out


def fetch_one(
    task: FetchTask,
    session: requests.Session,
    raw_root: Path,
    overwrite: bool = False,
) -> Path | None:
    dest = task.output_path(raw_root)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = session.get(task.url, timeout=60)
    except requests.RequestException as e:
        logger.error("network error %s: %s", task.url, e)
        return None

    if resp.status_code == 404:
        logger.info("not available (404): %s", task.filename)
        return None
    resp.raise_for_status()

    tmp = dest.with_suffix(".zip.part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def fetch_all(
    tasks: list[FetchTask],
    raw_root: Path,
    max_workers: int = 8,
    session: requests.Session | None = None,
    overwrite: bool = False,
) -> dict[FetchTask, Path | None]:
    session = session or build_session()
    results: dict[FetchTask, Path | None] = {}

    def _run(t: FetchTask) -> tuple[FetchTask, Path | None]:
        return t, fetch_one(t, session, raw_root, overwrite=overwrite)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run, t) for t in tasks]
        for fut in as_completed(futs):
            t, p = fut.result()
            results[t] = p

    n_ok = sum(1 for p in results.values() if p is not None)
    logger.info("fetched %d/%d files", n_ok, len(tasks))
    return results


def _read_zip_csv_bytes(zip_path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV in {zip_path}")
        with zf.open(names[0]) as fh:
            return names[0], fh.read()


def parse_book_ticker(zip_path: Path) -> pd.DataFrame:
    """Columns produced: ts_ms, bid_px, bid_sz, ask_px, ask_sz."""
    _, raw = _read_zip_csv_bytes(zip_path)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {
        "best_bid_price": "bid_px",
        "best_bid_qty": "bid_sz",
        "best_ask_price": "ask_px",
        "best_ask_qty": "ask_sz",
    }
    df = df.rename(columns=rename)
    ts_col = "transaction_time" if "transaction_time" in df.columns else "event_time"
    df["ts_ms"] = pd.to_numeric(df[ts_col], errors="coerce").astype("int64")
    for col in ["bid_px", "bid_sz", "ask_px", "ask_sz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["ts_ms", "bid_px", "bid_sz", "ask_px", "ask_sz"]].dropna().reset_index(drop=True)


def parse_agg_trades(zip_path: Path) -> pd.DataFrame:
    """Columns produced: ts_ms, price, quantity, is_buyer_maker, signed_qty.

    signed_qty = +qty if a taker bought (is_buyer_maker = False), -qty if a taker sold.
    """
    _, raw = _read_zip_csv_bytes(zip_path)
    first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    has_header = not first_line.split(",")[0].strip().lstrip("-").isdigit()

    cols = [
        "agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker",
    ]
    if has_header:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.lower().strip() for c in df.columns]
    else:
        df = pd.read_csv(io.BytesIO(raw), header=None, names=cols)

    df["ts_ms"] = pd.to_numeric(df["transact_time"], errors="coerce").astype("int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    df["signed_qty"] = df["quantity"].where(~df["is_buyer_maker"], -df["quantity"])
    return df[["ts_ms", "price", "quantity", "is_buyer_maker", "signed_qty"]].dropna().reset_index(drop=True)


def parse_book_depth(zip_path: Path) -> pd.DataFrame:
    """Columns produced: ts_ms, percentage, depth, notional.

    percentage is signed in basis-points relative to mid (e.g. -100 = 1% below mid).
    depth is the cumulative base-asset size within that percentage band.
    """
    _, raw = _read_zip_csv_bytes(zip_path)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]
    df["ts_ms"] = pd.to_datetime(df["timestamp"]).astype("int64") // 1_000_000
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    return df[["ts_ms", "percentage", "depth", "notional"]].dropna().reset_index(drop=True)


PARSERS = {
    "bookTicker": parse_book_ticker,
    "aggTrades": parse_agg_trades,
    "bookDepth": parse_book_depth,
}


def _parse_one(
    task_stream: str,
    task_symbol: str,
    task_day_iso: str,
    src: str,
    dest: str,
    overwrite: bool,
) -> tuple[str, str, str, str | None]:
    """Worker: parse one ZIP -> parquet. Returns (symbol, stream, day_iso, dest_or_None_on_error).

    Module-level + plain-str args so the function pickles cleanly under
    ProcessPoolExecutor (FetchTask + Path don't always round-trip well
    across the spawn boundary).
    """
    from pathlib import Path as _Path

    parser = PARSERS.get(task_stream)
    if parser is None:
        return task_symbol, task_stream, task_day_iso, None
    dest_path = _Path(dest)
    if dest_path.exists() and not overwrite:
        return task_symbol, task_stream, task_day_iso, dest
    try:
        df = parser(_Path(src))
    except Exception as e:
        # logging from worker subprocess is unreliable; surface error in retval
        return task_symbol, task_stream, task_day_iso, f"ERR:{e!s}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_path, index=False, compression="zstd")
    return task_symbol, task_stream, task_day_iso, dest


def parse_and_persist(
    fetched: dict[FetchTask, Path | None],
    dataset_root: Path,
    overwrite: bool = False,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = None,
    progress_every: int = 100,
) -> dict[FetchTask, Path]:
    """Parse each downloaded ZIP into a per-day Parquet under dataset_root, in parallel.

    CPU-bound (unzip + pandas + zstd) so ProcessPoolExecutor is the right
    tool; ThreadPoolExecutor would be GIL-bound.

    Default max_workers picks a CONSERVATIVE 8 instead of all cores. Some
    bookTicker days decompress to multi-GB pandas DataFrames; 16+ workers
    each holding one at peak can blow through the container's RAM quota
    and crash with BrokenProcessPool. Override via the PARSE_WORKERS env
    var if you know your memory headroom.

    max_tasks_per_child defaults to None (no recycling). Setting it
    forces Python's ProcessPoolExecutor into 'spawn' context, which
    re-imports pandas/boto3/numpy from scratch on every worker recycle
    (10-30 sec/worker startup cost). With only 8 workers our memory
    high-water mark is bounded at ~16 GB peak, well under typical pod
    quotas, so recycling is more harm than help. Set it explicitly if
    you have a tighter memory budget AND can tolerate the spawn cost.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Build pending list (skip None sources + already-existing dests)
    pending: list[tuple[FetchTask, Path]] = []
    out: dict[FetchTask, Path] = {}
    for task, src in fetched.items():
        if src is None:
            continue
        if PARSERS.get(task.stream) is None:
            logger.warning("no parser for stream %s", task.stream)
            continue
        dest = dataset_root / task.symbol / task.stream / f"{task.day.isoformat()}.parquet"
        if dest.exists() and not overwrite:
            out[task] = dest
            continue
        pending.append((task, dest))

    if not pending:
        return out

    # Resolution order: explicit arg > PARSE_WORKERS env > conservative default 8
    env_workers = os.environ.get("PARSE_WORKERS")
    if max_workers is None and env_workers:
        try:
            max_workers = int(env_workers)
        except ValueError:
            logger.warning("ignoring non-integer PARSE_WORKERS=%r", env_workers)
    n_workers = max_workers or min(os.cpu_count() or 4, len(pending), 8)
    logger.info(
        "parsing %d ZIPs with %d processes (max_tasks_per_child=%s)...",
        len(pending), n_workers, "unlimited" if max_tasks_per_child is None else str(max_tasks_per_child),
    )

    # Build {(sym, stream, day_iso) -> FetchTask} so we can map results back.
    by_key: dict[tuple[str, str, str], FetchTask] = {}
    args_list: list[tuple[str, str, str, str, str, bool]] = []
    for task, dest in pending:
        key = (task.symbol, task.stream, task.day.isoformat())
        by_key[key] = task
        # src and dest are passed as strings for cross-process safety
        # (we already filtered out None sources above)
        src = fetched[task]
        assert src is not None
        args_list.append((task.stream, task.symbol, task.day.isoformat(), str(src), str(dest), overwrite))

    done = 0
    failed = 0
    pool_kwargs: dict = {"max_workers": n_workers}
    if max_tasks_per_child is not None:
        pool_kwargs["max_tasks_per_child"] = max_tasks_per_child
    with ProcessPoolExecutor(**pool_kwargs) as ex:
        futures = [ex.submit(_parse_one, *args) for args in args_list]
        for fut in as_completed(futures):
            try:
                sym, stream, day_iso, dest_or_err = fut.result()
            except Exception as e:
                # A worker died (BrokenProcessPool / OOM-kill / segfault).
                # Surface it but keep going: the remaining futures may still
                # land if the pool can recover; otherwise the outer loop
                # will eventually drain or re-raise.
                logger.error("parse worker crashed: %s", e)
                failed += 1
                done += 1
                continue
            key = (sym, stream, day_iso)
            task = by_key.get(key)
            if dest_or_err and not dest_or_err.startswith("ERR:") and task is not None:
                out[task] = Path(dest_or_err)
            else:
                failed += 1
                if dest_or_err and dest_or_err.startswith("ERR:"):
                    logger.error("parse failed %s/%s/%s: %s", sym, stream, day_iso, dest_or_err[4:])
            done += 1
            if done % progress_every == 0:
                logger.info("  parsed %d / %d (failed=%d)", done, len(pending), failed)

    logger.info("parsing complete: %d ok / %d failed", len(pending) - failed, failed)
    return out


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Binance Vision L2-ish data collector for MomoDkr")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--raw-root", default="data/raw/binance_vision")
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--streams", nargs="+", default=list(STREAMS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    raw_root = Path(args.raw_root)
    ds_root = Path(args.dataset_root)

    started = time.time()
    tasks = build_tasks(args.symbols, start, end, streams=args.streams)
    logger.info("planned %d fetch tasks", len(tasks))

    fetched = fetch_all(tasks, raw_root, max_workers=args.workers, overwrite=args.overwrite)
    parsed = parse_and_persist(fetched, ds_root, overwrite=args.overwrite)
    logger.info("wrote %d parquet files to %s in %.1fs", len(parsed), ds_root, time.time() - started)


if __name__ == "__main__":
    main()
