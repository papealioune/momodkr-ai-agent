"""End-to-end orchestrator: Binance Vision -> parquet -> 100ms snapshots -> validate -> R2.

For each (symbol, day) in the requested window this script:
  1. Downloads the bookTicker / aggTrades / bookDepth ZIPs from Binance Vision
     (skips days already present locally unless --overwrite).
  2. Parses each ZIP to a per-day parquet in data/datasets/<SYMBOL>/<stream>/.
  3. Reconstructs a 100ms-aligned snapshot parquet under
     data/datasets/<SYMBOL>/snapshots/<YYYY-MM-DD>.parquet.
  4. Runs the L2 validator against the snapshot (Phase 1 gate).
  5. Uploads the parquet artifacts to Cloudflare R2 via rclone, preserving
     the same directory layout under the configured remote prefix.

Funding rates already exist on R2 for moleapp's universe, so we fetch them
once at the start via r2_funding_fetcher rather than re-downloading from
Binance Vision.

Resumption: the script is idempotent. Re-running with the same args skips
days already downloaded, parsed, validated, and uploaded (state is tracked
on disk; --overwrite forces re-execution).

Example:
    python -m scripts.prepare_l2_dataset \\
        --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --start 2024-01-01 --end 2024-01-31 \\
        --r2-remote moleapp-r2 \\
        --r2-prefix momodkr/datasets
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from data.collectors.asset_config import REVERSE_BINANCE_MAP
from data.collectors.binance_vision_l2_collector import (
    STREAMS,
    build_tasks,
    fetch_all,
    parse_and_persist,
)
from data.collectors.r2_funding_fetcher import fetch_asset_funding, rclone_available
from data.reconstructors.order_book_reconstructor import reconstruct_day
from data.validators.validate_l2_data import validate_snapshots

logger = logging.getLogger(__name__)


@dataclass
class DayResult:
    symbol: str
    day: date
    snapshot_path: Path | None
    validator_passed: bool
    validator_summary: str
    uploaded: bool
    error: str | None = None


def _funding_local_path(dataset_root: Path, symbol: str) -> Path | None:
    """Pick the freshest local funding parquet for a symbol, if any."""
    d = dataset_root / symbol / "fundingRate"
    if not d.exists():
        return None
    files = sorted(d.glob("*funding*.parquet"))
    return files[-1] if files else None


def _kline_local_path(dataset_root: Path, symbol: str) -> Path | None:
    d = dataset_root / symbol / "klines"
    if not d.exists():
        return None
    files = sorted(d.glob("*.parquet"))
    return files[-1] if files else None


def _rclone_upload(local_dir: Path, remote: str, remote_prefix: str, dry_run: bool = False) -> None:
    """Sync local_dir up to remote:remote_prefix/<local_dir.name>/ via rclone."""
    src = str(local_dir)
    dst = f"{remote}:{remote_prefix.rstrip('/')}"
    cmd = ["rclone", "copy", src, dst, "--include", "*.parquet", "--transfers", "8"]
    if dry_run:
        cmd.append("--dry-run")
    logger.info("rclone upload: %s -> %s", src, dst)
    subprocess.run(cmd, check=True)


def _ensure_funding(
    symbols: list[str],
    dataset_root: Path,
    r2_remote: str,
    r2_funding_prefix: str,
    skip: bool,
) -> None:
    if skip:
        logger.info("--skip-funding set; not pulling funding from R2")
        return
    if not rclone_available():
        logger.warning("rclone not available; skipping funding pull")
        return
    for sym in symbols:
        asset = REVERSE_BINANCE_MAP.get(sym, sym)
        try:
            fetch_asset_funding(r2_remote, r2_funding_prefix, asset, dataset_root)
        except subprocess.CalledProcessError as e:
            logger.warning("funding pull failed for %s: %s", asset, e)


def _process_day(
    symbol: str,
    day: date,
    dataset_root: Path,
    grid_ms: int,
    r2_remote: str | None,
    r2_prefix: str | None,
    overwrite_snapshot: bool,
) -> DayResult:
    day_iso = day.isoformat()
    snapshot_dir = dataset_root / symbol / "snapshots"
    snapshot_path = snapshot_dir / f"{day_iso}.parquet"

    if snapshot_path.exists() and not overwrite_snapshot:
        try:
            snaps = pd.read_parquet(snapshot_path)
        except Exception as e:
            return DayResult(symbol, day, None, False, "", False, error=f"read existing snapshot failed: {e}")
    else:
        funding_path = _funding_local_path(dataset_root, symbol)
        try:
            snaps = reconstruct_day(symbol, day_iso, dataset_root, funding_path, grid_ms=grid_ms)
        except FileNotFoundError as e:
            return DayResult(symbol, day, None, False, "", False, error=f"reconstruction missing input: {e}")
        except Exception as e:
            return DayResult(symbol, day, None, False, "", False, error=f"reconstruction failed: {e}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snaps.to_parquet(snapshot_path, index=False, compression="zstd")

    kline_path = _kline_local_path(dataset_root, symbol)
    klines = pd.read_parquet(kline_path) if kline_path else None
    res = validate_snapshots(snaps, kline_1h=klines, label=f"{symbol}/{day_iso}", grid_ms=grid_ms)
    summary = res.summary()
    if not res.passed:
        return DayResult(symbol, day, snapshot_path, False, summary, False, error="validation failed")

    uploaded = False
    if r2_remote and r2_prefix:
        for stream in (*STREAMS, "snapshots"):
            stream_dir = dataset_root / symbol / stream
            if stream_dir.exists():
                _rclone_upload(stream_dir, r2_remote, f"{r2_prefix.rstrip('/')}/{symbol}/{stream}")
        uploaded = True

    return DayResult(symbol, day, snapshot_path, True, summary, uploaded)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Prepare and upload MomoDkr L2 dataset")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--raw-root", default="data/raw/binance_vision")
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--grid-ms", type=int, default=100)
    p.add_argument("--workers", type=int, default=8, help="HTTP download concurrency")
    p.add_argument("--reconstruct-workers", type=int, default=4, help="per-day reconstruct concurrency")
    p.add_argument("--overwrite-downloads", action="store_true")
    p.add_argument("--overwrite-snapshots", action="store_true")
    p.add_argument("--streams", nargs="+", default=list(STREAMS))
    p.add_argument("--r2-remote", default="moleapp-r2", help="rclone remote name; '' to disable upload")
    p.add_argument("--r2-prefix", default="momodkr/datasets", help="remote prefix for uploaded parquets")
    p.add_argument("--r2-funding-prefix", default="datasets", help="prefix where moleapp funding parquets live")
    p.add_argument("--skip-funding", action="store_true")
    p.add_argument("--upload-only", action="store_true", help="skip download/parse/reconstruct; only sync local to R2")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    raw_root = Path(args.raw_root)
    ds_root = Path(args.dataset_root)
    r2_remote = args.r2_remote or None
    r2_prefix = args.r2_prefix or None

    t0 = time.time()

    if args.upload_only:
        if not (r2_remote and r2_prefix and rclone_available()):
            sys.exit("upload-only requires r2-remote, r2-prefix, and rclone on PATH")
        for sym in args.symbols:
            for stream in (*STREAMS, "snapshots", "fundingRate"):
                d = ds_root / sym / stream
                if d.exists():
                    _rclone_upload(d, r2_remote, f"{r2_prefix.rstrip('/')}/{sym}/{stream}")
        logger.info("upload-only complete in %.1fs", time.time() - t0)
        return

    _ensure_funding(args.symbols, ds_root, r2_remote or "moleapp-r2", args.r2_funding_prefix, args.skip_funding)

    tasks = build_tasks(args.symbols, start, end, streams=args.streams)
    logger.info("planned %d download tasks", len(tasks))
    fetched = fetch_all(tasks, raw_root, max_workers=args.workers, overwrite=args.overwrite_downloads)
    parsed = parse_and_persist(fetched, ds_root, overwrite=args.overwrite_downloads)
    logger.info("parsed %d parquet files", len(parsed))

    day_args = [
        (sym, d)
        for sym in args.symbols
        for d in pd.date_range(start, end, freq="D").date
    ]
    results: list[DayResult] = []
    with ProcessPoolExecutor(max_workers=args.reconstruct_workers) as ex:
        futs = {
            ex.submit(
                _process_day,
                sym,
                day,
                ds_root,
                args.grid_ms,
                r2_remote,
                r2_prefix,
                args.overwrite_snapshots,
            ): (sym, day)
            for sym, day in day_args
        }
        for fut in as_completed(futs):
            sym, day = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(DayResult(sym, day, None, False, "", False, error=f"process failed: {e}"))

    ok = [r for r in results if r.validator_passed]
    bad = [r for r in results if not r.validator_passed]
    logger.info("validated %d/%d days", len(ok), len(results))
    for r in bad[:10]:
        logger.error("FAILED %s %s: %s", r.symbol, r.day, r.error or r.validator_summary.splitlines()[0])
    if bad:
        sys.exit(f"{len(bad)} day(s) failed validation")

    logger.info("prepare_l2_dataset complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
