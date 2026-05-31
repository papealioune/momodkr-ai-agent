"""Phase-2 orchestrator: snapshots -> per-day features -> chronological episodes.

For each symbol:
  1. Iterate per-day snapshot parquets under data/datasets/<SYMBOL>/snapshots/
  2. Build per-day market-feature parquets under data/datasets/<SYMBOL>/features/
  3. Concatenate + chronological 80/20 split + persist as episodes
  4. Optionally upload features + episodes to R2 under momodkr/

Idempotent: skips per-day feature parquets that already exist unless --rebuild.

Example:
    python -m scripts.build_features \\
        --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --split-ratio 0.8
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from data.preprocessors.episode_builder import build_episodes_for_symbols
from data.preprocessors.feature_engineer import build_features_for_day
from scripts.r2_sync import upload_tree

logger = logging.getLogger(__name__)


def _build_one(symbol: str, day_iso: str, dataset_root: Path, rebuild: bool) -> tuple[str, str, bool, str | None]:
    feat_path = dataset_root / symbol / "features" / f"{day_iso}.parquet"
    if feat_path.exists() and not rebuild:
        return symbol, day_iso, True, None
    try:
        dest = build_features_for_day(symbol, day_iso, dataset_root)
        return symbol, day_iso, dest is not None, None if dest else "no output"
    except Exception as e:
        return symbol, day_iso, False, str(e)


def _enumerate_days(symbol: str, dataset_root: Path) -> list[str]:
    snap_dir = dataset_root / symbol / "snapshots"
    if not snap_dir.exists():
        return []
    return sorted(p.stem for p in snap_dir.glob("*.parquet"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build market features + chronological episodes")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--episodes-root", default="data/episodes")
    p.add_argument("--split-ratio", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rebuild", action="store_true", help="recompute per-day features even if present")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()

    ds_root = Path(args.dataset_root)
    ep_root = Path(args.episodes_root)
    t0 = time.time()

    tasks: list[tuple[str, str]] = []
    for sym in args.symbols:
        days = _enumerate_days(sym, ds_root)
        if not days:
            logger.warning("no snapshots for %s under %s", sym, ds_root / sym / "snapshots")
            continue
        tasks.extend((sym, d) for d in days)
    logger.info("planned %d (symbol, day) feature builds", len(tasks))

    n_ok = 0
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_build_one, sym, d, ds_root, args.rebuild) for sym, d in tasks]
        for fut in as_completed(futs):
            sym, d, ok, err = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                logger.error("feature build failed %s/%s: %s", sym, d, err)
    logger.info("features: %d ok / %d failed", n_ok, n_fail)
    if n_fail:
        raise SystemExit(f"{n_fail} day(s) failed feature build")

    manifests = build_episodes_for_symbols(args.symbols, ds_root, ep_root, args.split_ratio)
    for sym, m in manifests.items():
        logger.info("  %s: train_rows=%d eval_rows=%d split=%.2f", sym, m.train_rows, m.eval_rows, m.split_ratio)

    if not args.no_upload:
        # features land at s3://.../momodkr/<SYM>/features/<day>.parquet (same layout as snapshots)
        for sym in args.symbols:
            upload_tree(ds_root, filter_substr=f"{sym}/features/")
        # episodes land at s3://.../momodkr/episodes/<SYM>/<ver>/{train,eval}.parquet (+ manifest)
        upload_tree(ep_root.parent, filter_substr=f"{ep_root.name}/")

    logger.info("build_features done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
