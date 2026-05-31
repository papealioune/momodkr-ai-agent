"""Pull existing funding-rate parquet files from Cloudflare R2 via boto3.

Reuses moleapp's bucket + processed/1h/ prefix where funding parquets
already live (e.g. moleapp-rl-data/processed/1h/BTC_funding_730d.parquet).
No new credentials needed beyond the standard R2_* env vars consumed by
scripts/r2_sync.py.

Usage:
    python -m data.collectors.r2_funding_fetcher --assets BTC ETH SOL \\
        --dataset-root data/datasets
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.collectors.asset_config import ALLOWED_ASSETS, BINANCE_SYMBOL_MAP
from scripts.r2_sync import pull_moleapp_funding

logger = logging.getLogger(__name__)


def fetch_asset_funding(
    asset: str,
    dataset_root: Path,
) -> list[Path]:
    """Download all funding-rate parquet files for one asset into the local dataset tree."""
    symbol = BINANCE_SYMBOL_MAP.get(asset)
    if not symbol:
        logger.warning("no Binance symbol for asset %s; skipping", asset)
        return []
    dest = dataset_root / symbol / "fundingRate"
    return pull_moleapp_funding(asset, dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Fetch funding parquets from moleapp R2 prefix")
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    p.add_argument("--dataset-root", default="data/datasets")
    args = p.parse_args()

    unknown = [a for a in args.assets if a not in ALLOWED_ASSETS]
    if unknown:
        raise SystemExit(f"unknown assets: {unknown}; allowed: {ALLOWED_ASSETS}")

    root = Path(args.dataset_root)
    for asset in args.assets:
        logger.info("fetching funding for %s", asset)
        files = fetch_asset_funding(asset, root)
        logger.info("  -> %d file(s) under %s", len(files), root / BINANCE_SYMBOL_MAP[asset] / "fundingRate")


if __name__ == "__main__":
    main()
