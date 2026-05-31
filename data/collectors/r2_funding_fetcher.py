"""Pull existing funding-rate parquet files from Cloudflare R2 via rclone.

Reuses the same rclone remote already configured for moleapp-rl-training
(see `rclone listremotes`). No new auth needed for dev access.

Funding rate data already exists on R2 for the 15-asset moleapp universe
including BTC, ETH, SOL. This script downloads the per-asset funding
parquet into MomoDkr's local dataset tree.

Usage:
    python -m data.collectors.r2_funding_fetcher \\
        --remote moleapp-r2 \\
        --remote-prefix datasets \\
        --assets BTC ETH SOL \\
        --dataset-root data/datasets
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from data.collectors.asset_config import ALLOWED_ASSETS, BINANCE_SYMBOL_MAP

logger = logging.getLogger(__name__)


def rclone_available() -> bool:
    try:
        subprocess.run(["rclone", "version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def list_remote_funding_files(
    remote: str,
    remote_prefix: str,
    asset: str,
) -> list[str]:
    """List funding-rate parquet filenames available for an asset on the remote."""
    src = f"{remote}:{remote_prefix.rstrip('/')}/{asset}/"
    result = subprocess.run(
        ["rclone", "lsf", src, "--include", "*funding*.parquet"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def fetch_asset_funding(
    remote: str,
    remote_prefix: str,
    asset: str,
    dataset_root: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Copy all funding-rate parquet files for one asset from R2 to local."""
    symbol = BINANCE_SYMBOL_MAP.get(asset)
    if not symbol:
        logger.warning("no Binance symbol for asset %s; skipping", asset)
        return []

    dest_dir = dataset_root / symbol / "fundingRate"
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list_remote_funding_files(remote, remote_prefix, asset)
    if not files:
        logger.warning("no funding files found at %s:%s/%s/", remote, remote_prefix, asset)
        return []

    src = f"{remote}:{remote_prefix.rstrip('/')}/{asset}/"
    cmd = ["rclone", "copy", src, str(dest_dir), "--include", "*funding*.parquet", "--progress"]
    if not overwrite:
        cmd.append("--ignore-existing")
    subprocess.run(cmd, check=True)

    return sorted(dest_dir.glob("*funding*.parquet"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Fetch funding parquets from R2 via rclone")
    p.add_argument("--remote", default="moleapp-r2", help="rclone remote name (default reuses moleapp config)")
    p.add_argument("--remote-prefix", default="datasets", help="prefix within the bucket")
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not rclone_available():
        raise SystemExit("rclone not found on PATH; install it and re-run")

    unknown = [a for a in args.assets if a not in ALLOWED_ASSETS]
    if unknown:
        raise SystemExit(f"unknown assets: {unknown}; allowed: {ALLOWED_ASSETS}")

    root = Path(args.dataset_root)
    for asset in args.assets:
        logger.info("fetching funding for %s", asset)
        files = fetch_asset_funding(args.remote, args.remote_prefix, asset, root, overwrite=args.overwrite)
        logger.info("  -> %d file(s) under %s", len(files), root / BINANCE_SYMBOL_MAP[asset] / "fundingRate")


if __name__ == "__main__":
    main()
