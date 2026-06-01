"""Cloudflare R2 sync for MomoDkr (boto3, S3-compatible).

Dedicated `momodkr-data` bucket (Path B per 2026-06-01 decision):
MomoDkr owns its own R2 bucket end-to-end. Permissions are isolated from
moleapp's bucket, so a leaked / over-scoped token can't clobber moleapp
data and vice versa. The bucket root holds the data directly (no
`momodkr/` prefix; the whole bucket is ours).

Env vars (set in RunPod Pod Settings -> Environment Variables):

    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_ENDPOINT_URL       (default: moleapp's account endpoint -- same Cloudflare account)
    R2_BUCKET_NAME        (default: momodkr-data)
    MOMODKR_R2_PREFIX     (default: "" -- empty; the whole bucket is the project's namespace)

Usage:
    # upload local dataset tree to R2
    python -m scripts.r2_sync upload --local data/datasets

    # download a single symbol's snapshots from R2
    python -m scripts.r2_sync download --local data/datasets --filter BTCUSDT/snapshots

    # list a prefix
    python -m scripts.r2_sync list --filter BTCUSDT/

    # cross-bucket: pull moleapp's funding parquet for one asset (requires DIFFERENT creds
    # via R2_MOLEAPP_ACCESS_KEY_ID / R2_MOLEAPP_SECRET_ACCESS_KEY env vars)
    python -m scripts.r2_sync pull-moleapp-funding --asset BTC --local data/datasets/BTCUSDT/fundingRate
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://9507330fe5a8c228ea49f6e5c6c6b659.r2.cloudflarestorage.com"
DEFAULT_BUCKET = "momodkr-data"
DEFAULT_MOMODKR_PREFIX = ""
DEFAULT_MOLEAPP_BUCKET = "moleapp-rl-data"
DEFAULT_MOLEAPP_PROCESSED_PREFIX = "processed/1h/"


def get_client():
    endpoint = os.getenv("R2_ENDPOINT_URL", DEFAULT_ENDPOINT)
    access_key = os.getenv("R2_ACCESS_KEY_ID", "")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "")
    if not (access_key and secret_key):
        raise SystemExit(
            "R2 credentials missing. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY "
            "(see .env.example)."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        # max_pool_connections raised from boto3's default 10 -- the ingest
        # uses ProcessPoolExecutor(4) with each worker spawning ThreadPoolExecutor(8)
        # uploads, so 32 parallel sockets is the realistic peak.
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=64,
        ),
        region_name="auto",
    )


def get_bucket() -> str:
    return os.getenv("R2_BUCKET_NAME", DEFAULT_BUCKET)


def get_prefix() -> str:
    p = os.getenv("MOMODKR_R2_PREFIX", DEFAULT_MOMODKR_PREFIX)
    if not p:
        return ""
    return p if p.endswith("/") else p + "/"


def _iter_local_files(local_root: Path, patterns: tuple[str, ...] = ("*.parquet",)) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(local_root.rglob(pat))
    return sorted(files)


def upload_tree(
    local_root: Path,
    bucket: str | None = None,
    prefix: str | None = None,
    filter_substr: str | None = None,
    workers: int = 8,
    overwrite: bool = False,
) -> int:
    client = get_client()
    bucket = bucket or get_bucket()
    prefix = prefix or get_prefix()
    files = _iter_local_files(local_root)
    if filter_substr:
        files = [f for f in files if filter_substr in str(f.relative_to(local_root))]
    if not files:
        logger.warning("no files to upload under %s", local_root)
        return 0

    def _key(p: Path) -> str:
        return f"{prefix}{p.relative_to(local_root).as_posix()}"

    def _exists(k: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=k)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _upload(p: Path) -> tuple[Path, bool]:
        k = _key(p)
        if not overwrite and _exists(k):
            return p, False
        client.upload_file(str(p), bucket, k)
        return p, True

    uploaded = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_upload, p): p for p in files}
        for fut in as_completed(futs):
            p, did = fut.result()
            if did:
                uploaded += 1
                logger.info("uploaded %s", _key(p))
    logger.info("uploaded %d/%d files to s3://%s/%s", uploaded, len(files), bucket, prefix)
    return uploaded


def download_tree(
    local_root: Path,
    bucket: str | None = None,
    prefix: str | None = None,
    filter_substr: str | None = None,
    workers: int = 8,
    overwrite: bool = False,
) -> int:
    client = get_client()
    bucket = bucket or get_bucket()
    prefix = prefix or get_prefix()
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if filter_substr and filter_substr not in k:
                continue
            keys.append(k)
    if not keys:
        logger.warning("no keys to download from s3://%s/%s%s", bucket, prefix, filter_substr or "")
        return 0

    def _download(k: str) -> tuple[str, bool]:
        rel = k[len(prefix):] if k.startswith(prefix) else k
        dest = local_root / rel
        if dest.exists() and not overwrite:
            return k, False
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, k, str(dest))
        return k, True

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download, k) for k in keys]
        for fut in as_completed(futs):
            _, did = fut.result()
            if did:
                n += 1
    logger.info("downloaded %d/%d objects to %s", n, len(keys), local_root)
    return n


def list_keys(bucket: str | None = None, prefix: str | None = None, filter_substr: str | None = None) -> list[str]:
    client = get_client()
    bucket = bucket or get_bucket()
    prefix = prefix or get_prefix()
    out: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if filter_substr and filter_substr not in k:
                continue
            out.append(k)
    return out


def _get_moleapp_client():
    """Separate boto3 client for cross-bucket reads from moleapp's bucket.

    Uses R2_MOLEAPP_ACCESS_KEY_ID / R2_MOLEAPP_SECRET_ACCESS_KEY if set,
    otherwise falls back to the main R2_* creds (only useful if the main
    token has access to BOTH buckets, which we avoid in Path B).
    """
    endpoint = os.getenv("R2_ENDPOINT_URL", DEFAULT_ENDPOINT)
    access_key = os.getenv("R2_MOLEAPP_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID", "")
    secret_key = os.getenv("R2_MOLEAPP_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY", "")
    if not (access_key and secret_key):
        raise SystemExit(
            "pull-moleapp-funding needs R2_MOLEAPP_ACCESS_KEY_ID + R2_MOLEAPP_SECRET_ACCESS_KEY "
            "(token scoped to the moleapp-rl-data bucket)."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "adaptive"}),
        region_name="auto",
    )


def pull_moleapp_funding(asset: str, local_dir: Path, bucket: str | None = None) -> list[Path]:
    """Pull funding parquet(s) for an asset from moleapp's processed/1h prefix.

    Requires moleapp-scoped credentials (see _get_moleapp_client). This is
    an OPT-IN convenience; the default ingest path fetches fresh funding
    from Binance Vision monthly archives instead.
    """
    client = _get_moleapp_client()
    bucket = bucket or DEFAULT_MOLEAPP_BUCKET
    prefix = DEFAULT_MOLEAPP_PROCESSED_PREFIX

    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            name = k[len(prefix):]
            if name.startswith(f"{asset}_funding") and name.endswith(".parquet"):
                keys.append(k)
    if not keys:
        logger.warning("no funding parquet for asset=%s under s3://%s/%s", asset, bucket, prefix)
        return []

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for k in keys:
        dest = local_dir / k.split("/")[-1]
        client.download_file(bucket, k, str(dest))
        downloaded.append(dest)
        logger.info("pulled %s -> %s", k, dest)
    return downloaded


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="MomoDkr R2 sync (boto3)")
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload")
    up.add_argument("--local", required=True)
    up.add_argument("--filter", default=None, help="substring filter on relative paths")
    up.add_argument("--workers", type=int, default=8)
    up.add_argument("--overwrite", action="store_true")

    dn = sub.add_parser("download")
    dn.add_argument("--local", required=True)
    dn.add_argument("--filter", default=None, help="substring filter on R2 keys")
    dn.add_argument("--workers", type=int, default=8)
    dn.add_argument("--overwrite", action="store_true")

    ls = sub.add_parser("list")
    ls.add_argument("--filter", default=None)

    fund = sub.add_parser("pull-moleapp-funding")
    fund.add_argument("--asset", required=True)
    fund.add_argument("--local", required=True)

    args = p.parse_args()
    if args.cmd == "upload":
        upload_tree(Path(args.local), filter_substr=args.filter, workers=args.workers, overwrite=args.overwrite)
    elif args.cmd == "download":
        download_tree(Path(args.local), filter_substr=args.filter, workers=args.workers, overwrite=args.overwrite)
    elif args.cmd == "list":
        for k in list_keys(filter_substr=args.filter):
            print(k)
    elif args.cmd == "pull-moleapp-funding":
        pull_moleapp_funding(args.asset, Path(args.local))


if __name__ == "__main__":
    main()
