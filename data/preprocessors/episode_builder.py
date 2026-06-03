"""Build train/eval episodes from per-day feature parquets.

For each symbol:
  1. Concatenate per-day feature parquets in chronological order.
  2. Split 80/20 by row count (NEVER by shuffle) -- moleapp lesson on
     time-series leakage. Train is the past, eval is the strict future.
  3. Persist as:
        data/episodes/<SYMBOL>/<feature_version>/train.parquet
        data/episodes/<SYMBOL>/<feature_version>/eval.parquet
     plus a manifest.json with row counts, time ranges, checksum.

The env loads (train|eval).parquet at startup and indexes into it. No
shuffling, no random sampling of episode start -- the env's reset() picks
a chronological window from train/eval slabs and walks forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data.preprocessors.feature_stats import (
    DEFAULT_CLIP,
    NormStats,
    compute_norm_stats,
    norm_stats_path_for_episodes,
    save_norm_stats,
)
from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_NAMES,
    SIM_STATE_COLS,
)


class _WelfordStats:
    """Online (single-pass) per-column mean + std using Welford's algorithm.

    Memory: O(n_features) regardless of input dataset size. Lets us
    stream the 30 GB+ per-symbol concat without materialising it.
    """

    def __init__(self, n_features: int) -> None:
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update_batch(self, x: np.ndarray) -> None:
        """x: (batch_n, n_features) float64."""
        batch_n = x.shape[0]
        if batch_n == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self.mean
        total_n = self.n + batch_n
        self.mean += delta * batch_n / total_n
        self.m2 += batch_m2 + delta**2 * self.n * batch_n / total_n
        self.n = total_n

    def finalise(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(self.m2 / self.n) if self.n > 0 else np.ones_like(self.mean)
        return self.mean.astype(np.float32), std.astype(np.float32)

logger = logging.getLogger(__name__)


@dataclass
class EpisodeManifest:
    symbol: str
    feature_version: str
    feature_spec_checksum: str
    train_rows: int
    eval_rows: int
    train_start_ms: int
    train_end_ms: int
    eval_start_ms: int
    eval_end_ms: int
    split_ratio: float
    data_sha256_prefix: str  # first 16 chars of sha256 over the concatenated feature bytes
    norm_stats_path: str = ""
    norm_clip: float = 0.0


def _list_feature_days(
    symbol: str,
    dataset_root: Path,
    train_start: str | None = None,
    train_end: str | None = None,
) -> list[Path]:
    """List per-day feature parquets, optionally filtered by [train_start, train_end] inclusive (YYYY-MM-DD)."""
    feat_dir = dataset_root / symbol / "features"
    if not feat_dir.exists():
        return []
    all_paths = sorted(feat_dir.glob("*.parquet"))
    if train_start is None and train_end is None:
        return all_paths
    filtered: list[Path] = []
    for p in all_paths:
        day = p.stem
        if train_start is not None and day < train_start:
            continue
        if train_end is not None and day > train_end:
            continue
        filtered.append(p)
    return filtered


def _concat_features(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p)
        if "ts_ms" not in df.columns:
            raise KeyError(f"missing ts_ms in {p}")
        if df.get("feature_version", pd.Series([FEATURE_VERSION])).iloc[0] != FEATURE_VERSION:
            raise ValueError(
                f"feature_version mismatch in {p}: got {df['feature_version'].iloc[0]!r} expected {FEATURE_VERSION!r}"
            )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True).sort_values("ts_ms").drop_duplicates("ts_ms")
    if not full["ts_ms"].is_monotonic_increasing:
        raise ValueError("feature concatenation produced non-monotonic ts_ms")
    return full.reset_index(drop=True)


def build_episodes(
    symbol: str,
    dataset_root: Path,
    episodes_root: Path,
    split_ratio: float = 0.8,
    norm_clip: float = DEFAULT_CLIP,
    train_start: str | None = None,
    train_end: str | None = None,
    label: str | None = None,
    batch_rows: int = 200_000,
) -> EpisodeManifest:
    """Build chronological train/eval episodes for one symbol -- streaming.

    Memory: O(batch_rows * n_cols) ≈ 50-100 MB regardless of dataset size.
    The previous implementation pd.concat'd ~30 GB of per-day features into
    one DataFrame, hitting OOM on memory-capped containers; this version
    streams via pyarrow record batches and computes z-score stats via
    Welford's online algorithm.

    train_start / train_end (inclusive, YYYY-MM-DD) restrict the source
    days. The 80/20 split applies AFTER filtering.
    """
    if not 0.5 <= split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in [0.5, 1.0), got {split_ratio}")

    paths = _list_feature_days(symbol, dataset_root, train_start=train_start, train_end=train_end)
    if not paths:
        raise FileNotFoundError(
            f"no feature parquets under {dataset_root / symbol / 'features'} matching window "
            f"[{train_start}..{train_end}]"
        )

    # Validate feature_version on the first file (cheap -- reads 1 row).
    first_meta = pq.read_table(paths[0], columns=["feature_version"]).to_pandas()
    if not first_meta.empty and first_meta["feature_version"].iloc[0] != FEATURE_VERSION:
        raise ValueError(
            f"feature_version mismatch in {paths[0]}: got "
            f"{first_meta['feature_version'].iloc[0]!r} expected {FEATURE_VERSION!r}"
        )

    # Determine output columns (drop feature_version; keep ts_ms + market + sim).
    schema_names = {f.name for f in pq.read_schema(paths[0])}
    if "ts_ms" not in schema_names:
        raise KeyError(f"missing ts_ms in {paths[0]}")
    sim_cols_present = [c for c in SIM_STATE_COLS if c in schema_names]
    output_cols = ["ts_ms", *MARKET_FEATURE_NAMES, *sim_cols_present]

    # First pass: count rows per file from parquet metadata (does not read data).
    row_counts = [pq.read_metadata(p).num_rows for p in paths]
    total_rows = sum(row_counts)
    if total_rows == 0:
        raise ValueError("no rows after concat")
    split_rows = int(total_rows * split_ratio)

    suffix = f"_{label}" if label else ""
    out_dir = episodes_root / symbol / f"{FEATURE_VERSION}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    eval_path = out_dir / "eval.parquet"

    welford = _WelfordStats(len(MARKET_FEATURE_NAMES))
    sha = hashlib.sha256()

    train_writer: pq.ParquetWriter | None = None
    eval_writer: pq.ParquetWriter | None = None
    rows_emitted = 0
    train_first_ts: int | None = None
    train_last_ts: int | None = None
    eval_first_ts: int | None = None
    eval_last_ts: int | None = None
    prev_ts: int | None = None  # for monotonicity check across batches

    # Second pass: stream batches and write incrementally.
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=output_cols):
            n_in_batch = batch.num_rows
            if n_in_batch == 0:
                continue

            batch_end_pos = rows_emitted + n_in_batch
            if batch_end_pos <= split_rows:
                train_part, eval_part = batch, None
            elif rows_emitted >= split_rows:
                train_part, eval_part = None, batch
            else:
                cut = split_rows - rows_emitted
                train_part = batch.slice(0, cut)
                eval_part = batch.slice(cut)

            if train_part is not None and train_part.num_rows > 0:
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_path, train_part.schema, compression="zstd")
                train_writer.write_batch(train_part)
                # Welford update on the 26 market features
                market_np = np.stack(
                    [train_part.column(c).to_numpy(zero_copy_only=False) for c in MARKET_FEATURE_NAMES],
                    axis=1,
                ).astype(np.float64)
                welford.update_batch(market_np)
                # SHA-256 streaming over market columns
                for c in MARKET_FEATURE_NAMES:
                    sha.update(train_part.column(c).to_numpy(zero_copy_only=False).tobytes())
                # Track ts bounds
                ts_arr = train_part.column("ts_ms")
                if train_first_ts is None:
                    train_first_ts = int(ts_arr[0].as_py())
                train_last_ts = int(ts_arr[-1].as_py())

            if eval_part is not None and eval_part.num_rows > 0:
                if eval_writer is None:
                    eval_writer = pq.ParquetWriter(eval_path, eval_part.schema, compression="zstd")
                eval_writer.write_batch(eval_part)
                ts_arr = eval_part.column("ts_ms")
                if eval_first_ts is None:
                    eval_first_ts = int(ts_arr[0].as_py())
                eval_last_ts = int(ts_arr[-1].as_py())

            # Monotonic ts_ms check (cheap: just first vs prev_last)
            first_ts = int(batch.column("ts_ms")[0].as_py())
            if prev_ts is not None and first_ts <= prev_ts:
                raise ValueError(
                    f"feature concatenation produced non-monotonic ts_ms at {p}: "
                    f"prev={prev_ts} cur_first={first_ts}"
                )
            prev_ts = int(batch.column("ts_ms")[-1].as_py())

            rows_emitted += n_in_batch

    if train_writer is not None:
        train_writer.close()
    if eval_writer is not None:
        eval_writer.close()

    if eval_first_ts is not None and train_last_ts is not None and eval_first_ts <= train_last_ts:
        raise ValueError("chronological split violated: eval starts at or before train end")

    mean, std = welford.finalise()
    stats = NormStats(
        feature_version=FEATURE_VERSION,
        feature_spec_checksum=FEATURE_SPEC_CHECKSUM,
        n_train_rows=int(welford.n),
        clip=float(norm_clip),
        mean=mean.tolist(),
        std=std.tolist(),
    )
    stats_path = save_norm_stats(stats, norm_stats_path_for_episodes(out_dir))

    manifest = EpisodeManifest(
        symbol=symbol,
        feature_version=FEATURE_VERSION,
        feature_spec_checksum=FEATURE_SPEC_CHECKSUM,
        train_rows=int(welford.n),
        eval_rows=total_rows - int(welford.n),
        train_start_ms=int(train_first_ts) if train_first_ts is not None else 0,
        train_end_ms=int(train_last_ts) if train_last_ts is not None else 0,
        eval_start_ms=int(eval_first_ts) if eval_first_ts is not None else 0,
        eval_end_ms=int(eval_last_ts) if eval_last_ts is not None else 0,
        split_ratio=split_ratio,
        data_sha256_prefix=sha.hexdigest()[:16],
        norm_stats_path=str(stats_path),
        norm_clip=float(stats.clip),
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    return manifest


def build_episodes_for_symbols(
    symbols: list[str],
    dataset_root: Path,
    episodes_root: Path,
    split_ratio: float = 0.8,
    norm_clip: float = DEFAULT_CLIP,
    train_start: str | None = None,
    train_end: str | None = None,
    label: str | None = None,
) -> dict[str, EpisodeManifest]:
    out: dict[str, EpisodeManifest] = {}
    for sym in symbols:
        try:
            out[sym] = build_episodes(
                sym, dataset_root, episodes_root,
                split_ratio=split_ratio, norm_clip=norm_clip,
                train_start=train_start, train_end=train_end, label=label,
            )
            logger.info("episodes for %s: %d train / %d eval", sym, out[sym].train_rows, out[sym].eval_rows)
        except FileNotFoundError as e:
            logger.warning("skip %s: %s", sym, e)
    return out


def build_walk_forward_split(
    symbol: str,
    dataset_root: Path,
    episodes_root: Path,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    norm_clip: float = DEFAULT_CLIP,
    label: str | None = None,
) -> EpisodeManifest:
    """Build a strict walk-forward slice: train on [train_start..train_end],
    eval on [eval_start..eval_end]. Stats are computed on train ONLY.

    Both windows are inclusive YYYY-MM-DD. Eval must start strictly after
    train ends (no overlap; moleapp leakage lesson).
    """
    if eval_start <= train_end:
        raise ValueError(f"walk-forward requires eval_start ({eval_start}) > train_end ({train_end})")

    train_paths = _list_feature_days(symbol, dataset_root, train_start=train_start, train_end=train_end)
    eval_paths = _list_feature_days(symbol, dataset_root, train_start=eval_start, train_end=eval_end)
    if not train_paths:
        raise FileNotFoundError(f"no train features for {symbol} in [{train_start}..{train_end}]")
    if not eval_paths:
        raise FileNotFoundError(f"no eval features for {symbol} in [{eval_start}..{eval_end}]")

    train_full = _concat_features(train_paths)
    eval_full = _concat_features(eval_paths)
    sim_cols = [c for c in SIM_STATE_COLS if c in train_full.columns]
    train_full = train_full[["ts_ms", *MARKET_FEATURE_NAMES, *sim_cols]]
    eval_full = eval_full[["ts_ms", *MARKET_FEATURE_NAMES, *sim_cols]]

    suffix = f"_{label}" if label else "_walkforward"
    out_dir = episodes_root / symbol / f"{FEATURE_VERSION}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_full.to_parquet(out_dir / "train.parquet", index=False, compression="zstd")
    eval_full.to_parquet(out_dir / "eval.parquet", index=False, compression="zstd")

    stats = compute_norm_stats(train_full, clip=norm_clip)
    stats_path = save_norm_stats(stats, norm_stats_path_for_episodes(out_dir))

    sha = hashlib.sha256()
    for col in MARKET_FEATURE_NAMES:
        sha.update(train_full[col].to_numpy().tobytes())
    data_hex = sha.hexdigest()[:16]

    manifest = EpisodeManifest(
        symbol=symbol,
        feature_version=FEATURE_VERSION,
        feature_spec_checksum=FEATURE_SPEC_CHECKSUM,
        train_rows=len(train_full),
        eval_rows=len(eval_full),
        train_start_ms=int(train_full["ts_ms"].iloc[0]),
        train_end_ms=int(train_full["ts_ms"].iloc[-1]),
        eval_start_ms=int(eval_full["ts_ms"].iloc[0]),
        eval_end_ms=int(eval_full["ts_ms"].iloc[-1]),
        split_ratio=1.0,
        data_sha256_prefix=data_hex,
        norm_stats_path=str(stats_path),
        norm_clip=float(stats.clip),
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build chronological train/eval episodes from feature parquets")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--episodes-root", default="data/episodes")
    p.add_argument("--split-ratio", type=float, default=0.8)
    args = p.parse_args()
    build_episodes_for_symbols(args.symbols, Path(args.dataset_root), Path(args.episodes_root), args.split_ratio)


if __name__ == "__main__":
    main()
