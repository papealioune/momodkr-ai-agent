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

import pandas as pd

from data.preprocessors.feature_stats import (
    DEFAULT_CLIP,
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
) -> EpisodeManifest:
    """Build chronological train/eval episodes for one symbol.

    train_start / train_end (inclusive, YYYY-MM-DD) restrict the source
    days that feed into the concat. The 80/20 split still applies AFTER
    filtering, so the split is on the filtered window. For pure
    walk-forward (fixed train window then fixed eval window) call
    build_walk_forward_split() instead.
    """
    if not 0.5 <= split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in [0.5, 1.0), got {split_ratio}")

    paths = _list_feature_days(symbol, dataset_root, train_start=train_start, train_end=train_end)
    if not paths:
        raise FileNotFoundError(
            f"no feature parquets under {dataset_root / symbol / 'features'} matching window "
            f"[{train_start}..{train_end}]"
        )

    full = _concat_features(paths)
    if full.empty:
        raise ValueError("no rows after concat")

    sim_cols_present = [c for c in SIM_STATE_COLS if c in full.columns]
    keep_cols = ["ts_ms", *MARKET_FEATURE_NAMES, *sim_cols_present]
    full = full[keep_cols]

    n = len(full)
    split_idx = int(n * split_ratio)
    train = full.iloc[:split_idx].reset_index(drop=True)
    eval_df = full.iloc[split_idx:].reset_index(drop=True)
    if eval_df["ts_ms"].iloc[0] <= train["ts_ms"].iloc[-1]:
        raise ValueError("chronological split violated: eval starts at or before train end")

    suffix = f"_{label}" if label else ""
    out_dir = episodes_root / symbol / f"{FEATURE_VERSION}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    eval_path = out_dir / "eval.parquet"
    train.to_parquet(train_path, index=False, compression="zstd")
    eval_df.to_parquet(eval_path, index=False, compression="zstd")

    sha = hashlib.sha256()
    for col in MARKET_FEATURE_NAMES:
        sha.update(full[col].to_numpy().tobytes())
    data_hex = sha.hexdigest()[:16]

    # Compute z-score stats on TRAIN ONLY (chronologically first 80%).
    # These get baked into the env at training time and into the ONNX
    # graph at export time so train and live inference normalise identically.
    stats = compute_norm_stats(train, clip=norm_clip)
    stats_path = save_norm_stats(stats, norm_stats_path_for_episodes(out_dir))

    manifest = EpisodeManifest(
        symbol=symbol,
        feature_version=FEATURE_VERSION,
        feature_spec_checksum=FEATURE_SPEC_CHECKSUM,
        train_rows=len(train),
        eval_rows=len(eval_df),
        train_start_ms=int(train["ts_ms"].iloc[0]),
        train_end_ms=int(train["ts_ms"].iloc[-1]),
        eval_start_ms=int(eval_df["ts_ms"].iloc[0]),
        eval_end_ms=int(eval_df["ts_ms"].iloc[-1]),
        split_ratio=split_ratio,
        data_sha256_prefix=data_hex,
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
