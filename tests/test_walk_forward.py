import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocessors.episode_builder import (
    _list_feature_days,
    build_walk_forward_split,
)
from serving.feature_version import FEATURE_VERSION, MARKET_FEATURE_NAMES


def _make_day(dataset_root: Path, symbol: str, day_iso: str, n: int = 100, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    # offset timestamps by the day index so different days don't collide on ts_ms
    day_offset_ms = int(pd.Timestamp(day_iso).timestamp() * 1000)
    ts = day_offset_ms + np.arange(n, dtype=np.int64) * 100
    df = pd.DataFrame({"ts_ms": ts})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32)
    df["mid"] = np.full(n, 100.0, dtype=np.float32)
    df["bid_px"] = np.full(n, 99.95, dtype=np.float32)
    df["ask_px"] = np.full(n, 100.05, dtype=np.float32)
    df["abs_volume_100ms"] = np.full(n, 10.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    df["feature_version"] = FEATURE_VERSION
    dest = dataset_root / symbol / "features" / f"{day_iso}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


def test_list_feature_days_filters_by_date_range(tmp_path: Path) -> None:
    for d in ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]:
        _make_day(tmp_path, "BTCUSDT", d)
    paths = _list_feature_days("BTCUSDT", tmp_path, train_start="2024-01-15", train_end="2024-02-15")
    stems = [p.stem for p in paths]
    assert stems == ["2024-01-15", "2024-02-01", "2024-02-15"]


def test_list_feature_days_empty_when_no_match(tmp_path: Path) -> None:
    _make_day(tmp_path, "BTCUSDT", "2024-01-01")
    paths = _list_feature_days("BTCUSDT", tmp_path, train_start="2025-01-01")
    assert paths == []


def test_walk_forward_split_writes_train_and_eval_parquets(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    for d in ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-15"]:
        _make_day(ds, "BTCUSDT", d, seed=hash(d) % 32)
    manifest = build_walk_forward_split(
        symbol="BTCUSDT",
        dataset_root=ds,
        episodes_root=ep,
        train_start="2024-01-01",
        train_end="2024-02-15",
        eval_start="2024-03-01",
        eval_end="2024-03-15",
        label="h1_smoke",
    )
    out_dir = ep / "BTCUSDT" / f"{FEATURE_VERSION}_h1_smoke"
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "norm_stats.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert manifest.train_rows == 4 * 100  # 4 days x 100 rows
    assert manifest.eval_rows == 2 * 100
    assert manifest.split_ratio == 1.0
    assert manifest.train_end_ms < manifest.eval_start_ms
    loaded = json.loads((out_dir / "manifest.json").read_text())
    assert loaded["symbol"] == "BTCUSDT"


def test_walk_forward_rejects_overlapping_eval(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    for d in ["2024-01-01", "2024-02-01"]:
        _make_day(ds, "BTCUSDT", d)
    with pytest.raises(ValueError, match="eval_start"):
        build_walk_forward_split(
            symbol="BTCUSDT",
            dataset_root=ds,
            episodes_root=ep,
            train_start="2024-01-01",
            train_end="2024-02-01",
            eval_start="2024-02-01",  # overlap
            eval_end="2024-02-01",
        )


def test_walk_forward_raises_when_train_window_empty(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    _make_day(ds, "BTCUSDT", "2024-01-01")
    with pytest.raises(FileNotFoundError, match="train features"):
        build_walk_forward_split(
            symbol="BTCUSDT",
            dataset_root=ds,
            episodes_root=ep,
            train_start="2025-01-01",
            train_end="2025-01-31",
            eval_start="2025-02-01",
            eval_end="2025-02-28",
        )
