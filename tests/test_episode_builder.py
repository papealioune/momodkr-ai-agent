import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocessors.episode_builder import build_episodes
from serving.feature_version import FEATURE_VERSION, MARKET_FEATURE_NAMES


def _write_day(dataset_root: Path, symbol: str, day_iso: str, start_ms: int, n: int = 200) -> Path:
    rng = np.random.default_rng(hash(day_iso) % (2**32))
    ts = start_ms + np.arange(n, dtype=np.int64) * 100
    df = pd.DataFrame({"ts_ms": ts})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32)
    df["feature_version"] = FEATURE_VERSION
    dest = dataset_root / symbol / "features" / f"{day_iso}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


def test_build_episodes_chronological_split(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    base = 1_700_000_000_000
    one_day = 24 * 60 * 60 * 1000
    _write_day(ds, "BTCUSDT", "2024-01-01", base)
    _write_day(ds, "BTCUSDT", "2024-01-02", base + one_day)
    _write_day(ds, "BTCUSDT", "2024-01-03", base + 2 * one_day)
    _write_day(ds, "BTCUSDT", "2024-01-04", base + 3 * one_day)
    _write_day(ds, "BTCUSDT", "2024-01-05", base + 4 * one_day)

    m = build_episodes("BTCUSDT", ds, ep, split_ratio=0.8)
    assert m.symbol == "BTCUSDT"
    assert m.feature_version == FEATURE_VERSION
    assert m.train_rows + m.eval_rows == 1000
    assert m.train_rows == 800
    assert m.eval_rows == 200
    assert m.train_end_ms < m.eval_start_ms

    train = pd.read_parquet(ep / "BTCUSDT" / FEATURE_VERSION / "train.parquet")
    eval_df = pd.read_parquet(ep / "BTCUSDT" / FEATURE_VERSION / "eval.parquet")
    assert list(train.columns) == ["ts_ms", *MARKET_FEATURE_NAMES]
    assert list(eval_df.columns) == ["ts_ms", *MARKET_FEATURE_NAMES]
    assert train["ts_ms"].is_monotonic_increasing
    assert eval_df["ts_ms"].is_monotonic_increasing
    # zero overlap
    assert train["ts_ms"].max() < eval_df["ts_ms"].min()


def test_build_episodes_manifest_persisted(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    _write_day(ds, "ETHUSDT", "2024-02-01", 1_700_000_000_000)
    _write_day(ds, "ETHUSDT", "2024-02-02", 1_700_000_000_000 + 86_400_000)
    build_episodes("ETHUSDT", ds, ep, split_ratio=0.75)
    manifest_path = ep / "ETHUSDT" / FEATURE_VERSION / "manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded["symbol"] == "ETHUSDT"
    assert loaded["split_ratio"] == 0.75
    assert loaded["train_rows"] + loaded["eval_rows"] == 400


def test_build_episodes_rejects_feature_version_mismatch(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    p = _write_day(ds, "SOLUSDT", "2024-03-01", 1_700_000_000_000)
    df = pd.read_parquet(p)
    df["feature_version"] = "9.9.9"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="feature_version mismatch"):
        build_episodes("SOLUSDT", ds, ep)


def test_build_episodes_rejects_bad_split_ratio(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    _write_day(ds, "BTCUSDT", "2024-04-01", 1_700_000_000_000)
    with pytest.raises(ValueError, match="split_ratio"):
        build_episodes("BTCUSDT", ds, ep, split_ratio=0.4)
    with pytest.raises(ValueError, match="split_ratio"):
        build_episodes("BTCUSDT", ds, ep, split_ratio=1.0)


def test_build_episodes_no_features_raises(tmp_path: Path) -> None:
    ds = tmp_path / "datasets"
    ep = tmp_path / "episodes"
    (ds / "BTCUSDT" / "features").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_episodes("BTCUSDT", ds, ep)
