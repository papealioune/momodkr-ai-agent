import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocessors.feature_stats import compute_norm_stats, save_norm_stats
from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_DIM,
    MARKET_FEATURE_NAMES,
)
from serving.norm_bundle import NormStatsBundle


def _train_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"ts_ms": np.arange(n, dtype=np.int64) * 100})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32)
    return df


def _materialise_episode_dirs(tmp_path: Path, symbols: list[str]) -> Path:
    """Mimic build_features layout: episodes/<sym>/<feature_version>/norm_stats.json"""
    root = tmp_path / "episodes"
    for i, sym in enumerate(symbols):
        d = root / sym / FEATURE_VERSION
        d.mkdir(parents=True, exist_ok=True)
        stats = compute_norm_stats(_train_df(seed=i + 1))
        save_norm_stats(stats, d / "norm_stats.json")
    return root


def test_bundle_from_per_symbol_stats_round_trip(tmp_path: Path) -> None:
    root = _materialise_episode_dirs(tmp_path, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    bundle = NormStatsBundle.from_episode_dirs(root, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert bundle.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert bundle.feature_version == FEATURE_VERSION
    assert bundle.feature_spec_checksum == FEATURE_SPEC_CHECKSUM
    out = tmp_path / "bundle.json"
    bundle.save(out)
    loaded = NormStatsBundle.load(out)
    assert loaded.symbols == bundle.symbols
    for sym in bundle.symbols:
        assert loaded.by_symbol[sym].mean == bundle.by_symbol[sym].mean
        assert loaded.by_symbol[sym].std == bundle.by_symbol[sym].std


def test_bundle_missing_symbol_raises(tmp_path: Path) -> None:
    root = _materialise_episode_dirs(tmp_path, ["BTCUSDT"])
    with pytest.raises(FileNotFoundError, match="no norm_stats"):
        NormStatsBundle.from_episode_dirs(root, ["BTCUSDT", "ETHUSDT"])


def test_bundle_rejects_feature_version_drift(tmp_path: Path) -> None:
    out = tmp_path / "bundle.json"
    bad = {
        "feature_version": "9.9.9",
        "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        "clip": 10.0,
        "by_symbol": {"BTCUSDT": {"mean": [0.0] * MARKET_FEATURE_DIM, "std": [1.0] * MARKET_FEATURE_DIM, "n_train_rows": 1}},
    }
    out.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="feature_version"):
        NormStatsBundle.load(out)


def test_bundle_rejects_checksum_drift(tmp_path: Path) -> None:
    out = tmp_path / "bundle.json"
    bad = {
        "feature_version": FEATURE_VERSION,
        "feature_spec_checksum": "deadbeef" * 2,
        "clip": 10.0,
        "by_symbol": {"BTCUSDT": {"mean": [0.0] * MARKET_FEATURE_DIM, "std": [1.0] * MARKET_FEATURE_DIM, "n_train_rows": 1}},
    }
    out.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="feature_spec_checksum"):
        NormStatsBundle.load(out)


def test_bundle_empty_by_symbol_raises(tmp_path: Path) -> None:
    out = tmp_path / "bundle.json"
    bad = {
        "feature_version": FEATURE_VERSION,
        "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        "clip": 10.0,
        "by_symbol": {},
    }
    out.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="no per-symbol stats"):
        NormStatsBundle.load(out)


def test_bundle_wrong_length_raises(tmp_path: Path) -> None:
    out = tmp_path / "bundle.json"
    bad = {
        "feature_version": FEATURE_VERSION,
        "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        "clip": 10.0,
        "by_symbol": {"BTCUSDT": {"mean": [0.0] * 5, "std": [1.0] * 5, "n_train_rows": 1}},
    }
    out.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="length"):
        NormStatsBundle.load(out)


def test_bundle_get_norm_stats_unknown_symbol(tmp_path: Path) -> None:
    root = _materialise_episode_dirs(tmp_path, ["BTCUSDT"])
    bundle = NormStatsBundle.from_episode_dirs(root, ["BTCUSDT"])
    with pytest.raises(KeyError, match="not in bundle"):
        bundle.get_norm_stats("ETHUSDT")


def test_bundle_apply_matches_apply_zscore(tmp_path: Path) -> None:
    from data.preprocessors.feature_stats import apply_zscore, load_norm_stats

    root = _materialise_episode_dirs(tmp_path, ["BTCUSDT", "ETHUSDT"])
    bundle = NormStatsBundle.from_episode_dirs(root, ["BTCUSDT", "ETHUSDT"])
    raw = np.random.default_rng(7).standard_normal((50, MARKET_FEATURE_DIM)).astype(np.float32)
    # Apply via bundle vs apply via direct stats load -- must be bit-identical
    via_bundle = bundle.apply("BTCUSDT", raw)
    direct = apply_zscore(raw, load_norm_stats(root / "BTCUSDT" / FEATURE_VERSION / "norm_stats.json"))
    np.testing.assert_array_equal(via_bundle, direct)


def test_bundle_per_symbol_stats_differ(tmp_path: Path) -> None:
    root = _materialise_episode_dirs(tmp_path, ["BTCUSDT", "ETHUSDT"])
    bundle = NormStatsBundle.from_episode_dirs(root, ["BTCUSDT", "ETHUSDT"])
    raw = np.ones((1, MARKET_FEATURE_DIM), dtype=np.float32)
    btc = bundle.apply("BTCUSDT", raw)
    eth = bundle.apply("ETHUSDT", raw)
    # Different train data -> different per-symbol stats -> different normalised output
    assert not np.allclose(btc, eth)
