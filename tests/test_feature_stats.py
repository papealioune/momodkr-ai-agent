import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocessors.feature_stats import (
    DEFAULT_CLIP,
    NormStats,
    apply_zscore,
    compute_norm_stats,
    load_norm_stats,
    norm_stats_path_for_episodes,
    save_norm_stats,
)
from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_DIM,
    MARKET_FEATURE_NAMES,
)


def _train_df(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"ts_ms": np.arange(n, dtype=np.int64) * 100})
    # Generate features with VERY different scales: ofi in thousands, ret in 1e-4,
    # depth in millions. Exactly the pathology we want normalisation to fix.
    for i, c in enumerate(MARKET_FEATURE_NAMES):
        scale = 1e3 ** (i % 5 - 2)  # scales from 1e-2 to 1e2
        df[c] = (rng.standard_normal(n) * scale).astype(np.float32)
    return df


def test_compute_norm_stats_returns_per_feature_mean_and_std() -> None:
    df = _train_df()
    stats = compute_norm_stats(df)
    assert isinstance(stats, NormStats)
    assert stats.feature_version == FEATURE_VERSION
    assert stats.feature_spec_checksum == FEATURE_SPEC_CHECKSUM
    assert stats.n_train_rows == len(df)
    assert len(stats.mean) == MARKET_FEATURE_DIM
    assert len(stats.std) == MARKET_FEATURE_DIM
    for s in stats.std:
        assert s > 0


def test_compute_norm_stats_zero_variance_columns_get_std_one() -> None:
    n = 1000
    df = pd.DataFrame({"ts_ms": np.arange(n, dtype=np.int64) * 100})
    for c in MARKET_FEATURE_NAMES:
        df[c] = np.zeros(n, dtype=np.float32)
    stats = compute_norm_stats(df)
    assert all(s == 1.0 for s in stats.std)


def test_compute_norm_stats_raises_on_missing_features() -> None:
    df = pd.DataFrame({"ts_ms": [1, 2, 3]})
    with pytest.raises(KeyError):
        compute_norm_stats(df)


def test_apply_zscore_centres_and_clips() -> None:
    df = _train_df()
    stats = compute_norm_stats(df, clip=3.0)
    raw = df[list(MARKET_FEATURE_NAMES)].to_numpy(dtype=np.float32)[:1000]
    z = apply_zscore(raw, stats)
    # post-normalisation each feature should have near-zero mean and unit std on the training data
    mean_per_feat = z.mean(axis=0)
    std_per_feat = z.std(axis=0)
    assert np.all(np.abs(mean_per_feat) < 0.1)
    assert np.all(std_per_feat > 0.7)  # well-conditioned (close to 1, allowing for partial-batch sampling)
    assert z.max() <= 3.0 + 1e-6
    assert z.min() >= -3.0 - 1e-6


def test_apply_zscore_preserves_shape_for_single_vector() -> None:
    df = _train_df()
    stats = compute_norm_stats(df)
    raw = df[list(MARKET_FEATURE_NAMES)].to_numpy(dtype=np.float32)[0]
    z = apply_zscore(raw, stats)
    assert z.shape == (MARKET_FEATURE_DIM,)
    assert z.dtype == np.float32


def test_save_and_load_norm_stats_roundtrip(tmp_path: Path) -> None:
    df = _train_df()
    stats = compute_norm_stats(df, clip=DEFAULT_CLIP)
    path = norm_stats_path_for_episodes(tmp_path)
    save_norm_stats(stats, path)
    assert path.exists()
    loaded = load_norm_stats(path)
    assert loaded.mean == stats.mean
    assert loaded.std == stats.std
    assert loaded.clip == stats.clip


def test_load_norm_stats_rejects_feature_version_drift(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    bad = {
        "feature_version": "9.9.9",
        "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        "n_train_rows": 1,
        "clip": 10.0,
        "mean": [0.0] * MARKET_FEATURE_DIM,
        "std": [1.0] * MARKET_FEATURE_DIM,
    }
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="feature_version"):
        load_norm_stats(path)


def test_load_norm_stats_rejects_checksum_drift(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    bad = {
        "feature_version": FEATURE_VERSION,
        "feature_spec_checksum": "deadbeef" * 2,
        "n_train_rows": 1,
        "clip": 10.0,
        "mean": [0.0] * MARKET_FEATURE_DIM,
        "std": [1.0] * MARKET_FEATURE_DIM,
    }
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="checksum"):
        load_norm_stats(path)


def test_load_norm_stats_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    bad = {
        "feature_version": FEATURE_VERSION,
        "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        "n_train_rows": 1,
        "clip": 10.0,
        "mean": [0.0] * 5,  # wrong length
        "std": [1.0] * 5,
    }
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="mean length"):
        load_norm_stats(path)
