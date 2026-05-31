import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from training.utils import (
    categorical_entropy_normalised,
    linear_schedule,
    load_yaml,
    resolve_schedule_or_float,
)


def test_load_yaml_reads_known_file(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"a": 1, "b": [2, 3]}))
    cfg = load_yaml(p)
    assert cfg["a"] == 1
    assert cfg["b"] == [2, 3]


def test_load_yaml_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_linear_schedule_interpolates_progress_remaining() -> None:
    f = linear_schedule(0.005, 0.0005)
    assert f(1.0) == pytest.approx(0.005)
    assert f(0.0) == pytest.approx(0.0005)
    assert f(0.5) == pytest.approx(0.00275)


def test_linear_schedule_rejects_increase() -> None:
    with pytest.raises(ValueError):
        linear_schedule(0.001, 0.005)


def test_resolve_schedule_or_float_scalar() -> None:
    assert resolve_schedule_or_float(0.0003) == 0.0003


def test_resolve_schedule_or_float_linear_dict() -> None:
    fn = resolve_schedule_or_float({"schedule": "linear", "start": 0.005, "end": 0.0005})
    assert callable(fn)
    assert fn(1.0) == pytest.approx(0.005)
    assert fn(0.0) == pytest.approx(0.0005)


def test_resolve_schedule_unsupported_kind() -> None:
    with pytest.raises(ValueError, match="schedule"):
        resolve_schedule_or_float({"schedule": "cosine", "start": 1.0, "end": 0.0})


def test_categorical_entropy_uniform_is_one() -> None:
    probs = np.full(5, 0.2)
    assert categorical_entropy_normalised(probs) == pytest.approx(1.0)


def test_categorical_entropy_collapsed_is_zero() -> None:
    probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert categorical_entropy_normalised(probs) == pytest.approx(0.0, abs=1e-6)


def test_categorical_entropy_average_across_batch() -> None:
    batch = np.stack([np.full(5, 0.2), np.array([0.96, 0.01, 0.01, 0.01, 0.01])])
    val = categorical_entropy_normalised(batch)
    expected_single_collapsed = -(0.96 * math.log(0.96) + 4 * 0.01 * math.log(0.01)) / math.log(5)
    assert val == pytest.approx((1.0 + expected_single_collapsed) / 2, abs=1e-6)


def test_existing_yaml_configs_parse() -> None:
    """Every Phase 0 seed config must round-trip through load_yaml without error."""
    for rel in [
        "configs/data/binance_vision_v1.yaml",
        "configs/env/momodkr_v1.yaml",
        "configs/training/v1_engine_cold.yaml",
        "configs/live/governor_v1.yaml",
    ]:
        cfg = load_yaml(rel)
        assert isinstance(cfg, dict)
        assert cfg
