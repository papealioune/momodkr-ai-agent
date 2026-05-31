"""ONNX export + parity end-to-end smoke.

Trains a tiny PPO for a couple hundred steps, exports to ONNX, then runs
the parity validator against (a) the env's actual eval obs and (b) a
random-batch fallback. Asserts max_diff < 1e-4 and argmax match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.export_onnx import export
from scripts.validate_onnx_parity import (
    load_obs_from_eval_log_dir,
    load_obs_from_parquet,
    validate,
)
from serving.feature_version import MARKET_FEATURE_NAMES, OBS_DIM
from training.train_ppo import train


def _synthetic_episode(n: int = 5_000, drift_bps_per_tick: float = 0.5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 100
    log_ret = rng.standard_normal(n) * 1e-5 + drift_bps_per_tick / 10_000.0
    mid = 50_000.0 * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame({"ts_ms": ts})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32) * 0.1
    df["mid"] = mid.astype(np.float32)
    df["bid_px"] = (mid - 1.5).astype(np.float32)
    df["ask_px"] = (mid + 1.5).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n, 100.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    return df


@pytest.fixture
def trained_run(tmp_path: Path) -> dict[str, Path]:
    parquet = tmp_path / "train.parquet"
    _synthetic_episode().to_parquet(parquet, index=False)

    train_cfg = {
        "seed": [0],
        "policy": "MlpPolicy",
        "total_timesteps": 1024,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "learning_rate": 3e-4,
        "ent_coef": 0.005,
        "policy_kwargs": {"net_arch": [16, 16]},
        "vec_env": {"type": "dummy", "n_envs": 1},
        "eval": {"eval_freq": 512, "n_eval_episodes": 1},
        "callbacks": {
            "sigma_divergence_killswitch": {"high_threshold": 1.5, "low_threshold": -0.5, "consecutive_evals": 999},
            "trade_log": {"n_eval_episodes": 1, "record_obs": True},
        },
    }
    env_cfg = {
        "leverage": 6,
        "max_position_notional_pct": 0.17,
        "initial_nav_usd": 10_000.0,
        "episode": {"length_ticks": 300, "reset_on_dd": 0.05},
        "simulator": {
            "fee_taker_bps": 3.5,
            "fee_maker_bps": 1.0,
            "slippage_c": 0.001,
            "latency_bps_uniform": [5, 20],
            "fee_noise_pct": 0.0,
            "slippage_noise_pct": 0.0,
            "funding_interval_ticks": 288_000,
        },
        "reward": {
            "win_multiplier": 4.0,
            "loss_multiplier": 1.8,
            "per_entry_cost": 0.02,
            "dd_quadratic_coeff": 50.0,
            "dd_threshold": 0.03,
            "funding_coeff": 0.01,
            "losing_streak_coeff": 0.05,
            "losing_streak_offset": 2,
            "unrealized_breadcrumb_coeff": 0.3,
            "reward_floor": -5.0,
        },
    }
    train_yaml = tmp_path / "train.yaml"
    env_yaml = tmp_path / "env.yaml"
    train_yaml.write_text(yaml.safe_dump(train_cfg))
    env_yaml.write_text(yaml.safe_dump(env_cfg))
    run_dir = tmp_path / "run"
    train(train_yaml, env_yaml, parquet, parquet, run_dir)
    return {"run_dir": run_dir, "parquet": parquet}


@pytest.mark.slow
def test_export_onnx_writes_graph_and_manifest(trained_run: dict[str, Path], tmp_path: Path) -> None:
    best = trained_run["run_dir"] / "best_checkpoint" / "best_model.zip"
    assert best.exists()
    onnx_path = tmp_path / "policy.onnx"
    export(best, onnx_path)
    assert onnx_path.exists()
    manifest = json.loads(onnx_path.with_suffix(".json").read_text())
    assert manifest["obs_dim"] == OBS_DIM
    assert manifest["n_actions"] == 5
    assert manifest["feature_version"]
    assert manifest["feature_spec_checksum"]


@pytest.mark.slow
def test_onnx_parity_passes_on_eval_log_obs(trained_run: dict[str, Path], tmp_path: Path) -> None:
    best = trained_run["run_dir"] / "best_checkpoint" / "best_model.zip"
    onnx_path = tmp_path / "policy.onnx"
    export(best, onnx_path)
    obs = load_obs_from_eval_log_dir(trained_run["run_dir"] / "eval_episodes")
    result = validate(best, onnx_path, obs)
    assert result["passed"], result
    assert result["max_diff_logits"] < 1e-4
    assert result["action_match"]


@pytest.mark.slow
def test_onnx_parity_passes_on_parquet_obs(trained_run: dict[str, Path], tmp_path: Path) -> None:
    best = trained_run["run_dir"] / "best_checkpoint" / "best_model.zip"
    onnx_path = tmp_path / "policy.onnx"
    export(best, onnx_path)
    obs = load_obs_from_parquet(trained_run["parquet"], max_rows=200)
    result = validate(best, onnx_path, obs)
    assert result["passed"], result


@pytest.mark.slow
def test_onnx_parity_detects_mismatch(trained_run: dict[str, Path], tmp_path: Path) -> None:
    """If we corrupt the ONNX (e.g. by exporting from a different model), parity must fail."""
    import torch

    best = trained_run["run_dir"] / "best_checkpoint" / "best_model.zip"
    onnx_path = tmp_path / "policy.onnx"
    export(best, onnx_path)

    # Re-export with a different model: build a fresh PPO with random weights
    # by perturbing the loaded model in-place, then re-export.
    from stable_baselines3 import PPO

    perturbed = PPO.load(best, device="cpu")
    with torch.no_grad():
        for p in perturbed.policy.parameters():
            p.add_(torch.randn_like(p) * 1.0)
    bad_path = tmp_path / "bad.zip"
    perturbed.save(bad_path)
    bad_onnx = tmp_path / "bad.onnx"
    export(bad_path, bad_onnx)

    obs = load_obs_from_parquet(trained_run["parquet"], max_rows=100)
    # The original ONNX should still match the original checkpoint, but it
    # MUST NOT match the perturbed checkpoint.
    bad_result = validate(bad_path, onnx_path, obs)
    assert not bad_result["passed"], "parity should have detected a model mismatch"
