"""End-to-end PPO smoke test for Phase 4.

Trains a tiny PPO model on a synthetic trending episode for a couple
thousand steps to validate the full training loop: env vec wrapping,
PPO instantiation from YAML config, EvalCallback wiring, best-checkpoint
tracker, trade-log callback, and sigma killswitch. Does NOT assert any
performance threshold (that's the Phase 4 production gate, run on real
data on RunPod).

Marked `slow`; running time on CPU is ~30-60 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from serving.feature_version import MARKET_FEATURE_NAMES
from training.train_ppo import train


def _synthetic_trending(n: int = 8_000, drift_bps_per_tick: float = 0.5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 100
    log_ret = rng.standard_normal(n) * 1e-5 + drift_bps_per_tick / 10_000.0
    mid = 50_000.0 * np.exp(np.cumsum(log_ret))
    half_spread = 1.5
    df = pd.DataFrame({"ts_ms": ts})
    for col in MARKET_FEATURE_NAMES:
        df[col] = rng.standard_normal(n).astype(np.float32) * 0.1
    df["mid"] = mid.astype(np.float32)
    df["bid_px"] = (mid - half_spread).astype(np.float32)
    df["ask_px"] = (mid + half_spread).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n, 100.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    return df


@pytest.mark.slow
def test_train_ppo_end_to_end_writes_best_checkpoint(tmp_path: Path) -> None:
    parquet_path = tmp_path / "train.parquet"
    _synthetic_trending().to_parquet(parquet_path, index=False)

    train_cfg = {
        "run_name": "smoke",
        "seed": [0],
        "policy": "MlpPolicy",
        "total_timesteps": 2_048,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "learning_rate": 3e-4,
        "ent_coef": 0.005,
        "policy_kwargs": {"net_arch": [32, 32]},
        "vec_env": {"type": "dummy", "n_envs": 1},
        "eval": {"eval_freq": 512, "n_eval_episodes": 1},
        "callbacks": {
            "sigma_divergence_killswitch": {
                # disable the kill for this smoke -- untrained policy is uniform-random
                "high_threshold": 1.5,
                "low_threshold": -0.5,
                "consecutive_evals": 999,
            },
            "trade_log": {"n_eval_episodes": 1, "record_obs": False},
        },
    }
    env_cfg = {
        "leverage": 6,
        "max_position_notional_pct": 0.17,
        "initial_nav_usd": 10_000.0,
        "episode": {"length_ticks": 500, "reset_on_dd": 0.05},
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

    run_dir = tmp_path / "runs"
    best_path = train(train_yaml, env_yaml, parquet_path, parquet_path, run_dir)

    # SB3's EvalCallback writes best_model.zip; our tracker writes best_checkpoint.zip + marker.
    assert (run_dir / "best_checkpoint" / "best_model.zip").exists()
    # at least one eval JSON dropped
    eval_dir = run_dir / "eval_episodes"
    assert eval_dir.exists()
    eval_jsons = list(eval_dir.glob("*.json"))
    assert eval_jsons, "no trade-log JSON was written"
    sample = json.loads(eval_jsons[0].read_text())
    assert "episodes" in sample
    # final checkpoint exists
    assert (run_dir / "final_checkpoint.zip").exists()
    # best_path is the best_checkpoint we expose (may not exist if no improvement)
    assert best_path.parent == run_dir / "best_checkpoint"
