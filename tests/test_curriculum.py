from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from serving.feature_version import FEATURE_VERSION, MARKET_FEATURE_NAMES
from training.curriculum import episode_paths_for_symbols, parse_stages, run_curriculum


def test_parse_stages_minimal() -> None:
    stages = parse_stages(
        {
            "stages": [
                {"name": "btc", "symbols": ["BTCUSDT"], "total_timesteps": 100},
                {"name": "btc_eth", "symbols": ["BTCUSDT", "ETHUSDT"], "total_timesteps": 200},
            ]
        }
    )
    assert [s.name for s in stages] == ["btc", "btc_eth"]
    assert stages[0].warm_start is False  # first stage defaults to no warm-start
    assert stages[1].warm_start is True
    assert stages[1].symbols == ["BTCUSDT", "ETHUSDT"]


def test_parse_stages_rejects_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="name"):
        parse_stages({"stages": [{"symbols": ["BTC"], "total_timesteps": 1}]})
    with pytest.raises(ValueError, match="stages"):
        parse_stages({})


def test_episode_paths_for_symbols(tmp_path: Path) -> None:
    root = tmp_path
    for sym in ["BTCUSDT", "ETHUSDT"]:
        d = root / sym / FEATURE_VERSION
        d.mkdir(parents=True)
        (d / "train.parquet").touch()
        (d / "eval.parquet").touch()
    paths = episode_paths_for_symbols(["BTCUSDT", "ETHUSDT"], root, FEATURE_VERSION, "train")
    assert paths == [root / "BTCUSDT" / FEATURE_VERSION / "train.parquet", root / "ETHUSDT" / FEATURE_VERSION / "train.parquet"]


def test_episode_paths_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        episode_paths_for_symbols(["BTCUSDT"], tmp_path, FEATURE_VERSION, "train")


def _make_episode_parquet(path: Path, n: int = 5000, base_mid: float = 50_000.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000_000 + np.arange(n, dtype=np.int64) * 100
    mid = base_mid * np.exp(np.cumsum(rng.standard_normal(n) * 1e-5 + 0.5 / 10_000.0))
    df = pd.DataFrame({"ts_ms": ts})
    for c in MARKET_FEATURE_NAMES:
        df[c] = rng.standard_normal(n).astype(np.float32) * 0.1
    df["mid"] = mid.astype(np.float32)
    df["bid_px"] = (mid - 1.5).astype(np.float32)
    df["ask_px"] = (mid + 1.5).astype(np.float32)
    df["abs_volume_100ms"] = np.full(n, 100.0, dtype=np.float32)
    df["funding_rate"] = np.zeros(n, dtype=np.float32)
    df.to_parquet(path, index=False)


@pytest.mark.slow
def test_curriculum_runs_two_stage_smoke(tmp_path: Path) -> None:
    episodes_root = tmp_path / "episodes"
    for sym in ["BTCUSDT", "ETHUSDT"]:
        d = episodes_root / sym / FEATURE_VERSION
        d.mkdir(parents=True)
        _make_episode_parquet(d / "train.parquet", seed=hash(sym) % 32)
        _make_episode_parquet(d / "eval.parquet", seed=(hash(sym) + 1) % 32)

    base_train_cfg = {
        "seed": [0],
        "policy": "MlpPolicy",
        "n_steps": 128,
        "batch_size": 32,
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
        "eval": {"eval_freq": 256, "n_eval_episodes": 1},
        "callbacks": {
            "sigma_divergence_killswitch": {"high_threshold": 1.5, "low_threshold": -0.5, "consecutive_evals": 999},
            "trade_log": {"n_eval_episodes": 1, "record_obs": False},
        },
    }
    base_env_cfg = {
        "leverage": 6,
        "max_position_notional_pct": 0.17,
        "initial_nav_usd": 10_000.0,
        "apply_obs_normalisation": False,
        "episode": {"length_ticks": 200, "reset_on_dd": 0.05},
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
    curriculum_cfg = {
        "stages": [
            {"name": "btc", "symbols": ["BTCUSDT"], "total_timesteps": 512, "warm_start": False},
            {"name": "btc_eth", "symbols": ["BTCUSDT", "ETHUSDT"], "total_timesteps": 512, "warm_start": True},
        ]
    }

    train_yaml = tmp_path / "train.yaml"
    env_yaml = tmp_path / "env.yaml"
    curr_yaml = tmp_path / "curriculum.yaml"
    train_yaml.write_text(yaml.safe_dump(base_train_cfg))
    env_yaml.write_text(yaml.safe_dump(base_env_cfg))
    curr_yaml.write_text(yaml.safe_dump(curriculum_cfg))

    run_dir = tmp_path / "run"
    results = run_curriculum(curr_yaml, train_yaml, env_yaml, episodes_root, FEATURE_VERSION, run_dir)
    assert len(results) == 2
    assert (run_dir / "00_btc").is_dir()
    assert (run_dir / "01_btc_eth").is_dir()
    assert (run_dir / "curriculum_manifest.json").exists()
    # the promoted best_checkpoint exists
    assert (run_dir / "best_checkpoint" / "best_checkpoint.zip").exists()
