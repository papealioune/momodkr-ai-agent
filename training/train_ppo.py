"""SB3 PPO entrypoint for MomoDkr.

Loads a YAML config (e.g. configs/training/v1_engine_cold.yaml) plus an
env YAML (configs/env/momodkr_v1.yaml), wires up vectorised MomoDkrEnv
instances against an episode parquet, attaches the three Phase-4
callbacks (sigma killswitch, best checkpoint tracker, trade log) under a
SB3 EvalCallback, and runs PPO.learn().

Usage:
    python -m training.train_ppo \\
        --train-config configs/training/v1_engine_cold.yaml \\
        --env-config configs/env/momodkr_v1.yaml \\
        --train-parquet data/episodes/BTCUSDT/0.1.0/train.parquet \\
        --eval-parquet data/episodes/BTCUSDT/0.1.0/eval.parquet \\
        --run-dir runs/v1-engine-cold-btc

If --eval-parquet is omitted the train parquet is reused (development
only; production runs MUST use a held-out chronological eval).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from envs.momodkr_env import EnvConfig, MomoDkrEnv
from envs.reward.wrappers import apply_full_reward_shaping
from training.callbacks.best_checkpoint_tracker import BestCheckpointTracker
from training.callbacks.sigma_divergence_killswitch import SigmaDivergenceKillswitch
from training.callbacks.trade_log_callback import TradeLogCallback
from training.utils import load_yaml, resolve_schedule_or_float

logger = logging.getLogger(__name__)


def make_env_factory(episode_parquet: Path | list[Path], env_cfg: EnvConfig, seed_offset: int):
    def _f():
        env = MomoDkrEnv(episode_parquet, env_cfg, seed=seed_offset)
        env = apply_full_reward_shaping(env, env_cfg.risk)
        return Monitor(env)

    return _f


def build_vec_env(
    episode_parquet: Path | list[Path],
    env_cfg: EnvConfig,
    n_envs: int,
    kind: str,
    base_seed: int,
) -> VecEnv:
    fns = [make_env_factory(episode_parquet, env_cfg, base_seed + i) for i in range(n_envs)]
    if n_envs <= 1:
        return DummyVecEnv(fns)
    if kind == "subproc":
        return SubprocVecEnv(fns)
    if kind == "dummy":
        return DummyVecEnv(fns)
    raise ValueError(f"unknown vec_env kind: {kind!r}")


def env_cfg_from_yaml(env_yaml_path: Path) -> EnvConfig:
    raw = load_yaml(env_yaml_path)
    sim = raw.get("simulator", {})
    reward = raw.get("reward", {})
    cfg = EnvConfig(
        episode_length_ticks=int(raw.get("episode", {}).get("length_ticks", 9_000)),
        reset_on_dd=float(raw.get("episode", {}).get("reset_on_dd", 0.05)),
        max_position_notional_pct=float(raw.get("max_position_notional_pct", 0.17)),
        initial_nav_usd=float(raw.get("initial_nav_usd", 10_000.0)),
        apply_obs_normalisation=bool(raw.get("apply_obs_normalisation", True)),
        position_feature_clip=float(raw.get("position_feature_clip", 3.0)),
    )
    cfg.sim.fee_taker_bps = float(sim.get("fee_taker_bps", cfg.sim.fee_taker_bps))
    cfg.sim.fee_maker_bps = float(sim.get("fee_maker_bps", cfg.sim.fee_maker_bps))
    cfg.sim.slippage_c = float(sim.get("slippage_c", cfg.sim.slippage_c))
    if "latency_bps_uniform" in sim:
        lo, hi = sim["latency_bps_uniform"]
        cfg.sim.latency_bps_min = float(lo)
        cfg.sim.latency_bps_max = float(hi)
    cfg.sim.fee_noise_pct = float(sim.get("fee_noise_pct", cfg.sim.fee_noise_pct))
    cfg.sim.slippage_noise_pct = float(sim.get("slippage_noise_pct", cfg.sim.slippage_noise_pct))
    cfg.sim.funding_interval_ticks = int(sim.get("funding_interval_ticks", cfg.sim.funding_interval_ticks))
    cfg.sim.leverage = int(raw.get("leverage", cfg.sim.leverage))

    cfg.pnl.win_multiplier = float(reward.get("win_multiplier", cfg.pnl.win_multiplier))
    cfg.pnl.loss_multiplier = float(reward.get("loss_multiplier", cfg.pnl.loss_multiplier))
    cfg.risk.per_entry_cost = float(reward.get("per_entry_cost", cfg.risk.per_entry_cost))
    cfg.risk.dd_quadratic_coeff = float(reward.get("dd_quadratic_coeff", cfg.risk.dd_quadratic_coeff))
    cfg.risk.dd_threshold = float(reward.get("dd_threshold", cfg.risk.dd_threshold))
    cfg.risk.funding_coeff = float(reward.get("funding_coeff", cfg.risk.funding_coeff))
    cfg.risk.losing_streak_coeff = float(reward.get("losing_streak_coeff", cfg.risk.losing_streak_coeff))
    cfg.risk.losing_streak_offset = int(reward.get("losing_streak_offset", cfg.risk.losing_streak_offset))
    cfg.risk.churn_penalty = float(reward.get("churn_penalty", cfg.risk.churn_penalty))
    cfg.risk.peak_dd_coeff = float(reward.get("peak_dd_coeff", cfg.risk.peak_dd_coeff))
    cfg.risk.peak_dd_threshold = float(reward.get("peak_dd_threshold", cfg.risk.peak_dd_threshold))
    cfg.risk.reward_floor = float(reward.get("reward_floor", cfg.risk.reward_floor))
    cfg.breadcrumb.unrealized_breadcrumb_coeff = float(reward.get("unrealized_breadcrumb_coeff", cfg.breadcrumb.unrealized_breadcrumb_coeff))
    return cfg


def build_ppo(model_kwargs: dict, env: VecEnv) -> PPO:
    return PPO(env=env, **model_kwargs)


def model_kwargs_from_config(train_cfg: dict, log_dir: Path) -> dict:
    kwargs: dict = {
        "policy": train_cfg.get("policy", "MlpPolicy"),
        "n_steps": int(train_cfg.get("n_steps", 2048)),
        "batch_size": int(train_cfg.get("batch_size", 512)),
        "n_epochs": int(train_cfg.get("n_epochs", 4)),
        "gamma": float(train_cfg.get("gamma", 0.995)),
        "gae_lambda": float(train_cfg.get("gae_lambda", 0.95)),
        "clip_range": float(train_cfg.get("clip_range", 0.2)),
        "vf_coef": float(train_cfg.get("vf_coef", 0.5)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 0.5)),
        "learning_rate": resolve_schedule_or_float(train_cfg.get("learning_rate", 3e-4)),
        "ent_coef": resolve_schedule_or_float(train_cfg.get("ent_coef", 0.005)),
        "policy_kwargs": train_cfg.get("policy_kwargs"),
        "verbose": 1,
        "seed": int(train_cfg.get("seed", [42])[0]) if isinstance(train_cfg.get("seed"), list) else int(train_cfg.get("seed", 42)),
    }
    if train_cfg.get("tensorboard_log", False):
        try:
            import tensorboard  # noqa: F401

            kwargs["tensorboard_log"] = str(log_dir / "tb")
        except ImportError:
            logger.warning("tensorboard_log=True in config but tensorboard not installed; skipping")
    return kwargs


def train(
    train_config_path: Path,
    env_config_path: Path,
    train_parquet: Path | list[Path],
    eval_parquet: Path | list[Path] | None,
    run_dir: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = load_yaml(train_config_path)
    env_cfg = env_cfg_from_yaml(env_config_path)

    n_envs = int(train_cfg.get("vec_env", {}).get("n_envs", 1))
    kind = train_cfg.get("vec_env", {}).get("type", "dummy")
    base_seed = int(train_cfg.get("seed", [42])[0]) if isinstance(train_cfg.get("seed"), list) else int(train_cfg.get("seed", 42))

    train_env = build_vec_env(train_parquet, env_cfg, n_envs, kind, base_seed)
    eval_env = build_vec_env(eval_parquet or train_parquet, env_cfg, n_envs=1, kind="dummy", base_seed=base_seed + 9999)

    warm_start_from = train_cfg.get("warm_start_from")
    if warm_start_from:
        warm_path = Path(warm_start_from)
        if not warm_path.exists():
            raise FileNotFoundError(f"warm_start_from checkpoint not found: {warm_path}")
        logger.info("warm-starting from %s", warm_path)
        custom_objects = {
            "learning_rate": resolve_schedule_or_float(train_cfg.get("learning_rate", 3e-4)),
            "ent_coef": resolve_schedule_or_float(train_cfg.get("ent_coef", 0.005)),
        }
        model = PPO.load(warm_path, env=train_env, custom_objects=custom_objects)
    else:
        model = build_ppo(model_kwargs_from_config(train_cfg, run_dir), train_env)

    eval_cfg = train_cfg.get("eval", {})
    best_dir = run_dir / "best_checkpoint"
    killswitch = SigmaDivergenceKillswitch(
        eval_env=eval_env,
        high_threshold=float(train_cfg.get("callbacks", {}).get("sigma_divergence_killswitch", {}).get("high_threshold", 0.95)),
        low_threshold=float(train_cfg.get("callbacks", {}).get("sigma_divergence_killswitch", {}).get("low_threshold", 0.05)),
        consecutive_evals=int(train_cfg.get("callbacks", {}).get("sigma_divergence_killswitch", {}).get("consecutive_evals", 2)),
    )
    best_tracker = BestCheckpointTracker(save_dir=best_dir)
    trade_log = TradeLogCallback(
        eval_env=eval_env,
        log_dir=run_dir,
        n_eval_episodes=int(train_cfg.get("callbacks", {}).get("trade_log", {}).get("n_eval_episodes", 4)),
        record_obs=bool(train_cfg.get("callbacks", {}).get("trade_log", {}).get("record_obs", False)),
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=int(eval_cfg.get("eval_freq", 50_000)),
        n_eval_episodes=int(eval_cfg.get("n_eval_episodes", 8)),
        deterministic=True,
        callback_on_new_best=best_tracker,
        callback_after_eval=CallbackList([killswitch, trade_log]),
    )

    total_timesteps = int(train_cfg.get("total_timesteps", 20_000_000))
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb, log_interval=10)
    finally:
        train_env.close()
        eval_env.close()

    final_path = run_dir / "final_checkpoint.zip"
    model.save(final_path)
    logger.info("training done; final=%s best=%s", final_path, best_dir / "best_checkpoint.zip")
    return best_dir / "best_checkpoint.zip"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="MomoDkr SB3 PPO trainer")
    p.add_argument("--train-config", required=True)
    p.add_argument("--env-config", required=True)
    p.add_argument("--train-parquet", required=True, nargs="+", help="one or more episode parquets (multi-symbol pool)")
    p.add_argument("--eval-parquet", default=None, nargs="+")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    train_paths = [Path(x) for x in args.train_parquet]
    eval_paths = [Path(x) for x in args.eval_parquet] if args.eval_parquet else None
    train(
        Path(args.train_config),
        Path(args.env_config),
        train_paths if len(train_paths) > 1 else train_paths[0],
        eval_paths if eval_paths is not None and len(eval_paths) > 1 else (eval_paths[0] if eval_paths else None),
        Path(args.run_dir),
    )


if __name__ == "__main__":
    main()
