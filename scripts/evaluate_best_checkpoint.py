"""Run N deterministic eval episodes on a trained best_checkpoint.zip and
dump per-episode total rewards as JSON.

This is the trained-policy half of the Phase-4 Mann-Whitney U gate: the
random baseline (scripts/run_random_baseline.py) provides the null
distribution; this script provides the alternative for each seed.

Determinism + parity: we set deterministic=True so the policy uses
argmax(actions) -- same setting SB3's EvalCallback uses to populate
eval/mean_reward during training. The same env config + eval parquet
must be used for both random + trained runs so the comparison is fair.

Example:
    python -m scripts.evaluate_best_checkpoint \\
        --checkpoint runs/v1-engine-cold-btc-s42/best_checkpoint/best_checkpoint.zip \\
        --env-config configs/env/momodkr_v1.yaml \\
        --eval-parquet data/episodes/BTCUSDT/0.1.0/eval.parquet \\
        --n-episodes 1000 \\
        --out runs/v1-engine-cold-btc-s42/gate_eval_rewards.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs.momodkr_env import MomoDkrEnv
from training.train_ppo import env_cfg_from_yaml

logger = logging.getLogger(__name__)


def run_checkpoint_episodes(
    checkpoint_path: Path,
    env_config_path: Path,
    eval_parquet: Path,
    n_episodes: int,
    seed: int,
) -> tuple[list[float], list[int]]:
    env_cfg = env_cfg_from_yaml(env_config_path)
    env = MomoDkrEnv(eval_parquet, env_cfg, seed=seed)
    model = PPO.load(checkpoint_path, device="cpu")
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    lengths: list[int] = []
    t0 = time.time()
    for i in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(int(action))
            total += float(r)
            steps += 1
        rewards.append(total)
        lengths.append(steps)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                "%d/%d episodes -- elapsed=%.1fs mean=%.3f std=%.3f mean_len=%d",
                i + 1, n_episodes, elapsed,
                float(np.mean(rewards)), float(np.std(rewards)), int(np.mean(lengths)),
            )
    return rewards, lengths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Eval-distribution dump for a trained PPO checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-config", required=True)
    p.add_argument("--eval-parquet", required=True)
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rewards, lengths = run_checkpoint_episodes(
        Path(args.checkpoint), Path(args.env_config), Path(args.eval_parquet),
        args.n_episodes, args.seed,
    )
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "ppo_best_checkpoint",
        "checkpoint": str(args.checkpoint),
        "n_episodes": len(rewards),
        "seed": args.seed,
        "env_config": str(args.env_config),
        "eval_parquet": str(args.eval_parquet),
        "rewards": rewards,
        "episode_lengths": lengths,
        "summary": {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "median_reward": float(np.median(rewards_arr)),
            "p05_reward": float(np.percentile(rewards_arr, 5)),
            "p95_reward": float(np.percentile(rewards_arr, 95)),
            "mean_ep_length": int(lengths_arr.mean()),
            "median_ep_length": int(np.median(lengths_arr)),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "wrote %d episode rewards to %s -- mean=%.3f median=%.3f mean_len=%d",
        len(rewards), out_path, payload["summary"]["mean_reward"],
        payload["summary"]["median_reward"], payload["summary"]["mean_ep_length"],
    )


if __name__ == "__main__":
    main()
