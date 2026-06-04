"""Null distribution for the Phase-4 Mann-Whitney U gate.

Runs N independent episodes on the held-out eval parquet picking actions
uniformly from the Discrete(5) action space, and dumps the per-episode
total reward as JSON. That distribution is the "no-skill" baseline our
trained seeds have to beat.

The reward function in MomoDkrEnv already accounts for fees, slippage,
funding, and drawdown -- so a random policy will reliably lose money
and the U-test gate becomes "is the trained policy's reward distribution
stochastically greater than random's, with p < 0.01".

Example:
    python -m scripts.run_random_baseline \\
        --env-config configs/env/momodkr_v1.yaml \\
        --eval-parquet data/episodes/BTCUSDT/0.1.0/eval.parquet \\
        --n-episodes 1000 \\
        --out runs/random-baseline-btc/eval_rewards.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from envs.momodkr_env import MomoDkrEnv
from training.train_ppo import env_cfg_from_yaml

logger = logging.getLogger(__name__)


def run_random_episodes(
    env_config_path: Path,
    eval_parquet: Path,
    n_episodes: int,
    seed: int,
) -> list[float]:
    env_cfg = env_cfg_from_yaml(env_config_path)
    env = MomoDkrEnv(eval_parquet, env_cfg, seed=seed)
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    t0 = time.time()
    for i in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(rng.integers(0, env.action_space.n))
            _, r, terminated, truncated, _ = env.step(action)
            total += float(r)
        rewards.append(total)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                "%d/%d episodes -- elapsed=%.1fs mean=%.3f std=%.3f",
                i + 1, n_episodes, elapsed, float(np.mean(rewards)), float(np.std(rewards)),
            )
    return rewards


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Random-policy reward distribution for the Phase-4 U-test gate")
    p.add_argument("--env-config", required=True)
    p.add_argument("--eval-parquet", required=True)
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rewards = run_random_episodes(
        Path(args.env_config), Path(args.eval_parquet), args.n_episodes, args.seed
    )
    arr = np.asarray(rewards, dtype=np.float64)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "random_uniform",
        "n_episodes": len(rewards),
        "seed": args.seed,
        "env_config": str(args.env_config),
        "eval_parquet": str(args.eval_parquet),
        "rewards": rewards,
        "summary": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "wrote %d episode rewards to %s -- mean=%.3f std=%.3f median=%.3f",
        len(rewards), out_path, payload["summary"]["mean"],
        payload["summary"]["std"], payload["summary"]["median"],
    )


if __name__ == "__main__":
    main()
