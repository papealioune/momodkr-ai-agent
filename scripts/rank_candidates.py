"""Phase-5 candidate selection: rank trained checkpoints on the deployment rubric.

Takes one or more best_checkpoint.zip files, evaluates each on a target
parquet (200-1000 episodes deterministic), computes the 4-metric
deployment rubric, and emits a ranked table + JSON summary.

Deployment rubric (a checkpoint qualifies for live capital only if ALL four pass):
  mean_reward    > +2.0        well above baseline noise
  median_reward  > 0           true center is profitable, not just lucky tails
  ep_survival    > 0.90        full-episode rate (no DD-kill); capital safe
  p5_reward      > -15         tail risk bounded (5th percentile floor)

Phase-4 reference: only S42 cleared all four (mean +4.48, median +2.98,
survival 100%, p5 around -10). S43 + S44 cleared the U-test gate vs
random but failed median + survival, so they are NOT deployment-grade.

Example:
    python -m scripts.rank_candidates \\
        --checkpoints runs/v1-engine-cold-btc-s42/best_checkpoint/best_checkpoint.zip \\
                      runs/sweep-v1/*/best_checkpoint/best_checkpoint.zip \\
        --env-config configs/env/momodkr_v1.yaml \\
        --eval-parquet data/episodes/BTCUSDT/0.1.0/eval_selection.parquet \\
        --n-episodes 500 \\
        --out runs/phase5_ranking.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.evaluate_best_checkpoint import run_checkpoint_episodes

logger = logging.getLogger(__name__)

DEPLOYMENT_RUBRIC = {
    "mean_reward": 2.0,
    "median_reward": 0.0,
    "ep_survival": 0.90,
    "p5_reward": -15.0,
}
FULL_EPISODE_LEN = 9000  # ticks; an episode that hits this length survived (no DD-kill)


def evaluate_one(
    checkpoint_path: Path,
    env_config_path: Path,
    eval_parquet: Path,
    n_episodes: int,
    seed: int,
) -> dict:
    logger.info("evaluating %s (n=%d)", checkpoint_path, n_episodes)
    rewards, lengths = run_checkpoint_episodes(
        checkpoint_path, env_config_path, eval_parquet, n_episodes, seed
    )
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    survival_rate = float((lengths_arr >= FULL_EPISODE_LEN).mean())
    metrics = {
        "checkpoint": str(checkpoint_path),
        "n_episodes": n_episodes,
        "mean_reward": float(rewards_arr.mean()),
        "median_reward": float(np.median(rewards_arr)),
        "std_reward": float(rewards_arr.std()),
        "p5_reward": float(np.percentile(rewards_arr, 5)),
        "p95_reward": float(np.percentile(rewards_arr, 95)),
        "ep_survival": survival_rate,
        "mean_ep_length": int(lengths_arr.mean()),
    }
    metrics["passes_rubric"] = (
        metrics["mean_reward"] > DEPLOYMENT_RUBRIC["mean_reward"]
        and metrics["median_reward"] > DEPLOYMENT_RUBRIC["median_reward"]
        and metrics["ep_survival"] > DEPLOYMENT_RUBRIC["ep_survival"]
        and metrics["p5_reward"] > DEPLOYMENT_RUBRIC["p5_reward"]
    )
    metrics["rubric_breakdown"] = {
        "mean_reward": {"value": metrics["mean_reward"], "threshold": DEPLOYMENT_RUBRIC["mean_reward"], "pass": metrics["mean_reward"] > DEPLOYMENT_RUBRIC["mean_reward"]},
        "median_reward": {"value": metrics["median_reward"], "threshold": DEPLOYMENT_RUBRIC["median_reward"], "pass": metrics["median_reward"] > DEPLOYMENT_RUBRIC["median_reward"]},
        "ep_survival": {"value": metrics["ep_survival"], "threshold": DEPLOYMENT_RUBRIC["ep_survival"], "pass": metrics["ep_survival"] > DEPLOYMENT_RUBRIC["ep_survival"]},
        "p5_reward": {"value": metrics["p5_reward"], "threshold": DEPLOYMENT_RUBRIC["p5_reward"], "pass": metrics["p5_reward"] > DEPLOYMENT_RUBRIC["p5_reward"]},
    }
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Rank Phase-5 candidate checkpoints on the deployment rubric")
    p.add_argument("--checkpoints", nargs="+", required=True, help="one or more best_checkpoint.zip paths")
    p.add_argument("--env-config", required=True)
    p.add_argument("--eval-parquet", required=True, help="target parquet (selection for screening, holdout for final validation)")
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results = []
    for ckpt in args.checkpoints:
        try:
            metrics = evaluate_one(
                Path(ckpt), Path(args.env_config), Path(args.eval_parquet),
                args.n_episodes, args.seed,
            )
            results.append(metrics)
        except Exception as e:
            logger.error("eval failed for %s: %s", ckpt, e)
            results.append({"checkpoint": ckpt, "error": str(e), "passes_rubric": False})

    # Rank by mean_reward (deployable rubric-passers first, then everyone else)
    results.sort(key=lambda m: (m.get("passes_rubric", False), m.get("mean_reward", -1e9)), reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "deployment_rubric": DEPLOYMENT_RUBRIC,
        "n_episodes_per_eval": args.n_episodes,
        "eval_parquet": str(args.eval_parquet),
        "n_candidates": len(results),
        "n_deployment_grade": sum(1 for r in results if r.get("passes_rubric")),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    # Pretty print top-10 to stdout
    print("\n=== Phase-5 candidate ranking ===")
    print(f"{'rank':>4}  {'rubric':>7}  {'mean':>7}  {'med':>7}  {'p5':>7}  {'surv':>6}  ckpt")
    for i, r in enumerate(results[:10]):
        if r.get("error"):
            print(f"{i+1:>4}  ERROR -- {r['checkpoint']}: {r['error']}")
            continue
        flag = "PASS" if r["passes_rubric"] else "fail"
        print(
            f"{i+1:>4}  {flag:>7}  {r['mean_reward']:>+7.2f}  {r['median_reward']:>+7.2f}  "
            f"{r['p5_reward']:>+7.2f}  {r['ep_survival']:>5.1%}  {r['checkpoint']}"
        )
    logger.info(
        "wrote %s -- %d candidates, %d deployment-grade",
        out_path, len(results), payload["n_deployment_grade"],
    )


if __name__ == "__main__":
    main()
