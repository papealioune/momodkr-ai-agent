"""Phase-4 statistical gate.

Loads the per-episode reward distributions for one-or-more trained seeds
+ the random baseline, then runs a one-sided Mann-Whitney U test asking:
"Does the trained seed's reward distribution stochastically dominate the
random baseline?"

The gate passes only if every seed clears p < alpha (default 0.01).
Exits nonzero on failure so this script can be wired into CI.

Example:
    python -m scripts.mann_whitney_gate \\
        --baseline runs/random-baseline-btc/eval_rewards.json \\
        --seeds runs/v1-engine-cold-btc-s42/gate_eval_rewards.json \\
                runs/v1-engine-cold-btc-s43/gate_eval_rewards.json \\
                runs/v1-engine-cold-btc-s44/gate_eval_rewards.json \\
        --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _load_rewards(path: Path) -> tuple[list[float], dict]:
    payload = json.loads(path.read_text())
    rewards = payload.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError(f"{path}: missing or empty 'rewards' list")
    return rewards, payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Mann-Whitney U gate for Phase-4 statistical edge")
    p.add_argument("--baseline", required=True, help="random-baseline rewards JSON")
    p.add_argument("--seeds", nargs="+", required=True, help="one or more trained-seed rewards JSONs")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--min-seeds", type=int, default=3, help="warn if fewer seeds than this (statistical power)")
    args = p.parse_args()

    baseline_rewards, baseline_meta = _load_rewards(Path(args.baseline))
    baseline_arr = np.asarray(baseline_rewards, dtype=np.float64)
    logger.info(
        "baseline: n=%d mean=%.3f std=%.3f median=%.3f",
        len(baseline_arr), float(baseline_arr.mean()),
        float(baseline_arr.std()), float(np.median(baseline_arr)),
    )

    if len(args.seeds) < args.min_seeds:
        logger.warning(
            "only %d seed(s) supplied; the documented Phase-4 gate calls for %d-seed U-test. "
            "Proceeding but power is reduced.",
            len(args.seeds), args.min_seeds,
        )

    results = []
    any_failed = False
    for seed_path_str in args.seeds:
        seed_path = Path(seed_path_str)
        seed_rewards, seed_meta = _load_rewards(seed_path)
        seed_arr = np.asarray(seed_rewards, dtype=np.float64)
        u_stat, p_value = stats.mannwhitneyu(seed_arr, baseline_arr, alternative="greater")
        passed = bool(p_value < args.alpha)
        any_failed = any_failed or (not passed)
        results.append(
            {
                "seed_file": str(seed_path),
                "n": len(seed_arr),
                "mean": float(seed_arr.mean()),
                "std": float(seed_arr.std()),
                "median": float(np.median(seed_arr)),
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "passed": passed,
            }
        )
        status = "PASS" if passed else "FAIL"
        logger.info(
            "[%s] %s -- n=%d mean=%.3f median=%.3f U=%.0f p=%.3e (alpha=%.3f)",
            status, seed_path.name, len(seed_arr), float(seed_arr.mean()),
            float(np.median(seed_arr)), float(u_stat), float(p_value), args.alpha,
        )

    print(
        json.dumps(
            {
                "alpha": args.alpha,
                "baseline_mean": float(baseline_arr.mean()),
                "baseline_n": len(baseline_arr),
                "seeds": results,
                "all_passed": not any_failed,
            },
            indent=2,
        )
    )

    if any_failed:
        logger.error("Phase-4 gate FAILED: at least one seed did not beat random at p < %.3f", args.alpha)
        sys.exit(1)
    logger.info("Phase-4 gate PASSED: all %d seed(s) beat random at p < %.3f", len(args.seeds), args.alpha)


if __name__ == "__main__":
    main()
