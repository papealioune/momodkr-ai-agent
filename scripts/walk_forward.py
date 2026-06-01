"""Walk-forward validation orchestrator.

Before burning compute on a 20M-step full-history run, run a much shorter
training loop on a 3-6 month window and measure performance on the next
1 month. If the agent fails to generalise on the holdout, fix the wrapper
weights / hyperparameters BEFORE the big run.

Example:
    python -m scripts.walk_forward \\
        --symbol BTCUSDT \\
        --train-start 2024-01-01 --train-end 2024-06-30 \\
        --eval-start 2024-07-01 --eval-end 2024-07-31 \\
        --train-config configs/training/v1_engine_cold.yaml \\
        --env-config configs/env/momodkr_v1.yaml \\
        --run-dir runs/walkforward-btc-2024-h1

The episodes for the walk-forward windows are materialised under
data/episodes/<SYMBOL>/<feature_version>_<label>/, separate from the
production 80/20 split so the two never collide.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.preprocessors.episode_builder import build_walk_forward_split
from training.train_ppo import train

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Walk-forward training + validation slice")
    p.add_argument("--symbol", required=True)
    p.add_argument("--train-start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--train-end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--eval-start", required=True, help="YYYY-MM-DD inclusive (must be > train-end)")
    p.add_argument("--eval-end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--train-config", required=True)
    p.add_argument("--env-config", required=True)
    p.add_argument("--dataset-root", default="data/datasets")
    p.add_argument("--episodes-root", default="data/episodes")
    p.add_argument("--label", default=None, help="suffix for the episode dir (default: derived from window)")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    label = args.label or f"wf_{args.train_start}_{args.eval_end}"
    manifest = build_walk_forward_split(
        symbol=args.symbol,
        dataset_root=Path(args.dataset_root),
        episodes_root=Path(args.episodes_root),
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        label=label,
    )
    logger.info(
        "walk-forward episodes: %d train rows / %d eval rows", manifest.train_rows, manifest.eval_rows
    )

    episode_dir = Path(args.episodes_root) / args.symbol / f"0.1.0_{label}"
    train_parquet = episode_dir / "train.parquet"
    eval_parquet = episode_dir / "eval.parquet"

    best = train(
        Path(args.train_config),
        Path(args.env_config),
        train_parquet,
        eval_parquet,
        Path(args.run_dir),
    )
    logger.info("walk-forward run complete; best_checkpoint=%s", best)


if __name__ == "__main__":
    main()
