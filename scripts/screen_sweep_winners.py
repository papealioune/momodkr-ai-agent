"""Phase-5 sweep screener: pull a finished W&B sweep, rank by metric,
emit ready-to-train production configs for the top-N winners.

The sweep's job is to find good HYPERPARAMETER COMBINATIONS. The job
of this script is to take those combinations, screen for the ones
worth retraining cold at full budget, and prepare config YAMLs that
can be plugged into train_ppo.py without further editing.

It does NOT promote a sweep run's checkpoint to production -- per
moleapp §2.3, warm-starting is risky and we re-train cold instead. So
the output is configs, not checkpoints.

Example:
    python -m scripts.screen_sweep_winners \\
        --sweep dapps4africa/momodkr/nvpmnlw5 \\
        --top-n 5 \\
        --metric eval/mean_reward \\
        --output-dir configs/training/v2_winners \\
        --total-timesteps 20000000

This writes configs/training/v2_winners/v2_winner_01.yaml ... _05.yaml
plus a winners_summary.json with the W&B run id + metric for each.

Then for production cold-start training:
    for cfg in configs/training/v2_winners/*.yaml; do
      for seed in 42 43 44 45 46; do
        bash runpod/bg.sh train-prod-$(basename $cfg .yaml)-s$seed \\
          "python -m training.train_ppo --train-config $cfg \\
             --env-config configs/env/momodkr_v1.yaml \\
             --train-parquet data/episodes/BTCUSDT/0.1.0/train.parquet \\
             --eval-parquet  data/episodes/BTCUSDT/0.1.0/eval_selection.parquet \\
             --run-dir runs/prod/$(basename $cfg .yaml)-s$seed \\
             --seed $seed"
      done
    done
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Hyperparameters we read from each sweep run's config and bake into the
# production config YAML. Dotted keys (e.g. "ent_coef.start") need to
# walk into nested dicts when writing.
PROMOTED_HYPERPARAMS = [
    "learning_rate",
    "ent_coef.start",
    "ent_coef.end",
    "gamma",
    "gae_lambda",
    "clip_range",
    "vf_coef",
    "max_grad_norm",
    "n_steps",
    "batch_size",
    "n_epochs",
]


def _apply_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _load_baseline_yaml(base_path: Path) -> dict:
    with base_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Screen a finished W&B sweep and emit top-N production configs")
    p.add_argument("--sweep", required=True, help="entity/project/sweep_id")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--metric", default="eval/mean_reward", help="W&B metric key to sort by")
    p.add_argument("--higher-is-better", action="store_true", default=True)
    p.add_argument("--baseline-yaml", default="configs/training/v1_engine_cold.yaml",
                   help="Production YAML to clone + patch with sweep winner hyperparams")
    p.add_argument("--output-dir", default="configs/training/v2_winners")
    p.add_argument("--total-timesteps", type=int, default=20_000_000,
                   help="Override total_timesteps in the emitted production YAML (sweep ran 6M; prod wants 20M)")
    p.add_argument("--summary-only", action="store_true",
                   help="Print the ranked summary table but do not write YAML files")
    args = p.parse_args()

    import wandb

    api = wandb.Api()
    logger.info("fetching sweep %s ...", args.sweep)
    sweep = api.sweep(args.sweep)
    all_runs = list(sweep.runs)
    finished_runs = [r for r in all_runs if r.state == "finished"]
    logger.info("sweep has %d total runs, %d finished", len(all_runs), len(finished_runs))

    # Sort by chosen metric (drop runs missing the metric entirely)
    def _metric(run):
        v = run.summary.get(args.metric)
        if v is None:
            return float("-inf") if args.higher_is_better else float("inf")
        return float(v)

    ranked = sorted(finished_runs, key=_metric, reverse=args.higher_is_better)

    print(f"\n=== Sweep '{args.sweep}' ranked by {args.metric} ===")
    print(f"{'rank':>4}  {'metric':>9}  {'run_id':<12}  {'seed':>5}  best_checkpoint")
    for i, run in enumerate(ranked):
        metric_val = _metric(run)
        seed = run.config.get("seed", "?")
        ckpt = f"runs/sweep-v1/{run.id}/best_checkpoint/best_checkpoint.zip"
        marker = " <-- WINNER" if i < args.top_n else ""
        print(f"{i+1:>4}  {metric_val:>+9.3f}  {run.id:<12}  {seed:>5}  {ckpt}{marker}")

    if args.summary_only:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline_yaml(Path(args.baseline_yaml))
    summary = {"sweep": args.sweep, "metric": args.metric, "winners": []}

    for i, run in enumerate(ranked[: args.top_n]):
        cfg = json.loads(json.dumps(baseline))  # deep copy via JSON round-trip
        # Apply each promoted hyperparam from this run's config
        applied = {}
        for key in PROMOTED_HYPERPARAMS:
            if key in run.config:
                value = run.config[key]
                _apply_dotted(cfg, key, value)
                applied[key] = value
        # Pin total_timesteps to production target (sweep ran 6M)
        cfg["total_timesteps"] = int(args.total_timesteps)
        # Strip the seed list so --seed CLI override is the sole knob
        cfg.pop("seed", None)
        cfg["run_name"] = f"v2-winner-{i+1:02d}"
        out_path = out_dir / f"v2_winner_{i+1:02d}.yaml"
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        logger.info("wrote %s (from sweep run %s, %s=%.3f)", out_path, run.id, args.metric, _metric(run))
        summary["winners"].append({
            "rank": i + 1,
            "sweep_run_id": run.id,
            "sweep_run_url": run.url,
            "metric_value": _metric(run),
            "applied_overrides": applied,
            "production_yaml": str(out_path),
        })

    summary_path = out_dir / "winners_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s -- %d winner configs ready for cold-start prod training", summary_path, args.top_n)


if __name__ == "__main__":
    main()
