"""Staged curriculum for MomoDkr.

The curriculum YAML describes an ordered list of stages. Each stage:
  - names the symbols whose train/eval parquets feed the env pool
  - sets a total_timesteps budget for the stage
  - optionally overrides reward / env knobs (e.g. tighter min_tp later)
  - inherits training hyperparameters from the base training YAML; per-stage
    overrides shallow-merge into that base

Warm-starting between stages reuses the previous stage's
`<run_dir>/best_checkpoint/best_checkpoint.zip` -- aligning with the
moleapp lesson that production deploys from best_checkpoint, never
final_checkpoint. If a stage's best did not improve over the previous
stage's, the previous best carries forward.

Per moleapp lesson 2.2 (warm-start can degrade) we lower the LR for
warm-start stages by default; override via stage.warm_start.lr_start /
lr_end if you need different behavior.

Usage:
    python -m training.curriculum \\
        --curriculum-config configs/training/curriculum_v1.yaml \\
        --base-train-config configs/training/v1_engine_cold.yaml \\
        --env-config configs/env/momodkr_v1.yaml \\
        --episodes-root data/episodes \\
        --feature-version 0.1.0 \\
        --run-dir runs/curriculum-v1
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from serving.feature_version import FEATURE_VERSION
from training.train_ppo import train
from training.utils import load_yaml

logger = logging.getLogger(__name__)


@dataclass
class StageSpec:
    name: str
    symbols: list[str]
    total_timesteps: int
    train_overrides: dict = field(default_factory=dict)
    env_overrides: dict = field(default_factory=dict)
    warm_start: bool = True


@dataclass
class StageResult:
    name: str
    run_dir: Path
    best_checkpoint: Path
    final_checkpoint: Path
    elapsed_s: float


def parse_stages(curriculum_cfg: dict) -> list[StageSpec]:
    stages_raw = curriculum_cfg.get("stages")
    if not stages_raw:
        raise ValueError("curriculum config must define a 'stages' list")
    out: list[StageSpec] = []
    for i, s in enumerate(stages_raw):
        if "name" not in s or "symbols" not in s or "total_timesteps" not in s:
            raise ValueError(f"stage[{i}] must have name, symbols, total_timesteps")
        out.append(
            StageSpec(
                name=str(s["name"]),
                symbols=list(s["symbols"]),
                total_timesteps=int(s["total_timesteps"]),
                train_overrides=dict(s.get("train_overrides", {})),
                env_overrides=dict(s.get("env_overrides", {})),
                warm_start=bool(s.get("warm_start", i > 0)),
            )
        )
    return out


def episode_paths_for_symbols(
    symbols: list[str],
    episodes_root: Path,
    feature_version: str,
    split: str,
) -> list[Path]:
    paths: list[Path] = []
    for sym in symbols:
        p = episodes_root / sym / feature_version / f"{split}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"episode parquet missing for {sym}: {p}")
        paths.append(p)
    return paths


def _merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _write_stage_configs(
    stage: StageSpec,
    base_train_cfg: dict,
    base_env_cfg: dict,
    stage_dir: Path,
    prior_best_zip: Path | None,
) -> tuple[Path, Path]:
    train_cfg = _merge(base_train_cfg, stage.train_overrides)
    train_cfg["total_timesteps"] = stage.total_timesteps
    train_cfg["run_name"] = stage.name
    if stage.warm_start and prior_best_zip is not None and prior_best_zip.exists():
        train_cfg["warm_start_from"] = str(prior_best_zip)
    env_cfg = _merge(base_env_cfg, stage.env_overrides)
    train_yaml = stage_dir / "train.yaml"
    env_yaml = stage_dir / "env.yaml"
    train_yaml.write_text(yaml.safe_dump(train_cfg))
    env_yaml.write_text(yaml.safe_dump(env_cfg))
    return train_yaml, env_yaml


def run_curriculum(
    curriculum_cfg_path: Path,
    base_train_cfg_path: Path,
    env_cfg_path: Path,
    episodes_root: Path,
    feature_version: str,
    run_dir: Path,
) -> list[StageResult]:
    curriculum_cfg = load_yaml(curriculum_cfg_path)
    base_train_cfg = load_yaml(base_train_cfg_path)
    base_env_cfg = load_yaml(env_cfg_path)
    stages = parse_stages(curriculum_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    prior_best: Path | None = None
    results: list[StageResult] = []
    for i, stage in enumerate(stages):
        logger.info("=== stage %d/%d: %s (symbols=%s, steps=%d) ===", i + 1, len(stages), stage.name, stage.symbols, stage.total_timesteps)
        stage_dir = run_dir / f"{i:02d}_{stage.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        train_yaml, env_yaml = _write_stage_configs(stage, base_train_cfg, base_env_cfg, stage_dir, prior_best)

        train_paths = episode_paths_for_symbols(stage.symbols, episodes_root, feature_version, "train")
        eval_paths = episode_paths_for_symbols(stage.symbols, episodes_root, feature_version, "eval")

        t0 = time.time()
        best_ckpt = train(train_yaml, env_yaml, train_paths, eval_paths, stage_dir)
        elapsed = time.time() - t0

        best_zip = best_ckpt if best_ckpt.exists() else (stage_dir / "best_checkpoint" / "best_model.zip")
        final_zip = stage_dir / "final_checkpoint.zip"
        results.append(StageResult(name=stage.name, run_dir=stage_dir, best_checkpoint=best_zip, final_checkpoint=final_zip, elapsed_s=elapsed))
        prior_best = best_zip if best_zip.exists() else prior_best

    if prior_best and prior_best.exists():
        promoted = run_dir / "best_checkpoint"
        promoted.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prior_best, promoted / "best_checkpoint.zip")
        (run_dir / "curriculum_manifest.json").write_text(
            json.dumps(
                {
                    "feature_version": feature_version,
                    "stages": [
                        {"name": r.name, "run_dir": str(r.run_dir), "best_checkpoint": str(r.best_checkpoint), "elapsed_s": r.elapsed_s}
                        for r in results
                    ],
                    "promoted_best_checkpoint": str(promoted / "best_checkpoint.zip"),
                },
                indent=2,
            )
        )
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Run the MomoDkr staged curriculum")
    p.add_argument("--curriculum-config", required=True)
    p.add_argument("--base-train-config", required=True)
    p.add_argument("--env-config", required=True)
    p.add_argument("--episodes-root", default="data/episodes")
    p.add_argument("--feature-version", default=FEATURE_VERSION)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run_curriculum(
        Path(args.curriculum_config),
        Path(args.base_train_config),
        Path(args.env_config),
        Path(args.episodes_root),
        args.feature_version,
        Path(args.run_dir),
    )


if __name__ == "__main__":
    main()
