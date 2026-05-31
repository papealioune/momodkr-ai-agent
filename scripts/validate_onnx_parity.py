"""ONNX parity gate -- no model ships to mainnet without this passing.

Loads the SB3 PyTorch model + the exported ONNX, runs both on identical
observation batches, and asserts max(|logits_torch - logits_onnx|) <
tol_logits AND argmax(logits_torch) == argmax(logits_onnx) on every row.

Source of observations (priority order):
  1. --obs-parquet  -- explicit feature parquet (e.g. eval split)
  2. --eval-log-dir -- a TradeLogCallback output directory whose JSONs
                      contain obs_history; concatenated into a single
                      batch.
  3. --random-batch -- synthetic batch (development only; do NOT use this
                      as a production gate).

moleapp lesson 3.4: silent ONNX export bugs ship to prod undetected.
v9 iter-50 achieved max_diff=0.0. The default tolerance here is 1e-4 per
the moleapp gate.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from stable_baselines3 import PPO

from serving.feature_version import MARKET_FEATURE_NAMES, OBS_DIM

logger = logging.getLogger(__name__)

DEFAULT_TOL = 1e-4


def load_obs_from_parquet(path: Path, max_rows: int = 1000) -> np.ndarray:
    df = pd.read_parquet(path)
    missing = [c for c in MARKET_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"parquet {path} missing market features: {missing}")
    market = df[list(MARKET_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    # position features are zeroed for parity probes -- the policy is invariant under
    # the position-feature block when there's no carry-over state.
    n = min(len(market), max_rows)
    pos = np.zeros((n, OBS_DIM - len(MARKET_FEATURE_NAMES)), dtype=np.float32)
    return np.concatenate([market[:n], pos], axis=1)


def load_obs_from_eval_log_dir(eval_dir: Path, max_rows: int = 1000) -> np.ndarray:
    files = sorted(eval_dir.glob("*.json"))
    rows: list[list[float]] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for ep in data.get("episodes", []):
            obs_history = ep.get("obs_history")
            if not obs_history:
                continue
            rows.extend(obs_history)
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError(f"no obs_history found in eval logs under {eval_dir}; rerun training with record_obs=True")
    arr = np.asarray(rows[:max_rows], dtype=np.float32)
    if arr.shape[1] != OBS_DIM:
        raise ValueError(f"eval-log obs dim {arr.shape[1]} != expected {OBS_DIM}")
    return arr


def torch_logits(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(model.device)
        features = model.policy.features_extractor(obs_t)
        latent_pi, _ = model.policy.mlp_extractor(features)
        logits = model.policy.action_net(latent_pi)
        return logits.cpu().numpy()


def onnx_logits(session: ort.InferenceSession, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outs = session.run(None, {"obs": obs.astype(np.float32)})
    return outs[0], outs[1]  # logits, action


def validate(
    checkpoint_path: Path,
    onnx_path: Path,
    obs: np.ndarray,
    tol_logits: float = DEFAULT_TOL,
) -> dict:
    model = PPO.load(checkpoint_path, device="cpu")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    pt_logits = torch_logits(model, obs)
    ox_logits, ox_action = onnx_logits(session, obs)
    if pt_logits.shape != ox_logits.shape:
        raise ValueError(f"logits shape mismatch: torch={pt_logits.shape} onnx={ox_logits.shape}")

    diff = np.abs(pt_logits - ox_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    pt_action = pt_logits.argmax(axis=-1)
    action_match = bool(np.array_equal(pt_action.astype(np.int64), ox_action.astype(np.int64)))

    result = {
        "n_obs": int(obs.shape[0]),
        "max_diff_logits": max_diff,
        "mean_diff_logits": mean_diff,
        "action_match": action_match,
        "tol_logits": tol_logits,
        "passed": bool(max_diff < tol_logits and action_match),
    }
    if not result["passed"]:
        logger.error("ONNX parity FAILED: %s", result)
    else:
        logger.info("ONNX parity PASSED: max_diff=%.2e mean_diff=%.2e n=%d", max_diff, mean_diff, obs.shape[0])
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="ONNX vs PyTorch parity gate for MomoDkr")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--obs-parquet")
    src.add_argument("--eval-log-dir")
    src.add_argument("--random-batch", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=1000)
    args = p.parse_args()

    if args.obs_parquet:
        obs = load_obs_from_parquet(Path(args.obs_parquet), max_rows=args.max_rows)
    elif args.eval_log_dir:
        obs = load_obs_from_eval_log_dir(Path(args.eval_log_dir), max_rows=args.max_rows)
    else:
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((args.random_batch, OBS_DIM)).astype(np.float32)
        logger.warning("using random batch -- NOT a production gate")

    result = validate(Path(args.checkpoint), Path(args.onnx), obs, tol_logits=args.tol)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
