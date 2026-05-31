"""Persist the model with the highest eval mean reward to best_checkpoint/.

moleapp lesson 3.6: ALWAYS deploy from best_checkpoint/, never
final_checkpoint/. Late iterations frequently diverge (V9 iter-725).

Designed to run as the `callback_on_new_best` argument of SB3's
EvalCallback (so SB3 only fires us when its own best-reward bookkeeping
already crossed). We additionally save:
    - the model weights (best_checkpoint.zip)
    - a marker JSON with the iteration number, mean reward, timestamp,
      and feature_version + checksum

so it's trivially auditable what shipped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from serving.feature_version import FEATURE_SPEC_CHECKSUM, FEATURE_VERSION

logger = logging.getLogger(__name__)


class BestCheckpointTracker(BaseCallback):
    def __init__(self, save_dir: str | Path, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward: float = float("-inf")

    def _on_step(self) -> bool:
        # Fires only when SB3's EvalCallback decides a new best is reached
        # (we're attached via callback_on_new_best=this).
        parent = self.parent
        mean_reward = float(getattr(parent, "best_mean_reward", float("nan")))
        if mean_reward != mean_reward or mean_reward <= self.best_mean_reward:
            return True
        self.best_mean_reward = mean_reward

        model_path = self.save_dir / "best_checkpoint.zip"
        self.model.save(model_path)
        marker = {
            "best_mean_reward": mean_reward,
            "num_timesteps": int(self.model.num_timesteps),
            "saved_at_unix": int(time.time()),
            "feature_version": FEATURE_VERSION,
            "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        }
        (self.save_dir / "best_checkpoint.json").write_text(json.dumps(marker, indent=2))
        if self.verbose:
            logger.info("best_checkpoint saved: mean_reward=%.4f at %d steps -> %s", mean_reward, marker["num_timesteps"], model_path)
        return True
