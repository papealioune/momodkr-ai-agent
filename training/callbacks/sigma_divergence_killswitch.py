"""Phase 4 safety net -- aborts training when the policy collapses or diverges.

moleapp lesson 2.3: sigma > 2.0 on Box actions for 2 consecutive evals =
random sampling, kill and revert to best_checkpoint. The Discrete-action
analogue is the categorical-entropy fraction: 1.0 = uniform random over
the 5 actions; 0.0 = a single action dominates.

Rule encoded here:
    - HIGH-entropy divergence: normalised entropy > high_threshold for
      consecutive_evals consecutive evals -> abort.
    - LOW-entropy collapse: normalised entropy < low_threshold for
      consecutive_evals consecutive evals -> abort.

The thresholds default to (high=0.95, low=0.05). Tune via YAML in
training/configs.

The callback runs piggybacked on SB3's `EvalCallback` parent: pass it via
`callback_after_eval` so it fires once per eval pass.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.utils import categorical_entropy_normalised

logger = logging.getLogger(__name__)


class SigmaDivergenceKillswitch(BaseCallback):
    def __init__(
        self,
        eval_env,
        high_threshold: float = 0.95,
        low_threshold: float = 0.05,
        consecutive_evals: int = 2,
        n_obs_samples: int = 512,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.consecutive_evals = int(consecutive_evals)
        self.n_obs_samples = int(n_obs_samples)
        self.history: deque[float] = deque(maxlen=max(self.consecutive_evals, 1))
        self.aborted = False

    def _on_step(self) -> bool:
        # Fires once per eval when this callback is attached to
        # EvalCallback(callback_after_eval=...). SB3 invokes on_step() each
        # call; we use it as our event hook.
        ent_pct = self._compute_entropy_fraction()
        self.history.append(ent_pct)
        if self.verbose:
            logger.info("sigma killswitch: normalised entropy = %.3f (history=%s)", ent_pct, list(self.history))
        self.model.logger.record("killswitch/entropy_pct", ent_pct)

        if len(self.history) < self.consecutive_evals:
            return True

        if all(e > self.high_threshold for e in self.history):
            logger.error(
                "sigma killswitch ABORT: entropy > %.2f for %d consecutive evals -> uniform random policy",
                self.high_threshold,
                self.consecutive_evals,
            )
            self.aborted = True
            return False
        if all(e < self.low_threshold for e in self.history):
            logger.error(
                "sigma killswitch ABORT: entropy < %.2f for %d consecutive evals -> collapsed policy",
                self.low_threshold,
                self.consecutive_evals,
            )
            self.aborted = True
            return False
        return True

    def _compute_entropy_fraction(self) -> float:
        obs_batch = self._collect_eval_observations()
        if obs_batch is None or len(obs_batch) == 0:
            return float("nan")
        try:
            import torch

            obs_tensor = torch.as_tensor(obs_batch, device=self.model.device)
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()
        except Exception as e:
            logger.warning("entropy probe failed: %s", e)
            return float("nan")
        return categorical_entropy_normalised(probs)

    def _collect_eval_observations(self) -> np.ndarray | None:
        env = self.eval_env
        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        except Exception as e:
            logger.warning("eval env reset failed: %s", e)
            return None
        collected: list[np.ndarray] = []
        for _ in range(self.n_obs_samples):
            collected.append(np.asarray(obs))
            try:
                action, _ = self.model.predict(obs, deterministic=False)
                step_out = env.step(action)
            except Exception:
                break
            if len(step_out) == 5:
                obs, _, term, trunc, _ = step_out
                done = bool(np.any(term) or np.any(trunc))
            else:
                obs, _, done, _ = step_out
                done = bool(np.any(done))
            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
        if not collected:
            return None
        return np.stack([np.asarray(o).reshape(-1) for o in collected])
