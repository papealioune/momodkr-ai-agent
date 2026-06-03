"""Linearly decays ent_coef across training.

SB3 evaluates learning_rate schedules natively but stores ent_coef as a
raw float that's read every train step -- callables crash multiplication
(TypeError: 'function' * 'Tensor'). This callback walks the float down
between rollouts so we can express the 0.005 -> 0.0005 entropy decay
that moleapp's lessons mandate for HFT PPO convergence.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EntropyScheduleCallback(BaseCallback):
    def __init__(self, start: float, end: float, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if start < end:
            raise ValueError(f"EntropyScheduleCallback expects start >= end (decay), got {start} -> {end}")
        self.start = float(start)
        self.end = float(end)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        progress_remaining = float(getattr(self.model, "_current_progress_remaining", 1.0))
        ent = self.end + (self.start - self.end) * progress_remaining
        self.model.ent_coef = ent
        self.model.logger.record("train/ent_coef_scheduled", ent)
