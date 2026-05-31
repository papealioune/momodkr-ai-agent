"""Composable gymnasium.Wrappers that add reward shaping on top of the core env.

The base MomoDkrEnv only returns realized PnL carrot + unrealized
breadcrumb -- pure state-transition rewards. Every other moleapp-derived
penalty (quadratic DD, churn, peak-drawdown, funding cumulative,
per-entry cost, losing-streak entry, floor) is implemented as a wrapper
so:
  - the matching engine stays focused on state transitions
  - hyperparameter sweeps can enable/disable individual penalties
  - the reward composition is auditable in one place

Wrappers stack from left to right (the outermost wrapper sees the most
shaped reward). Recommended stack (closest-to-core first):

    env = MomoDkrEnv(...)
    env = PerEntryCostWrapper(env, cfg.risk)
    env = LosingStreakEntryWrapper(env, cfg.risk)
    env = FundingCumulativePenaltyWrapper(env, cfg.risk)
    env = QuadraticDrawdownWrapper(env, cfg.risk)
    env = PeakDrawdownPenaltyWrapper(env, cfg.risk)
    env = ChurnPenaltyWrapper(env, cfg.risk)
    env = RewardFloorWrapper(env, cfg.risk)

Or, equivalently, use `apply_full_reward_shaping(env, cfg.risk)`.
"""

from __future__ import annotations

import gymnasium as gym

from envs.reward.risk_penalties import (
    RiskPenaltyConfig,
    apply_floor,
    churn_penalty,
    funding_cumulative_penalty,
    losing_streak_entry_penalty,
    peak_drawdown_penalty,
    per_entry_cost,
    quadratic_drawdown_penalty,
)


class _InfoDrivenRewardWrapper(gym.Wrapper):
    """Base class: reads context from info dict, returns a delta to add to reward."""

    def __init__(self, env: gym.Env, cfg: RiskPenaltyConfig) -> None:
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        delta = self._delta(info)
        return obs, float(reward) + float(delta), terminated, truncated, info

    def _delta(self, info: dict) -> float:
        raise NotImplementedError


class QuadraticDrawdownWrapper(_InfoDrivenRewardWrapper):
    def _delta(self, info: dict) -> float:
        return quadratic_drawdown_penalty(float(info.get("drawdown_pct", 0.0)), self.cfg)


class ChurnPenaltyWrapper(_InfoDrivenRewardWrapper):
    def _delta(self, info: dict) -> float:
        return churn_penalty(int(info.get("n_cancellations_this_step", 0)), self.cfg)


class PeakDrawdownPenaltyWrapper(_InfoDrivenRewardWrapper):
    def _delta(self, info: dict) -> float:
        return peak_drawdown_penalty(
            current_unrealized_pct=float(info.get("unrealized_pct", 0.0)),
            peak_unrealized_pct=float(info.get("peak_unrealized_pct", 0.0)),
            has_position=bool(info.get("has_position", False)),
            cfg=self.cfg,
        )


class FundingCumulativePenaltyWrapper(_InfoDrivenRewardWrapper):
    def _delta(self, info: dict) -> float:
        if not info.get("has_position", False):
            return 0.0
        return funding_cumulative_penalty(float(info.get("cumulative_funding_pct", 0.0)), self.cfg)


class PerEntryCostWrapper(_InfoDrivenRewardWrapper):
    def _delta(self, info: dict) -> float:
        return per_entry_cost(bool(info.get("opened_this_step", False)), self.cfg)


class LosingStreakEntryWrapper(_InfoDrivenRewardWrapper):
    def _delta(self, info: dict) -> float:
        return losing_streak_entry_penalty(
            consecutive_losses=int(info.get("consecutive_losses", 0)),
            opening_trade_this_step=bool(info.get("opened_this_step", False)),
            cfg=self.cfg,
        )


class RewardFloorWrapper(gym.Wrapper):
    """Clamp the cumulative reward to a floor (applied LAST in the stack)."""

    def __init__(self, env: gym.Env, cfg: RiskPenaltyConfig) -> None:
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, apply_floor(float(reward), self.cfg), terminated, truncated, info


def apply_full_reward_shaping(env: gym.Env, cfg: RiskPenaltyConfig) -> gym.Env:
    """Stack every Phase-3.5 penalty in the recommended order, with the floor outermost."""
    env = PerEntryCostWrapper(env, cfg)
    env = LosingStreakEntryWrapper(env, cfg)
    env = FundingCumulativePenaltyWrapper(env, cfg)
    env = QuadraticDrawdownWrapper(env, cfg)
    env = PeakDrawdownPenaltyWrapper(env, cfg)
    env = ChurnPenaltyWrapper(env, cfg)
    env = RewardFloorWrapper(env, cfg)
    return env
