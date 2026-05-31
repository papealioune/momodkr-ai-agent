from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from envs.reward.risk_penalties import RiskPenaltyConfig
from envs.reward.wrappers import (
    ChurnPenaltyWrapper,
    FundingCumulativePenaltyWrapper,
    LosingStreakEntryWrapper,
    PeakDrawdownPenaltyWrapper,
    PerEntryCostWrapper,
    QuadraticDrawdownWrapper,
    RewardFloorWrapper,
    apply_full_reward_shaping,
)


class _StubEnv(gym.Env):
    """Minimal stub: step() returns a fixed (obs, reward, info) supplied at init."""

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, base_reward: float, info: dict[str, Any]) -> None:
        super().__init__()
        self.base_reward = float(base_reward)
        self.info = info

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), self.info

    def step(self, action):
        return np.zeros(1, dtype=np.float32), self.base_reward, False, False, self.info


def test_quadratic_dd_wrapper_zero_below_threshold() -> None:
    cfg = RiskPenaltyConfig(dd_quadratic_coeff=50.0, dd_threshold=0.03)
    env = QuadraticDrawdownWrapper(_StubEnv(0.0, {"drawdown_pct": 0.02}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == 0.0


def test_quadratic_dd_wrapper_fires_above_threshold() -> None:
    cfg = RiskPenaltyConfig(dd_quadratic_coeff=50.0, dd_threshold=0.03)
    env = QuadraticDrawdownWrapper(_StubEnv(0.0, {"drawdown_pct": 0.04}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(-0.005)


def test_churn_wrapper_charges_per_cancellation() -> None:
    cfg = RiskPenaltyConfig(churn_penalty=0.01)
    env = ChurnPenaltyWrapper(_StubEnv(0.0, {"n_cancellations_this_step": 3}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(-0.03)


def test_peak_dd_wrapper_zero_when_no_position() -> None:
    cfg = RiskPenaltyConfig(peak_dd_coeff=0.5, peak_dd_threshold=0.005)
    env = PeakDrawdownPenaltyWrapper(_StubEnv(0.0, {"has_position": False, "unrealized_pct": -1.0, "peak_unrealized_pct": 1.0}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == 0.0


def test_peak_dd_wrapper_fires_when_giving_back_peak() -> None:
    cfg = RiskPenaltyConfig(peak_dd_coeff=0.5, peak_dd_threshold=0.005)
    # peak was 0.02, now 0.01 -> excess = 0.02 - 0.01 - 0.005 = 0.005 -> penalty = -0.5 * 0.005 = -0.0025
    env = PeakDrawdownPenaltyWrapper(_StubEnv(0.0, {"has_position": True, "unrealized_pct": 0.01, "peak_unrealized_pct": 0.02}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(-0.0025)


def test_per_entry_cost_wrapper() -> None:
    cfg = RiskPenaltyConfig(per_entry_cost=0.02)
    env = PerEntryCostWrapper(_StubEnv(0.0, {"opened_this_step": True}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(-0.02)


def test_losing_streak_wrapper() -> None:
    cfg = RiskPenaltyConfig(losing_streak_coeff=0.05, losing_streak_offset=2)
    info = {"opened_this_step": True, "consecutive_losses": 4}
    env = LosingStreakEntryWrapper(_StubEnv(0.0, info), cfg)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(-0.1)  # -0.05 * (4 - 2)


def test_funding_wrapper_only_when_holding() -> None:
    cfg = RiskPenaltyConfig(funding_coeff=0.01)
    env_flat = FundingCumulativePenaltyWrapper(
        _StubEnv(0.0, {"has_position": False, "cumulative_funding_pct": 0.005}), cfg
    )
    _, r_flat, *_ = env_flat.step(0)
    assert r_flat == 0.0
    env_long = FundingCumulativePenaltyWrapper(
        _StubEnv(0.0, {"has_position": True, "cumulative_funding_pct": 0.005}), cfg
    )
    _, r_long, *_ = env_long.step(0)
    assert r_long == pytest.approx(-5e-5)


def test_reward_floor_wrapper_clamps_at_floor() -> None:
    cfg = RiskPenaltyConfig(reward_floor=-5.0)
    env = RewardFloorWrapper(_StubEnv(-100.0, {}), cfg)
    _, reward, *_ = env.step(0)
    assert reward == -5.0


def test_apply_full_reward_shaping_composes_all_penalties() -> None:
    cfg = RiskPenaltyConfig(
        per_entry_cost=0.02,
        churn_penalty=0.005,
        dd_quadratic_coeff=50.0,
        dd_threshold=0.03,
        peak_dd_coeff=0.5,
        peak_dd_threshold=0.005,
        funding_coeff=0.01,
        losing_streak_coeff=0.05,
        losing_streak_offset=2,
        reward_floor=-5.0,
    )
    info = {
        "opened_this_step": True,
        "n_cancellations_this_step": 1,
        "drawdown_pct": 0.04,                # -0.005
        "has_position": True,
        "unrealized_pct": 0.01,
        "peak_unrealized_pct": 0.02,         # -0.0025
        "cumulative_funding_pct": 0.005,     # -5e-5
        "consecutive_losses": 4,             # -0.1
    }
    base_env = _StubEnv(1.0, info)
    stacked = apply_full_reward_shaping(base_env, cfg)
    _, reward, *_ = stacked.step(0)
    expected = 1.0 - 0.02 - 0.05 * (4 - 2) - 5e-5 - 0.005 - 0.0025 - 0.005
    assert reward == pytest.approx(expected, abs=1e-9)
