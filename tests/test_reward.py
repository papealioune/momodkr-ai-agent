import pytest

from envs.reward.breadcrumb import BreadcrumbConfig, unrealized_breadcrumb
from envs.reward.pnl_reward import PnLRewardConfig, realized_pnl_reward
from envs.reward.risk_penalties import (
    RiskPenaltyConfig,
    apply_floor,
    funding_cumulative_penalty,
    losing_streak_entry_penalty,
    per_entry_cost,
    quadratic_drawdown_penalty,
)


def test_carrot_asymmetry_wins_get_higher_multiplier() -> None:
    cfg = PnLRewardConfig(win_multiplier=4.0, loss_multiplier=1.8)
    assert realized_pnl_reward(0.01, cfg) == pytest.approx(0.04)
    assert realized_pnl_reward(-0.01, cfg) == pytest.approx(-0.018)
    assert realized_pnl_reward(0.0, cfg) == 0.0


def test_per_entry_cost_only_charges_on_open() -> None:
    cfg = RiskPenaltyConfig(per_entry_cost=0.02)
    assert per_entry_cost(opening_trade_this_step=True, cfg=cfg) == pytest.approx(-0.02)
    assert per_entry_cost(opening_trade_this_step=False, cfg=cfg) == 0.0


def test_quadratic_drawdown_penalty_zero_below_threshold() -> None:
    cfg = RiskPenaltyConfig(dd_quadratic_coeff=50.0, dd_threshold=0.03)
    assert quadratic_drawdown_penalty(0.02, cfg) == 0.0
    assert quadratic_drawdown_penalty(0.03, cfg) == 0.0


def test_quadratic_drawdown_penalty_grows_with_excess() -> None:
    cfg = RiskPenaltyConfig(dd_quadratic_coeff=50.0, dd_threshold=0.03)
    # at dd=0.04 (excess=0.01) -> -50 * 0.0001 = -0.005
    assert quadratic_drawdown_penalty(0.04, cfg) == pytest.approx(-0.005)
    # at dd=0.05 (excess=0.02) -> -50 * 0.0004 = -0.02
    assert quadratic_drawdown_penalty(0.05, cfg) == pytest.approx(-0.02)
    # quadratic shape: doubling excess quadruples penalty
    p1 = quadratic_drawdown_penalty(0.05, cfg)
    p2 = quadratic_drawdown_penalty(0.07, cfg)
    assert p2 / p1 == pytest.approx(4.0)


def test_losing_streak_entry_penalty_only_after_offset() -> None:
    cfg = RiskPenaltyConfig(losing_streak_coeff=0.05, losing_streak_offset=2)
    assert losing_streak_entry_penalty(consecutive_losses=0, opening_trade_this_step=True, cfg=cfg) == 0.0
    assert losing_streak_entry_penalty(consecutive_losses=2, opening_trade_this_step=True, cfg=cfg) == 0.0
    assert losing_streak_entry_penalty(consecutive_losses=3, opening_trade_this_step=True, cfg=cfg) == pytest.approx(-0.05)
    assert losing_streak_entry_penalty(consecutive_losses=5, opening_trade_this_step=True, cfg=cfg) == pytest.approx(-0.15)
    # not opening = no penalty regardless of streak
    assert losing_streak_entry_penalty(consecutive_losses=10, opening_trade_this_step=False, cfg=cfg) == 0.0


def test_funding_penalty_uses_abs_value() -> None:
    cfg = RiskPenaltyConfig(funding_coeff=0.01)
    assert funding_cumulative_penalty(0.001, cfg) == pytest.approx(-1e-5)
    assert funding_cumulative_penalty(-0.001, cfg) == pytest.approx(-1e-5)


def test_apply_floor_clamps_below_minimum() -> None:
    cfg = RiskPenaltyConfig(reward_floor=-5.0)
    assert apply_floor(-100.0, cfg) == -5.0
    assert apply_floor(-5.0, cfg) == -5.0
    assert apply_floor(0.5, cfg) == 0.5


def test_breadcrumb_zero_when_no_position() -> None:
    cfg = BreadcrumbConfig(unrealized_breadcrumb_coeff=0.3)
    assert unrealized_breadcrumb(0.05, has_position=False, cfg=cfg) == 0.0


def test_breadcrumb_scales_with_unrealized_pnl() -> None:
    cfg = BreadcrumbConfig(unrealized_breadcrumb_coeff=0.3)
    assert unrealized_breadcrumb(0.05, has_position=True, cfg=cfg) == pytest.approx(0.015)
    assert unrealized_breadcrumb(-0.05, has_position=True, cfg=cfg) == pytest.approx(-0.015)
