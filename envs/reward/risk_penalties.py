"""Risk-side penalties that complement the carrot.

Lessons encoded:
  - Quadratic drawdown penalty above threshold (moleapp lesson 1.5 -- linear
    saturates at the -5 floor by 5% DD)
  - Losing-streak entry gate (Frame 2 -- stops wrong-way doubling-down
    during regime mismatch)
  - Funding cumulative penalty (V9 Builder lost 11.5% in +200% market by
    missing this)
  - Per-entry cost (caps frequency; calibrated to ~50% of expected
    per-trade reward at 50% win rate)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskPenaltyConfig:
    dd_quadratic_coeff: float = 50.0
    dd_threshold: float = 0.03
    funding_coeff: float = 0.01
    losing_streak_coeff: float = 0.05
    losing_streak_offset: int = 2
    per_entry_cost: float = 0.02
    # Churn discourages cancel-then-replace spam (live rate-limit risk).
    churn_penalty: float = 0.005
    # Peak-to-trough drawdown penalty on the OPEN position. Forces tight
    # intrinsic stops instead of "hold and hope".
    peak_dd_coeff: float = 0.5
    peak_dd_threshold: float = 0.005  # 50 bps from peak (leveraged)
    reward_floor: float = -5.0


def quadratic_drawdown_penalty(current_drawdown_pct: float, cfg: RiskPenaltyConfig) -> float:
    """50.0 * (dd - 0.03)^2 above threshold. Zero below. Subtractive."""
    if current_drawdown_pct <= cfg.dd_threshold:
        return 0.0
    excess = current_drawdown_pct - cfg.dd_threshold
    return -cfg.dd_quadratic_coeff * (excess * excess)


def losing_streak_entry_penalty(consecutive_losses: int, opening_trade_this_step: bool, cfg: RiskPenaltyConfig) -> float:
    """Penalise opening a trade after a losing streak."""
    if not opening_trade_this_step:
        return 0.0
    over = consecutive_losses - cfg.losing_streak_offset
    if over <= 0:
        return 0.0
    return -cfg.losing_streak_coeff * float(over)


def funding_cumulative_penalty(cumulative_funding_pct: float, cfg: RiskPenaltyConfig) -> float:
    """Subtract a small fraction of the absolute cumulative funding accrued
    on the current position. Drives the agent to close before funding bleeds.
    """
    return -cfg.funding_coeff * abs(cumulative_funding_pct)


def per_entry_cost(opening_trade_this_step: bool, cfg: RiskPenaltyConfig) -> float:
    if not opening_trade_this_step:
        return 0.0
    return -cfg.per_entry_cost


def churn_penalty(n_cancellations_this_step: int, cfg: RiskPenaltyConfig) -> float:
    """Subtract a flat penalty for each in-flight order cancelled this step.

    A cancel happens when the agent issues a new action that supersedes a
    previously-queued (still pending) action, or explicitly cancels a
    resting limit.
    """
    if n_cancellations_this_step <= 0:
        return 0.0
    return -cfg.churn_penalty * float(n_cancellations_this_step)


def peak_drawdown_penalty(
    current_unrealized_pct: float,
    peak_unrealized_pct: float,
    has_position: bool,
    cfg: RiskPenaltyConfig,
) -> float:
    """Penalise the agent when the OPEN position has given back more than
    peak_dd_threshold of its prior peak unrealised PnL.

    excess = max(0, peak - current - threshold). Penalty = -coeff * excess.
    Returns 0 when flat or when current > peak - threshold.
    """
    if not has_position:
        return 0.0
    excess = peak_unrealized_pct - current_unrealized_pct - cfg.peak_dd_threshold
    if excess <= 0:
        return 0.0
    return -cfg.peak_dd_coeff * float(excess)


def apply_floor(reward: float, cfg: RiskPenaltyConfig) -> float:
    return max(reward, cfg.reward_floor)
