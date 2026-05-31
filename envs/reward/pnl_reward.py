"""Realized-PnL carrot asymmetry on position close.

Carries the moleapp lesson: wins multiplied harder than losses. Without
positive asymmetry the agent turtles (Shield V8). MomoDkr defaults to
wins x4 / losses x1.8 per configs/env/momodkr_v1.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PnLRewardConfig:
    win_multiplier: float = 4.0
    loss_multiplier: float = 1.8


def realized_pnl_reward(net_pnl_pct: float, cfg: PnLRewardConfig) -> float:
    """net_pnl_pct is AFTER round-trip fees and slippage. Positive = win."""
    if net_pnl_pct >= 0.0:
        return net_pnl_pct * cfg.win_multiplier
    return net_pnl_pct * cfg.loss_multiplier
