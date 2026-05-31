"""Dense unrealized-PnL breadcrumb while a position is open.

moleapp lesson: sparse reward (only on close) makes credit assignment
hard. A small multiplier on unrealized PnL per step keeps the gradient
informative without dominating the realized-PnL signal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BreadcrumbConfig:
    unrealized_breadcrumb_coeff: float = 0.3


def unrealized_breadcrumb(unrealized_pnl_pct: float, has_position: bool, cfg: BreadcrumbConfig) -> float:
    if not has_position:
        return 0.0
    return cfg.unrealized_breadcrumb_coeff * unrealized_pnl_pct
