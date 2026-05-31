"""Shared scaffolding for any MomoDkr HFT env.

Currently just holds dataclasses for position + episode state, and the
canonical action label mapping. The concrete env (momodkr_env.MomoDkrEnv)
owns the gym.Env interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    MKT_BUY = 1
    MKT_SELL = 2
    POST_BID = 3
    POST_ASK = 4


ACTION_LABELS: tuple[str, ...] = ("hold", "mkt_buy", "mkt_sell", "post_bid", "post_ask")


class PositionSide(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    side: PositionSide = PositionSide.FLAT
    entry_price: float = 0.0
    notional_usd: float = 0.0
    entry_tick: int = 0
    hold_ticks: int = 0
    peak_unrealized_pct: float = 0.0      # best unrealized PnL pct during this hold
    cumulative_funding_pct: float = 0.0   # funding paid (positive) or received (negative) on this position
    entry_fee_pct: float = 0.0
    entry_slippage_pct: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.FLAT

    def reset(self) -> None:
        self.side = PositionSide.FLAT
        self.entry_price = 0.0
        self.notional_usd = 0.0
        self.entry_tick = 0
        self.hold_ticks = 0
        self.peak_unrealized_pct = 0.0
        self.cumulative_funding_pct = 0.0
        self.entry_fee_pct = 0.0
        self.entry_slippage_pct = 0.0


@dataclass
class PendingLimit:
    """A resting limit order placed at the previous tick's best bid/ask."""

    side: PositionSide = PositionSide.FLAT  # LONG = post_bid, SHORT = post_ask
    post_price: float = 0.0
    prev_mid: float = 0.0
    age_ticks: int = 0
    intended_notional_usd: float = 0.0
    is_open: bool = False

    def cancel(self) -> None:
        self.is_open = False
        self.side = PositionSide.FLAT
        self.post_price = 0.0
        self.prev_mid = 0.0
        self.age_ticks = 0
        self.intended_notional_usd = 0.0


@dataclass
class AccountState:
    initial_nav_usd: float = 10_000.0
    nav_usd: float = 10_000.0
    peak_nav_usd: float = 10_000.0
    realized_pnl_pct_cumulative: float = 0.0
    consecutive_losses: int = 0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_nav_usd <= 0:
            return 0.0
        return max(0.0, (self.peak_nav_usd - self.nav_usd) / self.peak_nav_usd)
