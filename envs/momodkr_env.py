"""Gymnasium env tying the 100ms snapshots, market simulator, and reward
modules together. Discrete(5) action space; 30-dim Box observation.

Episode lifecycle:
  reset(seed):
    - pick a random valid start tick in the loaded episode parquet (train or
      eval), reset position state, return initial obs
  step(action):
    - resolve any pending limit order against the next snapshot
    - apply the new action (open / close / no-op based on current side)
    - mark-to-market the position, accrue funding
    - check liquidation + drawdown kill
    - compute reward = carrot(realized) + breadcrumb(unrealized) + penalties
    - advance the cursor and return next obs

Episode termination:
  - liquidation: hard penalty, terminate
  - account drawdown >= reset_on_dd: terminate
  - tick cursor reaches episode_length_ticks: truncate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from envs.base_hft_env import (
    ACTION_LABELS,
    AccountState,
    Action,
    PendingLimit,
    Position,
    PositionSide,
)
from envs.market_simulator import (
    SimulatorConfig,
    is_liquidated,
    limit_buy_fill,
    limit_sell_fill,
    market_buy_fill_price,
    market_sell_fill_price,
    per_tick_funding_pct,
)
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
from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_NAMES,
    OBS_DIM,
    SIM_STATE_COLS,
)


@dataclass
class EnvConfig:
    episode_length_ticks: int = 9_000      # 15min @ 100ms
    reset_on_dd: float = 0.05               # 5% account DD ends episode
    max_position_notional_pct: float = 0.17  # 17% NAV per position (6x x 17% ~= 100% notional)
    initial_nav_usd: float = 10_000.0
    sim: SimulatorConfig = field(default_factory=SimulatorConfig)
    pnl: PnLRewardConfig = field(default_factory=PnLRewardConfig)
    risk: RiskPenaltyConfig = field(default_factory=RiskPenaltyConfig)
    breadcrumb: BreadcrumbConfig = field(default_factory=BreadcrumbConfig)
    liquidation_penalty: float = -5.0


class MomoDkrEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_parquet: str | Path,
        config: EnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.parquet_path = Path(episode_parquet)
        self._load_episode_data(self.parquet_path)

        self.action_space = gym.spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._cursor: int = 0
        self._start_cursor: int = 0
        self.position = Position()
        self.pending = PendingLimit()
        self.account = AccountState(
            initial_nav_usd=self.config.initial_nav_usd,
            nav_usd=self.config.initial_nav_usd,
            peak_nav_usd=self.config.initial_nav_usd,
        )

    # ------------------------------------------------------------------ data

    def _load_episode_data(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"episode parquet not found: {path}")
        df = pd.read_parquet(path)
        missing_feats = [c for c in MARKET_FEATURE_NAMES if c not in df.columns]
        if missing_feats:
            raise KeyError(f"episode parquet missing market features: {missing_feats}")
        missing_sim = [c for c in SIM_STATE_COLS if c not in df.columns]
        if missing_sim:
            raise KeyError(f"episode parquet missing simulator state columns: {missing_sim}")
        self._features: np.ndarray = df[list(MARKET_FEATURE_NAMES)].to_numpy(dtype=np.float32)
        self._sim_state: dict[str, np.ndarray] = {
            c: df[c].to_numpy(dtype=np.float64) for c in SIM_STATE_COLS
        }
        self._ts_ms: np.ndarray = df["ts_ms"].to_numpy(dtype=np.int64)
        self._n_rows: int = len(df)
        if self._n_rows <= self.config.episode_length_ticks + 1:
            raise ValueError(
                f"episode parquet has {self._n_rows} rows, need > {self.config.episode_length_ticks + 1}"
            )

    # ------------------------------------------------------------------ gym

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = self._n_rows - self.config.episode_length_ticks - 1
        self._start_cursor = int(self._rng.integers(0, max_start))
        self._cursor = self._start_cursor
        self.position.reset()
        self.pending.cancel()
        self.account = AccountState(
            initial_nav_usd=self.config.initial_nav_usd,
            nav_usd=self.config.initial_nav_usd,
            peak_nav_usd=self.config.initial_nav_usd,
        )
        return self._build_obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        try:
            act = Action(int(action))
        except ValueError as e:
            raise ValueError(f"invalid action {action}; valid={list(Action)}") from e

        opened_this_step = False
        realized_net_pnl_pct = 0.0
        position_closed = False

        # advance the cursor: the simulator looks at next-tick prices to resolve fills
        if self._cursor + 1 >= self._n_rows:
            return self._build_obs(), 0.0, False, True, self._info(extra={"reason": "out_of_data"})

        prev_idx = self._cursor
        next_idx = self._cursor + 1
        prev_mid = float(self._sim_state["mid"][prev_idx])
        next_mid = float(self._sim_state["mid"][next_idx])
        next_bid = float(self._sim_state["bid_px"][next_idx])
        next_ask = float(self._sim_state["ask_px"][next_idx])
        recent_volume_usd = float(self._sim_state["abs_volume_100ms"][prev_idx]) * prev_mid
        funding_rate = float(self._sim_state["funding_rate"][next_idx])

        # 1. resolve any resting limit
        if self.pending.is_open:
            filled, opened_this_step, realized_net_pnl_pct, position_closed = self._resolve_pending_limit(
                prev_mid=self.pending.prev_mid,
                next_mid=next_mid,
            )
            if filled:
                self.pending.cancel()
            else:
                self.pending.age_ticks += 1
                if self.pending.age_ticks >= self.config.sim.limit_max_age_ticks:
                    self.pending.cancel()

        # 2. apply the new action
        if act in (Action.MKT_BUY, Action.MKT_SELL):
            opened_via_market, realized_via_market, closed_via_market = self._apply_market_action(
                act, next_bid=next_bid, next_ask=next_ask, next_mid=next_mid, recent_volume_usd=recent_volume_usd
            )
            opened_this_step = opened_this_step or opened_via_market
            realized_net_pnl_pct += realized_via_market
            position_closed = position_closed or closed_via_market
        elif act in (Action.POST_BID, Action.POST_ASK) and not self.pending.is_open:
            self._post_limit(act, bid_px=float(self._sim_state["bid_px"][prev_idx]), ask_px=float(self._sim_state["ask_px"][prev_idx]), mid=prev_mid)
        # Action.HOLD: no-op

        # 3. accrue funding for an open position
        if self.position.is_open:
            funding_per_tick = per_tick_funding_pct(funding_rate, self.config.sim)
            self.position.cumulative_funding_pct += funding_per_tick * float(self.position.side)
            self.position.hold_ticks += 1

        # 4. mark-to-market for unrealized PnL and liquidation
        unreal_pct = self._unrealized_pnl_pct(next_mid)
        if self.position.is_open:
            self.position.peak_unrealized_pct = max(self.position.peak_unrealized_pct, unreal_pct)
            # leverage-amplified PnL on notional collapses the position when it crosses -1/L + mm
            unreal_on_notional = unreal_pct / max(self.config.sim.leverage, 1)
            if is_liquidated(unreal_on_notional, self.config.sim):
                realized_net_pnl_pct += -1.0  # lose the full margin
                position_closed = True
                self.position.reset()

        # 5. reward composition
        reward = 0.0
        if position_closed and realized_net_pnl_pct != 0.0:
            reward += realized_pnl_reward(realized_net_pnl_pct, self.config.pnl)
            # realized_net_pnl_pct is already leveraged (return on margin posted), so the
            # NAV update only scales by the fraction of NAV that was posted as margin.
            self.account.nav_usd *= (1.0 + realized_net_pnl_pct * self.config.max_position_notional_pct)
            self.account.peak_nav_usd = max(self.account.peak_nav_usd, self.account.nav_usd)
            self.account.realized_pnl_pct_cumulative += realized_net_pnl_pct
            if realized_net_pnl_pct < 0:
                self.account.consecutive_losses += 1
            else:
                self.account.consecutive_losses = 0
        reward += unrealized_breadcrumb(unreal_pct, self.position.is_open, self.config.breadcrumb)
        reward += per_entry_cost(opened_this_step, self.config.risk)
        reward += losing_streak_entry_penalty(self.account.consecutive_losses, opened_this_step, self.config.risk)
        reward += quadratic_drawdown_penalty(self.account.drawdown_pct, self.config.risk)
        if self.position.is_open:
            reward += funding_cumulative_penalty(self.position.cumulative_funding_pct, self.config.risk)
        reward = apply_floor(reward, self.config.risk)

        # 6. termination logic
        self._cursor = next_idx
        elapsed = self._cursor - self._start_cursor
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        if self.account.drawdown_pct >= self.config.reset_on_dd:
            terminated = True
            info["reason"] = "drawdown_kill"
        elif elapsed >= self.config.episode_length_ticks:
            truncated = True
            info["reason"] = "episode_end"

        return self._build_obs(), float(reward), terminated, truncated, self._info(extra=info)

    # ------------------------------------------------------------------ helpers

    def _post_limit(self, act: Action, bid_px: float, ask_px: float, mid: float) -> None:
        # If we have an open position, treat the limit as a close on the opposite side; flat -> open.
        if act == Action.POST_BID:
            if self.position.side == PositionSide.SHORT or self.position.side == PositionSide.FLAT:
                self.pending.is_open = True
                self.pending.side = PositionSide.LONG
                self.pending.post_price = bid_px
                self.pending.prev_mid = mid
                self.pending.age_ticks = 0
                self.pending.intended_notional_usd = self.account.nav_usd * self.config.max_position_notional_pct * self.config.sim.leverage
        elif act == Action.POST_ASK:
            if self.position.side == PositionSide.LONG or self.position.side == PositionSide.FLAT:
                self.pending.is_open = True
                self.pending.side = PositionSide.SHORT
                self.pending.post_price = ask_px
                self.pending.prev_mid = mid
                self.pending.age_ticks = 0
                self.pending.intended_notional_usd = self.account.nav_usd * self.config.max_position_notional_pct * self.config.sim.leverage

    def _resolve_pending_limit(self, prev_mid: float, next_mid: float) -> tuple[bool, bool, float, bool]:
        opened = False
        realized = 0.0
        closed = False
        if self.pending.side == PositionSide.LONG:
            filled, fill_px, fee_pct = limit_buy_fill(prev_mid, next_mid, self.pending.post_price, self.config.sim)
            if not filled:
                return False, False, 0.0, False
            if self.position.side == PositionSide.FLAT:
                self._open_position(PositionSide.LONG, fill_px, fee_pct, slippage_pct=0.0, notional_usd=self.pending.intended_notional_usd)
                opened = True
            elif self.position.side == PositionSide.SHORT:
                realized = self._close_position(fill_px, fee_pct, slippage_pct=0.0)
                closed = True
        elif self.pending.side == PositionSide.SHORT:
            filled, fill_px, fee_pct = limit_sell_fill(prev_mid, next_mid, self.pending.post_price, self.config.sim)
            if not filled:
                return False, False, 0.0, False
            if self.position.side == PositionSide.FLAT:
                self._open_position(PositionSide.SHORT, fill_px, fee_pct, slippage_pct=0.0, notional_usd=self.pending.intended_notional_usd)
                opened = True
            elif self.position.side == PositionSide.LONG:
                realized = self._close_position(fill_px, fee_pct, slippage_pct=0.0)
                closed = True
        return True, opened, realized, closed

    def _apply_market_action(
        self,
        act: Action,
        next_bid: float,
        next_ask: float,
        next_mid: float,
        recent_volume_usd: float,
    ) -> tuple[bool, float, bool]:
        opened = False
        realized = 0.0
        closed = False
        notional = self.account.nav_usd * self.config.max_position_notional_pct * self.config.sim.leverage
        if act == Action.MKT_BUY:
            if self.position.side == PositionSide.FLAT:
                fill, fee, slip = market_buy_fill_price(next_ask, next_mid, recent_volume_usd, notional, self.config.sim, self._rng)
                self._open_position(PositionSide.LONG, fill, fee, slip, notional)
                opened = True
            elif self.position.side == PositionSide.SHORT:
                fill, fee, slip = market_buy_fill_price(next_ask, next_mid, recent_volume_usd, notional, self.config.sim, self._rng)
                realized = self._close_position(fill, fee, slip)
                closed = True
            # else LONG: ignore (already long)
        elif act == Action.MKT_SELL:
            if self.position.side == PositionSide.FLAT:
                fill, fee, slip = market_sell_fill_price(next_bid, next_mid, recent_volume_usd, notional, self.config.sim, self._rng)
                self._open_position(PositionSide.SHORT, fill, fee, slip, notional)
                opened = True
            elif self.position.side == PositionSide.LONG:
                fill, fee, slip = market_sell_fill_price(next_bid, next_mid, recent_volume_usd, notional, self.config.sim, self._rng)
                realized = self._close_position(fill, fee, slip)
                closed = True
        return opened, realized, closed

    def _open_position(self, side: PositionSide, fill_price: float, fee_pct: float, slippage_pct: float, notional_usd: float) -> None:
        self.position.side = side
        self.position.entry_price = fill_price
        self.position.notional_usd = notional_usd
        self.position.entry_tick = self._cursor
        self.position.hold_ticks = 0
        self.position.peak_unrealized_pct = 0.0
        self.position.cumulative_funding_pct = 0.0
        self.position.entry_fee_pct = fee_pct
        self.position.entry_slippage_pct = slippage_pct

    def _close_position(self, fill_price: float, fee_pct: float, slippage_pct: float) -> float:
        gross_pct = self._gross_pnl_pct(fill_price)
        # round-trip fees + slippage already include the exit leg via fee_pct; entry leg from position state
        net_pct = gross_pct - self.position.entry_fee_pct - fee_pct - self.position.entry_slippage_pct - slippage_pct - self.position.cumulative_funding_pct
        self.position.reset()
        return float(net_pct)

    def _gross_pnl_pct(self, mark_price: float) -> float:
        if not self.position.is_open or self.position.entry_price <= 0:
            return 0.0
        signed_dir = float(self.position.side)
        return signed_dir * (mark_price - self.position.entry_price) / self.position.entry_price * self.config.sim.leverage

    def _unrealized_pnl_pct(self, mark_price: float) -> float:
        if not self.position.is_open:
            return 0.0
        gross = self._gross_pnl_pct(mark_price)
        return gross - self.position.entry_fee_pct - self.position.entry_slippage_pct - self.position.cumulative_funding_pct

    def _build_obs(self) -> np.ndarray:
        idx = min(self._cursor, self._n_rows - 1)
        market = self._features[idx]
        mark = float(self._sim_state["mid"][idx])
        unreal = self._unrealized_pnl_pct(mark)
        pos_features = np.array(
            [
                float(self.position.side) * self.config.max_position_notional_pct if self.position.is_open else 0.0,
                unreal,
                min(self.position.hold_ticks / max(self.config.episode_length_ticks, 1), 1.0),
                self.position.peak_unrealized_pct,
            ],
            dtype=np.float32,
        )
        return np.concatenate([market, pos_features]).astype(np.float32)

    def _info(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "feature_version": FEATURE_VERSION,
            "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
            "cursor": self._cursor,
            "nav_usd": self.account.nav_usd,
            "drawdown_pct": self.account.drawdown_pct,
            "consecutive_losses": self.account.consecutive_losses,
            "position_side": int(self.position.side),
        }
        if extra:
            info.update(extra)
        return info
