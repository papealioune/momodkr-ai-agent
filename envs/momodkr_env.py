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
    ActionQueue,
    PendingLimit,
    Position,
    PositionSide,
)
from envs.market_simulator import (
    SimulatorConfig,
    draw_latency_ticks,
    is_liquidated,
    limit_buy_fill,
    limit_sell_fill,
    market_buy_fill_price,
    market_sell_fill_price,
    per_tick_funding_pct,
)
from envs.reward.breadcrumb import BreadcrumbConfig, unrealized_breadcrumb
from envs.reward.pnl_reward import PnLRewardConfig, realized_pnl_reward
from envs.reward.risk_penalties import RiskPenaltyConfig
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
    # Symbol resolved per active episode for tick-size lookup. The env tries to
    # infer it from the parquet filename (.../<SYMBOL>/...); override here for
    # synthetic test data.
    override_symbol: str | None = None


class MomoDkrEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_parquet: str | Path | list[str | Path],
        config: EnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        if isinstance(episode_parquet, (str, Path)):
            self.parquet_paths: list[Path] = [Path(episode_parquet)]
        else:
            self.parquet_paths = [Path(p) for p in episode_parquet]
            if not self.parquet_paths:
                raise ValueError("episode_parquet list is empty")
        self._load_all_episodes()

        self.action_space = gym.spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._active_idx: int = 0
        self._cursor: int = 0
        self._start_cursor: int = 0
        self.position = Position()
        self.pending = PendingLimit()
        self.action_queue = ActionQueue()
        self.account = AccountState(
            initial_nav_usd=self.config.initial_nav_usd,
            nav_usd=self.config.initial_nav_usd,
            peak_nav_usd=self.config.initial_nav_usd,
        )

    # ------------------------------------------------------------------ data

    def _load_all_episodes(self) -> None:
        self._features_pool: list[np.ndarray] = []
        self._sim_state_pool: list[dict[str, np.ndarray]] = []
        self._ts_ms_pool: list[np.ndarray] = []
        self._n_rows_pool: list[int] = []
        for path in self.parquet_paths:
            if not path.exists():
                raise FileNotFoundError(f"episode parquet not found: {path}")
            df = pd.read_parquet(path)
            missing_feats = [c for c in MARKET_FEATURE_NAMES if c not in df.columns]
            if missing_feats:
                raise KeyError(f"episode parquet missing market features ({path}): {missing_feats}")
            missing_sim = [c for c in SIM_STATE_COLS if c not in df.columns]
            if missing_sim:
                raise KeyError(f"episode parquet missing simulator state columns ({path}): {missing_sim}")
            features = df[list(MARKET_FEATURE_NAMES)].to_numpy(dtype=np.float32)
            sim_state = {c: df[c].to_numpy(dtype=np.float64) for c in SIM_STATE_COLS}
            ts_ms = df["ts_ms"].to_numpy(dtype=np.int64)
            if len(df) <= self.config.episode_length_ticks + 1:
                raise ValueError(
                    f"episode parquet has {len(df)} rows, need > {self.config.episode_length_ticks + 1} ({path})"
                )
            self._features_pool.append(features)
            self._sim_state_pool.append(sim_state)
            self._ts_ms_pool.append(ts_ms)
            self._n_rows_pool.append(len(df))

    # Convenience properties that always reflect the active episode.
    @property
    def _features(self) -> np.ndarray:
        return self._features_pool[self._active_idx]

    @property
    def _sim_state(self) -> dict[str, np.ndarray]:
        return self._sim_state_pool[self._active_idx]

    @property
    def _ts_ms(self) -> np.ndarray:
        return self._ts_ms_pool[self._active_idx]

    @property
    def _n_rows(self) -> int:
        return self._n_rows_pool[self._active_idx]

    # ------------------------------------------------------------------ gym

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._active_idx = int(self._rng.integers(0, len(self.parquet_paths)))
        max_start = self._n_rows - self.config.episode_length_ticks - 1
        self._start_cursor = int(self._rng.integers(0, max_start))
        self._cursor = self._start_cursor
        self.position.reset()
        self.pending.cancel()
        self.action_queue.reset()
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
        liquidated = False
        n_cancellations_this_step = 0
        churned_limit = False

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

        # 1. enqueue the agent's action (HOLD doesn't enqueue)
        if act != Action.HOLD:
            # Cancel any in-flight prior orders that the agent is implicitly superseding
            # (only when this action is a market order; queued limits are typically left
            # alone unless the agent explicitly cancels with POST_BID/POST_ASK while flat).
            if act in (Action.MKT_BUY, Action.MKT_SELL) and self.action_queue.n_in_flight > 0:
                n_cancellations_this_step += self.action_queue.cancel_all()
            latency = draw_latency_ticks(self._rng, self.config.sim)
            self.action_queue.push(int(act), execute_at_tick=self._cursor + latency, issued_at_tick=self._cursor)

        # 2. resolve any resting limit against next tick's book (trade-through aware)
        if self.pending.is_open:
            filled, opened_via_limit, realized_via_limit, closed_via_limit = self._resolve_pending_limit(
                prev_mid=self.pending.prev_mid, next_mid=next_mid, next_bid=next_bid, next_ask=next_ask,
                abs_volume=float(self._sim_state["abs_volume_100ms"][next_idx]),
            )
            if filled:
                self.pending.cancel()
                opened_this_step = opened_this_step or opened_via_limit
                realized_net_pnl_pct += realized_via_limit
                position_closed = position_closed or closed_via_limit
            else:
                self.pending.age_ticks += 1
                if self.pending.age_ticks >= self.config.sim.limit_max_age_ticks:
                    self.pending.cancel()
                    churned_limit = True

        # 3. drain any actions whose latency has elapsed
        ready = self.action_queue.pop_ready(self._cursor)
        for delayed in ready:
            ready_act = Action(delayed.action)
            if ready_act in (Action.MKT_BUY, Action.MKT_SELL):
                opened_via_mkt, realized_via_mkt, closed_via_mkt = self._apply_market_action(
                    ready_act,
                    next_bid=next_bid,
                    next_ask=next_ask,
                    next_mid=next_mid,
                    recent_volume_usd=recent_volume_usd,
                    next_idx=next_idx,
                )
                opened_this_step = opened_this_step or opened_via_mkt
                realized_net_pnl_pct += realized_via_mkt
                position_closed = position_closed or closed_via_mkt
            elif ready_act in (Action.POST_BID, Action.POST_ASK) and not self.pending.is_open:
                self._post_limit(
                    ready_act,
                    bid_px=float(self._sim_state["bid_px"][next_idx]),
                    ask_px=float(self._sim_state["ask_px"][next_idx]),
                    mid=next_mid,
                )

        # 4. accrue funding for an open position
        if self.position.is_open:
            funding_per_tick = per_tick_funding_pct(funding_rate, self.config.sim)
            self.position.cumulative_funding_pct += funding_per_tick * float(self.position.side)
            self.position.hold_ticks += 1

        # 5. mark-to-market for unrealized PnL and liquidation
        unreal_pct = self._unrealized_pnl_pct(next_mid)
        if self.position.is_open:
            self.position.peak_unrealized_pct = max(self.position.peak_unrealized_pct, unreal_pct)
            unreal_on_notional = unreal_pct / max(self.config.sim.leverage, 1)
            if is_liquidated(unreal_on_notional, self.config.sim):
                realized_net_pnl_pct += -1.0
                position_closed = True
                liquidated = True
                self.position.reset()

        # 6. NAV accounting + bookkeeping
        if position_closed and realized_net_pnl_pct != 0.0:
            self.account.nav_usd *= (1.0 + realized_net_pnl_pct * self.config.max_position_notional_pct)
            self.account.peak_nav_usd = max(self.account.peak_nav_usd, self.account.nav_usd)
            self.account.realized_pnl_pct_cumulative += realized_net_pnl_pct
            if realized_net_pnl_pct < 0:
                self.account.consecutive_losses += 1
            else:
                self.account.consecutive_losses = 0

        # 7. base reward = realized carrot + breadcrumb. The risk-shaping wrappers
        # (envs.reward.wrappers.RewardShapingWrapper) handle DD / churn / peak-DD /
        # funding / per-entry / losing-streak / floor on top of this. Keeping the
        # core env focused on state transitions + matching.
        reward = 0.0
        if position_closed and realized_net_pnl_pct != 0.0:
            reward += realized_pnl_reward(realized_net_pnl_pct, self.config.pnl)
        reward += unrealized_breadcrumb(unreal_pct, self.position.is_open, self.config.breadcrumb)

        churn_cancellations = n_cancellations_this_step + (1 if churned_limit else 0)

        # 8. termination logic
        self._cursor = next_idx
        elapsed = self._cursor - self._start_cursor
        terminated = liquidated or self.account.drawdown_pct >= self.config.reset_on_dd
        truncated = elapsed >= self.config.episode_length_ticks
        extra: dict[str, Any] = {
            "opened_this_step": opened_this_step,
            "position_closed": position_closed,
            "liquidated": liquidated,
            "realized_net_pnl_pct": realized_net_pnl_pct,
            "unrealized_pct": unreal_pct,
            "peak_unrealized_pct": float(self.position.peak_unrealized_pct) if self.position.is_open else 0.0,
            "has_position": self.position.is_open,
            "n_cancellations_this_step": churn_cancellations,
            "cumulative_funding_pct": float(self.position.cumulative_funding_pct) if self.position.is_open else 0.0,
            "consecutive_losses": int(self.account.consecutive_losses),
            "drawdown_pct": self.account.drawdown_pct,
        }
        if terminated and liquidated:
            extra["reason"] = "liquidation"
        elif terminated:
            extra["reason"] = "drawdown_kill"
        elif truncated:
            extra["reason"] = "episode_end"

        return self._build_obs(), float(reward), terminated, truncated, self._info(extra=extra)

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

    def _ask_depth_tuple(self, idx: int) -> tuple[float, float, float, float]:
        return (
            float(self._sim_state.get("ask_depth_pct_pos_0_1", np.zeros(1))[idx] if "ask_depth_pct_pos_0_1" in self._sim_state else 0.0),
            float(self._sim_state.get("ask_depth_pct_pos_0_2", np.zeros(1))[idx] if "ask_depth_pct_pos_0_2" in self._sim_state else 0.0),
            float(self._sim_state.get("ask_depth_pct_pos_0_5", np.zeros(1))[idx] if "ask_depth_pct_pos_0_5" in self._sim_state else 0.0),
            float(self._sim_state.get("ask_depth_pct_pos_1_0", np.zeros(1))[idx] if "ask_depth_pct_pos_1_0" in self._sim_state else 0.0),
        )

    def _bid_depth_tuple(self, idx: int) -> tuple[float, float, float, float]:
        return (
            float(self._sim_state.get("bid_depth_pct_neg_0_1", np.zeros(1))[idx] if "bid_depth_pct_neg_0_1" in self._sim_state else 0.0),
            float(self._sim_state.get("bid_depth_pct_neg_0_2", np.zeros(1))[idx] if "bid_depth_pct_neg_0_2" in self._sim_state else 0.0),
            float(self._sim_state.get("bid_depth_pct_neg_0_5", np.zeros(1))[idx] if "bid_depth_pct_neg_0_5" in self._sim_state else 0.0),
            float(self._sim_state.get("bid_depth_pct_neg_1_0", np.zeros(1))[idx] if "bid_depth_pct_neg_1_0" in self._sim_state else 0.0),
        )

    def _resolve_symbol(self) -> str | None:
        if self.config.override_symbol:
            return self.config.override_symbol
        parts = self.parquet_paths[self._active_idx].parts
        for p in reversed(parts):
            if p in self.config.sim.tick_size_by_symbol:
                return p
        return None

    def _resolve_pending_limit(
        self,
        prev_mid: float,
        next_mid: float,
        next_bid: float,
        next_ask: float,
        abs_volume: float,
    ) -> tuple[bool, bool, float, bool]:
        opened = False
        realized = 0.0
        closed = False
        if self.pending.side == PositionSide.LONG:
            filled, fill_px, fee_pct = limit_buy_fill(
                prev_mid, next_mid, self.pending.post_price, self.config.sim,
                next_ask=next_ask, abs_volume_at_window=abs_volume,
            )
            if not filled:
                return False, False, 0.0, False
            if self.position.side == PositionSide.FLAT:
                self._open_position(PositionSide.LONG, fill_px, fee_pct, slippage_pct=0.0, notional_usd=self.pending.intended_notional_usd)
                opened = True
            elif self.position.side == PositionSide.SHORT:
                realized = self._close_position(fill_px, fee_pct, slippage_pct=0.0)
                closed = True
        elif self.pending.side == PositionSide.SHORT:
            filled, fill_px, fee_pct = limit_sell_fill(
                prev_mid, next_mid, self.pending.post_price, self.config.sim,
                next_bid=next_bid, abs_volume_at_window=abs_volume,
            )
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
        next_idx: int,
    ) -> tuple[bool, float, bool]:
        opened = False
        realized = 0.0
        closed = False
        notional = self.account.nav_usd * self.config.max_position_notional_pct * self.config.sim.leverage
        symbol = self._resolve_symbol()
        ask_depth = self._ask_depth_tuple(next_idx)
        bid_depth = self._bid_depth_tuple(next_idx)
        if act == Action.MKT_BUY:
            if self.position.side == PositionSide.FLAT:
                fill, fee, slip = market_buy_fill_price(
                    next_ask, next_mid, recent_volume_usd, notional, self.config.sim, self._rng,
                    ask_depth_cumulative=ask_depth, symbol=symbol,
                )
                self._open_position(PositionSide.LONG, fill, fee, slip, notional)
                opened = True
            elif self.position.side == PositionSide.SHORT:
                fill, fee, slip = market_buy_fill_price(
                    next_ask, next_mid, recent_volume_usd, notional, self.config.sim, self._rng,
                    ask_depth_cumulative=ask_depth, symbol=symbol,
                )
                realized = self._close_position(fill, fee, slip)
                closed = True
        elif act == Action.MKT_SELL:
            if self.position.side == PositionSide.FLAT:
                fill, fee, slip = market_sell_fill_price(
                    next_bid, next_mid, recent_volume_usd, notional, self.config.sim, self._rng,
                    bid_depth_cumulative=bid_depth, symbol=symbol,
                )
                self._open_position(PositionSide.SHORT, fill, fee, slip, notional)
                opened = True
            elif self.position.side == PositionSide.LONG:
                fill, fee, slip = market_sell_fill_price(
                    next_bid, next_mid, recent_volume_usd, notional, self.config.sim, self._rng,
                    bid_depth_cumulative=bid_depth, symbol=symbol,
                )
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
            "active_parquet": str(self.parquet_paths[self._active_idx]),
            "cursor": self._cursor,
            "nav_usd": self.account.nav_usd,
            "drawdown_pct": self.account.drawdown_pct,
            "consecutive_losses": self.account.consecutive_losses,
            "position_side": int(self.position.side),
        }
        if extra:
            info.update(extra)
        return info
