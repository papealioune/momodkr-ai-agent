"""Per-eval-episode JSON dump: actions, rewards, info, NAV trajectory.

moleapp lesson 3.1: eval recording surfaces failure modes that W&B
summary metrics flatten. Without recordings, post-mortems are impossible.

Each call to _on_event writes a single JSON file under
    <log_dir>/eval_episodes/<run_id>_<eval_idx>.json
The eval env is rolled forward for `n_eval_episodes` episodes using the
model's deterministic policy.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TradeLogCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        log_dir: str | Path,
        n_eval_episodes: int = 4,
        deterministic: bool = True,
        record_obs: bool = False,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.log_dir = Path(log_dir) / "eval_episodes"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.record_obs = bool(record_obs)
        self.run_id = uuid.uuid4().hex[:8]
        self.eval_counter = 0

    def _on_step(self) -> bool:
        # Fires once per eval (we're attached via callback_after_eval).
        episodes: list[dict[str, Any]] = []
        for ep_idx in range(self.n_eval_episodes):
            episodes.append(self._record_episode(ep_idx))
        out_path = self.log_dir / f"{self.run_id}_{self.eval_counter:06d}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "run_id": self.run_id,
                    "eval_idx": self.eval_counter,
                    "num_timesteps": int(self.model.num_timesteps),
                    "saved_at_unix": int(time.time()),
                    "episodes": episodes,
                },
                fh,
                indent=2,
                default=_json_default,
            )
        if self.verbose:
            logger.info("eval trade log written: %s (%d episodes)", out_path, len(episodes))
        self.eval_counter += 1
        return True

    def _record_episode(self, ep_idx: int) -> dict[str, Any]:
        env = self.eval_env
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        actions: list[int] = []
        rewards: list[float] = []
        infos: list[dict[str, Any]] = []
        obs_history: list[list[float]] = []
        for _ in range(50_000):
            if self.record_obs:
                obs_history.append(np.asarray(obs).reshape(-1).tolist())
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(np.any(terminated) or np.any(truncated))
            else:
                obs, reward, done, info = step_out
                done = bool(np.any(done))
            actions.append(int(np.asarray(action).flatten()[0]))
            rewards.append(float(np.asarray(reward).flatten()[0]))
            infos.append(_flatten_info(info))
            if done:
                break
        return {
            "episode_idx": ep_idx,
            "n_steps": len(actions),
            "actions": actions,
            "rewards": rewards,
            "cum_reward": float(np.sum(rewards)),
            "info_first": infos[0] if infos else {},
            "info_last": infos[-1] if infos else {},
            "obs_history": obs_history if self.record_obs else None,
        }


def _flatten_info(info: Any) -> dict[str, Any]:
    if isinstance(info, list) and info:
        info = info[0]
    if isinstance(info, dict):
        return {k: _json_default(v) for k, v in info.items()}
    return {}


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
