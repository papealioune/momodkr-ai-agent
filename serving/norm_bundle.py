"""Multi-symbol normalisation bundle shipped alongside the ONNX brain.

Architecture decision (Option 2 in docs/RUNPOD_TRAINING_GUIDE.md):
  - One ONNX policy, symbol-agnostic, normalisation NOT baked in.
  - One bundle JSON next to the ONNX with per-symbol z-score stats.
  - The Rust feature_builder loads the bundle into HashMap<Symbol, NormStats>
    and applies the right z-score before each ONNX call.

Adding a new symbol post-launch is therefore a no-retrain operation:
compute a fresh norm_stats.json for the new symbol on its recent history,
merge it into the bundle, drop the new bundle into the Rust client. The
brain (ONNX weights) is unchanged.

Bundle JSON schema:

    {
      "feature_version": "0.1.0",
      "feature_spec_checksum": "<16-hex>",
      "clip": 10.0,
      "by_symbol": {
        "BTCUSDT": { "mean": [...], "std": [...], "n_train_rows": int },
        "ETHUSDT": { "mean": [...], "std": [...], "n_train_rows": int },
        ...
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data.preprocessors.feature_stats import (
    DEFAULT_CLIP,
    NormStats,
    apply_zscore,
    load_norm_stats,
    norm_stats_path_for_episodes,
)
from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_DIM,
)


@dataclass
class SymbolStats:
    mean: list[float]
    std: list[float]
    n_train_rows: int = 0


@dataclass
class NormStatsBundle:
    feature_version: str
    feature_spec_checksum: str
    clip: float
    by_symbol: dict[str, SymbolStats] = field(default_factory=dict)

    @classmethod
    def from_per_symbol_stats(
        cls, stats_by_symbol: dict[str, NormStats], clip: float | None = None
    ) -> NormStatsBundle:
        if not stats_by_symbol:
            raise ValueError("from_per_symbol_stats requires at least one symbol")
        # All input stats must share feature_version + checksum (the env enforces
        # this on load, but double-check at bundle time so a stale dir is loud).
        clips: set[float] = set()
        for sym, s in stats_by_symbol.items():
            if s.feature_version != FEATURE_VERSION:
                raise ValueError(
                    f"{sym}: norm_stats.feature_version {s.feature_version!r} != current {FEATURE_VERSION!r}"
                )
            if s.feature_spec_checksum != FEATURE_SPEC_CHECKSUM:
                raise ValueError(
                    f"{sym}: norm_stats.feature_spec_checksum {s.feature_spec_checksum!r} != current "
                    f"{FEATURE_SPEC_CHECKSUM!r}"
                )
            if len(s.mean) != MARKET_FEATURE_DIM:
                raise ValueError(f"{sym}: mean length {len(s.mean)} != {MARKET_FEATURE_DIM}")
            clips.add(float(s.clip))
        if clip is None:
            if len(clips) > 1:
                raise ValueError(
                    f"per-symbol stats disagree on clip: {sorted(clips)}; pass clip= explicitly to override"
                )
            clip = next(iter(clips))
        return cls(
            feature_version=FEATURE_VERSION,
            feature_spec_checksum=FEATURE_SPEC_CHECKSUM,
            clip=float(clip),
            by_symbol={
                sym: SymbolStats(mean=list(s.mean), std=list(s.std), n_train_rows=int(s.n_train_rows))
                for sym, s in stats_by_symbol.items()
            },
        )

    @classmethod
    def from_episode_dirs(
        cls, episodes_root: Path, symbols: list[str], feature_version_dir: str | None = None
    ) -> NormStatsBundle:
        """Discover per-symbol norm_stats by convention.

        Looks at episodes_root/<symbol>/<feature_version_dir>/norm_stats.json
        and bundles every symbol whose stats exist. feature_version_dir
        defaults to the active FEATURE_VERSION (matching episode_builder's
        layout).
        """
        ver_dir = feature_version_dir or FEATURE_VERSION
        stats_by_symbol: dict[str, NormStats] = {}
        for sym in symbols:
            episode_dir = Path(episodes_root) / sym / ver_dir
            stats_path = norm_stats_path_for_episodes(episode_dir)
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"no norm_stats.json for {sym} at {stats_path} (run build_features first)"
                )
            stats_by_symbol[sym] = load_norm_stats(stats_path)
        return cls.from_per_symbol_stats(stats_by_symbol)

    def to_dict(self) -> dict:
        return {
            "feature_version": self.feature_version,
            "feature_spec_checksum": self.feature_spec_checksum,
            "clip": self.clip,
            "by_symbol": {
                sym: {"mean": s.mean, "std": s.std, "n_train_rows": s.n_train_rows}
                for sym, s in self.by_symbol.items()
            },
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> NormStatsBundle:
        raw = json.loads(Path(path).read_text())
        if raw.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"bundle.feature_version {raw.get('feature_version')!r} != current {FEATURE_VERSION!r}"
            )
        if raw.get("feature_spec_checksum") != FEATURE_SPEC_CHECKSUM:
            raise ValueError(
                f"bundle.feature_spec_checksum {raw.get('feature_spec_checksum')!r} != current "
                f"{FEATURE_SPEC_CHECKSUM!r}"
            )
        by_symbol_raw = raw.get("by_symbol", {})
        if not by_symbol_raw:
            raise ValueError("bundle has no per-symbol stats")
        for sym, s in by_symbol_raw.items():
            if len(s.get("mean", [])) != MARKET_FEATURE_DIM:
                raise ValueError(f"bundle[{sym}].mean length != {MARKET_FEATURE_DIM}")
        return cls(
            feature_version=raw["feature_version"],
            feature_spec_checksum=raw["feature_spec_checksum"],
            clip=float(raw.get("clip", DEFAULT_CLIP)),
            by_symbol={
                sym: SymbolStats(
                    mean=list(s["mean"]),
                    std=list(s["std"]),
                    n_train_rows=int(s.get("n_train_rows", 0)),
                )
                for sym, s in by_symbol_raw.items()
            },
        )

    def get_norm_stats(self, symbol: str) -> NormStats:
        if symbol not in self.by_symbol:
            raise KeyError(
                f"symbol {symbol!r} not in bundle (available: {sorted(self.by_symbol)})"
            )
        s = self.by_symbol[symbol]
        return NormStats(
            feature_version=self.feature_version,
            feature_spec_checksum=self.feature_spec_checksum,
            n_train_rows=s.n_train_rows,
            clip=self.clip,
            mean=list(s.mean),
            std=list(s.std),
        )

    def apply(self, symbol: str, market_features: np.ndarray) -> np.ndarray:
        return apply_zscore(market_features, self.get_norm_stats(symbol))

    @property
    def symbols(self) -> list[str]:
        return sorted(self.by_symbol.keys())
