"""Export an SB3 PPO checkpoint (Discrete actions, MLP policy) to ONNX.

Two export modes:

  1. **Production / multi-symbol** (recommended): ONNX is symbol-agnostic
     (no NormalizeMarketBlock baked in). A sidecar `<output>.bundle.json`
     ships per-symbol normalisation stats; the Rust feature_builder loads
     it into HashMap<Symbol, NormStats> and z-scores RAW features before
     each inference call. Adding a new symbol post-launch = compute one
     fresh norm_stats.json and re-bundle; the brain weights never change.

  2. **Single-symbol bake** (dev / pilot only): pass --norm-stats <path>
     to embed one symbol's stats as graph constants. The ONNX then takes
     RAW features and z-scores internally. Simpler artifact, locks the
     ONNX to one symbol.

The Rust engine consumes the resulting .onnx via the `ort` crate, no
Python in the hot path. The graph always emits:
    logits     [batch, n_actions]   raw policy head output
    action     [batch]              argmax(logits) as int64

Sidecar JSONs:
    <output>.json          manifest -- feature_version, checksum, action
                           labels, obs dim, n_actions, normalisation mode
    <output>.bundle.json   (mode 1 only) per-symbol norm_stats
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3 import PPO

from data.preprocessors.feature_stats import NormStats, load_norm_stats
from envs.base_hft_env import ACTION_LABELS
from serving.feature_version import (
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_DIM,
    OBS_DIM,
)
from serving.norm_bundle import NormStatsBundle

logger = logging.getLogger(__name__)


class NormalizeMarketBlock(nn.Module):
    """Z-score the first MARKET_FEATURE_DIM columns of obs; pass position block through.

    Stats are stored as buffers so they're captured by torch.onnx.export
    as graph constants. The Rust live engine sends RAW features; the
    normalisation happens inside the ONNX graph. This guarantees train
    and live inference normalise identically (moleapp lesson 3.4).
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, clip: float, market_dim: int) -> None:
        super().__init__()
        if mean.shape != (market_dim,) or std.shape != (market_dim,):
            raise ValueError(f"mean/std shape {mean.shape}/{std.shape} != ({market_dim},)")
        self.register_buffer("mean", mean.to(torch.float32))
        self.register_buffer("std", std.to(torch.float32))
        self.clip = float(clip)
        self.market_dim = int(market_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        market = obs[..., : self.market_dim]
        rest = obs[..., self.market_dim:]
        z = (market - self.mean) / self.std
        if self.clip > 0:
            z = torch.clamp(z, -self.clip, self.clip)
        return torch.cat([z, rest], dim=-1)


class DiscretePolicyWrapper(nn.Module):
    """Standalone nn.Module wrapping (optional NormalizeMarketBlock) + SB3 PPO's policy net + action head.

    We extract the parts of the SB3 policy that map obs -> action logits
    so the ONNX export contains only those forward modules (no value head,
    no distribution sampling, no SB3-specific helpers).
    """

    def __init__(self, model: PPO, normalise: NormalizeMarketBlock | None = None) -> None:
        super().__init__()
        self.normalise = normalise
        policy = model.policy
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.normalise(obs) if self.normalise is not None else obs
        features = self.features_extractor(x)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        action = torch.argmax(logits, dim=-1)
        return logits, action


def export(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    norm_stats_path: Path | None = None,
    bundle: NormStatsBundle | None = None,
    bake_normalisation: bool | None = None,
) -> Path:
    """Export a PPO checkpoint to ONNX.

    If `bundle` is provided -> production mode: ONNX is symbol-agnostic
    and the bundle is saved next to it as <output>.bundle.json. Live
    engine applies per-symbol normalisation externally.

    If `norm_stats_path` is provided -> single-symbol bake mode: stats
    are embedded in the ONNX graph. Simpler but locks the ONNX to one
    symbol.

    `bake_normalisation` is auto-resolved from the arg shape unless
    explicitly set: bundle -> False, norm_stats_path -> True,
    neither -> False (bare ONNX, warn).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if bundle is not None and norm_stats_path is not None:
        raise ValueError("pass either bundle= (production) or norm_stats_path= (single-symbol bake), not both")

    if bake_normalisation is None:
        bake_normalisation = bundle is None and norm_stats_path is not None

    model = PPO.load(checkpoint_path, device="cpu")

    normalise: NormalizeMarketBlock | None = None
    stats: NormStats | None = None
    if bake_normalisation:
        if norm_stats_path is None:
            raise ValueError("bake_normalisation=True requires norm_stats_path")
        stats = load_norm_stats(norm_stats_path)
        normalise = NormalizeMarketBlock(
            mean=torch.from_numpy(stats.mean_array),
            std=torch.from_numpy(stats.std_array),
            clip=stats.clip,
            market_dim=MARKET_FEATURE_DIM,
        )
    elif bundle is None and norm_stats_path is None:
        logger.warning(
            "exporting bare ONNX with NO normalisation artifacts -- the live engine MUST apply z-score externally; "
            "consider passing bundle= or norm_stats_path= to ship the stats alongside the graph"
        )

    wrapper = DiscretePolicyWrapper(model, normalise=normalise).eval()
    dummy = torch.zeros((1, OBS_DIM), dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["logits", "action"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )

    bundle_path: Path | None = None
    if bundle is not None:
        bundle_path = output_path.with_name(output_path.stem + ".bundle.json")
        bundle.save(bundle_path)

    manifest: dict[str, Any] = {
        "source_checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path),
        "opset": opset,
        "obs_dim": OBS_DIM,
        "n_actions": len(ACTION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "feature_version": FEATURE_VERSION,
        "feature_spec_checksum": FEATURE_SPEC_CHECKSUM,
        "normalisation_baked_in": normalise is not None,
        "norm_stats_path": str(norm_stats_path) if norm_stats_path else None,
        "norm_clip": stats.clip if stats else (bundle.clip if bundle else None),
        "bundle_path": str(bundle_path) if bundle_path else None,
        "bundle_symbols": bundle.symbols if bundle else None,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if bundle_path:
        logger.info("exported %s -> %s (+ manifest %s, + bundle %s)", checkpoint_path, output_path, manifest_path, bundle_path)
    else:
        logger.info("exported %s -> %s (+ manifest %s)", checkpoint_path, output_path, manifest_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Export SB3 PPO Discrete policy to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--opset", type=int, default=17)

    bundle_grp = p.add_argument_group(
        "multi-symbol production export (default)",
        "Auto-discovers per-symbol norm_stats and ships a sidecar bundle next to the ONNX.",
    )
    bundle_grp.add_argument(
        "--episodes-root", default="data/episodes",
        help="root containing <SYMBOL>/<feature_version>/norm_stats.json per symbol",
    )
    bundle_grp.add_argument(
        "--symbols", nargs="+", default=None,
        help="symbols to bundle (e.g. BTCUSDT ETHUSDT SOLUSDT). Required for production export.",
    )

    bake_grp = p.add_argument_group(
        "single-symbol bake (dev / pilot only)",
        "Embeds ONE symbol's stats into the ONNX graph. Mutually exclusive with the bundle args.",
    )
    bake_grp.add_argument(
        "--norm-stats", default=None,
        help="path to a single norm_stats.json to bake into the ONNX graph",
    )

    args = p.parse_args()

    if args.norm_stats and args.symbols:
        raise SystemExit("--norm-stats (bake mode) and --symbols (bundle mode) are mutually exclusive")

    bundle = None
    if args.symbols:
        bundle = NormStatsBundle.from_episode_dirs(Path(args.episodes_root), args.symbols)

    export(
        Path(args.checkpoint),
        Path(args.output),
        opset=args.opset,
        norm_stats_path=Path(args.norm_stats) if args.norm_stats else None,
        bundle=bundle,
    )


if __name__ == "__main__":
    main()
