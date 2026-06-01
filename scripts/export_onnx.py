"""Export an SB3 PPO checkpoint (Discrete actions, MLP policy) to ONNX.

The Rust live inference engine consumes the resulting .onnx graph directly
via the `ort` crate, with no Python in the hot path. To make the live
behaviour reproducible we export *both* the action logits and the
greedy-argmax action: in production we pick argmax; for offline parity
validation we compare logits float-for-float.

The exported model takes a single batched observation tensor of shape
[batch, OBS_DIM] (float32) and emits:
    logits     [batch, n_actions]   raw policy head output
    action     [batch]              argmax(logits) as int64

This matches moleapp's standalone-wrapper pattern (see
moleapp/scripts/export_onnx.py) adapted for Discrete actions.

We also write a sidecar JSON with:
    feature_version, feature_spec_checksum, action labels, obs dim,
    n_actions, source checkpoint path. The Rust engine asserts a match
    against its baked-in expected values at startup.
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
    bake_normalisation: bool = True,
) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    normalise: NormalizeMarketBlock | None = None
    stats: NormStats | None = None
    if bake_normalisation and norm_stats_path is not None:
        stats = load_norm_stats(norm_stats_path)
        normalise = NormalizeMarketBlock(
            mean=torch.from_numpy(stats.mean_array),
            std=torch.from_numpy(stats.std_array),
            clip=stats.clip,
            market_dim=MARKET_FEATURE_DIM,
        )
    elif bake_normalisation:
        logger.warning(
            "export(bake_normalisation=True) but no norm_stats_path supplied; "
            "the ONNX graph will NOT include normalisation -- live engine must apply it externally."
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
        "norm_clip": stats.clip if stats else None,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("exported %s -> %s (+ manifest %s)", checkpoint_path, output_path, manifest_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Export SB3 PPO Discrete policy to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--norm-stats",
        default=None,
        help="path to norm_stats.json; baked into the ONNX graph so the Rust live engine sends RAW features",
    )
    p.add_argument("--no-bake-normalisation", action="store_true")
    args = p.parse_args()
    export(
        Path(args.checkpoint),
        Path(args.output),
        opset=args.opset,
        norm_stats_path=Path(args.norm_stats) if args.norm_stats else None,
        bake_normalisation=not args.no_bake_normalisation,
    )


if __name__ == "__main__":
    main()
