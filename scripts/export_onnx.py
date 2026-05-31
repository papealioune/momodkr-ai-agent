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

from envs.base_hft_env import ACTION_LABELS
from serving.feature_version import FEATURE_SPEC_CHECKSUM, FEATURE_VERSION, OBS_DIM

logger = logging.getLogger(__name__)


class DiscretePolicyWrapper(nn.Module):
    """Standalone nn.Module wrapping SB3 PPO's policy net + action head.

    We extract the parts of the SB3 policy that map obs -> action logits
    so the ONNX export contains only those forward modules (no value head,
    no distribution sampling, no SB3-specific helpers).
    """

    def __init__(self, model: PPO) -> None:
        super().__init__()
        policy = model.policy
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        action = torch.argmax(logits, dim=-1)
        return logits, action


def export(checkpoint_path: Path, output_path: Path, opset: int = 17) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")
    wrapper = DiscretePolicyWrapper(model).eval()
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
    args = p.parse_args()
    export(Path(args.checkpoint), Path(args.output), opset=args.opset)


if __name__ == "__main__":
    main()
