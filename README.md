# MomoDkr — HFT Scalping AI Agent for Hyperliquid Vaults

A deep RL scalping agent trained on 2 years of Binance Vision L2 order book data, deployed against Hyperliquid via a Rust ONNX inference engine that signs EIP-712 transactions through an API agent wallet.

See [docs/LESSONS_LEARNED_FROM_MOLEAPP.md](docs/LESSONS_LEARNED_FROM_MOLEAPP.md) for the design rules carried forward from the predecessor project.

## Status
Phases 0–3.6 complete (data pipeline, env + simulator, reward wrappers, PPO training, curriculum, ONNX export + parity gate, train-set normalisation, walk-forward).

## Stack
- **RL**: Stable-Baselines3 PPO, MLP policy (256/256/128 tanh), Discrete(5) action space
- **Data**: Binance Vision bookTicker + aggTrades + bookDepth @ 100ms aggregated snapshots → 26-feature flat vector + 4-feature position state
- **Universe (v1)**: BTC, ETH, SOL
- **Leverage**: 6× (trained at ceiling)
- **Execution**: Rust + `ort` + `alloy`/`ethers-rs` for EIP-712
- **Secrets**: Akeyless (vaultless DFC)
- **Storage**: Cloudflare R2

## Quickstart (local dev)
```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## RunPod runbooks
- **Data prep** (collect L2 → reconstruct snapshots → engineer features → episode parquets + norm_stats): [docs/RUNPOD_DATA_PREP_GUIDE.md](docs/RUNPOD_DATA_PREP_GUIDE.md)
- **Training** (walk-forward dry-run → 20M-step PPO → ONNX export with normalisation baked in → parity gate): [docs/RUNPOD_TRAINING_GUIDE.md](docs/RUNPOD_TRAINING_GUIDE.md)
