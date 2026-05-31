# MomoDkr — HFT Scalping AI Agent for Hyperliquid Vaults

A deep RL scalping agent trained on 2 years of Binance Vision L2 order book data, deployed against Hyperliquid via a Rust ONNX inference engine that signs EIP-712 transactions through an API agent wallet.

See [docs/LESSONS_LEARNED_FROM_MOLEAPP.md](docs/LESSONS_LEARNED_FROM_MOLEAPP.md) for the design rules carried forward from the predecessor project.

## Status
Phase 0 — repo bootstrap.

## Stack
- **RL**: Stable-Baselines3 PPO, CNN+LSTM policy, Discrete(5) action space
- **Data**: Binance Vision L2 depth + aggTrade @ 100ms snapshots, top-10 levels
- **Universe (v1)**: BTC, ETH, SOL
- **Leverage**: 6× (trained at ceiling)
- **Execution**: Rust + `ort` + `alloy`/`ethers-rs` for EIP-712
- **Secrets**: Akeyless (vaultless DFC)
- **Storage**: Cloudflare R2

## Quickstart
```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
