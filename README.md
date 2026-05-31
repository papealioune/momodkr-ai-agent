# MomoDkr — HFT Scalping AI Agent for Hyperliquid Vaults

A deep RL scalping agent trained on 2 years of Binance Vision L2 order book data, deployed against Hyperliquid via a Rust ONNX inference engine that signs EIP-712 transactions through an API agent wallet.

See [docs/LESSONS_LEARNED_FROM_MOLEAPP.md](docs/LESSONS_LEARNED_FROM_MOLEAPP.md) for the design rules carried forward from the predecessor project.

## Status
Phase 0 — repo bootstrap.

## Stack
- **RL**: Stable-Baselines3 PPO, MLP policy (256/256/128 tanh), Discrete(5) action space
- **Data**: Binance Vision bookTicker + aggTrades + bookDepth @ 100ms aggregated snapshots → hand-engineered microstructure feature vector (~30 dims)
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

## Data ingest on RunPod
The 2-year L2 dataset (~600 GB raw + parsed + snapshots) is built on a CPU pod, not locally.

1. Launch a RunPod CPU pod (8-16 vCPU, ~1 TB persistent volume mounted at `/workspace`).
2. Clone the repo and `cd` into it.
3. Set the R2 credentials in the pod's env vars (see [`.env.example`](.env.example)):
   - `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
4. Bootstrap:
   ```bash
   bash runpod/setup.sh
   ```
5. Run the 7-day test (validates the Phase 1 gate: mid-vs-kline drift ≤ 1bp):
   ```bash
   bash runpod/run_ingest.sh test
   ```
6. If green, run the full 2-year pull:
   ```bash
   nohup bash runpod/run_ingest.sh full > /workspace/momodkr-ingest.log 2>&1 &
   ```
Data uploads incrementally to `s3://moleapp-rl-data/momodkr/{SYMBOL}/{stream}/<YYYY-MM-DD>.parquet`.
