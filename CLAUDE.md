# MomoDkr Operating Manual

This is the Claude operating manual for the MomoDkr AI agent project — an HFT scalping agent for Hyperliquid Vaults trained on Binance Vision L2 order book data.

## Project shape
- **Greenfield repo**, sibling to `moleapp-rl-training` (which is the lessons-learned source — never modify it from this repo)
- **Single-agent RL** (no MARL — proven dead end in moleapp; use hard-coded Governor for risk)
- **PPO** via Stable-Baselines3 (not RLlib — cleaner ONNX export path)
- **Discrete(5) action space**: `{hold, mkt_buy, mkt_sell, post_bid, post_ask}` — sizing handled by Governor (Kelly-capped), not learned
- **MLP policy** (no CNN, no LSTM) — hand-engineered microstructure features into flat vector; cleaner ONNX export, faster inference, simpler debugging
- **Universe v1**: BTC, ETH, SOL only (no HYPE tokens until v2)
- **Leverage**: 6× — trained at ceiling per moleapp rule
- **Execution**: Rust ONNX engine signs EIP-712 → Hyperliquid L1 via API agent wallet (trade-only, no withdraw)
- **Secrets**: Akeyless (never plaintext `.env`, never AWS SM / 1Password)

## Non-negotiable rules (carry-forward from moleapp lessons)

### Reward design
- **Carrot asymmetry**: wins × 3–5×, losses × 1.5–2× (never 1:1 — agent turtles)
- **NET PnL only** (after round-trip fees + slippage) — gross PnL teaches fake wins
- **No flat per-trade bonuses** — agent farms the constant
- **Per-entry cost** (~−0.02) calibrated to ~50% of expected per-trade reward
- **Quadratic DD penalty** above 3% threshold — linear saturates at −5 kill floor
- **Funding cost penalty** `−0.01 × cumulative_funding` — V9 Builder lost 11.5% in +200% market by missing this
- **Losing-streak entry gate** `−0.05 × max(0, streak − 2)` — stops wrong-way doubling-down
- **Dense breadcrumb** `+0.3 × unrealized_pnl_pct` per holding step

### Training discipline
- **Deploy from `best_checkpoint/`** (tracks max eval_pnl), never `final_checkpoint/`
- **σ(action) > 2.0** for 2 consecutive evals → KILL training, revert to best_checkpoint (V9 iter-725 death signal)
- **Cold-start** for HFT; warm-start risks inheriting divergence
- **Train at deployment leverage ceiling** (6× → train 6×, never lower)
- **Aggressive entropy decay**: 0.005 → 0.0005
- **Strict chronological 80/20 train/eval split** — never shuffle time-series
- **Per-episode JSON + MP4 recording** mandatory — W&B metrics flatten failure modes
- **Mann-Whitney U** for run comparison (not eyeballed charts)

### Architecture & safety
- **Feature version embedded in obs** prevents silent train/inference skew
- **ONNX parity gate**: `max_diff < 1e-4` on ≥1000 recorded eval obs — no exception, no deploy without
- **Domain randomization**: fees ±5%, sqrt impact c ±20%, latency 5–20bps (Kyle 1985)
- **Hard Governor rules** in Rust before live: consecutive-loss kill, macro-crash blocker, position cap, leverage clamp, funding-regime gate
- **API agent wallet**: `trade`/`cancel` only — never `withdraw`/`transfer`; daily permission audit

## Directory conventions
- `data/` — collectors + reconstructors + preprocessors + validators (Phase 1–2)
- `envs/` — Gymnasium env, market simulator, reward modules (Phase 3)
- `training/` — SB3 PPO entrypoint, callbacks, curriculum (Phase 4–5)
- `scripts/` — ONNX export, parity validator, R2 sync, calibration (Phase 6)
- `execution/rust_engine/` — Cargo workspace for live inference (Phase 7)
- `live/` — Hyperliquid API agent + vault admin (Phase 8)
- `tests/` — unit + integration + overfit smoke (gates every phase)

## Versioning
Follow moleapp convention: `vN-{role}-{variant}`
- `v1-engine-cold` (Phase 4 baseline)
- `v1-engine-btceth` (Phase 5 curriculum step 2)
- `v1-engine-fundingfix` (after adding funding penalty refinement)

Bump N on architecture changes (new env class, new reward shape). Bump variant on config tuning.

## Phase 0 gate
`pytest -q` green, `ruff check .` clean, repo committed.

## How to work in this repo
- Prefer editing existing files to creating new ones (per global Claude rules)
- Don't add comments unless the WHY is non-obvious
- Don't add backwards-compat shims, error handling for impossible cases, or premature abstractions
- Always reference moleapp lessons by file path when adapting patterns (e.g., "adapted from moleapp `scripts/export_onnx.py`")
- Never modify files under `/Users/papealioune/Documents/Dapps4frica/moleapp-rl-training/` from this repo

## Plan reference
The full implementation blueprint lives at `/Users/papealioune/.claude/plans/momodkr-ai-agent-from-abundant-teacup.md`.
