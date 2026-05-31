# Lessons Learned — Adapted from moleapp-rl-training (V1–V12 + Engine/Vault split)

This document distills the operational rules from 11 training iterations and 3+ failure post-mortems in the [moleapp-rl-training](/Users/papealioune/Documents/Dapps4frica/moleapp-rl-training/) project. It is **adapted for MomoDkr's HFT scalping context** — single-agent PPO on L2 order book data for Hyperliquid Vaults.

The original document is at [`moleapp-rl-training/docs/LESSONS_LEARNED.md`](/Users/papealioune/Documents/Dapps4frica/moleapp-rl-training/docs/LESSONS_LEARNED.md). Where MomoDkr diverges (e.g., Discrete vs Box actions, L2 vs OHLCV, Akeyless vs AWS), the rule has been re-cast for this project's stack.

Every new training run on MomoDkr must reference this document before changing reward, env, or training config.

---

## 1. Reward design

### 1.1 Flat per-trade constants are farmable
**moleapp evidence:** V1–V3 used `reward += close_pnl × 5.0 + 0.15`. Agent converged to 500+ trades/episode churning for the `+0.15` constant.

**MomoDkr rule:** No flat constants in per-trade reward. All components must scale with PnL magnitude. For Discrete(5) actions, this applies equally — bonuses on `mkt_buy` or `post_bid` actions farmable just as easily.

### 1.2 Use NET PnL, never gross
**moleapp evidence:** V3 gross `pnl_pct` reward let agent earn positive reward on 0.2% gross wins that were −0.3% NET after fees.

**MomoDkr rule:** Reward calculation in [`envs/reward/pnl_reward.py`](../envs/reward/pnl_reward.py) must subtract round-trip fees + slippage before applying multipliers. With Hyperliquid taker fees + sqrt market impact, gross-vs-net divergence is even larger than on moleapp's bar simulator.

### 1.3 Gated bonuses are only as good as the gate
**moleapp evidence:** V4's `bonus if close_pnl > 0.01` gate fired on virtually every win (min TP was 0.02 leveraged = ~1.5% net). De facto flat.

**MomoDkr rule:** Gate thresholds must be well above what the action space can trivially produce. For HFT scalping with 0.02–0.05% per-trade targets, gate any bonus at ≥2× the min TP.

### 1.4 Per-entry cost caps frequency
**moleapp evidence:** V4's `reward -= 0.02 on entry` stopped churn-for-constants. Calibrated to ~50% of expected per-trade reward at 50% WR.

**MomoDkr rule:** Per-entry cost in [`envs/reward/risk_penalties.py`](../envs/reward/risk_penalties.py) starts at −0.02; calibrate against actual per-trade reward distribution after Phase 4 baseline. Each of `mkt_buy`, `mkt_sell`, `post_bid`, `post_ask` triggers it; `hold` does not.

### 1.5 DD penalty saturates at the −5 floor at ~5% drawdown
**moleapp evidence:** V7's `if dd > 0.10: reward -= dd_delta × 3.0` saturates at the −5 floor by 5% DD, making V7 vs V7+V11 reward indistinguishable in the high-DD zone.

**MomoDkr rule:** Use **quadratic** DD penalty above 3% threshold: `raw -= 50.0 × (dd - 0.03)²`. At 4% DD → −0.5; at 5% DD → −5.0 (floor). Preserves behavioral differentiation up to the floor.

### 1.6 Funding penalty is non-optional
**moleapp evidence:** V9 Builder lost 11.5% in a +200% market because Builder's reward (unlike Shield's) omitted the funding cost penalty. Wrong-way position bled funding for hours.

**MomoDkr rule:** Funding penalty `−0.01 × cumulative_funding` is mandatory in [`envs/reward/risk_penalties.py`](../envs/reward/risk_penalties.py). Hyperliquid funding accrues continuously (not just 8h boundaries) — apply per step proportionally.

### 1.7 Macro-trend awareness needs explicit signal
**moleapp evidence:** Macro features in obs ≠ macro behavior. Agent still fights trend even with `btc_dominance`, `fear_greed`, `market_regime` in obs[37..40].

**MomoDkr rule:** For v1, rely on Governor (Rust hard rules) to block trades against macro-crash regime (BTC −3%/60s). v2 may add explicit trend-fight penalty in reward if eval shows trend-fight losses.

### 1.8 Asymmetric carrot is critical (new — confirmed via moleapp Builder vs Shield)
**moleapp evidence:** Shield V9 carrot at wins ×5 + 0.1 flat (vs all-stick V8 that turtled to 15% WR). Builder V5 uses ×3 (less aggressive because base returns higher at 2× leverage).

**MomoDkr rule:** Start at **wins × 4.0, losses × 1.8** for MomoDkr's 6× HFT context. Tune after Phase 4 baseline if eval shows turtling (low trade count) or churn (high trade count + low net PnL).

---

## 2. Training methodology

### 2.1 Cold-start PPO reliably finds HFT
**moleapp evidence:** Every cold-start run (v1, v3, v4, v9) converged to scalping. Easiest gradient.

**MomoDkr rule:** Embrace this — MomoDkr IS HFT, so cold-start is the right tool. Don't try to warm-start from any external baseline.

### 2.2 Warm-start can degrade
**moleapp evidence:** V7 warm-start from V4 degraded by iter 75 (median PnL −62, σ → 1.50).

**MomoDkr rule:** Treat warm-start as last resort. If used (e.g., after architecture change): LR warmup 1e-5 → 5e-5, clip_param 0.1, short budget. Otherwise cold-start.

### 2.3 σ(action) > 2.0 = death signal
**moleapp evidence:** V9 iter-725 had σ = 24.84 — 25× healthy 0.85–1.20. Policy was random sampling. HOLD probability → 0%.

**MomoDkr rule:** [`training/callbacks/sigma_divergence_killswitch.py`](../training/callbacks/sigma_divergence_killswitch.py) monitors action-distribution entropy at every eval. If σ > 2.0 in 2 consecutive evals → KILL training, revert to `best_checkpoint/`. For Discrete actions, monitor the entropy of the categorical distribution as the equivalent signal.

### 2.4 Entropy schedule must be tight
**moleapp evidence:** V1–V3 entropy at 0.01 → σ creep. V4 halved to 0.005 → σ cleaner. V9 at 0.005 still saw late creep.

**MomoDkr rule:** PPO config in `configs/training/v1_engine_cold.yaml`: `ent_coef` schedule 0.005 → 0.0005 over total_timesteps. Linear decay.

### 2.5 Half-step constraint changes
**moleapp evidence:** V5 jumped min_tp 0.02 → 0.04 (×2) → WR cratered to 0.36–0.44, eval PnL halved.

**MomoDkr rule:** Adjust env constraints ≤50% per iteration. If Phase 4 trains at TP target 0.05% and we want tighter, go 0.05 → 0.04, not 0.05 → 0.03.

### 2.6 Scale position size inversely to leverage
**moleapp evidence:** V8 at 4× leverage × 50% max position = 200% notional → 10% price move = 20% account DD = kill. eval PnL −113.

**MomoDkr rule:** At 6× leverage, cap max position notional at ~17% (6 × 17% ≈ 100% notional, matching moleapp's safe V9 zone). Encoded in [`envs/momodkr_env.py`](../envs/momodkr_env.py).

---

## 3. Tooling and process

### 3.1 Eval recording is invaluable
**moleapp evidence:** 3-MP4 post-mortem of V9 revealed 3 distinct failure modes (good / trend-fight / divergence) that W&B summary metrics flattened.

**MomoDkr rule:** [`training/callbacks/trade_log_callback.py`](../training/callbacks/trade_log_callback.py) writes per-episode JSON with every tick action, reward, observation. Optional MP4 of book + position state — defer until Phase 5 if storage cost matters.

### 3.2 Sequential eval (avoid Ray race) — N/A for SB3
**moleapp evidence:** RLlib 2.54 parallel eval race → KeyError.

**MomoDkr divergence:** SB3 doesn't have this issue. Use SB3's standard `EvalCallback` with `n_eval_episodes` — sequential by default.

### 3.3 R2 sync must run continuously
**moleapp evidence:** R2 sync tmux window died unnoticed during V9. 42 unsynced JSONs at risk if pod crashed.

**MomoDkr rule:** [`scripts/r2_sync.py`](../scripts/r2_sync.py) runs as a daemon process. Verify alive at start of every training session: `ps aux | grep r2_sync`. Backup to Akeyless-stored R2 credentials.

### 3.4 ONNX parity is a hard gate
**moleapp evidence:** V9 iter-50 ONNX achieved max_diff = 0.0. The parity script catches architecture mismatches, missing normalization stats, dtype issues.

**MomoDkr rule:** [`scripts/validate_onnx_parity.py`](../scripts/validate_onnx_parity.py) must PASS (`max_diff < 1e-4`) before any deploy. With CNN+LSTM, parity must include LSTM hidden-state inputs/outputs — moleapp's MLP-only export does NOT cover this. Phase 6 must specifically test recurrent-state ONNX export.

### 3.5 GPU is overspec for small-net PPO — N/A for CNN+LSTM
**moleapp evidence:** RTX 4080 = RTX 4090 for moleapp's MLP. A100/H100 wasted.

**MomoDkr divergence:** CNN + LSTM at sequence length 50 + batch 256 is heavier than moleapp's MLP. RTX 4080 likely still adequate for v1; profile in Phase 4 before upgrading.

### 3.6 Always deploy from `best_checkpoint/`, never `final_checkpoint/`
**moleapp evidence:** V9 trained to iter 725 but iter-50 was the deployable winner. Late iterations diverged.

**MomoDkr rule:** [`training/callbacks/best_checkpoint_tracker.py`](../training/callbacks/best_checkpoint_tracker.py) saves on max eval mean reward. ONNX export script defaults to `best_checkpoint/`. Document any deploy that uses anything else.

### 3.7 Preserve immutable checkpoint backups
**moleapp evidence:** V4 iter-50 checkpoint would have been overwritten by later runs without backup.

**MomoDkr rule:** First post-training task: `cp -r best_checkpoint v{version}_best_checkpoint && python scripts/r2_sync.py upload v{version}_best_checkpoint`.

---

## 4. Architectural lessons

### 4.1 Single-agent only — MARL is a dead end
**moleapp evidence:** Hierarchical "Prop Firm" MARL (3 traders + 1 RM) trained 30M+ steps, never beat 24% WR. Killed.

**MomoDkr rule:** MomoDkr is and stays single-agent. Risk management = hard-coded Rust Governor, not a learned policy.

### 4.2 Separate models per game type
**moleapp evidence:** Shield (HTF capital preservation) ≠ Engine (HFT scalping). A 9×-trained HFT model cannot, via downscaling, behave like a patient swing trader.

**MomoDkr rule:** MomoDkr is one game (HFT scalping). If a later product needs HTF behavior on Hyperliquid, train a separate model. Do not try to broaden MomoDkr's reward to cover both.

### 4.3 Train at deployment leverage ceiling
**moleapp evidence:** Engine V11 trained at 9× to serve Builder (6×) + Sprinter (9×). Training at 4× and deploying at 9× would expose users to risk the agent never learned.

**MomoDkr rule:** MomoDkr deploys at **6×** → train at **6×** in `configs/training/v1_engine_cold.yaml`. No exceptions. Bumping deployment leverage requires a full retraining cycle.

### 4.4 Strict capital isolation (vault layer)
**moleapp evidence:** Each tier in moleapp-aa had its own AA sub-wallet — cross-contamination impossible at smart-contract level.

**MomoDkr rule:** Hyperliquid Vault provides this natively — depositor funds are isolated in the vault contract. API agent can only trade vault funds, never withdraw. Multisig off-switch is a separate key from the API agent.

---

## 5. Deploy-side rules — the Governor (Rust)

These rules live in [`execution/rust_engine/crates/governor/`](../execution/rust_engine/crates/governor/) and are non-negotiable hard limits independent of the model's output. MomoDkr cannot ship to live capital until ALL are implemented and tested.

| Rule | Trigger | Why |
|---|---|---|
| Leverage clamp | `effective_leverage > 6×` | Trained ceiling; never exceed |
| Position notional cap | `single_asset_notional > 30% of vault NAV` | Concentration risk |
| Pre-submit circuit breaker | `price_move_1s > 1%` since model's decision tick | Model can't react to mid-decision moves |
| Funding regime gate | `8h funding > 0.05%` against position | Block new entries when funding bleeds |
| Consecutive-loss kill | `5 losses in a row` → flat + 30-min cooldown | Phase Lag enforcement |
| Macro-crash blocker | `BTC −3% in 60s` → flat all + 60-min cooldown | Reject all new entries during crashes |
| Trailing-DD circuit breaker | Vault NAV −8% from peak | Auto-pause, manual restart only |
| Redemption pause | depositor outflows >20%/24h | Prevent fire-sale slippage |

API agent wallet permissions: **`trade`, `cancel` only**. **No `withdraw`, no `transfer`**. Daily on-chain permission audit.

---

## 6. Empirical baselines (for "is this expected?" reference)

Adapted from moleapp V9 iter-50 reference; numbers will shift for L2/100ms HFT but the **shape** is the same.

| Metric | Healthy range (target for Phase 4) | Warning sign |
|---|---|---|
| Action distribution σ (categorical entropy) | 0.8 – 1.2 (5-action) | < 0.3 (collapsed) or near max ln(5)=1.609 (uniform random) |
| Trades/episode | 400 – 800 (15-min eval episode @ 100ms ticks) | < 50 (turtling) or > 2000 (churn) |
| Hold duration | 5 – 50 ticks (≈0.5–5s @ 100ms) | < 2 ticks (pure flicker) or > 500 ticks (stuck inventory) |
| Win rate (NET) | 0.48 – 0.55 | < 0.42 = wrong-way bias |
| Eval mean reward (first eval) | > 0 | < 0 for 3 consecutive evals = KILL |
| Max DD per episode | < 5% | > 8% = circuit breaker territory |
| Funding cumulative | < 0.05% per episode | > 0.2% = funding-bleed loss in progress |

---

## 7. Required pre-launch checklist for any training run

```
[ ] git pull + pip install -e .[dev] (deps fresh)
[ ] python -c "import stable_baselines3" (SB3 import sanity)
[ ] Backup any previous best_checkpoint to v{N}_best_checkpoint/
[ ] Export MOMODKR_TRADE_LOG_DIR / RUN_NAME / RECORD_OBS=1
[ ] R2 sync daemon alive (ps aux | grep r2_sync)
[ ] Run name in YAML matches MOMODKR_RUN_NAME env var
[ ] Akeyless creds loadable (python -c "from live.hyperliquid_api.agent_wallet import load_akeyless_secret")
[ ] Clear KILL CRITERION documented (e.g., σ > 2.0 × 2 evals OR eval mean reward < 0 × 3 evals)
```

---

## 8. Versioning convention

`v{N}-engine-{variant}` — examples:
- `v1-engine-cold` (Phase 4 cold-start baseline on BTC)
- `v1-engine-btceth` (Phase 5 curriculum: BTC + ETH)
- `v1-engine-btethsol` (Phase 5 final curriculum: BTC + ETH + SOL)
- `v1-engine-fundingfix` (after adding refined funding penalty)
- `v2-engine-cold` (after architectural change — e.g., Transformer instead of LSTM)

Bump N on architecture changes (new env class, new reward shape, new model class).
Bump variant on config tuning. Use suffixes to flag the experimental delta.

---

## What is NOT carried forward from moleapp

These moleapp decisions do NOT apply to MomoDkr:

- **RLlib / Ray** — MomoDkr uses SB3 (cleaner ONNX, no MARL needed)
- **Box(7) continuous action space** — MomoDkr uses Discrete(5)
- **3-layer MLP** — MomoDkr uses CNN+LSTM (L2 microstructure requires it)
- **15-min OHLCV bars** — MomoDkr uses 100ms L2 order book snapshots
- **AWS Secrets Manager / 1Password** — MomoDkr uses Akeyless (per user preference)
- **MARL "Prop Firm" architecture** — MomoDkr is single-agent
- **Skull `TIER_LEVERAGE_RATIO` downscaling** — MomoDkr has one tier (vault depositors), single 6× model
