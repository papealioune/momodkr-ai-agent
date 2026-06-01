# RunPod Data Prep Guide

End-to-end recipe for materialising the 2-year MomoDkr training dataset
on a RunPod CPU pod, from Binance Vision raw ZIPs through the
26-feature episode parquets + train-only normalisation stats ready for
PPO. Total wall time ≈ 8-16 hours; total disk ≈ 700 GB.

## 1. Provision the pod

| Spec | Value |
|---|---|
| Pod type | **CPU** (no GPU needed for ingest) |
| vCPU | 16-32 |
| RAM | 32-64 GB |
| Storage | **1 TB persistent volume** mounted at `/workspace` |
| Region | EU or US — closest to your dev box; the bottleneck is Binance Vision throughput, not the pod |

Once the pod is up, edit Pod Settings → Environment Variables and add:

```
R2_ACCESS_KEY_ID=<moleapp R2 access key>
R2_SECRET_ACCESS_KEY=<moleapp R2 secret key>
```

Optional overrides (defaults in [`.env.example`](../.env.example)):
`R2_ENDPOINT_URL`, `R2_BUCKET_NAME=moleapp-rl-data`, `MOMODKR_R2_PREFIX=momodkr/`.

## 2. Clone the repo + bootstrap

```bash
cd /workspace
git clone <your-fork-url> momodkr-ai-agent
cd momodkr-ai-agent
bash runpod/setup.sh
```

`setup.sh` checks Python ≥ 3.11, runs `pip install -e .[dev]`, probes R2
reachability via `head_bucket`, and runs `pytest -q`. Stop here if any of
those fail — fix the env before downloading 600 GB you can't upload.

## 3. Phase 1 — sanity ingest (~30-60 min, ~10 GB)

This pulls the last 7 days of `bookTicker + aggTrades + bookDepth` for
BTC/ETH/SOL, reconstructs 100ms snapshots, validates them against the
1-hour kline cross-check (`max_mid_drift ≤ 1bp`), and uploads to R2.

```bash
bash runpod/run_ingest.sh test
```

**Verification checkpoints:**
- `data/datasets/<SYM>/snapshots/<YYYY-MM-DD>.parquet` exists for every
  symbol/day in the test window
- The log shows `validated N/N days` with 0 failures
- The R2 bucket has `momodkr/<SYM>/snapshots/<YYYY-MM-DD>.parquet` keys

If any day fails validation, INVESTIGATE before proceeding — the same
glitch will hit 100+ days in the full pull.

## 4. Phase 1 — full 2-year pull (4-8 hours, ~600 GB)

Detached so a dropped SSH doesn't kill it:

```bash
nohup bash runpod/run_ingest.sh full > /workspace/momodkr-ingest.log 2>&1 &
disown
```

Monitor with:

```bash
tail -f /workspace/momodkr-ingest.log
du -sh data/datasets data/raw/binance_vision
```

The script is idempotent — re-running skips days already on disk + R2.
Safe to kill and restart if your network blips.

**When `full` is done:** the log ends with `validated N/N days` and
`prepare_l2_dataset complete in <T>s`. R2 holds:

```
s3://moleapp-rl-data/momodkr/<SYM>/
  bookTicker/<YYYY-MM-DD>.parquet × 730
  aggTrades/<YYYY-MM-DD>.parquet × 730
  bookDepth/<YYYY-MM-DD>.parquet × 730
  snapshots/<YYYY-MM-DD>.parquet × 730
```

## 5. Phase 2 — feature engineering (~1-2 hours)

Turns each per-day snapshot parquet into a 26-market-feature parquet
(OFI windows, micro-price returns, realized vol, depth, funding, time
encodings) then concatenates them into chronological `train.parquet` +
`eval.parquet` with a train-only `norm_stats.json`.

```bash
python -m scripts.build_features \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --dataset-root data/datasets \
    --episodes-root data/episodes \
    --split-ratio 0.8 \
    --workers 8
```

**What gets written:**

```
data/datasets/<SYM>/features/<YYYY-MM-DD>.parquet     # per-day features (cacheable)
data/episodes/<SYM>/0.1.0/
  train.parquet                                       # chronological first 80%
  eval.parquet                                        # chronological last 20%
  norm_stats.json                                     # mean/std on TRAIN ONLY (no leakage)
  manifest.json                                       # row counts, time bounds, sha256
```

R2 is updated automatically with both the per-day features and the final
episode bundles under `momodkr/<SYM>/features/...` and
`momodkr/episodes/<SYM>/0.1.0/...`.

**Verification checkpoints:**

```bash
# 1. norm_stats.json exists for every symbol
for s in BTCUSDT ETHUSDT SOLUSDT; do
  ls -la data/episodes/$s/0.1.0/norm_stats.json
done

# 2. manifest summary
python -c "
import json
from pathlib import Path
for s in ['BTCUSDT','ETHUSDT','SOLUSDT']:
    m = json.loads(Path(f'data/episodes/{s}/0.1.0/manifest.json').read_text())
    print(s, 'train_rows=', m['train_rows'], 'eval_rows=', m['eval_rows'],
          'train_end<eval_start:', m['train_end_ms'] < m['eval_start_ms'])
"
```

All three lines should print `train_end<eval_start: True`. That's the
chronological-split invariant.

## 6. Sanity-test the env on real data (~1 min)

Quick smoke test that the env can actually consume the produced episodes:

```bash
python - <<'PY'
from envs.momodkr_env import EnvConfig, MomoDkrEnv
env = MomoDkrEnv("data/episodes/BTCUSDT/0.1.0/train.parquet",
                 EnvConfig(episode_length_ticks=500))
obs, info = env.reset(seed=0)
print("obs shape:", obs.shape, "dtype:", obs.dtype)
print("info:", {k: v for k, v in info.items() if k != "active_parquet"})
for _ in range(10):
    obs, r, term, trunc, info = env.step(0)
print("OK — env steps without NaN; ready for training.")
PY
```

If the env errors with "no norm_stats.json", Phase 2 didn't write the
stats — re-run `build_features` and check the log.

## 7. Tear down the pod

The dataset lives on R2, not on this pod's volume. You CAN tear down the
pod once R2 sync confirms (the upload happens incrementally during the
ingest). Optionally:

```bash
# Belt-and-braces: re-upload everything in case the daemon missed any files
python -m scripts.r2_sync upload --local data/episodes
python -m scripts.r2_sync upload --local data/datasets --filter snapshots
```

You're now ready for the [Training guide](RUNPOD_TRAINING_GUIDE.md).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `setup.sh` fails on `head_bucket` | R2 creds wrong | Re-check the two env vars; re-run setup |
| Many 404s during `run_ingest.sh test` | Date window includes a day Binance hasn't archived yet (typically t-2) | Edit `END_TEST=$(date -u -d "${TODAY} - 5 days" ...)` in `runpod/run_ingest.sh` |
| `validate_l2_data` fails `mid_vs_kline_drift` | Reconstruction lost too much liquidity on a thin-book day | Confirm `bookDepth` parquet for that day exists; if not, day was unarchived — log and skip |
| `build_features` fails `feature_version mismatch` | A per-day parquet was produced by an older code | Delete `data/datasets/<SYM>/features/` and re-run; `_concat_features` rejects mixed versions by design |
| Disk fills during `full` pull | Estimated 700 GB; volume may be too small | Mount a 1.5 TB volume or split into two pods by symbol |
