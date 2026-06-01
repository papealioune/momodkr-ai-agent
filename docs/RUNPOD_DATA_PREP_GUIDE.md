# RunPod Data Prep Guide

End-to-end recipe for materialising the 2-year MomoDkr training dataset
on a RunPod CPU pod, from Binance Vision raw ZIPs through the
26-feature episode parquets + train-only normalisation stats ready for
PPO. Total wall time ≈ 8-16 hours; total disk ≈ 700 GB.

## 1. Provision the pod — pick once, use for both phases

**Strategy:** spin up a single **GPU Pod** sized for the training run
(Phase 4+) and use it for ingest too. The GPU sits idle during ingest
but you avoid migrating data between pods and the per-hour CPU/RAM
ratio on a GPU Pod is often better than the equivalent fixed CPU Pod
preset. Tear it down only when the training run is finished — or when
a retraining cycle clearly needs different hardware.

| Spec | Value |
|---|---|
| Pod type | **GPU Pod** (cheapest available with the CPU/RAM below) |
| GPU | RTX 4080 / RTX 4090 / A4000 Ada — moleapp lesson 3.5: A100/H100 is wasted money on small-net PPO. Pick on price, not specs. |
| vCPU | **≥ 8** (ingest uses ~12 of 8 download + 4 reconstruct workers; 8 cores is the floor) |
| RAM | **≥ 24 GB** (peak ~20-25 GB during parallel bookTicker reconstruction). 32 GB is the sweet spot. |
| Storage | **1 TB persistent volume** mounted at `/workspace` |
| Region | Closest to your dev box for SSH latency; Binance Vision egress is region-agnostic |

> **RunPod menu reality:** CPU Pod presets are fixed (8/32, 16/96 — exact
> pairings rotate per region). If a small GPU Pod happens to give you
> ≥ 8 vCPU + ≥ 24 GB at a price competitive with the CPU Pod options,
> take the GPU Pod — you'll already be there for training.
>
> **If your chosen pod has < 24 GB RAM**, lower the reconstruct
> concurrency before launching: edit `runpod/run_ingest.sh` and set
> `RECON_WORKERS=2` (or pass `--reconstruct-workers 2` to
> `scripts/prepare_l2_dataset.py`). The ingest still completes, just
> ~1.5× slower.

Once the pod is up, edit Pod Settings → Environment Variables and add:

```
R2_ACCESS_KEY_ID=<moleapp R2 access key>
R2_SECRET_ACCESS_KEY=<moleapp R2 secret key>
```

Optional overrides (defaults in [`.env.example`](../.env.example)):
`R2_ENDPOINT_URL`, `R2_BUCKET_NAME=momodkr-data`, `MOMODKR_R2_PREFIX=` (empty).

> **R2 token scope:** the token must have **Object Read & Write** on the
> `momodkr-data` bucket. In Cloudflare's R2 token creation UI: choose
> "Specify bucket(s) → `momodkr-data`" and "Permissions → Object Read &
> Write". Account-wide tokens work too but aren't necessary.
>
> **IP allowlist gotcha:** if you set "Client IP Address Filtering" on
> the R2 token, you MUST use the pod's **outbound HTTPS egress IP**,
> not the SSH host IP that RunPod shows you. They are different IPs.
> Get the real egress IP from inside the pod:
> ```bash
> curl -s https://ifconfig.me
> ```
> Then add that exact IP to the token's allowlist. RunPod pods can
> migrate to different hosts on restart (changing the egress IP), so
> the safe default is "Allow all IPs" on the token and rely on the
> bucket scope + Object R/W permission for security.

## 2. Clone the repo + bootstrap

```bash
cd /workspace
git clone <your-fork-url> momodkr-ai-agent
cd momodkr-ai-agent
bash runpod/setup.sh
```

`setup.sh` checks Python ≥ 3.11, **installs tmux** (used by every
long-running step), runs `pip install -e .[dev]`, probes R2 reachability
via `head_bucket`, and runs `pytest -q`. Stop here if any of those fail
— fix the env before downloading 600 GB you can't upload.

### About `runpod/bg.sh` — the SSH-drop-proof launcher

Every multi-hour step below is launched via `bash runpod/bg.sh <name>
'<command>'`, which spawns a detached tmux session. If your SSH
disconnects, the job keeps running. Lifecycle:

```bash
tmux ls                              # list active sessions
tmux attach -t <name>                # reattach + see live output
                                      #   (Ctrl-b then d to detach again, leaves it running)
tmux kill-session -t <name>          # stop the job
tail -f /workspace/logs/<name>.log   # watch the log without attaching
```

## Data window rationale (read this BEFORE running the ingest)

Binance Vision's `bookTicker` (top-of-book) archive has BOTH a start
and an end date — verified via direct HTTP probes:

| Stream | Window |
|---|---|
| `bookTicker` | **2023-05-22 .. 2024-03-15** (~10 months) |
| `bookDepth` | starts ~2022-Q4, ends 2024-03 (limited by bookTicker for our pipeline) |
| `aggTrades` | continuous, no cutoff |
| `klines`, `fundingRate` | continuous, no cutoff |

The env's `micro_price`, `log_spread_bps`, `top1_size_imbalance`, and
OFI features depend on `bookTicker`, so the full pull is anchored to
this ~10-month bookTicker window:

| Mode | Window | Days |
|---|---|---|
| `test` | 2024-03-09 .. 2024-03-15 | 7 |
| `full` | 2023-05-22 .. 2024-03-15 | 298 |

After 80/20 chronological split that's ~238 days train + ~60 days
eval — plenty for PPO. moleapp ran on 2 years but the eval split was
~5 months; our ~2 months eval is still statistically meaningful.

Override via env vars to use any sub-window:
```bash
BINANCE_BOOKTICKER_START=2023-08-01 \
BINANCE_BOOKTICKER_CUTOFF=2024-02-29 \
bash runpod/bg.sh ingest-full 'bash runpod/run_ingest.sh full'
```

The "10 months ending March 2024" data is intentional: training on it
gets you a deployable v1 brain, and the Sim-to-Real gap against current
Hyperliquid microstructure is closed empirically in Phase 9 via
small-capital live calibration. If Phase 9 shows the gap is too wide,
the upgrade path is Tardis.dev or CryptoLake (paid L2 archives) — but
defer until you have real numbers.

## 3. Phase 1 — sanity ingest (~30-60 min, ~10 GB)

This pulls the 7 days ending `BINANCE_BOOKTICKER_CUTOFF` of
`bookTicker + aggTrades + bookDepth` for BTC/ETH/SOL, reconstructs
100ms snapshots, validates them against the 1-hour kline cross-check
(`max_mid_drift ≤ 1bp`), and uploads to R2.

```bash
bash runpod/bg.sh ingest-test 'bash runpod/run_ingest.sh test'
tmux attach -t ingest-test          # watch live; Ctrl-b d to detach
```

**Verification checkpoints:**
- `data/datasets/<SYM>/snapshots/<YYYY-MM-DD>.parquet` exists for every
  symbol/day in the test window
- The log shows `validated N/N days` with 0 failures
- The R2 bucket has `momodkr/<SYM>/snapshots/<YYYY-MM-DD>.parquet` keys

If any day fails validation, INVESTIGATE before proceeding — the same
glitch will hit 100+ days in the full pull.

## 4. Phase 1 — full 2-year pull (4-8 hours, ~600 GB)

Launch detached so a dropped SSH doesn't kill it. The tmux session
keeps the process alive AND lets you reattach later to see live output:

```bash
bash runpod/bg.sh ingest-full 'bash runpod/run_ingest.sh full'
```

Output of `bg.sh` will print the reattach + tail commands. Disconnect
freely; reattach from any new SSH:

```bash
tmux ls
tmux attach -t ingest-full           # Ctrl-b d to detach without killing
tail -f /workspace/logs/ingest-full.log
du -sh data/datasets data/raw/binance_vision
```

The script is idempotent — re-running skips days already on disk + R2.
Safe to kill and restart if your network blips.

**When `full` is done:** the log ends with `validated N/N days` and
`prepare_l2_dataset complete in <T>s`. R2 holds:

```
s3://momodkr-data/<SYM>/
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
bash runpod/bg.sh features "python -m scripts.build_features \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --dataset-root data/datasets \
    --episodes-root data/episodes \
    --split-ratio 0.8 \
    --workers 8"
tmux attach -t features    # reattach to watch progress
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

## 7. Belt-and-braces R2 reconciliation (~5 min)

R2 upload happens incrementally during the ingest. As insurance against
the daemon missing a file:

```bash
python -m scripts.r2_sync upload --local data/episodes
python -m scripts.r2_sync upload --local data/datasets --filter snapshots
```

## 8. Keep the pod alive — head straight to training

**Do not tear down the pod.** The next step is training on this exact
machine — the dataset is already on local disk and on R2, and you've
already paid for setup. Skip directly to
[RUNPOD_TRAINING_GUIDE.md §3 onwards](RUNPOD_TRAINING_GUIDE.md#3-pull-episodes-from-r2-5-min-10-gb)
(skip §1-2 — same pod, same env vars, already bootstrapped).

When you'd tear down + spin up fresh hardware:
- The training run is finished and you've uploaded the ONNX + bundle to R2.
- A retrain cycle needs clearly different hardware (e.g. went from
  single-symbol to staged curriculum and need a fatter GPU, or pivoted
  to a model architecture that wants more VRAM).
- The pod has been idle for > 24h (stop the bleeding).

Otherwise: same pod, two phases, one bill.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `setup.sh` fails on `head_bucket` | R2 creds wrong | Re-check the two env vars; re-run setup |
| Many 404s during `run_ingest.sh test` | Date window includes a day Binance hasn't archived yet (typically t-2) | Edit `END_TEST=$(date -u -d "${TODAY} - 5 days" ...)` in `runpod/run_ingest.sh` |
| `validate_l2_data` fails `mid_vs_kline_drift` | Reconstruction lost too much liquidity on a thin-book day | Confirm `bookDepth` parquet for that day exists; if not, day was unarchived — log and skip |
| `build_features` fails `feature_version mismatch` | A per-day parquet was produced by an older code | Delete `data/datasets/<SYM>/features/` and re-run; `_concat_features` rejects mixed versions by design |
| Disk fills during `full` pull | Estimated 700 GB; volume may be too small | Mount a 1.5 TB volume or split into two pods by symbol |
| `bg.sh` says "session already exists" | Old job still running (or zombie pane after exit) | `tmux attach -t <name>` to check; `tmux kill-session -t <name>` if dead |
| `tmux: command not found` | setup.sh didn't run or wasn't a Debian/Ubuntu image | `apt-get update && apt-get install -y tmux` (or yum) — then re-run setup |
| Reattach shows just `=== job exited, press any key ===` | Job finished while you were disconnected | The log under `/workspace/logs/<name>.log` has the full history; press a key to close the pane |
| R2 token has IP allowlist set; setup.sh still 403s after token + bucket are clearly right | The IP you allowed is RunPod's SSH proxy / your laptop, NOT the pod's outbound IP | Inside the pod: `curl -s https://ifconfig.me` → add THAT IP to the token's Client IP Filtering, or just remove the IP filter entirely (default-safe given the bucket-scoped Read/Write token) |
| `R2_BUCKET_NAME=moleapp-rl-data` stuck in env across `unset` attempts | Pod Settings → Environment Variables still has the old override; the `unset` only affects current shell | RunPod UI → Pod → Edit → Environment Variables → remove the row or change to `momodkr-data` → restart the pod |
