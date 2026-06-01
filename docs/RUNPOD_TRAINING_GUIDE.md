# RunPod Training Guide

End-to-end recipe for training MomoDkr's PPO policy on the prepared
2-year dataset, exporting a deployment-ready ONNX with normalisation
baked in, and gating it through the moleapp parity rule. Assumes
[Data Prep](RUNPOD_DATA_PREP_GUIDE.md) is complete and episodes live at
`s3://moleapp-rl-data/momodkr/episodes/<SYM>/0.1.0/`.

## 1. Provision the pod

| Spec | Value |
|---|---|
| Pod type | **GPU** — RTX 4080 or 4090 (moleapp lesson 3.5: A100/H100 wastes money on small-net PPO) |
| vCPU | 16-32 (env step throughput matters more than GPU class) |
| RAM | 32-64 GB |
| Storage | 200 GB persistent volume at `/workspace` (episodes only — raw L2 stays on R2) |
| Region | Same as the data prep pod's region — R2 download is fastest there |

Pod env vars:

```
R2_ACCESS_KEY_ID=<moleapp R2 access key>
R2_SECRET_ACCESS_KEY=<moleapp R2 secret key>
WANDB_API_KEY=<your W&B key>          # optional but recommended
```

## 2. Clone + bootstrap

```bash
cd /workspace
git clone <your-fork-url> momodkr-ai-agent
cd momodkr-ai-agent
bash runpod/setup.sh
nvidia-smi    # confirm the GPU is visible
```

`setup.sh` installs tmux automatically — every long-running step below
goes through `runpod/bg.sh <session_name> <command>`, which spawns a
detached tmux session so an SSH drop never kills the job. Lifecycle:

```bash
tmux ls                              # list active sessions
tmux attach -t <name>                # reattach (Ctrl-b d detaches, leaves running)
tmux kill-session -t <name>          # stop the job
tail -f /workspace/logs/<name>.log   # watch the log without attaching
```

## 3. Pull episodes from R2 (~5 min, ~10 GB)

Only the episode bundles need to land locally — raw L2 stays on R2.

```bash
for s in BTCUSDT ETHUSDT SOLUSDT; do
  python -m scripts.r2_sync download \
      --local data/episodes \
      --filter "episodes/$s/0.1.0/"
done
```

Verify:

```bash
ls data/episodes/BTCUSDT/0.1.0/   # train.parquet eval.parquet norm_stats.json manifest.json
```

## 4. Walk-forward dry-run FIRST (~30-60 min, ~$0.20)

**Do not skip this.** moleapp lesson: if the reward shaping + wrapper
weights can't survive a holdout month, no amount of additional steps
will save them — and the full 20M run is ~20× more expensive.

```bash
bash runpod/bg.sh walkforward "python -m scripts.walk_forward \
    --symbol BTCUSDT \
    --train-start 2024-01-01 --train-end 2024-06-30 \
    --eval-start  2024-07-01 --eval-end  2024-07-31 \
    --train-config configs/training/v1_walkforward.yaml \
    --env-config   configs/env/momodkr_v1.yaml \
    --run-dir runs/walkforward-btc-2024H1"

tmux attach -t walkforward    # watch live; Ctrl-b d to detach
```

**Decision gate after walk-forward:**

```bash
# Pull every eval JSON and aggregate
python -c "
import json, statistics
from pathlib import Path
ep_dir = Path('runs/walkforward-btc-2024H1/eval_episodes')
rewards = []
for f in sorted(ep_dir.glob('*.json')):
    data = json.loads(f.read_text())
    for ep in data['episodes']:
        rewards.append(ep['cum_reward'])
print(f'n_episodes={len(rewards)}  mean={statistics.mean(rewards):.3f}'
      f'  stdev={statistics.stdev(rewards):.3f}'
      f'  pos_rate={sum(r>0 for r in rewards)/len(rewards):.2%}')
"
```

| Result | Action |
|---|---|
| `pos_rate >= 55%`, mean reward > 0, σ-killswitch never fired | ✓ Proceed to the full run |
| `pos_rate < 50%` or mean reward < 0 | ✗ Tune reward wrappers (start with `peak_dd_coeff`, `churn_penalty`, `funding_coeff`); re-run walk-forward |
| σ-killswitch aborted | ✗ Drop `ent_coef.start` to 0.002; tighten the `clip_range` to 0.1 |
| Eval DD > 5% on most episodes | ✗ Raise `dd_quadratic_coeff` to 100 or lower `max_position_notional_pct` to 0.14 |

Iterate the walk-forward until it's green. Cheap to fail here, expensive
to fail later.

## 5. Full 20M-step training (~12-24 hours)

Once walk-forward is green, launch the cold-start on the full 2-year
train split inside its own tmux session — SSH drops, network blips, or
even closing your laptop won't kill it:

```bash
mkdir -p runs/v1-engine-cold-btc

bash runpod/bg.sh train-btc "python -m training.train_ppo \
    --train-config configs/training/v1_engine_cold.yaml \
    --env-config   configs/env/momodkr_v1.yaml \
    --train-parquet data/episodes/BTCUSDT/0.1.0/train.parquet \
    --eval-parquet  data/episodes/BTCUSDT/0.1.0/eval.parquet \
    --run-dir runs/v1-engine-cold-btc"
```

**Watch (any time, any new SSH session):**

```bash
# A) reattach the tmux pane and see live SB3 output
tmux attach -t train-btc        # Ctrl-b d to detach, training keeps running

# B) or filter the log without attaching
tail -f /workspace/logs/train-btc.log | grep -E 'eval/mean_reward|killswitch/entropy_pct|New best'

# C) disk usage
du -sh runs/v1-engine-cold-btc

# D) GPU utilisation
nvidia-smi
```

The three callbacks fire automatically:

- `SigmaDivergenceKillswitch` — aborts if normalised action entropy
  stays > 0.95 (uniform random) or < 0.05 (collapsed) for 2 consecutive
  evals. **No manual intervention needed.** If it fires, deploy from the
  last `best_checkpoint.zip` and DON'T train past that point.
- `BestCheckpointTracker` — persists `runs/v1-engine-cold-btc/best_checkpoint/best_checkpoint.zip`
  + JSON marker on every new best eval reward.
- `TradeLogCallback` — drops per-eval JSON to `runs/v1-engine-cold-btc/eval_episodes/`
  for post-mortem.

**Optional — staged curriculum.** Use this AFTER you have a working
single-symbol run to fan out to BTC+ETH then BTC+ETH+SOL with
warm-starts. Curriculums run for 24-48 hours, so background it too:

```bash
bash runpod/bg.sh curriculum "python -m training.curriculum \
    --curriculum-config configs/training/curriculum_v1.yaml \
    --base-train-config configs/training/v1_engine_cold.yaml \
    --env-config        configs/env/momodkr_v1.yaml \
    --episodes-root data/episodes \
    --run-dir runs/curriculum-v1"
```

## 6. Backup best_checkpoint BEFORE doing anything else

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
cp -r runs/v1-engine-cold-btc/best_checkpoint \
      runs/v1-engine-cold-btc/best_checkpoint_${TIMESTAMP}
python -m scripts.r2_sync upload \
    --local runs/v1-engine-cold-btc \
    --filter best_checkpoint
```

moleapp rule 3.7: the next eval that beats this one will overwrite
`best_checkpoint/` locally. The dated copy + R2 backup is your insurance.

## 7. ONNX export with normalisation baked in (~30 sec)

```bash
python -m scripts.export_onnx \
    --checkpoint runs/v1-engine-cold-btc/best_checkpoint/best_checkpoint.zip \
    --norm-stats data/episodes/BTCUSDT/0.1.0/norm_stats.json \
    --output runs/v1-engine-cold-btc/policy.onnx
```

Inspect the sidecar manifest:

```bash
cat runs/v1-engine-cold-btc/policy.onnx.json
```

Confirm `normalisation_baked_in: true` and `norm_stats_path` matches the
path you passed. The Rust live engine will send RAW features; the ONNX
graph does the z-score internally.

## 8. ONNX parity gate — DO NOT SKIP

moleapp lesson 3.4: silent ONNX export bugs ship to mainnet undetected.
Two parity runs to cover both obs sources:

```bash
# (a) against the eval parquet -- pure-market obs
python -m scripts.validate_onnx_parity \
    --checkpoint runs/v1-engine-cold-btc/best_checkpoint/best_checkpoint.zip \
    --onnx       runs/v1-engine-cold-btc/policy.onnx \
    --norm-stats data/episodes/BTCUSDT/0.1.0/norm_stats.json \
    --obs-parquet data/episodes/BTCUSDT/0.1.0/eval.parquet \
    --max-rows 1000 \
    --tol 1e-4

# (b) against the recorded eval episode JSONs -- includes live position features
python -m scripts.validate_onnx_parity \
    --checkpoint runs/v1-engine-cold-btc/best_checkpoint/best_checkpoint.zip \
    --onnx       runs/v1-engine-cold-btc/policy.onnx \
    --norm-stats data/episodes/BTCUSDT/0.1.0/norm_stats.json \
    --eval-log-dir runs/v1-engine-cold-btc/eval_episodes \
    --max-rows 1000 \
    --tol 1e-4
```

Both must print `passed: true` with `max_diff_logits < 1e-4` and
`action_match: true`. If either fails, **DO NOT DEPLOY** — investigate
before re-exporting.

## 9. Upload deployable artifacts to R2

```bash
python -m scripts.r2_sync upload \
    --local runs/v1-engine-cold-btc \
    --filter policy.onnx

# Optionally pull the full run for offline review
python -m scripts.r2_sync upload \
    --local runs/v1-engine-cold-btc \
    --filter eval_episodes
```

The final shippable artifact lives at:

```
s3://moleapp-rl-data/momodkr/<run-dir>/policy.onnx
s3://moleapp-rl-data/momodkr/<run-dir>/policy.onnx.json   # manifest the Rust engine asserts at startup
s3://moleapp-rl-data/momodkr/<run-dir>/best_checkpoint/best_checkpoint.zip   # for re-export / debugging
```

## 10. Tear down the pod

Backups are on R2. The pod can be terminated. The next stage is the
[Rust inference engine](../execution/rust_engine/) which loads the
ONNX, ingests Hyperliquid l2Book WebSocket, and signs EIP-712.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `apply_obs_normalisation=True but no norm_stats.json` | Episode dir missing the stats file | Re-run `build_features` on the data prep pod (or `--no-upload --apply_obs_normalisation false` for a one-off raw sanity run) |
| Killswitch aborts in first 100k steps | Cold-start MLP hasn't differentiated yet | Lower `consecutive_evals` from 2 → 3 or raise `high_threshold` from 0.95 → 0.97. Be conservative — false negatives ship divergent policies. |
| `eval/mean_reward` stays flat at the breadcrumb floor | LR too high, policy not learning | Drop `learning_rate.start` from 3e-4 → 1e-4 |
| Parity fails: `max_diff_logits > 1e-4` | Almost always a torch / SB3 version mismatch between train and export | Re-run export ON THE TRAINING POD. NEVER export elsewhere. |
| Parity fails: `action_match: false` but logits close | Ties in argmax due to symmetric logits early in training | Train more steps; this resolves itself once the policy specialises |
| GPU at 5% utilisation | env.step is the bottleneck (expected for MLP PPO) | Raise `vec_env.n_envs` from 12 → 16; use more vCPUs |
| Disk fills with eval JSONs + tb logs | Long training run + record_obs=true | Disable record_obs after walk-forward confirms parity; the production run only needs it on the last eval pass |
| `bg.sh` says "session already exists" | Old training session still running (or zombie pane) | `tmux attach -t train-btc` to inspect; `tmux kill-session -t train-btc` if dead |
| `tmux attach` shows `=== job exited, press any key ===` | Training finished (or aborted) while disconnected | Full output is in `/workspace/logs/<session>.log`; press a key to close the pane |
| SSH dropped mid-training | Expected; tmux protects you | Reconnect → `tmux attach -t train-btc` → all good, no work lost |
