#!/usr/bin/env bash
# MomoDkr RunPod bootstrap (CPU pod sufficient — no GPU needed for ingest)
# Usage: bash runpod/setup.sh
#
# Required env vars (set them via RunPod's Pod Edit > Environment Variables):
#   R2_ACCESS_KEY_ID           moleapp R2 access key
#   R2_SECRET_ACCESS_KEY       moleapp R2 secret key
# Optional (sensible defaults):
#   R2_ENDPOINT_URL            (default: moleapp's R2 account endpoint)
#   R2_BUCKET_NAME             (default: moleapp-rl-data)
#   MOMODKR_R2_PREFIX          (default: momodkr/)

set -euo pipefail

echo "=== MomoDkr RunPod setup ==="

# 0a. Python 3.11+
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$(python -c "import sys; print(sys.version_info.major)")
PY_MIN=$(python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJ" -lt 3 ] || { [ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 11 ]; }; then
    echo "ERROR: Python >= 3.11 required, found $PY_VER"
    exit 1
fi
echo "[0a/4] Python $PY_VER OK"

# 0b. tmux -- every long-running command goes through runpod/bg.sh which uses tmux
#     so the job survives SSH drops AND remains reattachable from a new shell.
echo "[0b/4] Ensuring tmux is installed..."
if ! command -v tmux >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq tmux
    elif command -v yum >/dev/null 2>&1; then
        yum install -y -q tmux
    else
        echo "WARNING: tmux missing and no apt/yum found. Install manually before running long jobs." >&2
    fi
fi
command -v tmux >/dev/null 2>&1 && echo "  tmux: $(tmux -V)"

# 1. Install deps
echo "[1/4] Installing Python deps..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

# 2. R2 credential sanity (boto3 path only -- no rclone install needed)
echo "[2/4] Verifying R2 credentials..."
if [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
    echo "ERROR: R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set."
    echo "  Add them in RunPod: Pod > Edit > Environment Variables."
    exit 1
fi
python -c "
from scripts.r2_sync import get_client, get_bucket
c = get_client(); b = get_bucket()
c.head_bucket(Bucket=b)
print(f'  R2 reachable, bucket=\"{b}\"')
"

# 3. Smoke test the code
echo "[3/4] Running unit tests..."
pytest -q tests/

# 4. Make the background runner executable
chmod +x runpod/bg.sh runpod/run_ingest.sh
mkdir -p /workspace/logs
echo "[4/4] Background runner ready: runpod/bg.sh"

echo
echo "=== Setup complete ==="
echo
echo "Next: kick off the data ingest IN A TMUX SESSION (survives SSH drops):"
echo "  bash runpod/bg.sh ingest-test 'bash runpod/run_ingest.sh test'   # last-week sanity (~30-60 min)"
echo "  bash runpod/bg.sh ingest-full 'bash runpod/run_ingest.sh full'   # full 2-year pull (4-8 hours)"
echo
echo "Reattach later from any new SSH session:"
echo "  tmux ls                       # list active sessions"
echo "  tmux attach -t ingest-full    # reattach (Ctrl-b d detaches)"
echo "  tail -f /workspace/logs/ingest-full.log"
