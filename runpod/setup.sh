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

# 0. Python 3.11+
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$(python -c "import sys; print(sys.version_info.major)")
PY_MIN=$(python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJ" -lt 3 ] || { [ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 11 ]; }; then
    echo "ERROR: Python >= 3.11 required, found $PY_VER"
    exit 1
fi
echo "[0/3] Python $PY_VER OK"

# 1. Install deps
echo "[1/3] Installing Python deps..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

# 2. R2 credential sanity (boto3 path only -- no rclone install needed)
echo "[2/3] Verifying R2 credentials..."
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
echo "[3/3] Running unit tests..."
pytest -q tests/

echo
echo "=== Setup complete ==="
echo
echo "Next: kick off the data ingest"
echo "  bash runpod/run_ingest.sh test       # last-week sanity (small)"
echo "  bash runpod/run_ingest.sh full       # full 2-year pull (long-running)"
echo
echo "Both write logs to /workspace/momodkr-ingest.log and upload to R2 incrementally."
