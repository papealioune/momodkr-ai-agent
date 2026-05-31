#!/usr/bin/env bash
# Run the L2 data ingest on RunPod.
#
# Usage:
#   bash runpod/run_ingest.sh test     # 7-day sanity window (~8-10 GB)
#   bash runpod/run_ingest.sh full     # full 2-year pull (~600+ GB)
#
# Both modes are idempotent: re-runs skip days already downloaded + uploaded.
# The 'full' mode reuses the 'test' window's parquets (zero redundant work).

set -euo pipefail

MODE="${1:-test}"
LOG_FILE="${LOG_FILE:-/workspace/momodkr-ingest.log}"
SYMBOLS=(BTCUSDT ETHUSDT SOLUSDT)
DATASET_ROOT="${DATASET_ROOT:-data/datasets}"
RAW_ROOT="${RAW_ROOT:-data/raw/binance_vision}"
HTTP_WORKERS="${HTTP_WORKERS:-12}"
RECON_WORKERS="${RECON_WORKERS:-4}"

# Date math: Binance Vision typically archives with ~2-day lag.
TODAY=$(date -u +%Y-%m-%d)
END_TEST=$(date -u -d "${TODAY} - 3 days" +%Y-%m-%d 2>/dev/null || date -u -v-3d +%Y-%m-%d)
START_TEST=$(date -u -d "${END_TEST} - 6 days" +%Y-%m-%d 2>/dev/null || date -u -v-9d +%Y-%m-%d)
START_FULL=$(date -u -d "${END_TEST} - 2 years" +%Y-%m-%d 2>/dev/null || date -u -v-2y +%Y-%m-%d)

case "$MODE" in
    test)
        START="$START_TEST"
        END="$END_TEST"
        echo "Mode=test  window=${START}..${END}  symbols=${SYMBOLS[*]}"
        ;;
    full)
        START="$START_FULL"
        END="$END_TEST"
        echo "Mode=full  window=${START}..${END}  symbols=${SYMBOLS[*]}"
        ;;
    *)
        echo "Usage: $0 {test|full}"
        exit 1
        ;;
esac

echo "Logging to: $LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"

set -x
python -m scripts.prepare_l2_dataset \
    --symbols "${SYMBOLS[@]}" \
    --start "$START" \
    --end "$END" \
    --raw-root "$RAW_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --workers "$HTTP_WORKERS" \
    --reconstruct-workers "$RECON_WORKERS" \
    2>&1 | tee -a "$LOG_FILE"
set +x

echo
echo "=== Done (mode=$MODE) ==="
echo "Disk usage:"
du -sh "$DATASET_ROOT" "$RAW_ROOT" 2>/dev/null || true
echo
echo "If mode=test passed, run:  bash runpod/run_ingest.sh full"
