#!/usr/bin/env bash
# Run the L2 data ingest on RunPod.
#
# Usage:
#   bash runpod/run_ingest.sh test     # 7-day sanity window (~8-10 GB)
#   bash runpod/run_ingest.sh full     # full 2-year pull (~600+ GB)
#
# Both modes are idempotent: re-runs skip days already downloaded + uploaded.
# The 'full' mode reuses the 'test' window's parquets (zero redundant work).
#
# IMPORTANT: Binance Vision's bookTicker (top-of-book) archive has BOTH a
# start AND an end date:
#   - START: 2023-05-22 (earlier than this is 404 for bookTicker + bookDepth)
#   - END:   2024-03-15 (later than this is 404; archive discontinued)
# So the usable L2 window is ~10 months: 2023-05-22 .. 2024-03-15.
#
# The env's micro_price / spread / OFI features depend on bookTicker, so the
# "full" pull is capped to this window rather than the naive cutoff-minus-2-years.
# Phase 9 closes the Sim-to-Real gap empirically; if more data is needed,
# upgrade to Tardis.dev. See docs/RUNPOD_DATA_PREP_GUIDE.md "Data window rationale".
#
# Override either bound to widen/narrow the window:
#   BINANCE_BOOKTICKER_START=2023-08-01 BINANCE_BOOKTICKER_CUTOFF=2024-02-29 bash runpod/run_ingest.sh full

set -euo pipefail

MODE="${1:-test}"
LOG_FILE="${LOG_FILE:-/workspace/momodkr-ingest.log}"
SYMBOLS=(BTCUSDT ETHUSDT SOLUSDT)
DATASET_ROOT="${DATASET_ROOT:-data/datasets}"
RAW_ROOT="${RAW_ROOT:-data/raw/binance_vision}"
HTTP_WORKERS="${HTTP_WORKERS:-12}"
RECON_WORKERS="${RECON_WORKERS:-4}"

# Anchors: the window where bookTicker + aggTrades + bookDepth are ALL
# available on Binance Vision. Override via env vars if you want a sub-range.
BINANCE_BOOKTICKER_START="${BINANCE_BOOKTICKER_START:-2023-05-22}"
BINANCE_BOOKTICKER_CUTOFF="${BINANCE_BOOKTICKER_CUTOFF:-2024-03-15}"

END_TEST="${END_TEST:-$BINANCE_BOOKTICKER_CUTOFF}"
START_TEST=$(date -u -d "${END_TEST} - 6 days" +%Y-%m-%d 2>/dev/null || date -u -j -v-6d -f "%Y-%m-%d" "${END_TEST}" +%Y-%m-%d)
START_FULL="${START_FULL:-$BINANCE_BOOKTICKER_START}"
END_FULL="${END_FULL:-$BINANCE_BOOKTICKER_CUTOFF}"

case "$MODE" in
    test)
        START="$START_TEST"
        END="$END_TEST"
        echo "Mode=test  window=${START}..${END}  symbols=${SYMBOLS[*]}"
        ;;
    full)
        START="$START_FULL"
        END="$END_FULL"
        echo "Mode=full  window=${START}..${END}  symbols=${SYMBOLS[*]}"
        # Compute span in days for visibility
        if days=$(python -c "from datetime import date; print((date.fromisoformat('${END}') - date.fromisoformat('${START}')).days)" 2>/dev/null); then
            echo "  span: ${days} days  (~$((days * 3))  day-tasks at 3 streams/day across 3 symbols)"
        fi
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
