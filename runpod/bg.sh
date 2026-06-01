#!/usr/bin/env bash
# Run a long-running command in a detached tmux session.
#
# Why tmux (not nohup): tmux lets you REATTACH and watch live output
# from a new SSH session, not just tail a logfile. The process keeps
# running on a dropped SSH connection either way.
#
# Usage:
#   bash runpod/bg.sh <session_name> <command...>
#
# Examples:
#   bash runpod/bg.sh ingest "bash runpod/run_ingest.sh full"
#   bash runpod/bg.sh features "python -m scripts.build_features --symbols BTCUSDT ETHUSDT SOLUSDT"
#   bash runpod/bg.sh train "python -m training.train_ppo --train-config ... --run-dir ..."
#
# Lifecycle commands:
#   tmux ls                              # list every active session
#   tmux attach -t <session_name>        # reattach (Ctrl-b d detaches, leaves it running)
#   tmux kill-session -t <session_name>  # kill the job
#   tail -f /workspace/logs/<session_name>.log   # watch log without attaching

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <session_name> <command...>"
    echo "Example: $0 ingest 'bash runpod/run_ingest.sh full'"
    exit 1
fi

SESSION="$1"
shift
CMD="$*"

LOG_DIR="${LOG_DIR:-/workspace/logs}"
LOG_FILE="${LOG_DIR}/${SESSION}.log"
mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not found. Install it (runpod/setup.sh installs it automatically) and retry." >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "  Reattach: tmux attach -t $SESSION"
    echo "  Kill:     tmux kill-session -t $SESSION"
    exit 1
fi

# bash -lc loads ~/.bashrc so conda / venv / activate hooks fire as if the user logged in.
# printf %q safely re-quotes the command so spaces / quotes / globs survive.
tmux new-session -d -s "$SESSION" \
    "bash -lc $(printf '%q' "$CMD"); echo; echo '=== job exited; press any key to close pane ==='; read -n 1 -r -s"
# pipe-pane -o mirrors everything written to the pane into the log file.
tmux pipe-pane -o -t "$SESSION" "cat >> $LOG_FILE"

echo "Started session '$SESSION'"
echo "  Command:  $CMD"
echo "  Log:      $LOG_FILE"
echo
echo "  Reattach: tmux attach -t $SESSION    (Ctrl-b d to detach, leaves it running)"
echo "  Tail:     tail -f $LOG_FILE"
echo "  Kill:     tmux kill-session -t $SESSION"
echo "  List all: tmux ls"
