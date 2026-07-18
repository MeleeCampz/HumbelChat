#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# start_bot.sh — Launch the Discord AI bot with full logging
#
# Console  → visible in your terminal (stdout)
# Files    → logs/bot.log   (INFO+, rotated, 10 MB × 5)
#           → logs/dev.log   (DEBUG, rotated, 10 MB × 5)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Kill any existing bot (PID file)
if [ -f ".bot.pid" ]; then
    old_pid=$(cat .bot.pid | tr -d '[:space:]')
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Bot already running (PID $old_pid). Use ./stop_bot.sh first."
        exit 1
    else
        rm -f .bot.pid
    fi
fi

echo "Starting Discord AI bot..."
echo "  Console log:  visible in terminal"
echo "  File logs:    $LOG_DIR/bot.log (INFO)"
echo "                    $LOG_DIR/dev.log (DEBUG)"
echo "─────────────────────────────────────"

exec python -u "$SCRIPT_DIR/main.py" 2>&1 | tee "$LOG_DIR/start_output.log"
