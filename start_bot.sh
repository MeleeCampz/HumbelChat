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
        echo "Bot already running (PID $old_pid). Use ./botctl.sh stop first."
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

# Run from the project virtualenv (created by ./setup_venv.sh) so the bot's
# heavy ML deps stay isolated from the system Python. Fail clearly if the
# venv is missing rather than silently running on the wrong interpreter.
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: virtualenv not found at $VENV_PY" >&2
    echo "Run ./setup_venv.sh first, then start the bot." >&2
    exit 1
fi

# Run in the foreground so signals (Ctrl-C) reach the bot directly and its
# exit code propagates. main.py already writes logs/bot.log + dev.log via
# RotatingFileHandler, so no tee copy is needed. (The old `exec cmd | tee`
# form didn't actually exec — the pipe kept a subshell in charge, which
# mangled exit codes and signal handling.)
exec "$VENV_PY" -u "$SCRIPT_DIR/main.py"
