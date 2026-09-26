#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# setup_venv.sh — Create/refresh the project virtualenv for the bot
#
# Creates ./.venv (if missing) and installs requirements.txt into it.
# Idempotent: safe to re-run; re-running refreshes/installs any missing deps.
#
# The bot (start_bot.sh) runs from this venv, keeping its heavy ML deps
# (torch, sentence-transformers, transformers 4.x, …) isolated from the
# system Python so they don't conflict with other tools on the box.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PY="${PYTHON:-python3}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtualenv at $VENV_DIR ..."
    "$PY" -m venv "$VENV_DIR"
else
    echo "Virtualenv already exists at $VENV_DIR — refreshing dependencies."
fi

# Keep pip current, then install the bot's declared runtime dependencies.
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "Done. Start the bot with:  ./start_bot.sh   (or ./botctl.sh start)"
