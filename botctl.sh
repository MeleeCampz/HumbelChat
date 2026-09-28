#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# botctl.sh — The single local-dev control script for the Discord AI bot.
#
# Subcommands (user-facing):
#   setup    create/refresh the .venv and install requirements.txt
#   start    start in a detached tmux session 'bot' (auto-runs setup if the
#            venv is missing, so a fresh clone needs only this one command)
#   stop     graceful SIGINT, then kill-session fallback
#   restart  stop + start
#   status   RUNNING (PID …) / NOT RUNNING
#   logs     TTY: attach live; piped/redirected: last 2000 pane lines
#
# _run is an INTERNAL command (the foreground launch) that `start` types into
# the tmux session — it is not a user-facing subcommand.
#
# Why tmux: a process started from the web terminal is tied to that browser
# tab's PTY. Minimize/close the tab → SIGHUP → bot dies. A tmux server lives
# on the machine, so the bot keeps running no matter what you do with the
# browser. Re-attach anytime:  tmux attach -t bot   (detach: Ctrl-b d)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="bot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.bot.pid"
LOG_DIR="$SCRIPT_DIR/logs"

# Echoes the venv's python path (Unix bin/ or Windows Scripts/), or nothing.
venv_python() {
    local py="$SCRIPT_DIR/.venv/bin/python"
    if [ -x "$py" ]; then echo "$py"; return 0; fi
    py="$SCRIPT_DIR/.venv/Scripts/python.exe"
    if [ -x "$py" ]; then echo "$py"; return 0; fi
    return 0
}

# Create/refresh the project virtualenv and install runtime deps. Idempotent —
# safe to re-run; refreshing installs any missing/new dependencies. Uses the
# venv's own python (`-m pip`) so it works on both Unix and Windows layouts.
do_venv_setup() {
    local venv_dir="$SCRIPT_DIR/.venv"
    local py="${PYTHON:-python3}"
    if [ -z "$(venv_python)" ]; then
        echo "Creating virtualenv at $venv_dir ..."
        "$py" -m venv "$venv_dir"
    else
        echo "Virtualenv already exists at $venv_dir — refreshing dependencies."
    fi
    local vpy
    vpy="$(venv_python)"
    if [ -z "$vpy" ]; then
        echo "ERROR: could not locate the venv python after setup." >&2
        exit 1
    fi
    "$vpy" -m pip install --upgrade pip
    "$vpy" -m pip install -r requirements.txt
}

cmd_setup() {
    do_venv_setup
    echo ""
    echo "Done. Start the bot with:  ./botctl.sh start"
}

cmd_start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Bot session already exists. Use: ./botctl.sh stop  (or: tmux attach -t $SESSION)"
        exit 1
    fi
    # Auto-setup: a fresh clone has no venv yet. Do it here (host-side) so the
    # install output is visible in your terminal rather than hidden in tmux.
    if [ -z "$(venv_python)" ]; then
        echo "No virtualenv found — setting one up first (one-time) ..."
        do_venv_setup
    fi
    mkdir -p "$LOG_DIR"
    # Detached session; the internal `_run` refuses to double-start via
    # .bot.pid, so a stale session cannot spawn a second bot instance.
    #
    # The Windows build of tmux (arndawg/tmux-windows) cannot run compound
    # command strings: an explicit command's first token is CreateProcess'd
    # directly (so "cd ... && ..." never executes) and `-c` fails to spawn
    # outright.  A BARE session does start its default shell (cmd.exe there,
    # bash on Unix), so create it empty and type the run command in with
    # send-keys — which works on every tmux build.
    tmux new-session -d -s "$SESSION"
    sleep 1
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*)
            # Session shell is cmd.exe — invoke Git Bash explicitly with a
            # Windows path (cygpath), since cmd cannot cd to /c/... paths.
            local bash_exe dir_win
            bash_exe="$(cygpath -w "$(command -v bash)")"
            dir_win="$(cygpath -m "$SCRIPT_DIR")"
            tmux send-keys -t "$SESSION" "\"$bash_exe\" -c \"cd $dir_win && exec ./botctl.sh _run\"" C-m
            ;;
        *)
            tmux send-keys -t "$SESSION" "cd '$SCRIPT_DIR' && exec ./botctl.sh _run" C-m
            ;;
    esac
    echo "Bot starting in detached tmux session '$SESSION'."
    echo "  Live output:  ./botctl.sh logs   (TTY: attaches; piped: last 2000 lines)"
    echo "  File logs:    $LOG_DIR/bot.log, $LOG_DIR/dev.log"
}

cmd_stop() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "No bot session running."
        return
    fi
    # Graceful first (Ctrl-C → SIGINT to the foreground process), then force.
    tmux send-keys -t "$SESSION" C-c
    for _ in $(seq 1 10); do
        sleep 0.5
        tmux has-session -t "$SESSION" 2>/dev/null || break
    done
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Still alive after SIGINT — killing session."
        tmux kill-session -t "$SESSION"
    fi
    echo "Bot stopped."
}

# bash 'kill -0' cannot see processes in other console sessions on Windows
# (the tmux session runs the bot in its own console), so use tasklist there.
pid_alive() {
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*)
            tasklist //NH //FI "PID eq $1" 2>/dev/null | grep -qv "^INFO:"
            ;;
        *)
            kill -0 "$1" 2>/dev/null
            ;;
    esac
}

cmd_status() {
    local pid=""
    if [ -f "$PIDFILE" ]; then
        pid="$(tr -d '[:space:]' < "$PIDFILE")"
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null && [ -n "$pid" ] && pid_alive "$pid"; then
        echo "RUNNING (PID $pid, tmux session '$SESSION')"
        echo "  Attach to watch output:  tmux attach -t $SESSION   (detach: Ctrl-b d)"
    else
        echo "NOT RUNNING"
    fi
}

cmd_logs() {
    # P1 #5: `tmux follow` does not exist (tmux 3.5a has no such command) —
    # this used to error out.  Now: a real TTY gets an interactive attach
    # (live scrolling, Ctrl-b d to detach); anything piped/redirected gets a
    # non-interactive one-shot capture of the last 2000 pane lines.
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "No bot session running (no tmux session '$SESSION'). Start it with: ./botctl.sh start" >&2
        exit 1
    fi
    if [ -t 0 ] && [ -t 1 ]; then
        exec tmux attach -t "$SESSION"
    fi
    exec tmux capture-pane -p -S -2000 -t "$SESSION" | tail -n 2000
}

# INTERNAL: the actual foreground launch, typed into the tmux session by
# `start`. Not a user-facing subcommand. Preserves the old foreground-launch
# behaviour: PID fast-fail guard, venv-python resolution, dev INFER_URL
# default, then exec the bot in the foreground so Ctrl-C reaches it directly.
cmd_run() {
    cd "$SCRIPT_DIR"
    mkdir -p "$LOG_DIR"

    # Fast-fail if another instance is already up. (main.py enforces the real
    # single-instance lock on .bot.pid; this just gives a friendlier message.)
    if [ -f ".bot.pid" ]; then
        local old_pid
        old_pid="$(tr -d '[:space:]' < .bot.pid)"
        if pid_alive "$old_pid"; then
            echo "Bot already running (PID $old_pid). Use ./botctl.sh stop first."
            exit 1
        else
            rm -f .bot.pid
        fi
    fi

    local vpy
    vpy="$(venv_python)"
    if [ -z "$vpy" ]; then
        echo "ERROR: virtualenv not found (.venv/bin/python or .venv/Scripts/python.exe)" >&2
        echo "Run ./botctl.sh setup first, then ./botctl.sh start." >&2
        exit 1
    fi

    echo "Starting Discord AI bot..."
    echo "  Console log:  visible in terminal"
    echo "  File logs:    $LOG_DIR/bot.log (INFO)"
    echo "                    $LOG_DIR/dev.log (DEBUG)"
    echo "─────────────────────────────────────"

    # Dev runs happen ON the host, where the docker-only INFER_URL from .env
    # (host.docker.internal:8888) does not resolve. load_dotenv() does not
    # override real environment variables, so this host-local default wins;
    # set BOT_DEV_INFER_URL to point at a different backend.
    export INFER_URL="${BOT_DEV_INFER_URL:-http://127.0.0.1:8888/v1}"

    exec "$vpy" -u "$SCRIPT_DIR/main.py"
}

case "${1:-}" in
    setup)   cmd_setup ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    _run)    cmd_run ;;
    *)
        echo "Usage: $0 {setup|start|stop|restart|status|logs}"
        exit 2
        ;;
esac
