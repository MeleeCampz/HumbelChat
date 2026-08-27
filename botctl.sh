#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# botctl.sh — Run the Discord AI bot in a detached tmux session
#
# Why: a process started from the web terminal is tied to that
# browser tab's PTY. Minimize/close the tab → SIGHUP → bot dies.
# A tmux server lives on the machine, so the bot keeps running no
# matter what you do with the browser. You can re-attach anytime:
#     tmux attach -t bot
# (detach again with Ctrl-b d)
#
# Usage:  ./botctl.sh {start|stop|restart|status|logs}
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="bot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.bot.pid"
LOG_DIR="$SCRIPT_DIR/logs"

cmd_start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Bot session already exists. Use: ./botctl.sh stop  (or: tmux attach -t $SESSION)"
        exit 1
    fi
    mkdir -p "$LOG_DIR"
    # Detached session; start_bot.sh refuses to double-start via .bot.pid,
    # so a stale session cannot spawn a second bot instance.
    tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR" "./start_bot.sh"
    echo "Bot starting in detached tmux session '$SESSION'."
    echo "  Live output:  ./botctl.sh logs   (or: tmux attach -t $SESSION)"
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

cmd_status() {
    local pid=""
    if [ -f "$PIDFILE" ]; then
        pid="$(tr -d '[:space:]' < "$PIDFILE")"
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "RUNNING (PID $pid, tmux session '$SESSION')"
        echo "  Attach to watch output:  tmux attach -t $SESSION   (detach: Ctrl-b d)"
    else
        echo "NOT RUNNING"
    fi
}

cmd_logs() {
    # Follow the live tmux pane output (what you'd see in the terminal).
    exec tmux follow -t "$SESSION"
}

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 2
        ;;
esac
