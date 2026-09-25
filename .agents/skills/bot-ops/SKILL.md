---
name: bot-ops
description: Operate the HumbelChat Discord bot — start/stop/restart via tmux, read logs, check status, debug startup failures. Use when asked to start, stop, restart, or troubleshoot the running bot.
---

# Bot operations (HumbelChat)

## Control (tmux-based, survives terminal close)

```bash
./botctl.sh start      # start in detached tmux session 'bot'
./botctl.sh status     # RUNNING (PID …) / NOT RUNNING
./botctl.sh logs       # piped: last 2000 lines; interactive TTY: attaches
./botctl.sh restart    # graceful stop (SIGINT, 5s) then start
./botctl.sh stop       # SIGINT first, kill-session fallback
tmux attach -t bot     # watch live output directly (Ctrl-b d detaches)
```

`tmux` is installed via winget (`arndawg.tmux-windows`, linked into
`%LOCALAPPDATA%\Microsoft\WinGet\Links`). If `tmux: command not found`, the
terminal was opened before install — reopen it.

## Python environment gotcha

`start_bot.sh` runs bare `python`, which on this machine is **system Python 3.10**,
not the venv (3.12, where deps live). If startup fails with ImportError:

```bash
source .venv/activate && python main.py     # foreground dev run
```

(or patch `start_bot.sh` to use `.venv/Scripts/python.exe`).

## Logs

- `logs/bot.log` — INFO+, rotated 10 MB × 5
- `logs/dev.log` — DEBUG, rotated 10 MB × 5
- Written by `main.py` itself (RotatingFileHandler); no tee involved.

When debugging: read the tail of `dev.log` first (`tail -n 100 logs/dev.log`),
it has the full traceback context.

## Runtime data

- `data/` — knowledge base docs, recordings, session notes (git-ignored)
- `.bot.pid` — PID file; a stale one is cleaned by `start_bot.sh` automatically
- Slash commands not showing in Discord? Run `/sync` in the bot's channel.

## Required config

`.env` must define at minimum: `DISCORD_BOT_TOKEN`, `INFER_URL`, `INFER_API_KEY`.
Missing/invalid token → discord login error at startup, visible in `dev.log`.
