---
name: bot-ops
description: Operate the Discord bot — production via docker compose, local dev via tmux. Start/stop/restart, read logs, check status, debug startup failures. Use when asked to start, stop, restart, or troubleshoot the running bot.
---

# Bot operations

## Production (Docker Desktop, docker compose)

The live bot runs as a container from this repo's checkout
(`C:\Repos\GitHub_Me\HumbelChat`), defined in `docker-compose.yml`. All runtime
state is bind-mounted from the repo: `data/` (KB + vector index + recordings +
history.json), `logs/`, `hf-cache/` (HF model cache). Full docs: `docs/docker.md`.

```bash
docker compose up -d --build   # start; also the update flow after git pull
docker compose ps              # status
docker compose logs -f bot     # live stdout
docker compose restart bot     # restart
docker compose down            # stop (host data untouched)
```

- `restart: unless-stopped` — crashes/reboots recover automatically.
- No healthcheck (no HTTP endpoint): judge from logs — "logged in as …", and
  the backend probe settling UP. Give ~10–20 s after a start; bge-m3 loads from
  cache in seconds and the first backend-health probe can log a transient
  `DOWN (timeout)` that recovers on its own.
- No published ports (outbound WebSocket + UDP only).
- **One process at a time:** never run a local dev session while the container
  is up — same Discord token, shared `data/`.

## Knowledge database (direct host access)

`data\knowledge\.vector_index_cache\vector_index.db` — plain SQLite on the
host. Reading while the bot runs is fine; don't write to it live (stop the
container or work on a copy). New `/upload_kb` rows appear immediately.

## Local dev (no Docker) — tmux-based, survives terminal close

```bash
./botctl.sh start      # start in detached tmux session 'bot'
./botctl.sh status     # RUNNING (PID …) / NOT RUNNING
./botctl.sh logs       # piped: last 2000 lines; interactive TTY: attaches
./botctl.sh restart    # graceful stop (SIGINT, 5s) then start
./botctl.sh stop       # SIGINT first, kill-session fallback
tmux attach -t bot     # watch live output directly (Ctrl-b d detaches)
```

### Python environment gotcha

`start_bot.sh` runs the venv's python (`./.venv/bin/python`, created by
`./setup_venv.sh`). If startup fails with ImportError:

```bash
source .venv/activate && python main.py     # foreground dev run
```

(or re-run `./setup_venv.sh` to refresh deps).

## Logs

- `logs/bot.log` — INFO+, rotated 10 MB × 5
- `logs/dev.log` — DEBUG, rotated 10 MB × 5
- Written by `main.py` itself (RotatingFileHandler); no tee involved. Same files
  in Docker (bind-mounted) and dev runs.

When debugging: read the tail of `dev.log` first (`tail -n 100 logs/dev.log`),
it has the full traceback context.

## Runtime data

- `data/` — knowledge base docs, recordings, session notes (git-ignored)
- `.bot.pid` — PID file (dev runs only); a stale one is cleaned by `start_bot.sh` automatically
- Slash commands not showing in Discord? Run `/sync` in the bot's channel.

## Required config

`.env` must define at minimum: `DISCORD_BOT_TOKEN`, `INFER_URL`,
`INFER_API_KEY`. Missing/invalid values → login/startup error, visible in
`dev.log`. Never print or commit `.env` contents. In Docker, `INFER_URL` uses
`host.docker.internal` (Docker Desktop resolves it to the Windows host) or the
LAN IP of the machine running the unsloth API.
