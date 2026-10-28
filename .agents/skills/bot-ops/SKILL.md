---
name: bot-ops
description: Operate the Discord bot — production via docker compose, local dev via ./botctl.sh. Start/stop/restart/update, read logs, check status, access the knowledge DB, debug startup failures. Use when asked to start, stop, restart, update, or troubleshoot the running bot.
---

# Bot operations

## ⚠️ Sandbox limitation (read first)

If you are running inside the dev sandbox: its Docker daemon is **NOT the
host's Docker Desktop**. The live containers (`humbelchat-bot`, rerankers) are
invisible here — `docker compose ps` / `docker cp` / starting or stopping the
bot from the sandbox WILL fail or hit the wrong daemon. Do not try to manage
the production container from the sandbox.

What DOES work from the sandbox: reading `logs/` and `data/` (bind-mounted),
and E2E-testing the live bot over the Discord REST API — see
`docs/e2e-testing.md` (`scripts/e2e_discord_spike.py`). Anything requiring
`docker compose …` must be run by the user on the host.

Two ways the bot runs — **never both at once** (same Discord token, shared `data/`):

- **Production:** a Docker container from this repo's checkout (`docker-compose.yml`).
- **Local dev:** `./botctl.sh`, which runs the venv in a detached tmux session.

## Production (Docker Desktop)

The live bot runs as a single container from this repo's checkout
(`C:\Repos\GitHub_Me\HumbelChat`), defined in `docker-compose.yml`. All runtime
state is bind-mounted from the repo — nothing important lives inside the image:

- `data/` — KB docs, `.vector_index_cache/vector_index.db`, sessions/reminders JSON
- `logs/` — `bot.log` (INFO) + `dev.log` (DEBUG), rotated 10 MB × 5
- `hf-cache/` — HuggingFace model cache (bge-m3 ~2.2 GB)
- `.env`, `characters.json` — config

Full docs: `docs/docker.md`.

### Control (always from the repo root)

```bash
# start / UPDATE flow after git pull — ALWAYS with the build stamp (see below):
docker compose build --build-arg GIT_SHA=$(git rev-parse HEAD) bot
docker compose up -d bot
docker compose ps              # status ("Up …" — there is no healthcheck)
docker compose logs -f bot     # live stdout
docker compose restart bot     # restart
docker compose down            # stop (host data untouched)
```

**Build stamp (verify every deploy):** the image bakes the git SHA into
`/app/.build_sha` and the bot logs `Bot build sha: …` at startup. After any
deploy, confirm the running container is on the repo tip — a stale cached
build once served pre-fix code while looking healthy (2026-10-28):

```bash
diff <(docker exec humbelchat-bot cat /app/.build_sha) <(git rev-parse HEAD) \
  && echo "bot on repo tip" || echo "STALE IMAGE — rebuild with --build-arg GIT_SHA=$(git rev-parse HEAD)"
```

- `restart: unless-stopped` — the container self-recovers from crashes/reboots.
- No published ports (outbound WebSocket + UDP only); no healthcheck endpoint —
  judge health from logs.

### Updating the bot

```bash
git pull
docker compose build --build-arg GIT_SHA=$(git rev-parse HEAD) bot  # stamp it
docker compose up -d bot
docker compose logs -f bot     # watch it come up (look for "Bot build sha: …")
# then verify the running container matches the repo tip (stamp diff, above)
```

Discord drops for a few seconds; the bot re-logs in on start. Run `/sync` in the
bot's channel if slash commands misbehave after an update.

**STALE-IMAGE WARNING (as of 2026-10-25, backlog #61):** the running image was
built 2026-10-23 and predates this week's work (tool-calling, turn-logger,
contextual sidecar #52, message-uploads #60, all llama.cpp RAG env changes) —
none of it is live until you rebuild. When you do deploy, decide the ref first
(feat/tool-calling tip vs a curated commit); expect one Discord drop.

### RAG GPU services (reranker + embedder)

`reranker-llama` and `embedder-llama` pin ~6.4 GB of GPU-1 VRAM. To make room
for an image model: `./scripts/gpu_switch.sh image` (bot degrades gracefully to
dense+BM25; the embedder is only needed when indexing new chunks). Restore:
`./scripts/gpu_switch.sh rag` (~19 s to healthy). `status` shows what is
resident. Never run a bare `llama-server.exe` for these models — the compose
services are the only canonical path (a stray bare process from the first
cutover was found LAN-exposed on 0.0.0.0:8090 and killed, 2026-10-25).

### Verifying health (no healthcheck — read the logs)

Give ~10–20 s after a start, then:

```bash
docker compose logs bot --tail 50
```

Good startup sequence: `Logged in as HumbleChat#…` → `Commands synced` → KB file
list → `Preloaded local embedding model` → `Vector index store ready (N chunks)`.
The first backend-health probe can log a transient `DOWN (timeout)` that recovers
on its own (cold backend). If it's really down:

```bash
docker compose exec -T bot sh -c 'curl -sS -m 30 -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $INFER_API_KEY" "$INFER_URL/models"'
# expect 200 — INFER_URL is the unsloth studio API (host.docker.internal:8888)
```

### Knowledge database (direct host access)

`data\knowledge\.vector_index_cache\vector_index.db` — plain SQLite on the host.

- Reading while the bot runs is fine (bot is the sole writer); prefer read-only mode.
- Don't write to it live — `docker compose down` first, or work on a copy.
- New `/upload_kb` rows appear immediately; no restart needed.

### Docker gotchas

- `.env` in Docker: don't set `KB_PATH` / `CHARACTERS_FILE`
  to Windows paths — compose pins them to the bind mounts (`/app/...`).
- `INFER_URL` uses `host.docker.internal` (Docker Desktop resolves it to the
  Windows host) or a LAN IP for remote backends.
- First-ever start downloads models into `hf-cache/` (~2 GB) — slow boot once,
  fast afterwards.
- Bind mounts from the Windows filesystem are slower than native storage;
  reindexing may take longer than on a Linux host.

## Local dev (no Docker) — `./botctl.sh`

One script handles venv setup and a detached tmux session that survives
terminal close:

```bash
./botctl.sh start      # creates .venv if missing, then starts in tmux session 'bot'
./botctl.sh status     # RUNNING (PID …) / NOT RUNNING
./botctl.sh logs       # piped: last 2000 lines; interactive TTY: attaches
./botctl.sh restart    # graceful stop (SIGINT, ~5s) then start
./botctl.sh stop       # SIGINT first, kill-session fallback
./botctl.sh setup      # (re)create .venv + install requirements.txt
tmux attach -t bot     # watch live output directly (Ctrl-b d detaches)
```

### Python environment gotcha

`botctl.sh` runs the venv's python (`./.venv/...`, created by `./botctl.sh setup`).
If startup fails with ImportError, re-run `./botctl.sh setup` to refresh deps, or
run in the foreground: `source .venv/activate && python main.py`.

## Logs (both modes)

- `logs/bot.log` — INFO+, rotated 10 MB × 5
- `logs/dev.log` — DEBUG, rotated 10 MB × 5
- Written by `main.py` itself (RotatingFileHandler); no tee involved. Same files
  in Docker (bind-mounted) and dev runs.

When debugging: read the tail of `dev.log` first (`tail -n 100 logs/dev.log`),
it has the full traceback context. In Docker, `docker compose logs bot` shows the
same stdout the file logger mirrors. Config values are read at import — `.env`
changes require `docker compose up -d` (recreate), not just a restart.

## Runtime data

- `data/` — knowledge base docs, recordings, session notes (git-ignored)
- `.bot.pid` — PID file (dev runs only); a stale one is cleaned by `botctl.sh` automatically
- Slash commands not showing in Discord? Run `/sync` in the bot's channel.

## Required config

`.env` must define at minimum: `DISCORD_BOT_TOKEN`, `INFER_URL`,
`INFER_API_KEY`. Missing/invalid values → login/startup error, visible in
`dev.log`. Never print or commit `.env` contents. In Docker, `INFER_URL` uses
`host.docker.internal` (Docker Desktop resolves it to the Windows host) or the
LAN IP of the machine running the unsloth API.
