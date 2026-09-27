---
name: docker-ops
description: Operate the production HumbelChat Discord bot container on Docker Desktop — start/stop/restart/update via docker compose, read logs, verify health, access the knowledge database. Use when asked to start, stop, restart, update, or troubleshoot the LIVE bot.
---

# Docker ops (production bot)

The live bot runs as a single container from this repo's checkout
(`C:\Repos\GitHub_Me\HumbelChat`), defined in `docker-compose.yml`. All runtime
state is bind-mounted from the repo — nothing important lives inside the image:

- `data/` — KB docs, `.vector_index_cache/vector_index.db`, recordings, sessions/reminders JSON
- `logs/` — `bot.log` (INFO) + `dev.log` (DEBUG), rotated 10 MB × 5
- `hf-cache/` — HuggingFace model cache (bge-m3 ~2.2 GB, faster-whisper ~1.6 GB)
- `.env`, `characters.json` — config

Full docs: `docs/docker.md`.

## Control (always from the repo root)

```bash
docker compose up -d --build   # start; also the UPDATE flow after git pull
docker compose ps              # status ("Up … (healthy-ish)" — there is no healthcheck)
docker compose logs -f bot     # live stdout
docker compose restart bot     # restart
docker compose down            # stop (host data untouched)
```

- `restart: unless-stopped` — the container self-recovers from crashes/reboots.
- **One process at a time:** never run a local dev session (`python main.py` /
  `botctl.sh`) while the container is up — same Discord token, shared `data/`.
- No published ports (outbound WebSocket + UDP only); no healthcheck endpoint —
  judge health from logs.

## Updating the bot

```bash
git pull
docker compose up -d --build   # rebuilds image, recreates container
docker compose logs -f bot     # watch it come up
```

Discord drops for a few seconds; the bot re-logs in on start. Run `/sync` in the
bot's channel if slash commands misbehave after an update.

## Verifying health (no healthcheck — read the logs)

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

## Debugging

- File logs are the source of truth: `logs/dev.log` has full tracebacks.
  `tail -n 100 logs/dev.log` (Windows: `Get-Content logs\dev.log -Tail 100`).
- `docker compose logs bot` shows the same stdout the file logger mirrors.
- Config values are read at import — `.env` changes require
  `docker compose up -d` (recreate), not just a restart of the process.

## Knowledge database (direct host access)

`data\knowledge\.vector_index_cache\vector_index.db` — plain SQLite on the host.

- Reading while the bot runs is fine (bot is the sole writer); prefer read-only mode.
- Don't write to it live — `docker compose down` first, or work on a copy.
- New `/upload_kb` rows appear immediately; no restart needed.

## Gotchas

- `.env` in Docker: don't set `KB_PATH` / `RECORDINGS_DIR` / `CHARACTERS_FILE`
  to Windows paths — compose pins them to the bind mounts (`/app/...`).
- `INFER_URL` uses `host.docker.internal` (Docker Desktop resolves it to the
  Windows host) or a LAN IP for remote backends.
- First-ever start downloads models into `hf-cache/` (~4 GB) — slow boot once,
  fast afterwards. Same for the first voice→STT transcription (whisper model).
- Bind mounts from the Windows filesystem are slower than native storage;
  reindexing may take longer than on a Linux host.
- Never print or commit `.env` contents.
