# Docker deployment

The production bot runs in its own container via Docker Desktop. Everything is
driven from the repo root — the image holds only code + dependencies, all
runtime state lives in bind-mounted host directories.

## Layout

```
<repo>/                        # this checkout (e.g. C:\Repos\GitHub_Me\HumbelChat)
├── docker-compose.yml         # service definition
├── Dockerfile                 # python:3.12-slim + CPU torch + requirements.txt
├── .env                       # all bot config (env_file), incl. INFER_URL
├── characters.json            # personas (bind-mounted read path)
├── data/                      # KB docs, .vector_index_cache/, recordings, history.json
├── logs/                      # bot.log (INFO) + dev.log (DEBUG), rotated 10 MB × 5
└── hf-cache/                  # HuggingFace model cache (bge-m3, faster-whisper) — ~4 GB
```

## Operations

```bash
docker compose up -d --build   # build (if needed) + start; also the update flow after git pull
docker compose ps              # status
docker compose logs -f bot     # live stdout (file logs in logs/ are the source of truth)
docker compose restart bot     # restart
docker compose down            # stop (data is untouched — it's all on the host)
```

- `restart: unless-stopped` brings the container back after a crash or reboot.
- There is no HTTP endpoint, so no healthcheck — judge health from logs
  ("logged in as …", backend probe settling UP). Give ~10–20 s after a start
  before judging; bge-m3 loads from `hf-cache/` in seconds and the first
  backend-health probe can log a transient `DOWN (timeout)` that recovers.
- The container needs no published ports: the Discord gateway is an outbound
  WebSocket and voice uses outbound UDP.

## Config / `.env`

The host `.env` is passed through as-is (`env_file`). Container-specific notes:

- `INFER_URL` — the unsloth studio API. From Docker Desktop,
  `host.docker.internal` resolves to the Windows host (predefined; no
  `extra_hosts` needed). If the API runs on another machine, use its LAN IP.
- `KB_PATH`, `RECORDINGS_DIR`, `CHARACTERS_FILE` are pinned to the bind mounts
  in `docker-compose.yml` — don't set them to Windows paths in `.env`.
- Model cache: `HF_HOME=/models/hf` is set in the image, backed by `./hf-cache`.
  First start downloads the models (~4 GB); later starts load from cache.

## Direct access to the knowledge database

The vector index / KB state is a plain SQLite file on the host:

```
data\knowledge\.vector_index_cache\vector_index.db
```

Open it with any SQLite tool (DB Browser for SQLite, the `sqlite3` CLI, or
Python's built-in `sqlite3` module). Rules of
thumb:

- **Reading while the bot runs is fine.** The bot is the sole writer; use
  read-only mode if your tool supports it.
- **Don't write to it live.** To analyze or repair, stop the container first
  (`docker compose down`) or work on a copy.
- After `/upload_kb` / `/sync_kb`, new rows appear in this DB immediately —
  no restart needed.

## Updating the bot

```bash
git pull
docker compose up -d --build   # rebuilds the image, recreates the container
docker compose logs -f bot     # watch it come up
```

Discord drops for a few seconds during the recreate; the bot re-logs in on
start. Run `/sync` in the bot's channel if slash commands misbehave after an
update.

### Build speed

- Code-only changes: everything is layer-cached, rebuild takes ~1 s.
- `requirements.txt` changes: pip wheels (incl. the ~900 MB CPU torch) are
  served from a daemon-side BuildKit cache mount — no re-download.
- The `.dockerignore` keeps the uploaded build context small (no venvs,
  caches, or runtime state). If you add big directories to the repo, add them
  there too — the whole context is transferred on every build.
- A truly fresh machine (empty Docker cache) still pays one full download:
  base image + torch + requirements ≈ a few GB.

## Caveats

- **One process at a time.** Don't run a local `python main.py` dev session
  while the container is up — two processes would log into the same Discord
  token and fight over `data/`. Stop one before starting the other.
- The bot is only up while this machine + Docker Desktop run.
- Bind mounts from the Windows filesystem are slower than native storage; model
  load / reindexing may take a bit longer than on a Linux host.

## Local development (no Docker)

The tmux-based control script works for dev runs on any platform — it creates
the venv automatically if missing, so a fresh clone needs just one command:

```bash
./botctl.sh start   # sets up .venv if needed, then launches in a detached tmux session
```

See [Configuration](./configuration.md) for all env vars.
