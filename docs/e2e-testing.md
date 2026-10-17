# E2E testing of the live bot

Two tools let you exercise the **running production pipeline** (dispatch →
RAG → rerank → LLM → reply) without typing in Discord. Both are plain scripts
in `scripts/` — no framework, no extra services.

## 1. In-container emulation harness (`scripts/e2e_emulate.py`)

Runs a REAL registered slash command inside the running container with a mock
interaction — full pipeline against live config/KB/backends, **zero Discord
traffic**. Run from the repo root on the host:

```bash
docker compose exec -T bot python scripts/e2e_emulate.py /ai message="ping"
docker compose exec -T bot python scripts/e2e_emulate.py /item_stats
docker compose exec -T bot python scripts/e2e_emulate.py /character action=list
```

- Options are `key=value` pairs matching the command's parameter names;
  `None` for optional params. `/ai` needs `message=...` (and optionally
  `character=...`).
- Prints every captured output (content + embeds), elapsed time, and exits
  non-zero on error/timeout/no-output — usable as a smoke check.
- Env: `E2E_GUILD_ID` / `E2E_CHANNEL_ID` (fake ids, default `999000001`;
  history entries land under that key — test artifacts), `E2E_TIMEOUT`
  (default 300 s).
- The script ships in the image (`COPY . .`); after a git pull it is present
  on the next `docker compose up -d --build`. For immediate use without a
  rebuild: `docker cp scripts/e2e_emulate.py humbelchat-bot:/app/scripts/`.
- Limitation: commands whose delivery goes through `bot_core.channel_delivery`
  REST fallbacks (e.g. `/remind` posting later) need a logged-in bot and will
  fail loudly here — the harness is for turn-based slash commands.

## 2. Real Discord driver (`scripts/e2e_discord_spike.py`)

Sends REAL traffic to the live bot over the Discord REST API and polls for
its reply in a channel (default `#private-bot`). Runs from anywhere that can
reach discord.com — including this dev sandbox (no Docker involved).

### Prefix mode (the working real-E2E path)

```bash
python scripts/e2e_discord_spike.py --mode prefix --text "ping"
```

Sends `!ai <prompt>` as a plain message; the bot's `on_message` handler runs
the full production pipeline and answers in the channel.

**Caveat:** `on_message` ignores messages authored by the bot itself, so you
need a small **driver bot account**:

1. https://discord.com/developers/applications → *New Application* → e.g. `e2e-driver`.
2. *Bot* tab → reset token → copy it.
3. Invite it: OAuth2 URL generator → scope `bot`, permission
   *Send Messages / Read Message History*, open the URL, pick the server.
4. Run with the driver token (keep it out of `.env`/git — pass inline):

   ```bash
   E2E_DRIVER_TOKEN=<driver-bot-token> python scripts/e2e_discord_spike.py --text "who am I?"
   ```

Exit 0 = the live bot answered; the reply content/embeds are printed.

### Type-2 mode (documented dead end — kept for reference)

`--mode type2 --command help` creates a type-2 (application command) message
via REST. **Verified live 2026-10-17: with a bot token Discord accepts the
message but stores it as a plain empty type-0 message and dispatches NO
interaction to the bot** (the bot was confirmed alive via its vault-sync
activity; no log lines, no reply). Options cannot be carried this way either.
Conclusion: REST-created command messages do not trigger interactions; only
genuine client-side invocations (or prefix messages from a non-bot author) do.

## Running the harness OUTSIDE the container (e.g. this dev sandbox)

The harness can also run against the bind-mounted repo from any machine that
has the deps installed — no Docker needed. Verified 2026-10-17 from the dev
sandbox (`/ai` full turns worked, ~55 s each). Caveats:

- **Embeddings/reranker ARE reachable over IPv4** (verified 2026-10-17):
  `EMBED_URL=http://host.docker.internal:11435/v1` serves
  `qwen3-embedding:4b` and `RERANK_API_BASE=http://host.docker.internal:8000`
  is the Qwen reranker — full production-equivalent runs work from the sandbox
  (real vector index + rerank, no fallback). Plain `curl http://...` may fail
  with `[::1] refused` because dialing tries IPv6 first; force IPv4
  (`curl -4`) or let the HTTP client retry. Under heavy GPU load the embedder
  can hang — set `EMBED_TIMEOUT=10` so batches fall back to local quickly.
- **Local-fallback caveat:** if you DO run with `EMBED_BACKEND=local`, the
  stored vector index was built with `qwen3-embedding:4b`; with a local bge-m3
  query embedder the mismatch guard REJECTS it and attempts a rebuild against
  the live `data/knowledge/.vector_index_cache/vector_index.db`. A failed
  rebuild leaves the file untouched (verified), but a *successful* one would
  overwrite the production index — **back up the DB first** if you do this,
  or just run the harness inside the container (production env vars, no
  override needed).
- **Standalone processes rebuild on EVERY start:** the vector store is loaded
  once per process. A standalone script (harness/probe) that can't reach the
  qwen embedder falls back to local bge-m3 for the QUERY vector → dim mismatch
  vs the stored index → a full reindex attempt on every invocation (tens of
  minutes each, or an endless wait). Production is unaffected (single store,
  loaded at bot startup; if that load fails it degrades to keyword search).
  For long standalone RAG experiments, copy `data/knowledge` elsewhere and run
  with `--kb-path` / `KB_PATH` pointing at the copy so rebuilds hit the copy.
- Retrieval then degrades to keyword fallback (observed: German proper-noun
  queries attach session notes instead of the matching rulebook chapters —
  fine for plumbing verification, not for retrieval-quality testing).
- History entries land under the fake channel id; clean up afterwards with
  `/clear_history` through the harness.

## Environment variables (both tools)

| Var | Meaning |
| --- | --- |
| `E2E_CHANNEL_ID` | target channel id (overrides name discovery) |
| `E2E_CHANNEL_NAME` | channel to discover by name (default `private-bot`) |
| `E2E_TIMEOUT` | seconds to wait for the bot's reply / command run |
| `E2E_DRIVER_TOKEN` | second bot account token for prefix mode |
| `E2E_GUILD_ID`, `E2E_CHANNEL_ID` (harness) | fake ids for the mock interaction |

Tokens are read from `.env` automatically (`DISCORD_BOT_TOKEN`); never print
them.

## Sandbox limitation (important)

The Docker daemon reachable from this dev sandbox is **not** the host's
Docker Desktop — containers on the host (e.g. `humblechat-bot`) are invisible
here and must NOT be started/managed from the sandbox. Anything that needs
`docker compose …` (the emulation harness, restarts, log inspection via docker)
must be run **on the host**. What DOES work from the sandbox: the REST driver
above, and reading `logs/`, `data/` directly (they're bind-mounted into this
checkout).
