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
