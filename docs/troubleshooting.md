# Troubleshooting

## Commands missing from the slash command menu

Likely cause: command sync not yet propagated by Discord.

Fix:
- Re-run the bot with `python main.py` or `./start_bot.sh`
- Or wait up to about an hour for Discord to propagate the update
- Verify `DISCORD_BOT_TOKEN` is valid in `.env`

## Commands duplicated in the slash command menu

Likely cause: bot was restarted multiple times and previously synced commands on every reconnect (pre-fix behavior), causing duplicates to accumulate in Discord's cache.

Fix:
- Run `/sync` to flush Discord's command cache and re-register the current command set
- Wait a few minutes for Discord to update its cache
- If duplicates persist, try `/sync` again after a short wait

Note: The bot now uses a one-time sync on first startup (tracked via `.commands_synced` marker file) to prevent this issue going forward. If you need to force a fresh sync (e.g., after adding new commands), delete the `.commands_synced` file and restart the bot, or just run `/sync`.

## AI responses not appearing

Likely cause: API timeout or backend issue.

Fix:
- Look for the startup health-check line in `bot.log`: `AI backend health check at startup: OK/DOWN`. A `DOWN` result means the bot cannot reach `INFER_URL` — fix connectivity before anything else.
- Increase `AI_REQUEST_TIMEOUT` in `.env`
- Check the inference backend logs
- Verify `INFER_URL` and `INFER_API_KEY` if applicable

The health probe performs a lightweight `GET /models` request. Any HTTP response counts as "reachable"; only timeouts and connection failures mark the backend as down. Set `AI_HEALTH_CHECK_INTERVAL=60` (seconds) to keep probing in the background so `bot.log` records when the backend drops and recovers.

## Duplicate log lines

Likely cause: logging handlers attached to both the `bot` logger and the root logger, or `discord.py`'s own logger propagating upward.

Fix:
- The bot now attaches all handlers directly to the `bot` logger with `propagate=False`, and gives the `discord` logger its own console-only handler (see `main.py`). If you add custom logging, keep the same pattern: one handler chain, no propagation to root.

## `characters.json not found` warning

Likely cause: missing or misnamed file.

Fix:
- Ensure `characters.json` exists at the project root
- Use `characters.json.example` as a template
- Check that the file is valid JSON

## Knowledge base files not loading

Likely cause: wrong path or unsupported format.

Fix:
- Check `KB_PATH` points to the correct directory
- Use supported file types: `.txt`, `.md`, `.csv`, `.html`, `.xml`, `.rtf` (the bot may still accept other files via MIME-based inference or a `.txt` default, but unsupported extensions are not reliably read or indexed)

## Vector search returns no results

Likely cause: embedding backend unreachable or index empty.

Fix:
- Verify `INFER_URL` and `INFER_API_KEY`
- Try `RAG_RETRIEVAL_METHOD=keyword` as a fallback
- Reindex with `/reindex_kb` if needed

## Double bot instance error

Likely cause: stale PID or port conflict from a previous run.

Fix:
- Use `./start_bot.sh`; it auto-kills stale instances via PID file
- If needed, manually remove the stale PID/log state from the bot's runtime directory

## New commands not appearing after code changes

Likely cause: bot was restarted but commands weren't re-synced, or Discord hasn't propagated the update yet.

Fix:
- Run `/sync` to force a re-registration of all commands
- Wait up to an hour if the sync command itself doesn't appear immediately
- Alternatively, delete the `.commands_synced` marker file and restart the bot to trigger a fresh sync on startup
