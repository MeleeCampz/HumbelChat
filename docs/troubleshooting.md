# Troubleshooting

## Commands missing from the slash command menu

Likely cause: command sync not yet propagated by Discord.

Fix:
- Re-run the bot with `python main.py` or `./start_bot.sh`
- Or wait up to about an hour for Discord to propagate the update
- Verify `DISCORD_BOT_TOKEN` is valid in `.env`

## AI responses not appearing

Likely cause: API timeout or backend issue.

Fix:
- Increase `AI_REQUEST_TIMEOUT` in `.env`
- Check the inference backend logs
- Verify `INFER_URL` and `INFER_API_KEY` if applicable

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
