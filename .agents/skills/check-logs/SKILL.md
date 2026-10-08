---
name: check-logs
description: Find and read bot errors in its logs — locate log files, grep recipes (incl. rotated files), extract tracebacks. Use when something misbehaves or when asked to check/inspect/debug the logs.
---

# Check logs

`bot-ops` covers starting/stopping; this skill is about **reading what went wrong**.

## Where & how

- `logs/dev.log` — DEBUG, **start here**; `logs/bot.log` — INFO. Rotated 10 MB × 5 → e.g. `bot.log.1 … bot.log.5`.
- Format: `timestamp [LEVEL] logger.name: message`; tracebacks follow `[ERROR]` lines indented.
- Same files in Docker and dev runs; Docker also mirrors stdout (`docker compose logs bot`).

```bash
tail -n 100 logs/dev.log                          # most recent
grep -n "ERROR" logs/dev.log | tail -20           # recent errors
grep -n "ERROR\|WARNING" logs/bot.log*            # across rotated files
grep -n -A 30 "AI request failed" logs/dev.log    # error + its traceback
```

## Workflow

Bound the incident time window in `dev.log` → grep ERRORs + pull their tracebacks → read
the traceback/module names to classify (backend vs config vs sync vs code bug) → report
with exact log lines as evidence.
