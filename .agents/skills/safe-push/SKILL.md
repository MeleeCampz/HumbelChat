---
name: safe-push
description: Git branch strategy and push rules for HumbelChat — feature branches freely, master/main only with user confirmation via the guard system. Use before committing or pushing changes, or when asked to commit/push work.
---

# Safe push (HumbelChat)

## Branch strategy

- Default workflow: create/use a **feature branch** for changes
  (`git checkout -b feat/<short-topic>`), commit there, push freely:
  `git push -u origin feat/<short-topic>`.
- **Never push to `master`/`main` without explicit user intent.** The pi guard
  extension prompts the user on any such push; a blocked push is expected and
  fine — report it, don't retry or route around it.

## Commit conventions (observed in this repo)

- Prefixes: `fix(scope):`, `feat(scope):`, `chore:`, `docs:`
- Scopes used so far: `tests`, `sessions`, plus bare for cross-cutting work
- One logical change per commit; test fixes get `fix(tests):`

## Guard system (user-level pi extension)

State: `~/.pi/agent/guard-state.json` — path allowlist + branch allowlist.
The user manages it with `/guards` (status, toggle, allow/revoke, reset).

Rules for the agent:
1. If a push to master/main is **blocked by the user**, stop and ask how they
   want to proceed (feature branch + PR? approve manually?). Never re-run the
   same push in a loop.
2. Do not edit `guard-state.json` or run `/guards allow-branch master` unless
   the user explicitly asks for that grant.
3. If the user says "push to master", just do it — the guard will prompt them;
   their approval at the prompt is the authorization.

## Pre-push checklist

1. `git status --short` — only intended files staged (`.pi/`, `.env`, `data/`
   are git-ignored; never force-add secrets).
2. Run the test suite if product code changed: see the `run-tests` skill.
3. Push feature branch first; master pushes happen when the user decides.
