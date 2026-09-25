---
name: safe-push
description: Git branch strategy and push rules — feature branches freely, master/main only with explicit user intent. Use before committing or pushing changes, or when asked to commit/push work.
---

# Safe push

## Branch strategy

- Default workflow: create/use a **feature branch** for changes
  (`git checkout -b feat/<short-topic>`), commit there, push freely:
  `git push -u origin feat/<short-topic>`.
- **Never push to `master`/`main` without explicit user intent.** If a push
  to a protected branch is blocked or denied by any guard/prompt, that is
  expected — report it and ask how to proceed (feature branch + PR? approve
  manually?). Never retry in a loop or route around the block.

## Commit conventions

- Prefixes: `fix(scope):`, `feat(scope):`, `chore:`, `docs:`
- One logical change per commit; test fixes get `fix(tests):`

## Rules for the agent

1. A blocked or denied push is a user decision, not an error to fix. Stop and
   ask.
2. Do not modify guard/allowlist configuration files unless the user
   explicitly asks for that specific grant.
3. If the user says "push to master", just do it — any confirmation prompt
   that appears is their authorization point.

## Pre-push checklist

1. `git status --short` — only intended files staged; never force-add
   secrets, env files, or local config directories.
2. Run the test suite if product code changed: see the `run-tests` skill.
3. Push feature branches freely; protected-branch pushes happen only when
   the user decides.
