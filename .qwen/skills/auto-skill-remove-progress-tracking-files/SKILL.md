---
name: remove-progress-tracking-files
description: Replace scattered TODO/progress tracking files in repo root with a local-only todo/ folder behind .gitignore
source: auto-skill
extracted_at: '2026-07-12T13:24:36.283Z'
---

## Scenario

The project has multiple scattered progress-tracking files in the repo root (e.g., `ISSUES_AND_TODOS.md`, `TODOs.txt`, `TODO_next_chat.md`) that pollute version control and don't belong there. These track local AI work, pending server-side actions, or development state — all meant to be ephemeral and private.

## Procedure

### 1. Assess what's still relevant

Read each progress-tracking file and cross-reference with:
- **Git log** (`git log --oneline | head -20`) — verify claims of "done" against actual commits
- **Current repo state** — check if referenced files/features still exist in the working tree
- **Working tree** — note what's actually staged vs committed

Mark items as: done, redundant (already covered by another file), or pending user action.

### 2. Consolidate into `todo/`

Create a local-only directory and migrate only actionable, non-stale items:

```bash
mkdir -p todo
```

Create `todo/README.md`:
```markdown
# Local AI Working Tasks

Files in this directory are **not tracked** by git — they exist solely for local AI-assisted development. Use them to track actionable items that shouldn't be exposed on GitHub (credentials, private notes, work-in-progress checklists, etc.).

_The contents of this directory are managed by your AI assistant between sessions._
```

Create a `todo/pending-tasks.md` with the remaining actionable items, grouped by context.

### 3. Update `.gitignore`

Add the exclusion:
```text
# Local AI working tasks — for development-only notes, not tracked
todo/
```

### 4. Remove old files from repo tracking

For any progress-tracking file that was previously tracked in git:
```bash
git rm --cached ISSUES_AND_TODOS.md TODOs.txt TODO_next_chat.md
```

If `git add` blocked by `.gitignore`, commit with `--force` for those specific files as needed (they were already added; you're removing them).

### 5. Commit and push

```bash
git add .gitignore todo/
git commit -m "chore: remove progress tracking files from repo, move to local-only todo/ directory"
git push origin <branch>
```

## Key rules

- **Verify before deleting** — confirm tasks marked "done" actually have corresponding git commits
- **Move, don't discard** — pending items go into `todo/`; only fully completed or stale content is dropped
- **`todo/` must stay behind `.gitignore`** — never track progress files in version control
- **Root README/docs are the source of truth** for user-facing documentation; internal working notes live in `todo/` only
