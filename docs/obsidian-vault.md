# Obsidian Vault Sync (#19)

Write your campaign notes in **Obsidian** on any machine and have them show up in
the bot's knowledge base automatically — and read the session logs the bot
generates right back in Obsidian.

## How it works

Each vault is a **(private) GitHub repo**. Machines stay in sync with the free
*Obsidian Git* community plugin; the bot stays in sync with a small background
task (`bot_core/vault_sync.py`) that:

1. clones every configured vault into `<KB_PATH>/<vault-name>/` (once),
2. on a fixed interval (default 5 min) pulls new commits,
3. for the **campaign vault** only: commits + pushes any session notes the bot
   wrote since the last tick,
4. if anything changed, re-indexes only the changed files (the same incremental
   path as `/sync_kb`).

Dot-dirs are never indexed (`.git/`, `.obsidian/`), so vault internals stay out
of RAG. Only text formats (`.md`, `.txt`, …) are indexed — images and PDFs ride
along in git but are invisible to retrieval.

### Vault roles

| Role | Who writes | Behavior |
|---|---|---|
| **One-way vault** (default, any number) | you, in Obsidian | bot mirrors it: `fetch` + hard reset to remote |
| **Campaign vault** (exactly one, optional) | you **and** the bot | two-way: bot pulls your edits, commits + pushes session notes |

Put general reference material (player handbook, rules, …) in its own one-way
vault — it does not belong in the campaign vault. The campaign vault holds the
campaign-specific notes *and* the per-session folders (`session_notes/`).

## Client-side: which Obsidian plugin?

Researched 2026-10. All options below are community plugins that wrap git —
none of them talk to the bot directly, so any of them works with this feature.

| Plugin | What it does | Verdict |
|---|---|---|
| **Obsidian Git** (Vinzent03/obsidian-git) — the de-facto standard | Commit/pull/push from the command palette; **auto-pull on vault open**; scheduled **commit-and-sync** every N minutes (only when there are changes); optional *commit on file change* with delay; status bar showing ahead/behind counts; diff view, file history (pairs well with the *Version History Diff* plugin) | ✅ **Recommended.** Most mature, most configurable, exactly the automation this setup needs. Works on desktop and mobile. |
| Auto Git Sync (alavna/obsidian-auto-git-sync) | Minimal: auto-pull on open + interval commit/push + change detection | Fine if you want the smallest possible plugin; fewer knobs, smaller community |
| Git Sync (git-obsi-sync) | Auto-commit + push **on every save**, pull on open, side-by-side conflict UI | Heavier sync cadence (every save); interesting conflict UI, but more churn in the repo history |

**Recommendation: Obsidian Git**, with these settings per vault:

| Setting | Value | Why |
|---|---|---|
| *Auto pull* (on startup) | on | Pick up bot-pushed session logs + other machines' edits when you open the vault |
| *Commit-and-sync interval* | 10 min (any value ≥ bot interval works) | Pushes your edits so the bot can pull them |
| *Auto commit & push on update* (optional) | off by default; enable with a delay if you want near-instant sync | Avoids committing mid-typing |
| *Show update / status bar* | on | You can see ahead/behind counts at a glance |

Auth on your machines: the plugin uses your system git credentials (Git for
Windows / Git Credential Manager on Windows) — just log in to GitHub once in
git. No tokens needed client-side; tokens are only for the **bot** (below).

## Setup (complete checklist)

### A. Client side (each of your machines)

1. Create a **private GitHub repo per vault** (e.g. `humblewood`,
   `dnd_handbook`) — see “Outstanding questions” for naming/migration.
2. Clone the repo and open it in Obsidian as a vault (`Open folder as vault`).
3. Install the **Obsidian Git** community plugin; set the options above.
4. Write a first note, commit + push from the plugin so each repo has ≥1 commit
   (an empty repo still clones fine on the bot side, but this proves your auth).
5. Repeat per vault and per machine.

### B. Bot side (one-time)

1. Deploy the new bot version (`git pull` + `docker compose up -d --build`) —
   see “What changed in the bot” below for what that brings.
2. Create a **fine-grained personal access token** per repo (GitHub → Settings →
   Developer settings → Fine-grained tokens), scoped to that single repo only:
   - one-way vaults: **Contents: Read-only**
   - campaign vault: **Contents: Read and write**
3. Embed each token in its URL:
   `https://<github-user>:<token>@github.com/<user>/<repo>.git`
4. Set the four `.env` variables (below) and recreate the container:
   `docker compose up -d` (config is read at import).
5. Migrate existing session folders into the campaign vault (section D).
6. Verify: watch `logs/bot.log` for `Vault sync: cloned …`, check
   `/list_kb_docs` shows the vault folders, then edit a note in Obsidian and
   confirm it becomes searchable within one interval.

### C. Tokens & bot `.env`

Tokens were created in step B2. Now put everything in the bot's `.env`:

```dotenv
# Semicolon-separated name=git_url entries (feature is OFF when empty).
# One entry per REPO — a repo may contain several vault/area folders
# (e.g. HumblewoodObsidianVault holds DnDRules/ + HumbleWood/); everything
# under the clone is indexed.
OBSIDIAN_VAULTS=humblewood=https://MeleeCampz:TOKEN@github.com/MeleeCampz/HumblewoodObsidianVault.git
# The two-way repo that holds session notes (one of the names above; empty = all one-way)
OBSIDIAN_CAMPAIGN_VAULT=humblewood
# Point session notes into the campaign vault (Docker path — compose pins KB_PATH
# to /app/data/knowledge; for local dev via botctl.sh use <repo>/data/knowledge/…)
SESSIONS_NOTES_DIR=/app/data/knowledge/humblewood/HumbleWood/SessionLogs
# Pull/commit period in seconds (default 300; 0 = one startup pass only)
OBSIDIAN_VAULT_PULL_INTERVAL=300
```

Then recreate the container: `docker compose up -d` (config is read at import —
a plain `restart` is not enough for `.env` changes).

### D. One-time migration of existing KB content

If the vault replaces folders that previously lived directly in
`data/knowledge/` (rules dumps, worldbuilding notes, session notes):

1. Move the content into the repo (commit from Obsidian or on the bot host).
2. **Delete the old folders from `data/knowledge/`** — move, never copy.
   Leftover copies get indexed twice and retrieval returns duplicates.
3. Point `SESSIONS_NOTES_DIR` at the session folder inside the clone (above);
   the old `data/knowledge/session_notes/` goes away with it.

The first sync tick picks everything up as added files (the index self-heals
via the normal changed-file path). Done for this deployment 2026-10-07:
`HumblewoodObsidianVault` (DnDRules + HumbleWood incl. SessionLogs) replaced
the old `DnD5_5/`, `HumbleWood_GeneralInformation/`, `Trixy/` and
`session_notes/` folders.

## What to expect

- After ~one interval, new/edited notes are RAG-searchable; `/list_kb_docs` shows
  each vault as a KB folder (`📂 humblewood`, `📂 dnd_handbook`).
- Session notes written by the bot (notes, recap, merged session log) appear in
  the campaign repo within one interval — pull in Obsidian on any machine to see
  them. Your edits are pulled back into RAG on the next tick.
- Log lines (`logs/bot.log`): `Vault sync: cloned …`, `Vault sync: <name> pushed
  session-note commit …`, `Vault sync: <names> changed — re-indexed N file(s);
  index now M chunks`.

## Images and attachments

Embedded pictures (`![[map.png]]`, markdown images, PDFs) are a non-problem:

- git syncs binaries fine; if the repo grows large (dozens of MB), enable
  **Git LFS** in the vault repos (`git lfs track "*.png" …`) — the bot's plain
  clone/fetch/push works with LFS transparently.
- RAG skips non-text files entirely, so they can never break chunking.
- The AI sees an image reference as plain text (`![[map.png]]`) — it knows a map
  is referenced but cannot see it. Use alt text / captions in notes so the
  reference stays meaningful.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `clone failed: … could not read Username` / auth errors | Token expired or wrong scope; check the fine-grained token (Contents read / read+write) and that the URL embeds it |
| `pull conflict (local changes kept)` warning | Both you (Obsidian) and the bot changed the same file. Resolve in Obsidian or on the bot host (`git -C data/knowledge/<vault> …`); the next tick retries automatically |
| Notes edited but not searchable yet | Wait one interval, or run `/sync_kb` manually |
| Vault folder missing from `/list_kb_docs` | Check `logs/bot.log` for clone/pull warnings; verify `OBSIDIAN_VAULTS` spelling (names must be simple — no slashes) |
| Moving the bot to a new machine | Nothing special: `.env` + the repos are all that matter — clones re-create themselves on first start |

## What changed in the bot itself

| Change | Where | Notes |
|---|---|---|
| `git` installed in the Docker image | `Dockerfile` | Needed for clone/pull/commit/push; requires a one-time image rebuild (`docker compose up -d --build`) |
| New background task `obsidian-vault-sync` | `bot_core/vault_sync.py` (new), started from `main.py` on boot | No-op when `OBSIDIAN_VAULTS` is empty — existing deployments are unaffected. Cancelled cleanly on shutdown like the other tracked tasks |
| New env vars | `config/settings.py` | `OBSIDIAN_VAULTS`, `OBSIDIAN_CAMPAIGN_VAULT`, `OBSIDIAN_VAULT_PULL_INTERVAL` (default 300 s), `SESSIONS_NOTES_DIR` (default `<KB_PATH>/session_notes` — **unchanged behavior when unset**) |
| Session notes location is now configurable | `bot_core/sessions.py::notes_dir()` | Single helper; all session I/O goes through it. Only moves if you set `SESSIONS_NOTES_DIR` |
| Nothing else touched | — | `/upload_kb`, `/sync_kb`, `/reindex_kb`, RAG pipeline, chunker: unchanged. The vault sync reuses the exact same incremental index path as `/sync_kb` |

Behavioral notes:

- **Off by default.** Without `OBSIDIAN_VAULTS` in `.env` the bot behaves exactly
  as before.
- **Failure-isolated.** A failing vault (bad token, network) is logged and retried
  next tick; it never blocks other vaults or the bot itself.
- **Worst-case latency** for a note to become searchable (or a session log to
  appear in Obsidian) is one `OBSIDIAN_VAULT_PULL_INTERVAL` (default 5 min).
- The bot commits in the campaign vault as `HumbelChat (bot) <humbelchat-bot@localhost>`.

## Setup status (2026-10-07)

- ✅ Repo: `MeleeCampz/HumblewoodObsidianVault` (private) with `DnDRules/`
  (reference area) + `HumbleWood/` (campaign vault, incl. `SessionLogs/`).
- ✅ `.gitignore` for per-machine Obsidian state (workspace.json, plugins/…).
- ✅ Old KB folders removed from `data/knowledge/`; content now lives in the repo.
- ✅ Bot `.env` entries in place (`OBSIDIAN_VAULTS`, `OBSIDIAN_CAMPAIGN_VAULT`,
  `SESSIONS_NOTES_DIR`, interval) **with a working fine-grained PAT**
  (Contents: read+write, this repo only).
- ✅ De-base64'd the large image embeds (`Summery.md`, `Trixy/Images.md`,
  Marvin) — images now live in `Images/` as real files.
- ✅ First index build succeeded (56 files → 2 988 chunks; two batches fell
  back to local embedding when the GPU endpoint returned 500 on the oversized
  base64 chunks — transparent, nothing lost).
- ✅ Fixed post-deploy: `_run_git` returned stderr instead of stdout, so every
  sync tick failed with "cannot determine branch" (clone + first index were
  unaffected). Campaign commits are also scoped to the session subpath and all
  log output is credential-redacted.
- ⏳ **Deploy** — `git pull` the bot branch + `docker compose up -d --build`,
  then confirm in `logs/bot.log` that ticks run *without* the
  "cannot determine branch" warning; the first tick pulls all repo edits made
  while sync was broken and re-indexes the changed files.

## Manual fallback

- `/sync_kb` — re-index changed files without pulling (use after resolving conflicts by hand).
- `/reindex_kb` — full rebuild if the index ever looks wrong.
