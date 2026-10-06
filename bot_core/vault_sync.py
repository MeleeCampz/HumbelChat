"""Obsidian vault sync (#19) — git-backed KB vaults, one-way or two-way.

The user keeps their Obsidian vaults as (private) GitHub repos, synced between
machines with the Obsidian Git plugin.  This module mirrors those repos into
the knowledge base and keeps the RAG index in step:

- Every configured vault is cloned into ``<KB_PATH>/<vault-name>/`` (dot-dirs
  like ``.git/`` / ``.obsidian/`` are already skipped by the KB indexer).
- **One-way vaults** (the default): pure mirrors — ``fetch`` + hard reset to
  the remote; the bot never writes into them.
- **Campaign vault** (``OBSIDIAN_CAMPAIGN_VAULT``, exactly one, two-way): the
  session-notes dir is pointed inside it via ``SESSIONS_NOTES_DIR``.  Each tick
  pulls the user's Obsidian edits, then commits + pushes any local changes the
  bot made (session notes / recaps / merged logs), so those appear in Obsidian
  on every machine.

A single tracked background task (``vault_sync_loop``) does all of this on a
fixed interval; after any vault change it re-indexes only changed files via
the existing ``kb.retrievers.sync_kb_store`` path.  Git failures are logged and
retried next tick — they never crash the loop or the bot.

No new dependencies: ``subprocess`` + ``asyncio`` only, all git calls off the
event loop (P0 #4 convention).
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import subprocess

log = logging.getLogger("bot.vault_sync")

#: Wall-clock cap for any single git call (clone/fetch/pull/push).
GIT_TIMEOUT = 120.0

#: Local identity used when committing in the two-way campaign vault.
_BOT_GIT_NAME = "HumbelChat (bot)"
_BOT_GIT_EMAIL = "humbelchat-bot@localhost"


def _run_git(args: list[str], cwd: pathlib.Path) -> tuple[int, str]:
    """Run ``git <args>`` in *cwd*; return ``(returncode, stderr_tail)``.

    Synchronous — always call via :func:`asyncio.to_thread`.  Timeouts and
    missing git binaries are reported as non-zero returns, never raised.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return 1, f"git {' '.join(args)} timed out after {GIT_TIMEOUT:.0f}s"
    except (OSError, FileNotFoundError) as exc:
        return 1, f"git unavailable: {exc}"
    stderr = (proc.stderr or "").strip()
    if len(stderr) > 500:
        stderr = "…" + stderr[-500:]
    return proc.returncode, stderr


async def _git(args: list[str], cwd: pathlib.Path) -> tuple[int, str]:
    """Off-loop wrapper around :func:`_run_git`."""
    return await asyncio.to_thread(_run_git, args, cwd)


def parse_vaults(raw: str) -> dict[str, str]:
    """Parse ``OBSIDIAN_VAULTS`` into an ordered ``{name: git_url}`` mapping.

    Format: semicolon-separated ``name=git_url`` entries, e.g.
    ``humblewood=https://…;dnd_handbook=https://…``.  Blank and malformed
    entries (no ``=``, empty name/url) are skipped with a warning.  Names that
    would escape the KB dir (path separators / ``..``) are rejected — each
    vault is cloned to ``<KB_PATH>/<name>/``.
    """
    vaults: dict[str, str] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, url = entry.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            log.warning("Vault sync: skipping malformed OBSIDIAN_VAULTS entry: %r", entry)
            continue
        if ".." in pathlib.PurePosixPath(name).parts or "/" in name or "\\" in name:
            log.warning("Vault sync: rejecting unsafe vault name: %r", name)
            continue
        vaults[name] = url
    return vaults


async def ensure_cloned(url: str, dir_path: pathlib.Path, *, two_way: bool = False) -> str | None:
    """Clone *url* into *dir_path* if not already a git checkout.

    Returns ``None`` on success (already present or freshly cloned), otherwise
    an error string for the caller to log and retry next tick.  For two-way
    vaults a local commit identity is configured so session-note commits work
    without any host-level git config.
    """
    if (dir_path / ".git").exists():
        return None
    dir_path.parent.mkdir(parents=True, exist_ok=True)
    rc, err = await _git(["clone", "--quiet", url, str(dir_path)], dir_path.parent)
    if rc != 0:
        return f"clone failed: {err or 'unknown error'}"
    log.info("Vault sync: cloned %s → %s", url.split("@")[-1][:80], dir_path)
    if two_way:
        for args in (
            ["config", "user.name", _BOT_GIT_NAME],
            ["config", "user.email", _BOT_GIT_EMAIL],
        ):
            rc, err = await _git(args, dir_path)
            if rc != 0:
                log.warning("Vault sync: git config failed in %s: %s", dir_path, err)
    return None


async def mirror_pull(dir_path: pathlib.Path) -> tuple[bool, bool, str | None]:
    """One-way vault tick: hard-reset the checkout to the remote HEAD.

    The bot never writes into one-way vaults, so ``reset --hard`` is always
    safe.  Returns ``(ok, changed, error)`` — *changed* means the HEAD moved
    (or local drift was discarded), i.e. the KB index may need a refresh.
    """
    rc, branch = await _git(["rev-parse", "--abbrev-ref", "HEAD"], dir_path)
    if rc != 0 or not branch.strip():
        return False, False, f"cannot determine branch: {branch}"
    branch = branch.strip()

    before_rc, before = await _git(["rev-parse", "HEAD"], dir_path)
    if before_rc != 0:
        return False, False, f"cannot read HEAD: {before}"

    rc, err = await _git(["fetch", "--quiet", "origin", branch], dir_path)
    if rc != 0:
        return False, False, f"fetch failed: {err or 'unknown error'}"

    rc, err = await _git(["reset", "--hard", "--quiet", "FETCH_HEAD"], dir_path)
    if rc != 0:
        return False, False, f"reset failed: {err or 'unknown error'}"

    _, after = await _git(["rev-parse", "HEAD"], dir_path)
    changed = before.strip() != after.strip()
    if changed:
        log.info("Vault sync: %s updated to %s", dir_path.name, after.strip()[:8])
    return True, changed, None


def sessions_scope(vault_dir: pathlib.Path, sessions_dir: pathlib.Path) -> str | None:
    """Relative git pathspec of *sessions_dir* inside *vault_dir*, or ``None``.

    Used to scope the campaign vault's commit to the session-notes subpath so
    bot commits only ever contain session data.  Returns ``"."`` when the two
    are the same directory; ``None`` when the sessions dir is outside the
    vault (misconfiguration → caller falls back to whole-tree commits).
    """
    try:
        rel = pathlib.Path(sessions_dir).resolve().relative_to(
            pathlib.Path(vault_dir).resolve()
        )
    except ValueError:
        return None
    return rel.as_posix() or "."


async def campaign_sync(
    dir_path: pathlib.Path, sessions_relpath: str | None = None
) -> tuple[bool, bool, str | None]:
    """Two-way (campaign vault) tick: pull remote edits, push local commits.

    Order per tick:
      1. ``fetch`` + ``pull --rebase --autostash`` — bring in the user's
         Obsidian edits from other machines.
      2. If the worktree is dirty (the bot wrote session notes since the last
         tick) → ``add -A`` + commit + push.  When *sessions_relpath* is given
         the status check and ``add`` are scoped to that subpath, so local
         changes anywhere else in the vault are never committed by the bot.
         A rejected push gets one retry after a fresh rebase; a conflicting
         rebase is aborted, local changes are KEPT, and the tick ends with a
         warning (retried next tick).

    Returns ``(ok, index_changed, error)`` — *index_changed* covers both
    pulled remote changes and locally committed notes.
    """
    scope_args = ["--", sessions_relpath] if sessions_relpath else []
    rc, branch = await _git(["rev-parse", "--abbrev-ref", "HEAD"], dir_path)
    if rc != 0 or not branch.strip():
        return False, False, f"cannot determine branch: {branch}"
    branch = branch.strip()

    _, before = await _git(["rev-parse", "HEAD"], dir_path)

    # 1. Pull remote edits (user's Obsidian commits from other machines).
    rc, err = await _git(["fetch", "--quiet", "origin", branch], dir_path)
    if rc != 0:
        return False, False, f"fetch failed: {err or 'unknown error'}"
    rc, err = await _git(
        ["pull", "--rebase", "--autostash", "--quiet", "origin", branch], dir_path
    )
    if rc != 0:
        # Conflicting rebase (e.g. both sides touched the same note): abort,
        # keep our local changes, warn — retried on the next tick.
        await _git(["rebase", "--abort"], dir_path)
        log.warning(
            "Vault sync: %s pull conflict — kept local changes, will retry; "
            "resolve in Obsidian or on the bot host if it persists: %s",
            dir_path.name, err,
        )
        return False, False, f"pull conflict (local changes kept): {err}"

    # 2. Commit + push anything the bot wrote since the last tick.
    rc, status = await _git(["status", "--porcelain", *scope_args], dir_path)
    if rc != 0:
        return False, False, f"status failed: {status}"
    pushed = False
    if status.strip():
        for args in (
            ["add", "-A", *scope_args],
            ["commit", "--quiet", "-m", "session notes update"],
        ):
            rc, err = await _git(args, dir_path)
            if rc != 0:
                return False, True, f"commit failed: {err or 'unknown error'}"
        rc, err = await _git(["push", "--quiet", "origin", branch], dir_path)
        if rc != 0:
            # Remote advanced meanwhile → rebase and retry once.
            rc2, err2 = await _git(
                ["pull", "--rebase", "--autostash", "--quiet", "origin", branch], dir_path
            )
            if rc2 != 0:
                await _git(["rebase", "--abort"], dir_path)
                log.warning(
                    "Vault sync: %s push conflict — local commit kept, will retry "
                    "next tick; resolve in Obsidian or on the bot host if it persists: %s",
                    dir_path.name, err2,
                )
                return False, True, f"push conflict (local commit kept): {err2}"
            rc, err = await _git(["push", "--quiet", "origin", branch], dir_path)
            if rc != 0:
                log.warning("Vault sync: %s push failed: %s", dir_path.name, err)
                return False, True, f"push failed: {err or 'unknown error'}"
        pushed = True
        log.info("Vault sync: %s pushed session-note commit to origin/%s", dir_path.name, branch)

    _, after = await _git(["rev-parse", "HEAD"], dir_path)
    index_changed = before.strip() != after.strip() or bool(status.strip())
    if index_changed and not pushed:
        log.info("Vault sync: %s pulled remote changes", dir_path.name)
    return True, index_changed, None


async def vault_sync_loop() -> None:
    """Background loop: sync every configured vault, then re-index once.

    Runs as a tracked task (cancelled on bot shutdown).  A failing vault is
    logged and skipped — it never blocks the others or crashes the loop.
    With ``OBSIDIAN_VAULT_PULL_INTERVAL`` ≤ 0 it does a single startup pass.
    """
    from config.settings import (
        KB_PATH,
        OBSIDIAN_CAMPAIGN_VAULT,
        OBSIDIAN_VAULTS,
        OBSIDIAN_VAULT_PULL_INTERVAL,
    )

    vaults = parse_vaults(OBSIDIAN_VAULTS)
    if not vaults:
        return  # defensive — start_vault_sync() already guards this

    campaign = OBSIDIAN_CAMPAIGN_VAULT.strip()
    if campaign and campaign not in vaults:
        log.warning(
            "Vault sync: OBSIDIAN_CAMPAIGN_VAULT %r is not in OBSIDIAN_VAULTS — "
            "treating all vaults as one-way", campaign,
        )
        campaign = ""

    # Scope the campaign vault's commits to the session-notes subpath so bot
    # commits only ever contain session data (None = whole-tree fallback).
    scope: str | None = None
    if campaign:
        from config.settings import SESSIONS_NOTES_DIR
        scope = sessions_scope(pathlib.Path(KB_PATH) / campaign, pathlib.Path(SESSIONS_NOTES_DIR))
        if scope is None:
            log.warning(
                "Vault sync: SESSIONS_NOTES_DIR (%s) is outside the campaign vault "
                "(%s) — committing whole-tree changes instead",
                SESSIONS_NOTES_DIR, KB_PATH / campaign,
            )

    interval = OBSIDIAN_VAULT_PULL_INTERVAL
    log.info(
        "Vault sync: started (%d vault(s)%s, every %ss)",
        len(vaults),
        f", two-way: {campaign}" if campaign else "",
        interval if interval > 0 else "startup-only",
    )

    while True:
        changed_names: list[str] = []
        for name, url in vaults.items():
            vdir = pathlib.Path(KB_PATH) / name
            err = await ensure_cloned(url, vdir, two_way=(name == campaign))
            if err is not None:
                log.warning("Vault sync: %s: %s (retrying next tick)", name, err)
                continue
            if name == campaign:
                ok, changed, err = await campaign_sync(vdir, sessions_relpath=scope)
            else:
                ok, changed, err = await mirror_pull(vdir)
            if not ok and err:
                log.warning("Vault sync: %s: %s (retrying next tick)", name, err)
            elif changed:
                changed_names.append(name)

        if changed_names:
            try:
                from kb.retrievers import sync_kb_store
                idx, report = await sync_kb_store()
                count = idx.count() if idx is not None else 0
                log.info(
                    "Vault sync: %s changed — re-indexed %s file(s); index now %s chunk(s)",
                    ", ".join(changed_names),
                    report.get("changed_count", 0),
                    f"{count:,}",
                )
            except Exception:
                log.exception("Vault sync: KB re-index after vault changes failed")

        if interval <= 0:
            break
        await asyncio.sleep(interval)


def start_vault_sync() -> None:
    """Entry point called from ``main.py`` on startup.

    No-op (with an info log) when ``OBSIDIAN_VAULTS`` is empty; otherwise
    spawns the tracked background loop.
    """
    from config.settings import OBSIDIAN_VAULTS
    from utils.background_tasks import spawn_tracked_task

    if not parse_vaults(OBSIDIAN_VAULTS):
        log.info("Obsidian vault sync disabled (OBSIDIAN_VAULTS empty)")
        return
    spawn_tracked_task(vault_sync_loop(), name="obsidian-vault-sync")


__all__ = [
    "GIT_TIMEOUT",
    "parse_vaults",
    "ensure_cloned",
    "mirror_pull",
    "sessions_scope",
    "campaign_sync",
    "vault_sync_loop",
    "start_vault_sync",
]
