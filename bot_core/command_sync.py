"""Slash-command syncing: purge stale/duplicate registrations, then re-sync.

The bot registers every slash command **globally** (see ``main.py`` — no command
is ever registered with a ``guild=`` scope). But two properties of discord.py
mean stale registrations can never disappear on their own:

* ``Tree.sync()`` is a **bulk upsert** — it upserts the current command set but
  **never deletes** a command that is no longer in the local tree.
* ``Tree.clear_commands()`` is a **synchronous local cache clear** — it touches
  only in-memory state, never Discord.

So a command that was once registered per-guild (the old ``copy_global_to``
pattern), or that was renamed/removed in code, stays registered on Discord's
side **forever**. Because Discord merges global + per-guild commands in the ``/``
menu, such a stale per-guild command makes the same name appear **twice** — the
"commands are listed two times" bug.

This module is the single place that knows how to actually **remove** stale
registrations from Discord and bring the registered set back in line with the
code. It is used by:

* ``/sync`` (``commands/sync_command.py``) — on demand, by a user.
* startup (``main.on_ready``) — an idempotent cleanup on first run.

It is safe to run repeatedly: deleting an already-absent command raises
``discord.NotFound`` which is swallowed, and re-syncing is idempotent.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discord.ext.commands import Bot

log = logging.getLogger("bot.commands.sync")


async def _delete_quietly(cmd: discord.app_commands.AppCommand) -> bool:
    """Delete one registered command, swallowing the 'already gone' case.

    Returns True if a delete was attempted (found), False if it was already
    absent.
    """
    try:
        await cmd.delete()
        return True
    except discord.NotFound:
        return False
    except discord.HTTPException as e:
        log.warning(
            "Could not delete stale command %r (id=%s): %s",
            getattr(cmd, "name", "?"), getattr(cmd, "id", "?"), e,
        )
        return False


async def purge_guild_commands(bot: "Bot") -> list[tuple[str, list[str]]]:
    """Delete every guild-scoped command for the bot.

    This bot registers **only** globally, so *any* guild-scoped command
    currently registered for the application is, by definition, stale. We
    fetch each guild's registered commands and delete all of them.

    Returns a list of ``(guild_name, [deleted command names])`` for the report.
    """
    results: list[tuple[str, list[str]]] = []
    for guild in bot.guilds or []:
        try:
            registered = await bot.tree.fetch_commands(guild=guild)
        except discord.HTTPException as e:
            log.warning("Could not fetch guild commands for %s: %s", guild.name, e)
            continue
        if not registered:
            continue
        deleted: list[str] = []
        for cmd in registered:
            if await _delete_quietly(cmd):
                deleted.append(cmd.name)
        if deleted:
            results.append((guild.name, deleted))
            log.info(
                "Purged %d stale guild-scoped command(s) from %s: %s",
                len(deleted), guild.name, ", ".join(deleted),
            )
    return results


async def purge_stale_globals(bot: "Bot") -> list[str]:
    """Delete any *global* command that is not in the local (code-defined) tree.

    After ``tree.sync()`` the registered global set is exactly the local set, so
    any global command whose name is not among the synced ones is a stale
    leftover (renamed/removed in code). Returns the names deleted.

    NOTE: ``tree.sync()`` failures (e.g. HTTP 50035 — invalid command payload
    such as a description over 100 characters) are **re-raised**, not
    swallowed. Swallowing them made ``/sync`` report success while the new
    commands were never actually registered.
    """
    synced = await bot.tree.sync()  # may raise HTTPException — propagate it
    live_names = {c.name for c in synced}
    try:
        registered = await bot.tree.fetch_commands()  # guild=None → global
    except discord.HTTPException as e:
        log.warning("Could not fetch global commands: %s", e)
        return []
    deleted: list[str] = []
    for cmd in registered:
        if cmd.name not in live_names:
            if await _delete_quietly(cmd):
                deleted.append(cmd.name)
    if deleted:
        log.info("Purged %d stale global command(s): %s", len(deleted), ", ".join(deleted))
    return deleted


async def sync_commands(bot: "Bot") -> dict:
    """Bring Discord's registered commands back in line with the code.

    1. Delete all stale guild-scoped commands (the bot is global-only).
    2. Re-sync (upsert) the current global command set.
    3. Delete any stale global commands no longer in the code.

    Returns a small report dict for the caller to summarise:

    * ``"guild_names"``: ``[(guild_name, [deleted command names]), ...]``
    * ``"guilds"``: just the guild names (flat convenience view)
    * ``"global_deleted"``: ``[stale global command names]``
    * ``"error"``: ``str`` when the global re-sync itself failed (the command
      set was **not** registered) — callers must surface this to the user and
      must not treat the run as successful.
    """
    guild_purged = await purge_guild_commands(bot)
    # Re-sync the globals *before* comparing, so `purge_stale_globals` sees the
    # current set as the source of truth. A failed upload (invalid payload)
    # must reach the caller — see purge_stale_globals.
    try:
        stale_globals = await purge_stale_globals(bot)
    except discord.HTTPException as e:
        log.error("Global command sync FAILED — new commands are NOT registered: %s", e)
        return {
            "guilds": [name for name, _ in guild_purged],
            "guild_names": guild_purged,
            "global_deleted": [],
            "error": str(e),
        }
    return {
        "guilds": [name for name, _ in guild_purged],
        "guild_names": guild_purged,
        "global_deleted": stale_globals,
    }
