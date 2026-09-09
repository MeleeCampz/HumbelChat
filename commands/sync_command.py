"""Slash command handler for /sync — purges stale commands and re-syncs.

The real work lives in :mod:`bot_core.command_sync`. The key insight (the reason
``/sync`` used to appear to do nothing): ``Tree.sync()`` only *upserts* and
``Tree.clear_commands()`` only clears the *local* cache — neither can remove a
registration from Discord. This handler instead deletes the stale registrations
(directly answering "are they registered two times?" → yes, and here's the fix).
"""

from __future__ import annotations

import logging

import discord

from bot_core import command_sync

log = logging.getLogger("bot.commands.sync_command")


async def handle_sync_command(interaction: discord.Interaction) -> None:
    """Purge stale/duplicate command registrations and re-register the current set.

    - Deletes every **guild-scoped** command (this bot is global-only, so any
      per-guild registration is a stale leftover that causes duplicate listings).
    - Re-syncs (upserts) the current **global** command set.
    - Deletes any stale **global** command no longer present in the code.
    """
    await interaction.response.defer()
    client = interaction.client

    try:
        report = await command_sync.sync_commands(client)
    except Exception as e:  # noqa: BLE001 - report any failure to the user
        log.exception("Command sync failed")
        await interaction.followup.send(f"❌ **Sync failed:** {e}")
        return

    # A failed global upload (invalid payload — e.g. a description over 100
    # characters) means NOTHING was registered; never report that as success.
    if report.get("error"):
        await interaction.followup.send(
            f"❌ **Command sync FAILED — commands were NOT updated.\n\n**{report['error']}**\n\n"
            "Fix the offending command in the code (check names/descriptions "
            "against Discord's limits), then restart the bot or run `/sync` again."
        )
        return

    removed = [
        f"  • `{n}`"
        for _, names in report["guild_names"]
        for n in names
    ]
    stale_global = report["global_deleted"]

    lines: list[str] = []
    if removed:
        lines.append(f"🧹 Removed {len(removed)} stale guild-scoped command(s):")
        lines.extend(removed)
    if stale_global:
        lines.append(f"🧹 Removed {len(stale_global)} stale global command(s): "
                     + ", ".join(f"`{n}`" for n in stale_global))
    if not removed and not stale_global:
        lines.append("No stale or duplicate commands found — registrations already clean.")

    body = (
        "✅ **Commands synced.**\n\n"
        "Stale/duplicate registrations were removed from Discord and the current "
        "command set was re-registered globally:\n"
        + "\n".join(lines)
        + "\n\n"
        "If a command still looks duplicated in the `/` menu, Discord's client-side "
        "cache can take a few minutes to refresh. Run `/sync` again if needed."
    )
    log.info("Command sync by %s (ID: %s): removed guild=%s global=%s",
             interaction.user, interaction.user.id,
             report["guild_names"], stale_global)
    await interaction.followup.send(body)
