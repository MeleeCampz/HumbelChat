"""Slash command handler for /sync — flushes Discord's command cache and re-syncs."""

from __future__ import annotations

import logging

import discord

log = logging.getLogger("bot.commands.sync_command")


async def handle_sync_command(interaction: discord.Interaction) -> None:
    """Re-sync slash commands with Discord.

    Clears any stale or duplicated commands from Discord's cache and
    re-registers the current set. This is the fix for duplicated command
    listings in the slash command menu.
    """
    await interaction.response.defer(ephemeral=True)

    # Clear guild-scoped commands first (if any)
    for guild in interaction.client.guilds or []:
        try:
            await interaction.client.tree.clear_commands(guild=guild)
        except discord.NotFound:
            pass

    # Re-sync global commands
    try:
        await interaction.client.tree.sync()
        log.info(
            "Command sync triggered by %s (ID: %s) — commands re-registered globally.",
            interaction.user,
            interaction.user.id,
        )
        msg = (
            "✅ **Commands synced successfully.**\n\n"
            "Discord's command cache has been flushed and the current command set "
            "has been re-registered globally.\n\n"
            "If you still see duplicated commands, Discord's cache may take a few "
            "minutes to update. Try the `/sync` command again if needed."
        )
    except Exception as e:
        log.error("Command sync failed: %s", e)
        msg = f"❌ **Sync failed:** {e}"

    await interaction.followup.send(msg, ephemeral=True)
