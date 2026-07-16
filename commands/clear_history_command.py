"""Clear history slash command."""
from __future__ import annotations

import discord
from bot_core import clear_history


async def handle_clear_history_command(
    interaction: discord.Interaction,
) -> None:
    """Clear conversation history for this channel."""
    guild_id = interaction.guild_id or 0
    cid      = interaction.channel_id
    await clear_history(guild_id, cid)
    await interaction.response.send_message("Conversation history cleared.", ephemeral=True)
