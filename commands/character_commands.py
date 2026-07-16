"""Character management slash commands."""
from __future__ import annotations

import discord
import discord.app_commands as app_commands
from config.characters import _CHARACTERS, get_character, default_character


async def handle_character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
) -> None:
    """Handle the /character slash command."""
    from bot_core import set_active_char_key, get_active_char_key

    await interaction.response.defer(ephemeral=True)

    active_key = default_character().key
    for c in _CHARACTERS:
        if c.key == get_active_char_key(interaction.guild_id, interaction.channel_id):
            active_key = c.key
            break

    current_char = get_character(active_key)

    if action == "list":
        lines = ["**Available characters:**\n"]
        for char in _CHARACTERS:
            marker = f" ← current" if char.key == active_key else ""
            lines.append(f"  • `{char.key}` — display: `{char.display or char.key}`{marker}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    elif action == "set":
        if name is None:
            await interaction.followup.send("Please provide a character key: `/character set <name>`", ephemeral=True)
            return
        char_obj = get_character(name)
        if char_obj is None:
            # Try matching by key explicitly
            for c in _CHARACTERS:
                if getattr(c, "key", "") == name:
                    char_obj = c
                    break
        if char_obj is None:
            avail = ", ".join(f"`{c.key}`" for c in _CHARACTERS)
            await interaction.followup.send(
                f"Unknown character ``{name}``. Available: {avail}", ephemeral=True
            )
            return
        if interaction.guild_id is not None:
            set_active_char_key(interaction.guild_id, interaction.channel_id, char_obj.key)
        await interaction.followup.send(
            f"Switched to **{char_obj.display}** (model: ``{char_obj.model or '(none set)'}``)", ephemeral=True
        )

    elif action == "show":
        display = current_char.display if current_char else "Default"
        model   = current_char.model if current_char else "(not set)"
        await interaction.followup.send(
            f"**Current character:** `{display}`\n**Model:** ``{model}``", ephemeral=True
        )

    elif action == "reset":
        if interaction.guild_id is not None:
            set_active_char_key(interaction.guild_id, interaction.channel_id, default_character().key)
        default_name = default_character().display or "Default"
        await interaction.followup.send(
            f"Reverted to default character: **{default_name}**", ephemeral=True
        )

    else:
        await interaction.followup.send(
            f"Unknown action ``{action}``. Use: list, set, show, reset.", ephemeral=True
        )
