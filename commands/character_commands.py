"""Character management slash commands."""
from __future__ import annotations

import discord
# NOTE: read the registry through accessors — importing ``_CHARACTERS``
# directly binds the pre-load list (the same trap main.py's _CHAR_CHOICES
# used to hit).
from config.characters import all_characters, get_character, default_character
from bot_core.history import get_active_char_key, set_active_char_key

async def handle_character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
) -> None:
    """Handle the /character slash command."""
    await interaction.response.defer()

    # get_active_char_key already returns the default key when nothing is
    # set per-channel, so no separate lookup loop is needed.
    active_key = get_active_char_key(interaction.guild_id, interaction.channel_id)
    current_char = get_character(active_key)

    if action == "list":
        lines = ["**Available characters:**\n"]
        for char in all_characters():
            marker = " ← current" if char.key == active_key else ""
            lines.append(f"  • `{char.key}` — display: `{char.display or char.key}`{marker}")
        await interaction.followup.send("\n".join(lines))

    elif action == "set":
        if name is None:
            await interaction.followup.send("Please provide a character key: `/character set <name>`")
            return
        # get_character() matches by key OR display name — no second loop.
        char_obj = get_character(name)
        if char_obj is None:
            avail = ", ".join(f"`{c.key}`" for c in all_characters())
            await interaction.followup.send(
                f"Unknown character ``{name}``. Available: {avail}"
            )
            return
        if interaction.guild_id is not None:
            set_active_char_key(interaction.guild_id, interaction.channel_id, char_obj.key)
        await interaction.followup.send(
            f"Switched to **{char_obj.display}** (model: ``{char_obj.model or '(none set)'}``)",
        )

    elif action == "show":
        display = current_char.display if current_char else "Default"
        model = current_char.model if current_char else "(not set)"
        await interaction.followup.send(
            f"**Current character:** `{display}`\n**Model:** ``{model}``"
        )

    elif action == "reset":
        if interaction.guild_id is not None:
            set_active_char_key(interaction.guild_id, interaction.channel_id, default_character().key)
        default_name = default_character().display or "Default"
        await interaction.followup.send(
            f"Reverted to default character: **{default_name}**"
        )

    else:
        await interaction.followup.send(
            f"Unknown action ``{action}``. Use: list, set, show, reset."
        )
