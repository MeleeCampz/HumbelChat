"""Slash command handler for /help — lists all commands, with optional detail view.

The listing is built **dynamically from the live command tree**
(``tree.get_commands()``), so it can never drift out of sync with the actual
commands: ``_CATEGORIES`` only controls *grouping*, and any command name that
is not in the map (e.g. a newly added one) still appears under "Other".

Two modes:

* ``/help`` — one embed, one field per category, lines of ``/name — description``.
* ``/help <command>`` — detail embed for one command: its description plus one
  line per option (type, required/default, min–max when present).

All parameter attribute access is defensive (``getattr``) so a future
discord.py shape change degrades to a plain listing instead of crashing.
"""
from __future__ import annotations

import logging
from typing import Any

import discord
from discord.utils import MISSING

log = logging.getLogger("bot.commands.help_command")

# ── Grouping ─────────────────────────────────────────────────────────────────

#: command name → category (display order follows ``_CATEGORY_ORDER``)
_CATEGORIES: dict[str, str] = {
    "ai": "AI & Chat",
    "ai_stop": "AI & Chat",
    "character": "AI & Chat",
    "clear_history": "AI & Chat",
    "roll_items": "D&D Items",
    "item_search": "D&D Items",
    "item_stats": "D&D Items",
    "reset_rolls": "D&D Items",
    "upload_kb": "Knowledge Base",
    "list_kb_docs": "Knowledge Base",
    "reindex_kb": "Knowledge Base",
    "sync_kb": "Knowledge Base",
    "start_session": "Sessions",
    "end_session": "Sessions",
    "remind_next_session": "Sessions",
    "session_notes": "Sessions",
    "start_recording": "Voice Recording",
    "stop_recording": "Voice Recording",
    "remind": "Utility",
    "ocr": "Utility",
    "summarize": "Utility",
    "translate": "Utility",
    "sync": "Maintenance",
}

#: fixed display order for the known categories
_CATEGORY_ORDER: list[str] = [
    "AI & Chat",
    "D&D Items",
    "Knowledge Base",
    "Sessions",
    "Voice Recording",
    "Utility",
    "Maintenance",
]

_OTHER_CATEGORY = "Other"

# Discord's hard embed-field limit is 1024 chars; keep headroom.
_FIELD_LIMIT = 1000


# ── Small helpers ────────────────────────────────────────────────────────────


def _type_label(param: object) -> str:
    """Human-readable label for a parameter type (``int`` → "integer"-ish)."""
    t = getattr(param, "type", None)
    name = getattr(t, "__name__", None)
    if not name:  # typing generics (e.g. Range[...]) expose __origin__ instead
        origin = getattr(t, "__origin__", None)
        name = getattr(origin, "__name__", None)
    return str(name).lower() if name else "any"


def _option_line(param: object) -> str:
    """One detail-view line for a single parameter."""
    name = getattr(param, "name", "?")
    label = _type_label(param)
    desc = (getattr(param, "description", None) or "").strip()

    default = getattr(param, "default", MISSING)
    required = getattr(param, "required", default is MISSING)

    bounds = ""
    lo = getattr(param, "min_value", None)
    hi = getattr(param, "max_value", None)
    if lo is not None or hi is not None:
        bounds = f" ({lo if lo is not None else '…'}–{hi if hi is not None else '…'})"

    if required:
        suffix = " — *required*"
    elif default is MISSING:
        suffix = ""
    else:
        suffix = f" — *default: {default}*"

    line = f"**{name}** `({label}{bounds})`"
    if desc:
        line += f": {desc}"
    return line + suffix


def _chunk_lines(lines: list[str], limit: int = _FIELD_LIMIT) -> list[str]:
    """Split lines into chunks whose joined text fits the embed-field limit."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        add = len(line) + (1 if current else 0)  # +1 for the joining newline
        if current and size + add > limit:
            chunks.append("\n".join(current))
            current, size = [], 0
            add = len(line)
        current.append(line)
        size += add
    if current:
        chunks.append("\n".join(current))
    return chunks


# ── Embed builders ───────────────────────────────────────────────────────────


def _build_overview_embed(
    commands: list[discord.app_commands.Command[Any, Any, Any]],
) -> discord.Embed:
    """One embed, one field per non-empty category (fixed order)."""
    by_category: dict[str, list[str]] = {}
    for cmd in commands:
        category = _CATEGORIES.get(cmd.name, _OTHER_CATEGORY)
        desc = (getattr(cmd, "description", None) or "").strip()
        line = f"`/{cmd.name}`" + (f" — {desc}" if desc else "")
        by_category.setdefault(category, []).append(line)

    ordered = [c for c in _CATEGORY_ORDER if c in by_category]
    ordered += [c for c in by_category if c not in _CATEGORY_ORDER]  # e.g. "Other"

    embed = discord.Embed(
        title="Available Commands",
        description="Use `/help <command>` for detailed usage of a single command.",
        color=discord.Color.blurple(),
    )
    for category in ordered:
        lines = by_category[category]
        for i, chunk in enumerate(_chunk_lines(lines)):
            title = category if i == 0 else f"{category} (cont.)"
            embed.add_field(name=title, value=chunk, inline=False)
    return embed


def _build_detail_embed(cmd: discord.app_commands.Command[Any, Any, Any]) -> discord.Embed:
    """Detail embed for one command: description + one line per option."""
    desc = (getattr(cmd, "description", None) or "").strip()
    embed = discord.Embed(
        title=f"/{cmd.name}",
        description=desc or "*No description.*",
        color=discord.Color.blurple(),
    )
    params = list(getattr(cmd, "parameters", None) or [])
    if params:
        embed.add_field(
            name="Options",
            value="\n".join(_option_line(p) for p in params),
            inline=False,
        )
    else:
        embed.add_field(name="Options", value="No options.", inline=False)
    return embed


# ── Handler ──────────────────────────────────────────────────────────────────


async def handle_help_command(
    interaction: discord.Interaction,
    command_name: str | None = None,
) -> None:
    """Send the command overview, or the detail view for one command.

    Public reply (visible in channel), like most other command replies.
    """
    tree = getattr(interaction.client, "tree", None)
    commands = list(tree.get_commands()) if tree is not None else []

    if command_name:
        cmd = next((c for c in commands if c.name == command_name), None)
        if cmd is None:
            await interaction.response.send_message(
                f"❓ `/{command_name}` is not a known command — "
                "run `/help` to see everything."
            )
            return
        log.info("Help detail for %s requested by %s (ID: %s)",
                 cmd.name, interaction.user, getattr(interaction.user, "id", "?"))
        await interaction.response.send_message(embed=_build_detail_embed(cmd))
    else:
        log.info("Help overview requested by %s (ID: %s)",
                 interaction.user, getattr(interaction.user, "id", "?"))
        await interaction.response.send_message(embed=_build_overview_embed(commands))
