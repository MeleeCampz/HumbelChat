"""/roll_items — random item table rolls with item page links (+ CR scaling)."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import discord

from bot_core import item_tables

log = logging.getLogger("bot.roll_items_command")


def _fmt_cr(value: float) -> str:
    """Render a CR float D&D-style: 2.0 → '2', 0.5 → '1/2', 6.5 → '6 1/2'."""
    if value == int(value):
        return str(int(value))
    whole = int(value)
    frac = value - whole
    for denom in (2, 4, 8):
        num = frac * denom
        if abs(num - round(num)) < 1e-9:
            n = int(round(num))
            return f"{whole} {n}/{denom}" if whole else f"{n}/{denom}"
    return f"{value:g}"


def _build_embed(
    title: str,
    rolled: list[item_tables.Item],
    footer: str | None = None,
) -> discord.Embed:
    embed = discord.Embed(title=title, color=discord.Color.gold())
    for item in rolled:
        label = f"**{item.name}**"
        if item.rarity:
            label += f" — {item.rarity}"
        lines = []
        if item.url:
            lines.append(f"[Item page]({item.url})")
        if item.notes:
            lines.append(item.notes)
        embed.add_field(name=label, value="\n".join(lines) or "—", inline=False)
    if footer:
        embed.set_footer(text=footer)
    return embed


async def handle_roll_items_command(
    interaction: discord.Interaction,
    table: str | None = None,
    count: int = 3,
    cr: str | None = None,
) -> None:
    """Roll random items from the CSV tables in ``ITEMS_DIR``.

    ``cr`` (optional) overrides ``count`` and filters by the tier's rarities.
    """
    if table:
        table = table.strip() or None
    count = max(1, int(count))
    await interaction.response.defer()

    def _roll() -> dict[str, Any]:
        """Runs off the event loop; returns a plain-serializable result dict."""
        # First use: copy the built-in sample tables into ITEMS_DIR so the
        # user can edit them in place (existing files are never touched).
        item_tables.seed_from_samples()
        rng = random.Random()
        if table is None and not item_tables.list_tables():
            return {
                "ok": False,
                "error": "No item tables found — add CSV files to `data/items/` (see docs/commands.md).",
            }
        try:
            if cr is not None:
                cr_value = item_tables.parse_cr(cr)
                tiers = item_tables.load_cr_tiers()
                items = item_tables.load_items(table)
                rolled, tier, exhausted = item_tables.roll_for_cr(
                    cr_value, items, tiers, rng=rng
                )
                scope = table if table else "all tables"
                by_rarity: dict[str, int] = {}
                for item in rolled:
                    by_rarity[item.rarity] = by_rarity.get(item.rarity, 0) + 1
                breakdown = ", ".join(f"{r} ×{n}" for r, n in sorted(by_rarity.items()))
                footer_extra = (
                    f" — some rarities ran out of items in {scope}"
                    if exhausted else ""
                )
                footer: str | None = (
                    f"CR {_fmt_cr(cr_value)} → tier (min CR {_fmt_cr(tier.min_cr)}): {breakdown}"
                    + footer_extra
                )
                return {
                    "ok": True,
                    "title": f"🎲 Random items from {scope} (CR {_fmt_cr(cr_value)})",
                    "items": rolled,
                    "footer": footer,
                }
            items = item_tables.load_items(table)
            rolled, exhausted = item_tables.roll_items(items, count, rng=rng)
            scope = table if table else "all tables"
            footer = (
                f"— only {len(rolled)} item(s) available in {scope}" if exhausted else None
            )
            return {
                "ok": True,
                "title": f"🎲 Random items from {scope}",
                "items": rolled,
                "footer": footer,
            }
        except item_tables.NoCrTiersError:
            return {
                "ok": False,
                "error": (
                    "No CR tier table found — expected `data/items/cr_tiers.csv` with columns "
                    "`min_cr, rarities` (rarities = per-rarity ranges like `common:3-4;rare:1-2`)."
                ),
            }
        except LookupError:
            tables = item_tables.list_tables()
            if table is not None:
                avail = ", ".join(f"`{t}`" for t in tables) or "*(none — add CSVs to `data/items/`)*"
                return {"ok": False, "error": f"No table named `{table}`. Available tables: {avail}."}
            return {
                "ok": False,
                "error": "No item tables found — add CSV files to `data/items/` (see docs/commands.md).",
            }
        except ValueError as exc:
            if cr is not None and "CR" in str(exc):
                return {"ok": False, "error": f"Couldn't parse CR `{cr}` — use e.g. `4` or `1/2`."}
            return {"ok": False, "error": f"Roll failed: {exc}"}

    try:
        result = await asyncio.to_thread(_roll)
    except Exception:
        log.exception("roll_items: unexpected error")
        await interaction.followup.send("Item rolling failed unexpectedly — check the bot logs.")
        return
    if not result["ok"]:
        await interaction.followup.send(result["error"])
        return
    await interaction.followup.send(embed=_build_embed(result["title"], result["items"], result["footer"]))
