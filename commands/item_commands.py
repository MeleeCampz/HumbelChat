"""Item-table utility commands: /item_search, /item_stats, /reset_rolls.

Read-only lookups plus pool maintenance for the /roll_items tables. Search
and stats never touch pool state; reset clears consumed entries (see
``bot_core.item_state``). These utilities do **not** seed sample tables —
that is /roll_items' job on first use.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import discord

from bot_core import item_search, item_state, item_tables

log = logging.getLogger("bot.item_commands")


def _scope_label(table: str | None) -> str:
    return table if table else "all tables"


async def handle_item_search_command(
    interaction: discord.Interaction,
    query: str,
    table: str | None = None,
    limit: int = 10,
) -> None:
    """Fuzzy-search the item tables; replies with matching items + links."""
    if table:
        table = table.strip() or None
    limit = max(1, min(int(limit), 25))
    await interaction.response.defer()

    def _search() -> dict[str, Any]:
        try:
            items = item_tables.load_items(table)
        except LookupError:
            tables = item_tables.list_tables()
            avail = ", ".join(f"`{t}`" for t in tables) or "*(none — add CSVs to `data/items/`)*"
            return {"ok": False, "error": f"No table named `{table}`. Available tables: {avail}."}
        if not items:
            return {
                "ok": False,
                "error": "No item tables found — run `/roll_items` once (it seeds samples) or add CSVs to `data/items/`.",
            }
        matches = item_search.search_items(query, items, limit=limit)
        if not matches:
            return {"ok": False, "error": f"No items matching `{query}` in {_scope_label(table)}."}
        embed = discord.Embed(
            title=f"🔎 {query!r} — {len(matches)} match(es) in {_scope_label(table)}",
            color=discord.Color.gold(),
        )
        for item, _score in matches:
            label = f"**{item.name}** — {item.rarity}"
            lines = []
            if item.url:
                lines.append(f"[Item page]({item.url})")
            if table is None and item.table:
                lines.append(f"table: `{item.table}`")
            embed.add_field(name=label, value="\n".join(lines) or "—", inline=False)
        return {"ok": True, "embed": embed}

    try:
        result = await asyncio.to_thread(_search)
        if not result["ok"]:
            await interaction.followup.send(result["error"])
            return
        await interaction.followup.send(embed=result["embed"])
    except Exception:
        log.exception("item_search: unexpected error")
        try:
            await interaction.followup.send("Item search failed unexpectedly — check the bot logs.")
        except Exception:  # noqa: BLE001 - last-resort, nothing left to do
            pass


async def handle_item_stats_command(
    interaction: discord.Interaction,
    table: str | None = None,
) -> None:
    """Report per-table item counts (total / consumed / remaining + rarities)."""
    if table:
        table = table.strip() or None
    await interaction.response.defer()

    def _stats() -> dict[str, Any]:
        try:
            items = item_tables.load_items(table)
        except LookupError:
            tables = item_tables.list_tables()
            avail = ", ".join(f"`{t}`" for t in tables) or "*(none — add CSVs to `data/items/`)*"
            return {"ok": False, "error": f"No table named `{table}`. Available tables: {avail}."}
        if not items:
            return {
                "ok": False,
                "error": "No item tables found — run `/roll_items` once (it seeds samples) or add CSVs to `data/items/`.",
            }
        state = item_state.load_state()
        embed = discord.Embed(title="📊 Item table stats", color=discord.Color.gold())
        grand_total = grand_remaining = 0
        for t in sorted({i.table for i in items}):
            rows = [i for i in items if i.table == t]
            consumed = len(state.get(t, []))
            remaining = len(rows) - min(consumed, len(rows))
            by_rarity: dict[str, int] = {}
            for i in rows:
                by_rarity[i.rarity] = by_rarity.get(i.rarity, 0) + 1
            rarity_line = " · ".join(f"{r} {n}" for r, n in sorted(by_rarity.items()))
            embed.add_field(
                name=t,
                value=f"{len(rows)} total · {consumed} rolled out · {remaining} left\n{rarity_line}",
                inline=False,
            )
            grand_total += len(rows)
            grand_remaining += remaining
        embed.set_footer(text=f"grand total: {grand_total} items · {grand_remaining} remaining")
        return {"ok": True, "embed": embed}

    try:
        result = await asyncio.to_thread(_stats)
        if not result["ok"]:
            await interaction.followup.send(result["error"])
            return
        await interaction.followup.send(embed=result["embed"])
    except Exception:
        log.exception("item_stats: unexpected error")
        try:
            await interaction.followup.send("Item stats failed unexpectedly — check the bot logs.")
        except Exception:  # noqa: BLE001 - last-resort, nothing left to do
            pass


async def handle_item_reset_rolls_command(
    interaction: discord.Interaction,
    table: str | None = None,
) -> None:
    """Return consumed items to the pool (one table or all)."""
    if table:
        table = table.strip() or None
    await interaction.response.defer()

    def _reset() -> dict[str, Any]:
        released = item_state.reset(table)
        if released == 0 and table is not None:
            known = {t.lower() for t in item_tables.list_tables()}
            if str(table).lower().removesuffix(".csv") not in known:
                tables = item_tables.list_tables()
                avail = ", ".join(f"`{t}`" for t in tables) or "*(none)*"
                return {
                    "ok": False,
                    "error": f"No table named `{table}`. Available tables: {avail}.",
                }
        if released == 0:
            scope = _scope_label(table)
            return {"ok": True, "text": f"Nothing to reset — the pool in {scope} is already fresh."}
        scope = _scope_label(table)
        return {
            "ok": True,
            "text": f"Pool reset — {released} item(s) returned to {scope}.",
        }

    try:
        result = await asyncio.to_thread(_reset)
        if not result["ok"]:
            await interaction.followup.send(result["error"])
            return
        await interaction.followup.send(result["text"])
    except Exception:
        log.exception("reset_rolls: unexpected error")
        try:
            await interaction.followup.send("Reset failed unexpectedly — check the bot logs.")
        except Exception:  # noqa: BLE001 - last-resort, nothing left to do
            pass
