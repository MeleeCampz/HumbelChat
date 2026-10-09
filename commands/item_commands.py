"""Item-table utility commands: /item_search, /item_stats, /reset_rolls,
/item_exclude, /item_include.

Read-only lookups plus pool maintenance for the /roll_items tables. Search
and stats never touch pool state; reset clears consumed entries; exclude/
include mark or release single items in the same consumed state (see
``bot_core.item_state``). These utilities do **not** seed sample tables —
that is /roll_items' job on first use.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import discord

from bot_core import item_search, item_state, item_tables, roll_buttons

log = logging.getLogger("bot.item_commands")


def _scope_label(table: str | None) -> str:
    return table if table else "all tables"


def _load_scope(table: str | None) -> dict[str, Any]:
    """Load the item scope for pool-mutation commands; error dict on failure."""
    try:
        items = item_tables.load_items(table)
    except LookupError:
        tables = item_tables.list_tables()
        avail = ", ".join(f"`{t}`" for t in tables) or "*(none)*"
        return {"ok": False, "error": f"No table named `{table}`. Available tables: {avail}."}
    if not items:
        return {
            "ok": False,
            "error": "No item tables found — run `/roll_items` once (it seeds samples) or add CSVs to `data/items/`.",
        }
    return {"ok": True, "items": items}


def _resolve_target(query: str, items: list[item_tables.Item]) -> dict[str, Any]:
    """Fuzzy-match one item to act on, using /item_search scoring.

    Acts (``single``) when exactly one strong-tier candidate — exact /
    prefix / substring on the normalized names — strictly outranks the
    second-best match. The difflib-ratio tier never acts on its own (ratios
    can reach 0.8+, so the score alone cannot separate the tiers); it only
    counts as a tie-breaker against a strong candidate. Multiple strong
    candidates, or a fuzzy match scoring at least as high as the sole strong
    one, yield ``ambiguous`` with up to five options; nothing found yields
    ``none``.
    """
    matches = item_search.search_items(query, items, limit=5)
    if not matches:
        return {"status": "none"}
    q = item_search.norm(query)
    strong = [
        (item, score) for item, score in matches
        if (n := item_search.norm(item.name)) == q or n.startswith(q) or q in n
    ]
    if len(strong) == 1 and (len(matches) == 1 or strong[0][1] > matches[1][1]):
        return {"status": "single", "item": strong[0][0]}
    return {"status": "ambiguous", "candidates": [(i, s) for i, s in matches]}


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
        # Single-table results: name the table once in the title. Mixed
        # results: keep a per-item table line.
        match_tables = {item.table for item, _ in matches if item.table}
        if len(match_tables) == 1:
            scope_label = f"`{match_tables.pop()}`"
            per_item_table = False
        else:
            scope_label = _scope_label(table)
            per_item_table = table is None
        embed = discord.Embed(
            title=f"🔎 {query!r} — {len(matches)} match(es) in {scope_label}",
            color=discord.Color.gold(),
        )
        for item, _score in matches:
            label = f"**{item.name}** — {item.rarity}"
            lines = []
            if item.notes:
                lines.append(item.notes)
            if item.url:
                lines.append(f"[Item page]({item.url})")
            if per_item_table and item.table:
                lines.append(f"table: `{item.table}`")
            embed.add_field(name=label, value="\n".join(lines) or "—", inline=False)
        # One toggle button per match (per-item table — results may be mixed).
        entries = [(item.table, item.name) for item, _ in matches if item.table]
        return {"ok": True, "embed": embed, "entries": entries}

    try:
        result = await asyncio.to_thread(_search)
        if not result["ok"]:
            await interaction.followup.send(result["error"])
            return
        entries: list[tuple[str, str]] = result.get("entries") or []
        # Built on the loop thread — views created outside a running event
        # loop never dispatch clicks in this discord.py version.
        view = roll_buttons.build_view(entries) if entries else None
        sent: discord.WebhookMessage | None = None
        if view is not None:
            sent = await interaction.followup.send(
                embed=result["embed"], view=view, wait=True
            )
        else:
            await interaction.followup.send(embed=result["embed"])
        if view is not None and sent is not None:
            try:
                await asyncio.to_thread(
                    roll_buttons.record_roll_message, sent.channel.id, sent.id, entries
                )
            except OSError:
                log.warning("item_search: could not persist button record for message %s", sent.id)
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
            # Only count consumed entries that still exist as rows — stale
            # state (deleted CSV rows) is pruned on the next roll, but stats
            # must not show it in the meantime.
            live_names = {i.name for i in rows}
            consumed = len([n for n in state.get(t, []) if n in live_names])
            remaining = len(rows) - consumed
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


def _mark_result_text(action: str, item: item_tables.Item) -> str:
    table_label = f" ({item.table})" if item.table else ""
    if action == "exclude":
        return (
            f"🚫 Marked **{item.name}**{table_label} as used — it won't come up in "
            "consume rolls until you use `/item_include` or `/reset_rolls`."
        )
    return (
        f"✅ Put **{item.name}**{table_label} back into the pool — it can come up "
        "in consume rolls again."
    )


def _candidate_lines(candidates: list[tuple[item_tables.Item, float]]) -> str:
    lines = []
    for i, (item, _score) in enumerate(candidates, start=1):
        where = f" — {item.table}" if item.table else ""
        lines.append(f"{i}. **{item.name}** ({item.rarity}){where}")
    return "\n".join(lines)


async def _handle_mark_command(
    interaction: discord.Interaction,
    query: str,
    table: str | None,
    action: str,
) -> None:
    """Shared body for /item_exclude and /item_include."""
    if table:
        table = table.strip() or None
    await interaction.response.defer()

    def _mark() -> dict[str, Any]:
        scope = _load_scope(table)
        if not scope["ok"]:
            return scope
        resolved = _resolve_target(query, scope["items"])
        if resolved["status"] == "none":
            return {
                "ok": False,
                "error": f"No items matching `{query}` in {_scope_label(table)}.",
            }
        if resolved["status"] == "ambiguous":
            return {
                "ok": False,
                "error": (
                    f"Multiple or uncertain matches for `{query}` — tell me the exact name:\n"
                    f"{_candidate_lines(resolved['candidates'])}"
                ),
            }
        item = resolved["item"]
        state = item_state.load_state()
        already = item.name in state.get(item.table, [])
        if action == "exclude":
            if already:
                return {"ok": True, "text": f"`{item.name}` is already marked as used."}
            item_state.consume([item])
            return {"ok": True, "text": _mark_result_text("exclude", item)}
        # action == "include"
        if not already:
            return {"ok": True, "text": f"`{item.name}` wasn't marked — nothing to do."}
        released = item_state.release_item(item.table, item.name)
        return {
            "ok": True,
            "text": _mark_result_text("include", item) if released else f"Couldn't unmark `{item.name}` — check the bot logs.",
        }

    try:
        result = await asyncio.to_thread(_mark)
        if not result["ok"]:
            await interaction.followup.send(result["error"])
            return
        await interaction.followup.send(result["text"])
    except Exception:
        log.exception("%s: unexpected error", action)
        try:
            await interaction.followup.send(f"{action.capitalize()} failed unexpectedly — check the bot logs.")
        except Exception:  # noqa: BLE001 - last-resort, nothing left to do
            pass


async def handle_item_exclude_command(
    interaction: discord.Interaction,
    name: str,
    table: str | None = None,
) -> None:
    """Mark one item as used (won't come up in consume rolls until unmarked)."""
    await _handle_mark_command(interaction, name, table, "exclude")


async def handle_item_include_command(
    interaction: discord.Interaction,
    name: str,
    table: str | None = None,
) -> None:
    """Release one item from the consumed state without resetting the table."""
    await _handle_mark_command(interaction, name, table, "include")


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
