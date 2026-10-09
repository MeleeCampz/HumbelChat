"""/roll_items — random item table rolls with item page links (+ CR scaling)."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import discord

from bot_core import item_state, item_tables, roll_buttons

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


def _exhausted_error(scope: str, rarity: str | None) -> str:
    what = f"no {rarity} items in {scope}" if rarity else f"all items in {scope}"
    return (
        f"Pool exhausted — {what} have already been rolled. "
        "Use `/reset_rolls` to put them back."
    )


async def handle_roll_items_command(
    interaction: discord.Interaction,
    table: str | None = None,
    count: int = 3,
    cr: str | None = None,
    rarity: str | None = None,
    consume: bool = False,
) -> None:
    """Roll random items from one CSV table in ``ITEMS_DIR``.

    ``table`` omitted → the default table (``DEFAULT_ITEM_TABLE``, usually
    ``magic_items``). ``cr`` (optional) overrides ``count`` and filters by the
    tier's rarities; ``rarity`` (optional) filters plain rolls only.
    ``consume`` (default off): when on, already-consumed items are excluded
    from the roll and the rolled items are marked as consumed — they stay out
    until cleared with ``/reset_rolls``. When off, the roll is pure random
    from the full table and pool state is not touched at all.
    """
    if table:
        table = table.strip() or None
    if not table:
        # Omitted table → the configured default (read at call time so tests
        # can monkeypatch it). Every roll is from exactly one table.
        from config import settings
        table = settings.DEFAULT_ITEM_TABLE
    if rarity:
        rarity = rarity.strip().lower() or None
    count = max(1, int(count))
    await interaction.response.defer()

    def _roll() -> dict[str, Any]:
        """Runs off the event loop; returns a plain-serializable result dict."""
        # First use: copy the built-in sample tables into ITEMS_DIR so the
        # user can edit them in place (existing files are never touched).
        item_tables.seed_from_samples()
        if not item_tables.list_tables():
            return {
                "ok": False,
                "error": "No item tables found — add CSV files to `data/items/` (see docs/item-rolls.md).",
            }
        scope = table
        rng = random.Random()
        try:
            items = item_tables.load_items(table)
            if consume:
                # Only consuming rolls touch pool state (prune stale entries
                # first so deleted rows never wedge the pool).
                item_state.prune(items)

            if cr is not None:
                cr_value = item_tables.parse_cr(cr)
                tiers = item_tables.load_cr_tiers()
                if not tiers:
                    raise item_tables.NoCrTiersError("no CR tiers configured")

                if consume:
                    available, total = item_state.exclude_consumed(items)
                    if not available:
                        return {"ok": False, "error": _exhausted_error(scope, None)}
                else:
                    available, total = items, len(items)
                rolled, tier, exhausted = item_tables.roll_for_cr(
                    cr_value, available, tiers, rng=rng
                )
                by_rarity: dict[str, int] = {}
                for item in rolled:
                    by_rarity[item.rarity] = by_rarity.get(item.rarity, 0) + 1
                breakdown = ", ".join(f"{r} ×{n}" for r, n in sorted(by_rarity.items()))
                footer_parts = [
                    f"CR {_fmt_cr(cr_value)} → tier (min CR {_fmt_cr(tier.min_cr)}): {breakdown}"
                ]
                if exhausted:
                    footer_parts.append(f"some rarities ran out of items in {scope}")
                if consume:
                    footer_parts.append(
                        f"{len(available) - len(rolled)} of {total} remaining in {scope}"
                    )
                return {
                    "ok": True,
                    "title": f"🎲 Random items from {scope} (CR {_fmt_cr(cr_value)})",
                    "items": rolled,
                    "footer": " — ".join(footer_parts),
                    "consume": rolled if consume else None,
                }

            pool = items
            if rarity is not None:
                pool = [i for i in items if i.rarity == rarity]
                if not pool:
                    have = sorted({i.rarity for i in items})
                    return {
                        "ok": False,
                        "error": (
                            f"No `{rarity}` items in {scope}. "
                            f"Rarities available: {', '.join(have) or 'none'}."
                        ),
                    }
            if consume:
                available, total = item_state.exclude_consumed(pool)
                if not available:
                    return {"ok": False, "error": _exhausted_error(scope, rarity)}
            else:
                available, total = pool, len(pool)
            rolled, exhausted = item_tables.roll_items(available, count, rng=rng)
            # With a rarity filter the remaining clause refers to the filtered
            # subset — say so explicitly.
            roll_scope = f"{scope} ({rarity})" if rarity is not None else scope
            footer_parts = []
            if exhausted:
                footer_parts.append(f"only {len(rolled)} item(s) available in {roll_scope}")
            if consume:
                footer_parts.append(
                    f"{len(available) - len(rolled)} of {total} remaining in {roll_scope}"
                )
            return {
                "ok": True,
                "title": f"🎲 Random items from {scope}",
                "items": rolled,
                "footer": " — ".join(footer_parts),
                "consume": rolled if consume else None,
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
            avail = ", ".join(f"`{t}`" for t in tables) or "*(none — add CSVs to `data/items/`)*"
            return {"ok": False, "error": f"No table named `{table}`. Available tables: {avail}."}
        except ValueError as exc:
            if cr is not None and "CR" in str(exc):
                return {"ok": False, "error": f"Couldn't parse CR `{cr}` — use e.g. `4` or `1/2`."}
            return {"ok": False, "error": f"Roll failed: {exc}"}

    try:
        result = await asyncio.to_thread(_roll)
        if not result["ok"]:
            await interaction.followup.send(result["error"])
            return
        names = [i.name for i in result["items"]]
        # Per-item toggle buttons. For consume rolls the rolled items are
        # shown as already-consumed (↩️) even though the state write happens
        # after the send lands — a failed send loses nothing.
        view = await asyncio.to_thread(
            roll_buttons.build_view, table, names,
            {n for n in names} if result.get("consume") else None,
        )
        sent = await interaction.followup.send(
            embed=_build_embed(result["title"], result["items"], result["footer"]),
            view=view,
        )
        # Consume only after the reply landed — a failed send loses nothing.
        if result.get("consume"):
            await asyncio.to_thread(item_state.consume, result["consume"])
        try:
            await asyncio.to_thread(
                roll_buttons.record_roll_message, sent.channel.id, sent.id, table, names
            )
        except OSError:
            log.warning("roll_items: could not persist button record for message %s", sent.id)
    except Exception:
        log.exception("roll_items: unexpected error")
        try:
            await interaction.followup.send(
                "Item rolling failed unexpectedly — check the bot logs."
            )
        except Exception:  # noqa: BLE001 - last-resort, nothing left to do
            pass
