"""Slash command handler for `/botperms` — why can't the bot post here?

Thin shell over :mod:`bot_core.perms_report`, which explains *why* the check is
needed: slash-command replies ride the interaction webhook and work no matter
what, while background senders (session highlights, next-session reminders,
roll-button re-attach) must resolve the channel by ID and hold the real channel
permissions.  When those two disagree — chat fine, highlights archived to disk
only — this command is the answer.

Usage in Discord::

    /botperms                      # inspect the channel the command ran in
    /botperms 869602082552954973   # inspect a channel by id or <#mention>
    /botperms humblewood-chat      # inspect by name (cached channels only)
"""
from __future__ import annotations

import logging
from typing import Any

import discord

from bot_core import perms_report

log = logging.getLogger("bot.commands.perms_command")


async def _notify_channel(interaction: discord.Interaction, message: str) -> None:
    """Best-effort fallback for when the interaction webhook already expired."""
    try:
        chan = getattr(interaction, "channel", None)
        if chan is not None:
            await chan.send(message)
            return
    except Exception as e:  # noqa: BLE001 - purely defensive
        log.warning("/botperms: channel fallback failed: %s", e)
    log.warning("/botperms: no usable delivery path for an expired interaction")


async def handle_botperms_command(
    interaction: discord.Interaction,
    channel: str | None = None,
) -> None:
    """Build and post the permission report for one channel."""
    try:
        # The probe does a REST channel fetch, so give the 3-second window up front.
        await interaction.response.defer()
    except discord.NotFound:
        await _notify_channel(
            interaction,
            "⚠️ That `/botperms` timed out before the bot could answer. Please try again.",
        )
        return

    bot: Any = interaction.client
    guild = getattr(interaction, "guild", None)

    if guild is None:
        await interaction.followup.send(
            "ℹ️ `/botperms` is a guild-tool — channel permissions only exist in a "
            "server, so run it in a server channel."
        )
        return

    if channel:
        target, note = await perms_report.find_channel(bot, guild, channel)
        if target is None:
            # The failure reason IS the diagnosis: an unreadable channel answers 403.
            log.info("/botperms: could not resolve %r: %s", channel, note)
            await interaction.followup.send(f"❓ **Could not inspect `{channel}`:** {note}")
            return
    else:
        target = getattr(interaction, "channel", None)
        if target is None:
            await interaction.followup.send(
                "ℹ️ Discord did not hand the bot this channel object (that alone is a "
                "strong hint it cannot view it). Run "
                "`/botperms <channel id>` with the id from the channel's Copy Link."
            )
            return

    try:
        report = await perms_report.build_report(bot, target)
    except Exception as e:  # noqa: BLE001 - a diagnostic must not take the bot down
        log.exception("/botperms: report build failed")
        await interaction.followup.send(f"❌ **Permission check failed:** {e!r}")
        return

    log.info(
        "/botperms by %s (ID: %s) on channel %s (%s): resolve=%s/%s perms=%s",
        interaction.user, interaction.user.id, report.channel_id, report.name,
        report.resolve_via or "ok", "ok" if report.resolve_ok else "failed",
        {attr: granted for attr, _label, granted in report.perms},
    )
    await interaction.followup.send(perms_report.format_report(report))
