"""``/ai stop`` — cancel the in-flight /ai run for the current channel (P3 #25).

A stuck or slow generation could previously not be cancelled by the user. This
command cancels the child task that :mod:`commands.ai_command` (and the prefix
path in :mod:`main`) registered in :mod:`bot_core.ai_runs` for the channel, so
the request + its delivery are abandoned mid-flight and the user is told so.
"""
from __future__ import annotations

import logging

from bot_core.ai_runs import cancel_run

log = logging.getLogger("bot.commands.ai_stop")


async def handle_ai_stop_command(interaction) -> None:
    """Cancel the in-flight AI run for this channel, if any."""
    channel_key = interaction.channel_id if interaction.channel_id is not None else 0
    if cancel_run(channel_key):
        # The run task's own handler sends the "stopped" acknowledgement after
        # it observes the cancellation, so here we only confirm we acted.
        await interaction.response.send_message("⏹️ Cancelling the in-flight reply…")
    else:
        await interaction.response.send_message(
            "⚠️ There's no in-flight /ai reply in this channel to stop."
        )
