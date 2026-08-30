"""Voice-recording slash commands — /start_recording and /stop_recording.

These translate Discord interactions into :mod:`bot_core.voice_recorder` calls.
The bot joins the voice channel the invoking user is currently in, captures each
participant's audio separately (for per-speaker STT), and writes WAV files plus a
timestamped manifest to disk on stop.

The heavy lifting lives in ``bot_core.voice_recorder``; this module only:
  * validates the interaction (guild + the user being in voice),
  * makes sure the bot is in the right voice channel,
  * starts / stops the recorder and reports the result.
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime

import discord

from bot_core.channel_delivery import get_bot
from bot_core.voice_recorder import (
    VoiceRecorder,
    _wire_voice_client,
    attach_to_bot,
    recorder_voice_cls,
)
from config.settings import RECORDINGS_DIR

log = logging.getLogger("bot.recording_commands")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_recorder(bot: discord.Client) -> VoiceRecorder:
    """Return the bot's recorder, attaching it on first use (idempotent)."""
    return attach_to_bot(bot, RECORDINGS_DIR)


def _user_voice_channel(interaction: discord.Interaction):
    """Return the voice channel the invoking user is in, or ``None``."""
    member = interaction.user
    # In a guild interaction.user is a Member with a .voice attribute.
    voice = getattr(member, "voice", None)
    if voice is not None:
        return getattr(voice, "channel", None)
    return None


def _guild_voice_client(bot: discord.Client, guild_id: int):
    """Return the bot's VoiceClient for this guild (or ``None``)."""
    try:
        guild = bot.get_guild(guild_id)
    except Exception:  # pragma: no cover - defensive
        return None
    if guild is None:
        return None
    return getattr(guild, "voice_client", None)


async def _ensure_bot_in_channel(bot: discord.Client, guild_id: int, channel) -> None:
    """Join (or move) the bot into ``channel``. No-op if already there.

    New joins go through ``channel.connect(cls=recorder_voice_cls(bot))`` so
    the recording voice client is instantiated *by discord.py itself* with the
    op-5 hook and UDP listener installed before the handshake begins.
    """
    vc = _guild_voice_client(bot, guild_id)
    if vc is not None:
        if getattr(vc, "channel", None) is not None and vc.channel.id == channel.id:
            # Already in the right channel. Make sure it's recording-wired —
            # e.g. if the bot was already in voice for another reason.
            if not getattr(vc, "_recorder_wired", False):
                rec = getattr(bot, "_voice_recorder", None)
                if rec is not None:
                    try:
                        _wire_voice_client(vc, rec)
                    except Exception as e:  # pragma: no cover - defensive
                        log.warning("Could not retrofit recorder onto existing voice client: %s", e)
            return
        log.info("Moving bot from #%s to #%s for recording", vc.channel.name, channel.name)
        await vc.move_to(channel)
        return

    log.info("Joining voice channel #%s (id=%s) for recording", channel.name, channel.id)
    await channel.connect(cls=recorder_voice_cls(bot), timeout=30.0)


def _new_recording_dir() -> pathlib.Path:
    """A fresh per-recording subdirectory under RECORDINGS_DIR."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RECORDINGS_DIR / f"recording_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── /start_recording ────────────────────────────────────────────────────────
async def handle_start_recording(interaction: discord.Interaction) -> None:
    """Join the user's voice channel and start capturing per-speaker audio."""
    if interaction.guild_id is None:
        await interaction.response.send_message(
            "⚠️ Voice recording only works inside a server (not DMs).",
            ephemeral=True,
        )
        return

    channel = _user_voice_channel(interaction)
    if channel is None:
        await interaction.response.send_message(
            "⚠️ You're not in a voice channel. Join one first, then run `/start_recording`.",
            ephemeral=True,
        )
        return

    bot = get_bot()
    if bot is None:
        await interaction.response.send_message("⚠️ Bot isn't ready yet — try again in a moment.", ephemeral=True)
        return

    # Joining voice can take a few seconds; defer so we don't blow the 15 s window.
    await interaction.response.defer(ephemeral=True)

    try:
        recorder = _get_recorder(bot)
        out_dir = _new_recording_dir()
        recorder.out_dir = out_dir

        await _ensure_bot_in_channel(bot, interaction.guild_id, channel)

        recorder.start(
            guild_id=interaction.guild_id,
            channel_id=channel.id,
            channel_name=getattr(channel, "name", ""),
        )
    except discord.Forbidden:
        await interaction.followup.send(
            "⚠️ I don't have permission to join that voice channel (need **Connect** and **Speak**).",
            ephemeral=True,
        )
        return
    except Exception as e:  # pragma: no cover - defensive
        log.exception("Failed to start voice recording")
        await interaction.followup.send(f"⚠️ Failed to start recording: {e}", ephemeral=True)
        return

    log.info("Recording started: %s", recorder.snapshot())
    await interaction.followup.send(
        f"🎙️ **Recording started** in #{getattr(channel, 'name', '?')}.\n"
        f"Each participant's audio is captured separately with timestamps.\n"
        f"Files will be saved to `{out_dir.name}/` when you run `/stop_recording`.",
        ephemeral=True,
    )


# ── /stop_recording ─────────────────────────────────────────────────────────
async def handle_stop_recording(
    interaction: discord.Interaction,
    leave_channel: bool = True,
) -> None:
    """Stop capturing, write the WAV files + manifest, and (optionally) leave voice."""
    bot = get_bot()
    if bot is None:
        await interaction.response.send_message("⚠️ Bot isn't ready yet — try again in a moment.", ephemeral=True)
        return

    recorder = getattr(bot, "_voice_recorder", None)
    if recorder is None or not recorder.is_recording:
        await interaction.response.send_message(
            "⚠️ No recording is active. Start one with `/start_recording`.",
            ephemeral=True,
        )
        return

    # Writing WAVs + manifest can exceed the 15 s window for long recordings.
    await interaction.response.defer(ephemeral=True)

    try:
        manifest = recorder.stop()
    except Exception as e:  # pragma: no cover - defensive
        log.exception("Failed to stop voice recording")
        await interaction.followup.send(f"⚠️ Failed to stop recording: {e}", ephemeral=True)
        return

    if manifest is None:
        await interaction.followup.send("⚠️ The recording was already stopped.", ephemeral=True)
        return

    # Optionally leave the voice channel so we don't keep occupying a slot.
    if leave_channel and interaction.guild_id is not None:
        vc = _guild_voice_client(bot, interaction.guild_id)
        if vc is not None:
            try:
                await vc.disconnect()
            except Exception as e:  # pragma: no cover - defensive
                log.warning("Could not disconnect from voice after stop: %s", e)

    speakers = manifest.get("speakers", [])
    sp_lines = []
    for s in speakers:
        name = s.get("display_name") or f"user {s['user_id']}"
        sp_lines.append(f"  • **{name}** — {s['spoken_duration_s']}s spoken → `{s['wav_file']}`")
    if not sp_lines:
        sp_lines.append("  • (no audio was captured — nobody spoke, or SSRC mapping didn't resolve)")

    body = (
        f"⏹️ **Recording stopped** — {manifest['duration_s']}s total.\n"
        f"Captured {len(speakers)} speaker(s):\n" + "\n".join(sp_lines) +
        f"\n\n📁 Saved to `{pathlib.Path(manifest.get('manifest_path', '')) or 'recordings/'}`"
    )

    # Attach the (small) manifest so it's easy to grab; WAVs stay on disk.
    manifest_file = None
    mp = manifest.get("manifest_path")
    if mp and pathlib.Path(mp).exists():
        try:
            manifest_file = discord.File(pathlib.Path(mp), filename="manifest.json")
        except Exception:  # pragma: no cover - defensive
            manifest_file = None

    await interaction.followup.send(body, file=manifest_file, ephemeral=True)
