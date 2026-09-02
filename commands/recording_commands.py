"""Voice-recording slash commands — /start_recording and /stop_recording.

These translate Discord interactions into :mod:`bot_core.voice_recorder` calls.
The bot joins the voice channel the invoking user is currently in, captures each
participant's audio separately (for per-speaker STT), and writes WAV files plus a
timestamped manifest to disk on stop. When ``STT_ENABLED`` (and the optional
per-command ``transcribe`` flag) are set, each speaker's WAV is then transcribed
in the background via :mod:`bot_core.transcriber` — locally with faster-whisper
(``STT_BACKEND=local``, segment timestamps -> interleaved transcript) or against
the OpenAI-compatible backend (``STT_BACKEND=http``) — and the results are
posted to the channel when done.

The heavy lifting lives in ``bot_core.voice_recorder`` / ``bot_core.transcriber``;
this module only:
  * validates the interaction (guild + the user being in voice),
  * makes sure the bot is in the right voice channel,
  * starts / stops the recorder, kicks off transcription, and reports results.
"""
from __future__ import annotations

import logging
import pathlib
import time
from datetime import datetime

import discord

from bot_core.channel_delivery import get_bot
from bot_core.voice_recorder import (
    VoiceRecorder,
    _wire_voice_client,
    attach_to_bot,
    recorder_voice_cls,
)
from config.settings import (
    RECORDINGS_DIR,
    STT_ADD_TO_SESSION,
    STT_BACKEND,
    STT_ENABLED,
    STT_LOCAL_MODEL,
    STT_MODEL,
)
from utils.background_tasks import spawn_tracked_task

log = logging.getLogger("bot.recording_commands")


def _stt_model_name() -> str:
    """The model name the configured STT backend will actually use."""
    return STT_LOCAL_MODEL if (STT_BACKEND or "local").strip().lower() == "local" else STT_MODEL


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

        # Start BEFORE joining: Discord announces the initial SSRC -> user
        # mapping (op-11/op-5) as soon as the voice connection establishes,
        # which happens inside _ensure_bot_in_channel(). Starting afterwards
        # would wipe those mappings and every audio packet would be dropped
        # as "unknown SSRC" — silent failure, no WAVs written.
        recorder.start(
            guild_id=interaction.guild_id,
            channel_id=channel.id,
            channel_name=getattr(channel, "name", ""),
            out_dir=out_dir,
        )

        await _ensure_bot_in_channel(bot, interaction.guild_id, channel)
    except discord.Forbidden:
        recorder.discard()
        await interaction.followup.send(
            "⚠️ I don't have permission to join that voice channel (need **Connect** and **Speak**).",
            ephemeral=True,
        )
        return
    except Exception as e:  # pragma: no cover - defensive
        log.exception("Failed to start voice recording")
        recorder.discard()
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
    transcribe: bool = True,
) -> None:
    """Stop capturing, write the WAV files + manifest, (optionally) leave voice,
    and (optionally) kick off per-speaker STT in the background."""
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

    # Pin the transcript to the session active AT RECORD TIME: transcription
    # runs in the background, so by the time it finishes the user may have
    # ended/started sessions — the notes still belong to this recording's
    # session (its file is written even if that session has ended).
    from bot_core import sessions as S
    session_at_stop = S.get_current_session()

    # Only transcribe when enabled (globally + per-command) and there is
    # actual speech to transcribe.
    stt_speakers = [s for s in speakers if s.get("frames_captured", 0) > 0]
    run_stt = STT_ENABLED and transcribe and bool(stt_speakers)
    if STT_ENABLED and not transcribe:
        log.info("Transcription skipped (transcribe=false) for %s", manifest.get("manifest_path"))

    body = (
        f"⏹️ **Recording stopped** — {manifest['duration_s']}s total.\n"
        f"Captured {len(speakers)} speaker(s):\n" + "\n".join(sp_lines) +
        f"\n\n📁 Saved to `{pathlib.Path(manifest.get('manifest_path', '')) or 'recordings/'}`"
    )
    if run_stt:
        body += (
            f"\n\n🎧 Transcribing {len(stt_speakers)} speaker(s) with `{_stt_model_name()}` "
            f"({STT_BACKEND} backend) in the background — I'll post the transcript here when it's done."
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

    if run_stt:
        spawn_tracked_task(
            _run_transcription(interaction, manifest, session_at_stop),
            name="stt-transcription",
        )


async def _safe_followup(interaction: discord.Interaction, body: str, files: list | None = None) -> None:
    """Best-effort followup — the webhook can expire for very long recordings."""
    try:
        await interaction.followup.send(body, files=files or [], ephemeral=True)
    except Exception as e:  # pragma: no cover - expired webhooks
        log.warning("Could not deliver STT result (interaction expired?): %s", e)


async def _run_transcription(
    interaction: discord.Interaction,
    manifest: dict,
    session_at_stop: dict | None = None,
) -> None:
    """Background job: transcribe every speaker's WAV, write transcript.json,
    append the transcript to the session that was active when the recording
    stopped (when enabled), and post a summary with the transcript file
    attached."""
    from bot_core import sessions as S
    from bot_core import transcriber

    try:
        report = await transcriber.transcribe_recording(manifest)
        out_dir = pathlib.Path(pathlib.Path(manifest["manifest_path"]).parent)
        path = transcriber.write_transcript(out_dir, manifest, report)
    except Exception as e:  # pragma: no cover - defensive
        log.exception("STT run failed")
        await _safe_followup(interaction, f"⚠️ Transcription failed: {e}")
        return

    total = len(report.speakers)
    lines = []
    for s in report.speakers:
        name = s.display_name or f"user {s.user_id}"
        if s.ok and s.text:
            preview = s.text[:160] + ("…" if len(s.text) > 160 else "")
            lines.append(f"  • **{name}** ({s.elapsed_s}s): {preview}")
        elif s.ok:
            lines.append(f"  • **{name}**: (no speech detected)")
        else:
            lines.append(f"  • **{name}**: ⚠️ {s.error}")

    # Automatically fold the transcript into the notes of the session that
    # was active when the recording stopped (see handle_stop_recording) so it
    # shows up in /session_notes and stays RAG-searchable. Best effort: a
    # missing session or a bookkeeping error only changes the message line.
    session_line = ""
    if STT_ADD_TO_SESSION:
        started = datetime.fromtimestamp(
            manifest.get("started_at") or time.time()
        ).strftime("%Y-%m-%d %H:%M")
        title = (
            f"Voice channel transcript — #{manifest.get('channel_name') or '?'} "
            f"({started}, {int(manifest.get('duration_s') or 0)}s)"
        )
        try:
            session, n_bullets = S.add_transcript(
                transcriber.build_session_transcript(report), title=title,
                session=session_at_stop,
            )
            if session is not None:
                fname = pathlib.Path(session["file"]).name
                status = "active" if S.get_current_session() is session else "ended"
                session_line = (
                    f"\n\n📝 Added to the {status} session's notes "
                    f"(**{session.get('name') or '(untitled)'}**, {n_bullets} note(s)) — `{fname}`"
                )
            else:
                log.info("Transcript not added: no active session at STT completion")
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Could not add transcript to session notes: %s", e)

    body = (
        f"🎧 **Transcription done** — {report.ok_count}/{total} speaker(s) OK, "
        f"{report.finished_at - report.started_at:.0f}s total.\n" + "\n".join(lines)
        + session_line
    )

    files: list = []
    for f in (path, path.with_name("transcript.txt")):
        try:
            if f.exists():
                files.append(discord.File(f, filename=f.name))
        except Exception:  # pragma: no cover - defensive
            pass
    await _safe_followup(interaction, body, files=files)
