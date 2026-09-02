# Voice Recording

The bot can capture a Discord voice channel **per speaker** for later speech-to-text (STT). Each participant gets their own 16-bit PCM WAV file, and a JSON manifest ties everything to absolute timestamps so a multi-speaker conversation can be reconstructed on a shared timeline.

Implemented in [`bot_core/voice_recorder.py`](../bot_core/voice_recorder.py) (capture pipeline) and [`commands/recording_commands.py`](../commands/recording_commands.py) (slash-command layer).

## Commands

### `/start_recording`

Joins the voice channel **you** are currently in and starts capturing. Requirements:

- You must be in a guild (not DMs) and in a voice channel.
- The bot needs **Connect** and **Speak** permissions in that channel.

The bot joins automatically; if it was already in a different voice channel of the same server, it moves to yours. Each recording goes into its own subdirectory: `recordings/recording_YYYYMMDD-HHMMSS/`.

### `/stop_recording [leave_channel: true] [transcribe: true]`

Stops capturing, writes the WAV files + `manifest.json`, and replies with a per-speaker summary (and attaches the manifest). By default the bot then leaves the voice channel so it doesn't occupy a slot.

When `STT_ENABLED` is on (default) and `transcribe` isn't set to `false`, each speaker's WAV is additionally **transcribed in the background** (see [Speech-to-text](#speech-to-text)) and a second message with the transcript is posted when done. If a session is active, the finished transcript is also appended to that session's notes automatically (see [Session notes](#session-notes)).

> **Note:** each speaker's audio is streamed to disk *as it is captured* (see [Crash recovery](#crash-recovery)), so a crash mid-recording loses at most the last few frames — not hours of audio. The final WAVs + `manifest.json` are written when you stop (or on a clean shutdown, or automatically after a crash). Long recordings can take a moment to write (the command defers to stay under Discord's 15 s response window).

## Crash recovery

A bot crash used to mean **total loss** of an in-progress recording: every decoded frame lived in an unbounded in-memory list (≈96 KB/s per active speaker, ~2 GB/hour for six people) that was only written to disk on `/stop_recording`. The realistic failure mode — an **OOM kill mid-meeting** — wiped all of RAM.

That's fixed. Three mechanisms make a crash non-destructive:

1. **Stream-to-disk.** Each speaker's decoded frames are appended to an append-only binary log (`<name>_<user_id>.log`) the moment they decode, instead of being held in RAM. A crash now loses only the single frame still in flight (a trailing partial record is detected and dropped on read). The RAM footprint drops to "current frame + file handles", which also removes the OOM growth that was causing the crashes.
2. **Graceful SIGTERM flush.** Docker sends SIGTERM on `docker stop` / compose restarts. A signal handler runs `stop()` first, so *clean* shutdowns always write complete WAVs + manifest and leave no orphan behind.
3. **Startup auto-recovery.** On boot the bot scans for *orphans* — recording directories that have a `.recording` marker but no `manifest.json` (i.e. captured to disk but never stopped, because the process died). Each is rebuilt into WAVs + a manifest flagged `"recovered": true`. To avoid silently truncating a still-live meeting (e.g. the bot restarted while a call was ongoing), only orphans whose session started **more than 5 minutes ago** are auto-recovered; newer ones are left alone and can be recovered manually via `bot_core.voice_recorder.recover_orphans()`.

What's durable vs. what can still be lost:

| Shutdown type | Outcome |
|---|---|
| `/stop_recording` (normal) | Complete WAVs + manifest; logs cleaned up |
| `docker stop` / compose restart (SIGTERM) | Complete WAVs + manifest via the flush handler |
| OOM kill / segfault / power loss | Orphan left on disk → auto-recovered at next startup (if >5 min old). Audio is complete up to the last flushed frame; only in-flight frames at the instant of death are lost. Decode/decrypt failure counts aren't available for recovered runs |

A recovered `manifest.json` has `"recovered": true` and a `recovery_note`; per-speaker `decode_failures` / `decrypt_failures` are `null` (not known after a crash). Everything else — timeline placement, WAV format, speaker attribution by `user_id` — is identical to a normal stop.

## Output format

Each recording directory contains:

```
recordings/recording_20260831-014251/
├── manifest.json          # session + per-speaker metadata (see below)
├── transcript.json        # STT results (written after /stop_recording, if enabled)
├── MeleeChan_268856797626892288.wav
└── ...                    # one WAV per speaker who produced audio
```

While a recording is in progress the directory also holds transient durability files — a `.recording` session marker and one `<name>_<user_id>.log` per speaker (the raw decoded frames, streamed to disk as they arrive). These are removed automatically once the recording is finalized (by `stop()`, a clean shutdown, or crash recovery), so a *finished* directory contains only the WAVs + manifest (+ transcript). If you ever see them left behind, that's an orphan awaiting recovery.

WAV files are **48 kHz, mono, 16-bit PCM** — the format STT engines expect. The filename is `<sanitized_display_name>_<user_id>.wav` (non-alphanumeric characters become `_`, name truncated to 40 chars; falls back to `user-<id>` if no name could be resolved).

### Timeline model

All speakers' WAVs share one time base: the moment `/start_recording` fired. A speaker's first frame is anchored at its arrival offset, and every subsequent frame is placed **exactly 20 ms** after the previous one (one nominal Opus frame).

This is deliberate, not an approximation: voice packets arrive with network jitter (intervals anywhere from ~15 to ~40 ms), while each packet holds a fixed 20 ms of audio. Placing frames by raw arrival time would punch silence holes into the middle of words and make bursty arrivals overwrite each other — which is exactly the "gappy/rough" artifact seen in early recordings. The nominal grid removes it entirely.

Consequences:

- `first_speech_offset_s` / `last_speech_offset_s` tell you where that speaker's audio sits on the shared timeline, so overlapping speech across speakers can be interleaved for reconstruction.
- The WAV length is *anchor + frames × 20 ms* — network jitter never inflates file length.
- If a packet is truly lost (decrypt/decode failure), the gap shows up as silence at that point and is counted in the manifest.

## `manifest.json` schema

```jsonc
{
  "guild_id": 123,
  "channel_id": 456,
  "channel_name": "general-voice",
  "started_at": 1788133000.123,   // unix epoch (absolute)
  "ended_at": 1788133020.456,
  "duration_s": 20.333,           // wall-clock session length
  "sample_rate": 48000,
  "channels": 1,
  "bit_depth": 16,
  "encryption_mode": "aead_xchacha20_poly1305_rtpsize",
  "total_packets_seen": 1000,
  "unknown_ssrc_packets_dropped": 0,
  "unmapped_ssrcs_seen": [],      // SSRCs that arrived with no op-5 mapping
  "ssrc_to_user_at_stop": {"987654321": 268856797626892288},
  "ssrc_packet_counts": {"987654321": 1000},
  "pending_packets_replayed": 3,  // packets that arrived before their op-5 mapping
  "stage_failures": {"transport": 0, "dave": 0, "decode": 0},
  "first_stage_error": {},        // first failure text per stage (if any)
  "dave_state": {                 // diagnostics: E2EE negotiation state
    "mode": "aead_xchacha20_poly1305_rtpsize",
    "dave_present": true,
    "dave_ready": true
  },
  "passthrough_frames_recovered": 48,   // unencrypted frames recovered directly
  "passthrough_silence_fallbacks": 0,   // replaced with DTX silence (decoder sync)
  "speakers": [
    {
      "user_id": 268856797626892288,
      "display_name": "MeleeChan",
      "ssrc": 987654321,
      "wav_file": "MeleeChan_268856797626892288.wav",
      "wav_path": "/abs/path/to/wav",
      "first_speech_offset_s": 2.704,   // offset from session start
      "last_speech_offset_s": 18.634,
      "spoken_duration_s": 15.936,      // frames × 20 ms (nominal)
      "frames_captured": 797,
      "decode_failures": 0,
      "decrypt_failures": 2,
      "jitter_gap_frames": 141,         // arrivals >30 ms late (diagnostic only)
      "jitter_overlap_frames": 0        // arrivals <10 ms early (diagnostic only)
    }
  ]
}
```

`jitter_*` fields describe the *network*, not the output: frames are always written on the nominal grid, so these numbers never affect audio quality — they just tell you how rough the connection was.

## Speech-to-text

Implemented in [`bot_core/transcriber.py`](../bot_core/transcriber.py). STT runs on the **same OpenAI-compatible backend** as chat (same `INFER_URL` / `INFER_API_KEY`) via its `/v1/audio/transcriptions` endpoint — no extra service to run.

Flow after `/stop_recording` (http backend):

```
per-speaker WAV ──► downmix/resample to 16 kHz mono (numpy)
                 ──► trim leading/trailing silence below STT_SILENCE_DBFS
                 ──► split into chunks that fit under STT_MAX_UPLOAD_MB /
                     STT_CHUNK_SECONDS (recursively, so long files always fit)
                 ──► POST each chunk to /v1/audio/transcriptions (verbose_json)
                 ──► merge segments back onto the shared timeline
                 ──► transcript.json next to manifest.json
                 ──► active session's notes (if a session is running)
                 ──► follow-up Discord message with per-speaker preview + file
```

(The `local` backend skips trim/chunking — faster-whisper runs in-process on the
full WAV and returns real segment timestamps directly.)

Key details:

- **16 kHz resampling.** Whisper-class models want 16 kHz, so each WAV is resampled to 16 kHz mono before upload — a ~3x size reduction.
- **Silence trim (http).** The recorder writes a *full-length* WAV with the first/last frames zero-padded out to the session start/end, so a short meeting still produces a long file. `STT_TRIM_SILENCE` (on by default) trims everything quieter than `STT_SILENCE_DBFS` from both ends before upload; the removed span is added back as an offset so segment timestamps land on the correct point of the shared timeline.
- **Chunked upload (http).** A speaker's audio is split into chunks that each fit under `STT_MAX_UPLOAD_MB` (default 25) and/or `STT_CHUNK_SECONDS` (default 600), so arbitrarily long recordings transcribe instead of hitting the backend's per-request size cap. Chunks are uploaded sequentially and their segments merged back onto one clock, so the interleaved transcript is unaffected.
- **Per-speaker attribution comes free.** Each file is one person, so no diarization is needed; `transcript.json` maps text back to speaker by `user_id`/`display_name` from the manifest.
- **Sequential processing.** One local backend = one inference slot, so speakers are transcribed one at a time (same reasoning as the global AI lock).
- **Per-speaker error isolation.** A failing file (e.g. model not downloaded) is reported in `transcript.json` and the Discord message; the other speakers still get transcribed.
- **Automatic session notes.** When a session is active (`/start_session`) and `STT_ADD_TO_SESSION` is on (default), the finished transcript is appended to the notes of the session that was active **when the recording stopped** — via [`bot_core.sessions.add_transcript()`](../bot_core/sessions.py) as timestamped note bullets of ~1800 chars each, so long transcripts stay displayable in `/session_notes` and RAG-searchable. The target is pinned at stop time: if you end that session (or start a new one) while STT is still running, the transcript still lands in the original session's notes file. The same text is what `transcript.txt` contains: chronological `[mm:ss] Speaker: line` with per-segment timestamps (local backend), or one attributed block per speaker (http backend). If no session was active when the recording stopped, the transcript is still posted to the channel but nothing is added; set `STT_ADD_TO_SESSION=0` to opt out entirely.

### Session notes

When a session is active (started with `/start_session`) and `STT_ADD_TO_SESSION` is on (default), the finished transcript is appended to that session's notes automatically once transcription completes — no extra command needed. The text is identical to `transcript.txt`: chronological `[mm:ss] Speaker: line` entries when the local backend provides segment timestamps, or one attributed block per speaker on the http backend. Long transcripts are split into timestamped note bullets of ~1800 chars each (`part 1/N`, `continued, part N/N`), so they stay readable in `/session_notes` and index cleanly for RAG.

The target session is pinned at **stop time**: the transcript always goes to the session that was active when `/stop_recording` ran, even if it ends (or is replaced) before transcription finishes. If no session was active at that moment, nothing is added (the transcript is still posted to the channel and saved on disk).

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `STT_ENABLED` | `1` | `0` = record only, never transcribe |
| `STT_BACKEND` | `local` | `local` (faster-whisper) or `http` (OpenAI-compatible endpoint via `STT_URL`) |
| `STT_MODEL` | `qwen3-asr-1.7b` | slug accepted by the backend's `/v1/audio/transcriptions` (unsloth-studio: `tiny`, `base`, `small`, `large-v3-turbo`, `large-v3`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, or any HF `owner/model`) |
| `STT_LANGUAGE` | *(empty)* | force a language (e.g. `en`, `de`); empty = auto-detect |
| `STT_TIMEOUT` | `300` | per-file HTTP timeout in seconds |
| `STT_MAX_UPLOAD_MB` | `25` | http: max bytes per `/v1/audio/transcriptions` upload; audio is chunked to fit. `0` = no size-based splitting |
| `STT_CHUNK_SECONDS` | `600` | http: also split each speaker's audio into chunks of at most this many seconds (bounds per-request latency). `0` = never split by time |
| `STT_TRIM_SILENCE` | `1` | http: `0` = don't trim leading/trailing silence before upload |
| `STT_SILENCE_DBFS` | `-45` | http: silence threshold (dBFS) for trimming — samples quieter than this are treated as silence |
| `STT_ADD_TO_SESSION` | `1` | `0` = don't append the finished transcript to the active session's notes |

> **Model must be downloaded first** (unsloth-studio: Settings → Voice). A missing model surfaces as a per-speaker error like `STT model 'small' is not downloaded.` — the recording itself is always safe on disk.

### `transcript.json` schema

```jsonc
{
  "model": "qwen3-asr-1.7b",
  "language_requested": null,        // STT_LANGUAGE, or null when auto
  "started_at": 1788133050.123,
  "finished_at": 1788133055.4,
  "elapsed_s": 5.3,
  "speakers": [
    {
      "user_id": 268856797626892288,
      "display_name": "MeleeChan",
      "wav_file": "MeleeChan_268856797626892288.wav",
      "text": "So, Test, Test, wieder Recorden. ...",
      "language": null,              // backend-reported (null when not provided)
      "elapsed_s": 5.3,
      "error": null,                 // set instead of text on failure
      "segments": [                  // present when the backend returns timestamps
        {"start": 2.704, "end": 4.1, "text": "So, Test, Test"}
      ]                              // times are on the shared recording timeline
    }
  ]
}
```

The on-disk `manifest.json` also gains `"transcript": "transcript.json"` and `"stt_model"` once transcription completes.

## How it works

Discord voice audio arrives over UDP as RTP packets. The SSRC in each packet header doesn't say *who* is speaking; that mapping is published by the voice WebSocket:

```
UDP packet ──► 1. parse RTP header (SSRC)
             ──► 2. transport decrypt (XChaCha20-Poly1305 or AES-256-GCM,
                   "rtpsize" layout: [header][ciphertext][tag(16)][nonce(4)])
             ──► 3. DAVE E2EE decrypt (mandatory since 2026-03-01) via the
                   davey session discord.py maintains
             ──► 4. Opus decode → 48 kHz stereo PCM → downmix to mono
             ──► 5. buffer per speaker, tagged with arrival time

voice WS op-5 (Speaking) ──► SSRC → user_id mapping
voice WS op-11/13 ──► client connect/disconnect bookkeeping
```

Key implementation details:

- **SSRC → user mapping.** Voice-gateway v8 includes `user_id` directly in op-5, so the mapping is explicit. Packets that arrive *before* their op-5 (observed ~1 ms early in live testing) are buffered per SSRC and replayed once the mapping lands (`pending_packets_replayed`).
- **DAVE framing.** For the xchacha rtpsize mode, pycord's proven pipeline strips the first 8 bytes before `davey.decrypt`; we try that variant first and keep the full payload as a fallback, recording which one won per SSRC.
- **Passthrough frames.** Discord sends ~5% of frames unencrypted even when DAVE is active (davey rejects them with `UnencryptedWhenPassthroughDisabled`). Their layout is `[raw_opus][dave_supp_block][rtp_padding]`, where the supp block ends with `<size_byte> 0xFA 0xFA` and the final byte is the RFC 3550 padding length. We recover the Opus directly; if parsing fails we substitute a DTX silence frame so the *stateful* decoder stays in sync (pycord does the same).
- **Per-speaker decoders.** Opus decoders are stateful — each speaker gets their own `opus.Decoder`, fed in frame order. Stereo is downmixed to mono by averaging L/R.
- **Threading.** The UDP callback runs on discord.py's socket-reader thread; the op-5 hook runs on the event loop. All shared state sits behind one lock, and no packet path may raise (a bad packet must never kill the reader thread).

## Troubleshooting

| Symptom | Manifest/log signal | What it means |
|---|---|---|
| No WAVs at all | `unknown_ssrc_packets_dropped` high, `unmapped_ssrcs_seen` non-empty | op-5 mapping never resolved — check `voice ws op-5` lines in `logs/bot.log` |
| Audio gappy/rough *within* words | (fixed) nominal-grid placement | If you ever see this again, compare `frames_captured × 20 ms` vs. `spoken_duration_s`, and check `stage_failures` |
| Constant silence for a speaker | `decrypt_failures` high, `first_stage_error.dave` set | DAVE/MLS state problem — the davey session isn't decrypting that user's frames |
| Occasional clicks/dropouts | small `decode_failures`, or `passthrough_silence_fallbacks` > 0 | packet loss / unparseable passthrough frame; each costs one 20 ms silence |
| WAV longer than expected | — | shouldn't happen (nominal grid); if it does, compare `duration_s` vs. `spoken_duration_s` |
| Wrong speaker name in filename | `display_name` empty → `user-<id>` fallback | member object wasn't cached at capture time; the user_id in the manifest is authoritative |
| No transcript message after stop | `STT_ENABLED=0`, or `transcribe=false`, or nobody spoke (`frames_captured=0`) | expected — check the stop message, which says whether transcription was queued |
| Transcript posted but not in session notes | no active session when the recording stopped, or `STT_ADD_TO_SESSION=0` | expected — start a session with `/start_session` before stopping the recording (or re-enable the flag); nothing is lost, the transcript is still on disk + in the channel |
| Transcript landed in an ended session's notes | the session was active at stop time but ended before STT finished | intended — transcripts are pinned to the session of the recording; check that session's file under `<KB_PATH>/session_notes/` |
| Transcript error: `STT model '...' is not downloaded` | per-speaker `error` in `transcript.json` | download it in unsloth-studio Settings → Voice (or fix `STT_MODEL`) |
| Transcript error: `Audio is too large` | per-speaker `error` in `transcript.json` | only if a *single* chunk still exceeds the cap — raise `STT_MAX_UPLOAD_MB` (or lower `STT_CHUNK_SECONDS`) so audio splits smaller; normal-length recordings now chunk automatically and never hit this |
| Recording lost after a crash / restart | `.recording` marker + `*.log` files present, no `manifest.json` | an orphan — it will be auto-recovered at the next startup once it's >5 min old (see [Crash recovery](#crash-recovery)); or run `recover_orphans()` to rebuild it now |
| Recovered manifest shows `decode_failures: null` | `"recovered": true` in `manifest.json` | expected — failure counters live in RAM and don't survive a hard crash; the audio itself is intact up to the last flushed frame |

The first frame per SSRC is hexdumped to the log (`UDP frame first-seen`) and the first failure per pipeline stage is logged loudly (`PIPELINE <stage> FAILED (first): ...`), so `logs/bot.log` plus the manifest are usually enough to diagnose any bad recording without re-running it.

## Testing

The pipeline is unit-tested end-to-end without a live Discord connection — packets are synthesised with the same crypto primitives Discord uses:

```bash
python3 -m pytest tests/test_voice_recorder.py -q
```

Covers transport decryption (both modes), stereo→mono downmix, Opus decode, SSRC mapping from op-5/11/13, passthrough recovery, the full packet-in → PCM-out path, timeline placement (nominal grid, no-overwrite, jitter diagnostics), `stop()` WAV + manifest output, and crash durability: on-disk frame-log round-trip (incl. a trailing partial record from a simulated crash), `stop()`/`discard()` log cleanup, `recover_orphans()` (old-orphan rebuild, 5-minute threshold skip, completed-recording skip, multi-speaker + empty-log handling), and the SIGTERM flush handler.

The STT layer is tested separately with a fake backend client:

```bash
python3 -m pytest tests/test_transcriber.py tests/test_stop_recording_stt.py -q
```

Covers 48 kHz→16 kHz conversion (resample, stereo downmix, size shrink), silence trim (`_trim_silence`), chunk planning (`_plan_chunks`), WAV encoding (`_pcm_to_wav_bytes`), segment merge (`_merge_segments`), the full http pipeline end-to-end (trim → resample → chunk → merge with a fake backend), `transcribe_wav` success/error paths, per-speaker aggregation in `transcribe_recording`, `transcript.json` output + manifest pointer, and the `/stop_recording` wiring (spawn/skip logic, result delivery).
