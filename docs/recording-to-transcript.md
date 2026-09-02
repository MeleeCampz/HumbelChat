# Voice Recording → Final Transcript: End-to-End

This document traces the **entire pipeline** from `/start_recording` to a
finished, timestamped, RAG-searchable transcript — step by step, with the code
location for each stage. It is the single place to understand *why* the system
is shaped the way it is.

```
/start_recording                     /stop_recording
      │                                    │
      ▼                                    ▼
┌───────────────────  CAPTURE  ───────────────────┐
│ 1. arm recorder          3. transport decrypt   │
│ 2. join voice            4. DAVE E2EE decrypt   │
│                        5. Opus decode→mono      │
│                        6. stream to disk (.log) │
└──────────────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ 7. stop(): lay frames │
                        │   on a 20 ms grid     │
                        │   → 1 WAV/speaker     │
                        │   + manifest.json     │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ 8. transcribe (STT):  │
                        │   resample→trim→      │
                        │   chunk→upload→merge  │
                        │   → transcript.json   │
                        │   + transcript.txt    │
                        └──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        transcript.json      session notes           Discord message
        (per-speaker +       (RAG-searchable)       (preview + files)
        segments)
```

---

## Part A — Capture

### A1. `/start_recording` arms the recorder *before* joining
`commands/recording_commands.py :: handle_start_recording`

The recorder must be **armed before the voice join**. Discord announces the
initial `SSRC → user_id` mapping (`op-11`/`op-5`) *during* the handshake, which
happens inside `channel.connect()`. If `start()` ran afterwards it would clear
the mapping and **every packet would be dropped as "unknown SSRC"** — a silent
failure that produces no WAVs. The code calls `recorder.start(...)` first, then
`_ensure_bot_in_channel(...)`. A regression test
(`test_start_called_before_channel_connect`) pins this ordering.

A fresh per-recording directory is created:
`RECORDINGS_DIR/recording_YYYYMMDD-HHMMSS/` (default
`data/recordings/`, configurable via `RECORDINGS_DIR`).

`VoiceRecorder.start()` (`bot_core/voice_recorder.py`) resets all state and —
**first thing** — writes a `.recording` marker file into that directory. The
marker is what makes a later crash recoverable (Part C).

### A2. The bot joins, wired for capture
`_ensure_bot_in_channel` connects with
`channel.connect(cls=recorder_voice_cls(bot))`. That builds a
`discord.VoiceClient` subclass whose `__init__` (i.e. *before* the handshake)
installs two hooks via `_wire_voice_client`:

1. **A voice-WebSocket hook** (`state.hook = recorder.on_voice_ws`) to see
   every `op-5`/`op-11`/`op-13` event.
2. **A raw UDP socket listener** (`state.add_socket_listener(_on_udp)`) to see
   every audio packet.

### A3. Who is speaking? — SSRC → user_id
`on_voice_ws`

Audio packets carry an **SSRC** (synchronisation source id), not a user id.
The voice gateway maps SSRC → user in **op-5 (Speaking)** events; voice-gateway
v8 includes `user_id` directly, so the mapping is explicit. `op-11`/`op-13`
track client connect/disconnect. There is a ~1 ms race (a packet can arrive
*before* its op-5), so unmatched packets are **buffered per SSRC** (up to 20)
and replayed the moment the mapping lands (`pending_packets_replayed` in the
manifest).

### A4. Decrypting one packet (the 3 layers)
`handle_packet` → `_process_mapped_packet`

```
raw UDP packet
   │ 1. parse RTP header (SSRC)                _process_mapped_packet
   │ 2. transport decrypt (rtpsize layout)     _decrypt_transport
   │      XChaCha20-Poly1305 or AES-256-GCM
   │ 3. DAVE E2EE decrypt (mandatory since     _decrypt_dave
   │      2026-03-01) via the davey session
   │ 4. Opus decode → 48 kHz stereo PCM        _decode_opus
   └ 5. downmix to mono (avg L/R)              _downmix_to_mono
```

- **Transport layer** (`_decrypt_transport`): the "rtpsize" layout is
  `[header][ciphertext][auth_tag(16)][nonce_counter(4)]`; both current modes
  (xchacha and aes-gcm) are handled.
- **DAVE layer** (`_decrypt_dave`): for the xchacha rtpsize mode, pycord's
  proven pipeline strips the first 8 bytes before `davey.decrypt`; we try that
  variant first and keep the full payload as a fallback, recording which won.
- **Passthrough frames**: Discord sends ~5% of frames *unencrypted* even with
  DAVE on. Their layout is `[raw_opus][dave_supp_block][rtp_padding]`;
  `_extract_passthrough_opus` recovers the Opus directly, and if that fails we
  substitute a **DTX silence frame** (`b"\xf8\xff\xfe"`) so the *stateful* Opus
  decoder stays in sync (matching pycord). Each fallback is counted in the
  manifest.

Each speaker gets their **own** `opus.Decoder` (decoders are stateful and must
be fed in frame order).

### A5. Stream to disk as it decodes (crash durability)
`_process_mapped_packet`, step 5 → `_SpeakerLog`

Every decoded frame is **immediately appended** to an append-only binary log
`<name>_<user_id>.log` in the recording directory and flushed:

```
header : "DRAV" + u32 version(1) + u32 rate + u32 ch + u32 width + u32 frame_samples
record : i64 ts_us | u32 pcm_len | pcm bytes        (repeated)
```

This is the heart of crash safety (Part C): frames no longer sit in an unbounded
in-memory list (which was ~96 KB/s per speaker and caused the OOM kills). A
crash now loses only the single frame in flight at the instant of death.

---

## Part B — Writing the WAVs (timeline model)

### B1. `stop()` lays frames on a shared 20 ms grid
`VoiceRecorder.stop()` → `_write_timeline_wav_from_frames`

`stop()` reads every speaker's `.log` back (`_SpeakerLog.read_frames` — the
same reader used by crash recovery, so both paths produce **identical**
output), then writes one WAV per speaker. **The single most important design
decision:**

- All speakers share **one time base**: the moment `/start_recording` fired
  (`origin`). A speaker's first frame is anchored at its arrival offset, and
  every subsequent frame is placed **exactly one nominal frame (20 ms)** after
  the previous one — *never* by raw arrival time.
- Why: voice packets arrive with jitter (15–40 ms intervals) but each holds a
  fixed 20 ms of audio. Placing by arrival time would punch **silence holes
  into the middle of words** (late arrivals) and make **bursts overwrite** each
  other. The nominal grid removes that entirely.
- Because every speaker's file is aligned to the same `origin`, **file time 0
  == recording start for all of them** — so segment timestamps from different
  speakers live on one clock and can be merged directly (this is what makes the
  interleaved transcript possible with no per-speaker offset correction).
- The WAV length is *anchor + frames × 20 ms* (nominal), so network jitter
  never inflates file length. A truly lost packet shows up as 20 ms of silence
  and is counted in `jitter_gap_frames` / `decode_failures` (diagnostics only —
  they never affect audio quality).

### B2. `manifest.json`
`stop()` writes the manifest (schema in `voice-recording.md`): guild/channel,
`started_at`/`ended_at`/`duration_s`, per-speaker `user_id`/`display_name`/`ssrc`,
`first_speech_offset_s`/`last_speech_offset_s`/`spoken_duration_s`, and all the
diagnostics. Once WAVs + manifest are durable, the intermediate `.log` files and
`.recording` marker are **deleted** so a later crash-recovery run won't
re-process a finished recording. The returned manifest carries an absolute
`manifest_path`, which STT uses to locate the WAVs.

---

## Part C — Crash recovery (recording that died)

| Shutdown | What happens | Result |
|---|---|---|
| `/stop_recording` (normal) | `stop()` runs | complete WAVs + manifest; logs removed |
| `docker stop` / compose restart (**SIGTERM**) | `install_sigterm_flush()` handler runs `stop()` first, then exits | complete WAVs + manifest; no orphan |
| OOM / segfault / power loss | nothing runs → orphan left on disk | rebuilt at next startup |

**Orphan** = a directory under `RECORDINGS_DIR` that has a `.recording` marker
but **no** `manifest.json` (captured to disk, never stopped). On startup
(`main.py :: _recover_crashed_recordings`, called from `on_ready`),
`recover_orphans()` rebuilds each orphan's logs into WAVs + a manifest flagged
`"recovered": true` — using the *same* `_write_timeline_wav_from_frames`, so the
output is byte-identical to a clean stop.

- **5-minute threshold:** only orphans whose marker `started_at` is **> 5 min
  old** are auto-recovered. A very recent one might belong to a recording that
  is *still live* (the bot restarted while a meeting was ongoing); auto-recovering
  it would silently truncate it. Recent orphans are left alone and can be
  recovered manually via `bot_core.voice_recorder.recover_orphans()`.
- A trailing **partial record** (a frame mid-write at crash time) is detected and
  dropped — bounded loss of a single 20 ms frame.
- Recovered manifests set `decode_failures`/`decrypt_failures` to `null` (those
  counters live in RAM) and carry a `recovery_note`. Everything else — timeline,
  WAV format, `user_id` attribution — is identical.

---

## Part D — Speech-to-text (the trim → resample → chunk pipeline)

`bot_core/transcriber.py`. Triggered by `/stop_recording` when `STT_ENABLED`
and the per-command `transcribe` flag are both on and at least one speaker
captured audio. It runs as a **background task**
(`spawn_tracked_task(_run_transcription, ...)`); the stop message tells the user
it's in progress.

### D0. Two backends (`STT_BACKEND`)
- **`local`** — faster-whisper in-process: no upload, real segment timestamps,
  no size cap, VAD skips long silence. `STT_LOCAL_MODEL`.
- **`http`** — OpenAI-compatible `/v1/audio/transcriptions`. `STT_URL` (falls
  back to `INFER_URL`), reuses `INFER_API_KEY`. Model slug via `STT_MODEL`.

### D1. `transcribe_recording` — per speaker
For each speaker in the manifest it calls `transcribe_wav` **sequentially**
(one inference slot; concurrent recordings share it, like the global AI lock).
Per-speaker failures never abort the others.

### D2. `transcribe_wav` (http path) — the pipeline

```
load + resample to 16 kHz mono (_load_to_16k_mono)
   │      returns (int16 samples, 16000)  ← ALWAYS 16 kHz
   ▼
trim leading/trailing silence (_trim_silence)   [if STT_TRIM_SILENCE]
   │      → (trimmed_pcm, trim_samples)
   │      → t_start = trim_samples / 16000       (the offset to restore)
   ▼
plan chunks (_plan_chunks)
   │      each chunk fits under STT_MAX_UPLOAD_MB and/or STT_CHUNK_SECONDS
   ▼
for each chunk (start s, end e):
   │   upload WAV  (name__chunkNN.wav)
   │   POST /v1/audio/transcriptions  (verbose_json + segment timestamps)
   │   _extract_segments → (text, lang, per-chunk segs)   [times RELATIVE to this chunk]
   │   _merge_segments(segs, t_start + s/16000)  → shift onto shared timeline
   ▼
text = join(chunk texts)     segments = concatenation of merged segments
```

**Why each step exists:**

- **Resample to 16 kHz** — Whisper-class models want 16 kHz; a ~3x size cut.
  The recorder's 48 kHz WAV hits a backend's per-request cap in only ~4 minutes.
- **Silence trim** — the recorder writes a *full-length* WAV (zero-padded to the
  session start/end), so a short meeting would otherwise upload hours of
  silence. Trimming removes that; `STT_SILENCE_DBFS` (default −45) sets the
  threshold.
- **Chunking** — without it, a long speaker's audio exceeds the per-request size
  cap and the whole upload fails. Chunks make arbitrarily long recordings
  transcribable. `STT_MAX_UPLOAD_MB=0` disables the size split;
  `STT_CHUNK_SECONDS=0` disables the time split.

### D3. ⭐ Why timestamps stay correct through trim + chunk

This is the correctness question. There are two sources of offset that must both
be compensated:

1. **Trim offset** — trimming removes the leading silence, so a chunk's samples
   no longer start at recording time 0. The number of trimmed samples is divided
   by 16 kHz to get `t_start` seconds, which is **added back**.
2. **Chunk offset** — each chunk is uploaded independently, so its segment times
   are relative to *that chunk's* start (0.0). Adding the chunk's sample start
   (`s / 16000`) on top of `t_start` places it on the shared timeline.

So a merged segment = `chunk_relative_time + t_start + chunk_start_seconds`.

Because:
- the recorder already aligned **every** speaker's WAV to the shared start
  (Part B1), and
- chunks are cut at **fixed, contiguous sample offsets** (no overlap/crossfade),

the merged segments reassemble on **one clock, with no gaps and no duplicates**,
and the constant offset is exact. A sentence that began 2 s into a recording
transcribes with `start ≈ 2.0 s`, matching the other speakers.

> **This is a subtle spot.** The whole thing only holds if the array is treated
> as 16 kHz everywhere. `_load_to_16k_mono` deliberately returns the array's
> *true* rate (16000) — returning the source rate (48000) would make `t_start`
> and the chunk math drift by 3× (this exact bug was caught in testing).

**Edge:** if a backend ignores `verbose_json` and returns plain text, that
speaker simply has no `segments` — still transcribed, just without per-segment
timing.

### D4. Local backend
`_local_transcribe` runs faster-whisper off the event loop (`asyncio.to_thread`)
under a single inference lock; VAD filters silence; real segment timestamps come
back on the same shared-timeline basis (file time 0 == recording start), so no
trim/chunk/offset logic is needed.

---

## Part E — Output & delivery

`write_transcript` + `_run_transcription`.

### E1. `transcript.json` (per-speaker, with segments)
Written next to `manifest.json`. Each speaker entry has `text`, `language`,
`elapsed_s`, `error`, and — when the backend returns timestamps — `segments`
(`start`/`end`/`text`, on the shared timeline). The manifest gains
`"transcript": "transcript.json"` and `"stt_model"`.

### E2. `transcript.txt` (chronological)
`build_session_transcript` → `build_interleaved_transcript` merges **all**
speakers' timestamped segments into one sorted-by-start `[mm:ss] Speaker: line`
stream. Because all speakers share the clock (Part B1 + D3), a real multi-talk
conversation reads correctly interleaved. Speakers with no timing (a
non-`verbose_json` backend) are appended as attributed blocks at the end.

### E3. Session notes (RAG)
`_run_transcription` calls `sessions.add_transcript(text, title, session=session_at_stop)`.
The text is split into ~1800-char timestamped note bullets (`part 1/N`,
`continued, part N/N`) so long transcripts stay readable in `/session_notes` and
index cleanly for RAG (`_index_session_file`).

**Session pinning:** `handle_stop_recording` captures the *active session's dict*
at **stop time** and hands it to the background job. The transcript always lands
in the session that was active when the recording stopped — even if it ends (or
is replaced) before STT finishes. See `voice-recording.md` → *Subtleties & edge
cases* for the "no session at stop, one started mid-STT" edge case.

### E4. Discord message
A follow-up posts the per-speaker summary (text preview, or `(no speech
detected)`, or the `error`), the session-notes line, and attaches
`transcript.json` + `transcript.txt`. Best-effort (`_safe_followup`) since the
webhook can expire for very long recordings.

---

## Files on disk after a recording + STT

```
data/recordings/recording_20260831-014251/
├── manifest.json          # session + per-speaker metadata (+ "transcript" pointer)
├── transcript.json        # per-speaker text + segments (shared timeline)
├── transcript.txt         # chronological [mm:ss] Speaker: line
├── MeleeChan_268856797626892288.wav
└── ...                    # one 48 kHz mono WAV per speaker
data/knowledge/session_notes/2026-08-31_01_MySession.md   # ← transcript bullets
```

---

## Quick reference: where each thing lives

| Concern | File | Key symbol |
|---|---|---|
| Slash commands | `commands/recording_commands.py` | `handle_start_recording`, `handle_stop_recording`, `_run_transcription` |
| Capture pipeline | `bot_core/voice_recorder.py` | `VoiceRecorder.start/stop`, `handle_packet`, `_process_mapped_packet` |
| Decrypt (transport/DAVE) | `bot_core/voice_recorder.py` | `_decrypt_transport`, `_decrypt_dave`, `_extract_passthrough_opus` |
| Crash durability | `bot_core/voice_recorder.py` | `_SpeakerLog`, `recover_orphans`, `install_sigterm_flush` |
| Timeline / WAV writing | `bot_core/voice_recorder.py` | `_write_timeline_wav_from_frames` |
| STT pipeline | `bot_core/transcriber.py` | `transcribe_wav`, `transcribe_recording`, `_trim_silence`, `_plan_chunks`, `_merge_segments` |
| Transcript assembly | `bot_core/transcriber.py` | `write_transcript`, `build_session_transcript`, `build_interleaved_transcript` |
| Session notes | `bot_core/sessions.py` | `add_transcript` |
| Startup recovery | `main.py` | `_recover_crashed_recordings` (from `on_ready`) |
| Config | `config/settings.py` | `STT_*`, `RECORDINGS_DIR`, `KB_PATH` |
