# Voice Recording

The bot can capture a Discord voice channel **per speaker** for later speech-to-text (STT). Each participant gets their own 16-bit PCM WAV file, and a JSON manifest ties everything to absolute timestamps so a multi-speaker conversation can be reconstructed on a shared timeline.

Implemented in [`bot_core/voice_recorder.py`](../bot_core/voice_recorder.py) (capture pipeline) and [`commands/recording_commands.py`](../commands/recording_commands.py) (slash-command layer).

## Commands

### `/start_recording`

Joins the voice channel **you** are currently in and starts capturing. Requirements:

- You must be in a guild (not DMs) and in a voice channel.
- The bot needs **Connect** and **Speak** permissions in that channel.

The bot joins automatically; if it was already in a different voice channel of the same server, it moves to yours. Each recording goes into its own subdirectory: `recordings/recording_YYYYMMDD-HHMMSS/`.

### `/stop_recording [leave_channel: true]`

Stops capturing, writes the WAV files + `manifest.json`, and replies with a per-speaker summary (and attaches the manifest). By default the bot then leaves the voice channel so it doesn't occupy a slot.

> **Note:** recordings are written to disk on stop — nothing is saved if you never stop. Long recordings can take a moment to write (the command defers to stay under Discord's 15 s response window).

## Output format

Each recording directory contains:

```
recordings/recording_20260831-014251/
├── manifest.json          # session + per-speaker metadata (see below)
├── MeleeChan_268856797626892288.wav
└── ...                    # one WAV per speaker who produced audio
```

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

The first frame per SSRC is hexdumped to the log (`UDP frame first-seen`) and the first failure per pipeline stage is logged loudly (`PIPELINE <stage> FAILED (first): ...`), so `logs/bot.log` plus the manifest are usually enough to diagnose any bad recording without re-running it.

## Testing

The pipeline is unit-tested end-to-end without a live Discord connection — packets are synthesised with the same crypto primitives Discord uses:

```bash
python3 -m pytest tests/test_voice_recorder.py -q
```

Covers transport decryption (both modes), stereo→mono downmix, Opus decode, SSRC mapping from op-5/11/13, passthrough recovery, the full packet-in → PCM-out path, timeline placement (nominal grid, no-overwrite, jitter diagnostics), and `stop()` WAV + manifest output.
