"""Voice recording sub-package (P3 #35).

Split out of the monolithic ``bot_core.voice_recorder`` (1 294 lines) into
focused modules:

  * :mod:`voice.capture` — audio constants, the per-speaker capture buffer
    + on-disk frame log, and timeline WAV layout (crash durability).
  * :mod:`voice.dave`    — RTP/DAVE decryption, Opus decode, and the audio
    primitives (pure, unit-testable).
  * :mod:`voice.session` — the :class:`VoiceRecorder` state machine and the
    bot wiring (``attach_to_bot`` / ``recorder_voice_cls``).
  * :mod:`voice.recover` — crash recovery (rebuild a recording whose
    process died mid-capture).

``bot_core.voice_recorder`` remains a thin re-exporting facade so every
existing import path is unchanged.
"""

__all__ = [
    "capture",
    "dave",
    "recover",
    "session",
]
