"""Voice-channel audio capture for later STT transcription (facade).

P3 #35: the monolithic module was split into the :mod:`bot_core.voice`
package (``capture`` / ``dave`` / ``recover`` / ``session``). This module
now re-exports the full public + internal surface so every existing
``from bot_core.voice_recorder import ...`` (production code and the test
suite) keeps working unchanged. New code should import from
:mod:`bot_core.voice.*` directly.
"""
from __future__ import annotations

import os  # noqa: F401  (tests patch ``bot_core.voice_recorder.os``)

from bot_core.voice.capture import (  # noqa: F401
    CHANNELS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    VoiceRecorderError,
    _Speaker,
    _SpeakerLog,
    _wav_filename,
    _write_timeline_wav,
    _write_timeline_wav_from_frames,
)
from bot_core.voice.dave import (  # noqa: F401
    _as_int,
    _decode_opus,
    _decrypt_dave,
    _decrypt_transport,
    _downmix_to_mono,
    _extract_passthrough_opus,
)
from bot_core.voice.recover import (  # noqa: F401
    recover_orphans,
)
from bot_core.voice.session import (  # noqa: F401
    VoiceRecorder,
    _make_name_resolver,
    _wire_voice_client,
    attach_to_bot,
    recorder_voice_cls,
)

__all__ = [
    "VoiceRecorder",
    "VoiceRecorderError",
    "SAMPLE_RATE",
    "CHANNELS",
    "SAMPLE_WIDTH",
    "FRAME_SAMPLES",
    "attach_to_bot",
    "recorder_voice_cls",
    "recover_orphans",
    "_Speaker",
    "_wire_voice_client",
    "_make_name_resolver",
    "_SpeakerLog",
    "_as_int",
    "_decrypt_transport",
    "_extract_passthrough_opus",
    "_decrypt_dave",
    "_decode_opus",
    "_downmix_to_mono",
    "_wav_filename",
    "_write_timeline_wav",
    "_write_timeline_wav_from_frames",
    "os",
]
