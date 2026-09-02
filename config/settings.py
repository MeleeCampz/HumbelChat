"""Application settings — all environment variables with defaults, typed."""
from __future__ import annotations

import os
import pathlib

# ═══ Helper functions ════════════════════════════════════════════════════

def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default


def _history_reset_flag(value: str | None) -> bool:
    """Return True if *value* is the sentinel string "clear" (case-insensitive).

    Used to read CHAT_HISTORY_RESET, which is a boolean-ish env var:
    "clear" (or "1"/"true") means "clear history on startup", anything
    else means "keep it". Previously this was a ``str | None`` helper whose
    only meaningful return value was the string ``"clear"`` — confusing and
    over-typed (see code review §3.6).
    """
    if not value:
        return False
    return value.strip().lower() in ("clear", "1", "true", "yes")


# ════════════════════════════════════
#  DISCORD
# ════════════════════════════════════
DISCORD_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")

# ════════════════════════════════════
#  AI PROVIDER (any OpenAI-compatible)
# ════════════════════════════════════
INFER_URL: str = os.getenv("INFER_URL", "http://127.0.0.1:11434/v1")
INFER_API_KEY: str = os.getenv("INFER_API_KEY", "")  # sometimes empty for local

# ════════════════════════════════════
#  CHARACTER defaults (per-char in characters.json)
# ════════════════════════════════════
DEFAULT_MODEL: str | None = os.getenv("MODEL_NAME")
DEFAULT_SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", "")
CONTEXT_WINDOW: int = _safe_int(os.getenv("CONTEXT_WINDOW"), 10)
REQUEST_TIMEOUT: int = _safe_int(os.getenv("AI_REQUEST_TIMEOUT"), 120)
# §3.9: lightweight backend liveness probe. 0 = only probe once on startup;
# a positive value starts a periodic background check at that interval.
AI_HEALTH_CHECK_INTERVAL: int = _safe_int(os.getenv("AI_HEALTH_CHECK_INTERVAL"), 0)
AI_HEALTH_CHECK_TIMEOUT: int = _safe_int(os.getenv("AI_HEALTH_CHECK_TIMEOUT"), 5)
MAX_TOKENS: int = _safe_int(os.getenv("MAX_TOKENS"), 2000)
MAX_TOKENS_HARD_CAP: int = _safe_int(os.getenv("MAX_TOKENS_HARD_CAP"), 4096)

# Fallback models tried (in order) when DEFAULT_MODEL fails during summarize/translate.
# Comma-separated list of model slugs.
FALLBACK_MODELS: list[str] = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()
]

# ── Session overview prompt (customizable) ───────────────────────────────
#: Default system prompt used by /end_session to write the session overview.
DEFAULT_SESSION_SUMMARY_PROMPT: str = (
    "You write concise session overviews. Given the notes and recent chat "
    "of a work session, produce a short overview (max ~250 words) with: "
    "what was done, key points/decisions, and open items or follow-ups for "
    "the next session. Use markdown bullet points. Return ONLY the overview."
)

#: Override for SESSION_SUMMARY_PROMPT — set in .env to customize how the
#: /end_session AI overview is written. Empty = use DEFAULT_SESSION_SUMMARY_PROMPT.
SESSION_SUMMARY_PROMPT: str = os.getenv("SESSION_SUMMARY_PROMPT", "")

# ════════════════════════════════════
#  BOT BEHAVIOUR
# ════════════════════════════════════
BOT_PREFIX: str = os.getenv("BOT_PREFIX", "!ai")
# §3.6: boolean flag instead of the old ``"clear" | None`` sentinel string.
CHAT_HISTORY_RESET: bool = _history_reset_flag(os.getenv("CHAT_HISTORY_RESET"))

# Beyond20-style embed rendering for /ai replies.
# Structured replies (headings, tables, lists) become a discord.Embed with a
# title, description and inline fields — the way the Beyond20 bot formats
# rolls. Plain prose still works; tiny/empty replies fall back to text.
# Set EMBED_FORMAT=0 in .env to restore classic plain-text delivery.
EMBED_FORMAT: bool = os.getenv("EMBED_FORMAT", "1") not in ("0", "false", "no")

def _or_default(value: str | None, default: str) -> str:
    """Return value if non-empty, else default (for optional path overrides)."""
    return value if value else default


# ════════════════════════════════════
#  EMBEDDING MODEL
# ════════════════════════════════════
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# ════════════════════════════════════
#  KNOWLEDGE BASE
# ════════════════════════════════════
# Repo root, resolved from this file so all paths work regardless of CWD.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

KB_PATH: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("KB_PATH"), str(_REPO_ROOT / "data" / "knowledge"))
)
CHARACTERS_FILE: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("CHARACTERS_FILE"), str(_REPO_ROOT / "characters.json"))
)

# ── Voice recording (per-speaker capture for STT) ───────────────────────────
# Where /start_recording writes its per-recording subdirectories (one WAV per
# speaker + a manifest.json with absolute timestamps). Defaults to
# <repo_root>/data/recordings.
RECORDINGS_DIR: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("RECORDINGS_DIR"), str(_REPO_ROOT / "data" / "recordings"))
)

# ── Speech-to-text ───────────────────────────────────────────────────────────
# Transcribe each speaker's WAV automatically after /stop_recording.
# Set STT_ENABLED=0 to keep recording without transcribing.
STT_ENABLED: bool = os.getenv("STT_ENABLED", "1") not in ("0", "false", "no")
# Which engine performs the transcription:
#   local — faster-whisper on this machine (default). Real per-segment
#           timestamps -> interleaved chronological transcript, no upload cap.
#   http  — an OpenAI-compatible /v1/audio/transcriptions endpoint (e.g. a
#           separate Whisper API container, addressed via STT_URL).
#           Requests verbose_json + segment timestamps; ~25 MB upload cap.
STT_BACKEND: str = os.getenv("STT_BACKEND", "local")
# Model name for the local backend: any faster-whisper model id, e.g.
# tiny / base / small / medium / large-v3 / large-v3-turbo (or an HF repo in
# owner/model form). Downloaded on first use (~1.6 GB for large-v3-turbo).
STT_LOCAL_MODEL: str = os.getenv("STT_LOCAL_MODEL", "large-v3-turbo")
# Base URL of the OpenAI-compatible STT backend (used only when
# STT_BACKEND=http). Point this at your Whisper API container, e.g.
#   http://192.168.1.50:8000/v1        (separate container on the network)
#   http://whisper-api:8000/v1          (sidecar in the same compose file)
# Falls back to INFER_URL when unset, so a single backend serving both
# chat and STT keeps working.
STT_URL: str = os.getenv("STT_URL") or INFER_URL
# STT model slug as accepted by the backend's /v1/audio/transcriptions route
# (used only when STT_BACKEND=http). unsloth-studio defaults: tiny, base,
# small, large-v3-turbo, large-v3, qwen3-asr-0.6b, qwen3-asr-1.7b.
STT_MODEL: str = os.getenv("STT_MODEL", "qwen3-asr-1.7b")
# Force a language for transcription (e.g. "en", "de"); empty = auto-detect.
STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "")
# Per-file HTTP timeout in seconds (long recordings + cold model load).
STT_TIMEOUT: int = _safe_int(os.getenv("STT_TIMEOUT"), 300)
# ── http-backend upload handling (trim -> resample -> chunk) ────────────────
# Hard cap on a single /v1/audio/transcriptions upload, in MB. A speaker's
# audio is split into chunks that each fit under this cap, so arbitrarily long
# recordings still transcribe. 0 = unlimited (no size-based splitting). The
# default 25 matches the ~25 MB per-request limit most Whisper backends impose.
STT_MAX_UPLOAD_MB: int = _safe_int(os.getenv("STT_MAX_UPLOAD_MB"), 25)
# Also split a speaker's audio into chunks of at most this many seconds, even
# when they'd fit under the size cap — useful to bound per-request latency on
# very long recordings. 0 = never split by time (size cap only).
STT_CHUNK_SECONDS: int = _safe_int(os.getenv("STT_CHUNK_SECONDS"), 600)
# Trim leading/trailing silence below STT_SILENCE_DBFS from each speaker's
# audio before resampling/uploading (1 = on, 0 = off). The recorder writes a
# full-length WAV with the first/last frames zero-padded to the session start/
# end; trimming removes that padding so short meetings don't upload hours of
# silence. Segment timestamps are shifted back onto the shared timeline.
STT_TRIM_SILENCE: bool = os.getenv("STT_TRIM_SILENCE", "1") not in ("0", "false", "no")
# Silence threshold for STT_TRIM_SILENCE, in dBFS (negative). Samples with a
# level below this are treated as silence. -45 is well under typical speech
# but above the digital-noise floor of quiet channels.
STT_SILENCE_DBFS: float = _safe_float(os.getenv("STT_SILENCE_DBFS"), -45.0)
# Append the finished transcript to the ACTIVE session's notes automatically
# when transcription completes (see bot_core.sessions.add_transcript). The
# full transcript is stored as timestamped note bullets, so it shows up in
# /session_notes and stays RAG-searchable. Set STT_ADD_TO_SESSION=0 to keep
# transcripts out of the session notes.
STT_ADD_TO_SESSION: bool = os.getenv("STT_ADD_TO_SESSION", "1") not in ("0", "false", "no")

CHUNK_TARGET: int = _safe_int(os.getenv("CHUNK_SIZE"), 2000)
RAG_MAX_DOCS: int = _safe_int(os.getenv("RAG_MAX_DOCS"), 4)
RAG_RETRIEVAL_METHOD: str = os.getenv("RAG_RETRIEVAL_METHOD", "vector").lower()
RAG_MAX_CHARS: int = _safe_int(os.getenv("RAG_MAX_CHARS"), 24000)
RAG_WINDOW_LINES: int = _safe_int(os.getenv("RAG_WINDOW_LINES"), 80)

# ── Low-confidence query rewriting (vector path only) ─────────────────────
# When the top vector similarity score for a query is below RAG_REWRITE_MIN_SCORE,
# the rewriter asks the LLM for up to RAG_QUERY_MAX_EXPANSIONS alternative phrasings,
# embeds them in one batch, and merges the rankings (reciprocal rank fusion).
# Confident queries pay nothing extra.  Set RAG_QUERY_REWRITER=0 to disable entirely.
RAG_QUERY_REWRITER: bool = os.getenv("RAG_QUERY_REWRITER", "1") not in ("0", "false", "no")
RAG_REWRITE_MIN_SCORE: float = float(os.getenv("RAG_REWRITE_MIN_SCORE", "0.35") or 0.35)
RAG_QUERY_MAX_EXPANSIONS: int = _safe_int(os.getenv("RAG_QUERY_MAX_EXPANSIONS"), 3)
# Wall-clock budget (seconds) for the LLM rewrite call itself.
RAG_REWRITE_BUDGET_SECONDS: int = _safe_int(os.getenv("RAG_REWRITE_BUDGET_SECONDS"), 10)

# ════════════════════════════════════
#  INPUT VALIDATION
# ════════════════════════════════════
# Hard cap on user prompt length (chars).  Prevents a user from pasting
# 100 K+ chars which would blow past any context window once RAG is added.
MAX_INPUT_CHARS: int = _safe_int(os.getenv("MAX_INPUT_CHARS"), 50_000)

# ════════════════════════════════════
#  RATE LIMITING
# ════════════════════════════════════
# Max AI requests per user in a sliding window.
AI_RATE_LIMIT_MAX: int = _safe_int(os.getenv("AI_RATE_LIMIT_MAX"), 5)
AI_RATE_LIMIT_WINDOW: int = _safe_int(os.getenv("AI_RATE_LIMIT_WINDOW"), 60)


