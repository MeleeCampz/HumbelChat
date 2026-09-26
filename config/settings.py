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


def _safe_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean-ish env var robustly (P3 #32).

    The codebase previously scattered ad-hoc ``os.getenv(...) not in
    ("0", "false", "no")`` checks, which mis-parsed capitalised values (e.g.
    ``False``/``No`` were treated as *true*) and were inconsistent with one
    another.  This single helper is the one place that decides how bool-ish
    strings are read: explicit falsy tokens are False, explicit truthy tokens
    are True, and anything else (empty, unset, unrecognised) falls back to
    *default*.  Case and surrounding whitespace are ignored.
    """
    if value is None:
        return default
    v = value.strip().lower()
    if not v:
        return default
    if v in ("1", "true", "yes", "on", "y", "t"):
        return True
    if v in ("0", "false", "no", "off", "n", "f"):
        return False
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
# P1 #6: fallback (seconds) used when a 429 carries no parseable Retry-After
# header. The header value itself is clamped to [5, 120] s (see
# bot_core.errors._parse_retry_after); this is only the no-header fallback.
AI_RETRY_AFTER_FALLBACK_S: int = _safe_int(os.getenv("AI_RETRY_AFTER_FALLBACK_S"), 30)
# Generous defaults: thinking models (e.g. Qwen3) spend part of max_tokens on
# internal reasoning before the visible answer — small budgets can be fully
# consumed by thinking, producing empty answers (finish_reason "length").
# ai_client auto-retries once at MAX_TOKENS_HARD_CAP when that happens.
MAX_TOKENS: int = _safe_int(os.getenv("MAX_TOKENS"), 8000)
MAX_TOKENS_HARD_CAP: int = _safe_int(os.getenv("MAX_TOKENS_HARD_CAP"), 16384)

# Fallback models tried (in order) when DEFAULT_MODEL fails during summarize/translate.
# Comma-separated list of model slugs.
FALLBACK_MODELS: list[str] = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()
]

# ── Session overview prompt (customizable) ───────────────────────────────
#: Default system prompt used by /end_session to write the session overview.
#: Strict by design: it must (1) only use THIS session's own sources, (2) never
#: invent or pull in events from other sessions, and (3) itself determine the
#: language of the provided sources and write the overview in that language
#: (no hard-coded language detection on the bot side).
#: The prompt is written in English and is language-neutral by design: it
#: instructs the model to write the overview in the language of the sources
#: (German or English), so the bot works for sessions in either language.
DEFAULT_SESSION_SUMMARY_PROMPT: str = (
    "You are writing the closing overview for exactly ONE session.\n"
    "The user message contains the ONLY sources for this session, in this order:\n"
    "  1) Session files (attachments/ and transcripts/) — the complete, verbatim session record.\n"
    "  2) Session notes — short, timestamped bullet points created during the session.\n"
    "  3) Last chat — the last few messages in the channel where the session was ended.\n"
    "Write a concise overview (max. ~250 words) with:\n"
    "  - What happened in THIS session, in the order it happened, as a short bullet list.\n"
    "  - Important points / decisions.\n"
    "  - Open points / consequences for the NEXT session.\n"
    "Use Markdown bullet points. Return ONLY the overview.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the sources in the user message. You know nothing about this session "
    "from any other source. Add NOTHING, infer nothing, and do not recall any events, "
    "names or places that do not appear in the provided sources.\n"
    "- The session files are the authoritative, detailed record; the notes and the "
    "last chat serve only as a supplement. On conflict, the session files take precedence.\n"
    "- Never describe events from other/previous sessions and never fill gaps with "
    "generic filler or action from the larger campaign frame. If a source names a "
    "hook for the next session, list it as an open point — but do not narrate what "
    "happened before or after.\n"
    "- If the sources are thin (few notes, short chat), keep the overview short and "
    "factual; never pad it with invented details.\n"
    "- LANGUAGE: Determine the language of the provided sources yourself and write the ENTIRE "
    "overview in that language (German sources → German overview, English sources → "
    "English overview). Do not mix languages and do not translate proper names (characters, places, items).\n"
    "- Keep proper names, numbers, amounts and item names exactly as they appear in the sources."
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

# Structured embed rendering for /ai replies.
# Structured replies (headings, tables, lists) become a discord.Embed with a
# title, description and inline fields. Plain prose still works; tiny/empty
# replies fall back to text. Set EMBED_FORMAT=0 in .env to restore classic
# plain-text delivery.
EMBED_FORMAT: bool = _safe_bool(os.getenv("EMBED_FORMAT"), True)

def _or_default(value: str | None, default: str) -> str:
    """Return value if non-empty, else default (for optional path overrides)."""
    return value if value else default


# ════════════════════════════════════
#  EMBEDDING MODEL
# ════════════════════════════════════
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
# P2 #21: per-request timeout (seconds) for the /embeddings endpoint. Was
# hardcoded to 30 in kb/embedder; configurable here now. The HTTP client is
# shared across batches (keep-alive), so only this per-request timeout applies.
EMBED_TIMEOUT: int = _safe_int(os.getenv("EMBED_TIMEOUT"), 30)

# ── Embedding backend selection (CPU RAG) ────────────────────────────────
# "remote" (default) = current behaviour: embed via the OpenAI-compatible
#   /embeddings endpoint on INFER_URL (the chat backend). This is what causes
#   the per-request model swap on a single-model local backend.
# "local" = embed in-process on CPU with sentence-transformers (no backend
#   round-trip, no swap). Requires the `rag-cpu` extra (sentence-transformers).
EMBED_BACKEND: str = os.getenv("EMBED_BACKEND", "remote").strip().lower()
# In-process embedding model used when EMBED_BACKEND == "local". bge-m3 is a
# strong hybrid dense/sparse, multilingual embedder that runs well on CPU.
LOCAL_EMBED_MODEL: str = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")

# ── Index-build embedding source (GPU reindex) ─────────────────────────────
# Per-request QUERIES always embed in-process on CPU (EMBED_BACKEND=local: fast,
# no model swap). Building/rebuilding the vector INDEX is a one-time batch job,
# so it can instead run on the GPU backend: point INDEX_EMBED_BACKEND at the
# inference backend (INFER_URL) which has bge-m3 loaded, and index builds use its
# /embeddings endpoint. Same model (bge-m3) on both sides => vectors are
# interchangeable, so a GPU-built index works with CPU query embeddings (and an
# index already built on CPU stays valid — no forced rebuild on the switch).
#   "backend" (default) = build the index via INFER_URL's /embeddings (GPU).
#   "local"             = build the index in-process on CPU (previous behaviour).
# If the backend is unreachable during a build, indexing falls back to local CPU
# automatically so a reindex never hard-fails.
INDEX_EMBED_BACKEND: str = os.getenv("INDEX_EMBED_BACKEND", "backend").strip().lower()
# Model slug sent to the backend when INDEX_EMBED_BACKEND == "backend". It must
# be the SAME model as the query embedder (LOCAL_EMBED_MODEL) or similarity
# breaks; it defaults to that so the two stay in lockstep. Override only if your
# backend names the loaded bge-m3 differently.
INDEX_EMBED_MODEL: str = os.getenv("INDEX_EMBED_MODEL", LOCAL_EMBED_MODEL)


def effective_embedding_model() -> str:
    """Model name used for embedding AND stored in the index metadata.

    Returning a *different* name per backend is deliberate: ``kb.index`` records
    this in its SQLite metadata and rebuilds the cache when it changes, so
    switching between the remote and local backends (which produce incompatible
    vectors) automatically triggers a one-time reindex on next load.
    """
    if EMBED_BACKEND == "local":
        return LOCAL_EMBED_MODEL
    return EMBEDDING_MODEL


# ── Cross-encoder reranker (CPU, RAG quality boost) ────────────────────────
# Reranks the top-K retrieved chunks against the query with a cross-encoder
# before they are injected into the prompt. Runs in-process on CPU; falls back
# to plain vector order on any failure/timeout so retrieval never breaks.
RERANK_ENABLED: bool = _safe_bool(os.getenv("RERANK_ENABLED"), False)
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# Number of candidate chunks handed to the reranker (was hardcoded 6*4=24).
RERANK_TOP_K: int = _safe_int(os.getenv("RAG_VECTOR_TOP_K"), 15)
RERANK_TIMEOUT_SECONDS: int = _safe_int(os.getenv("RERANK_TIMEOUT_SECONDS"), 8)
# Enable the hybrid lexical (BM25) leg that is RRF-fused with dense vectors.
# On by default (part of the high-quality pipeline); set to 0 to use dense-only.
RAG_HYBRID_ENABLED: bool = _safe_bool(os.getenv("RAG_HYBRID_ENABLED"), True)

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
STT_ENABLED: bool = _safe_bool(os.getenv("STT_ENABLED"), True)
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
#   http://<stt-host>:8000/v1          (separate container on the network)
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
STT_TRIM_SILENCE: bool = _safe_bool(os.getenv("STT_TRIM_SILENCE"), True)
# Silence threshold for STT_TRIM_SILENCE, in dBFS (negative). Samples with a
# level below this are treated as silence. -45 is well under typical speech
# but above the digital-noise floor of quiet channels.
STT_SILENCE_DBFS: float = _safe_float(os.getenv("STT_SILENCE_DBFS"), -45.0)
# Silero VAD filter for the local (faster-whisper) backend: skips silence so
# long quiet recordings don't waste decode time — and, more importantly, stop
# Whisper from hallucinating filler words in the gaps (observed: "Vielen
# Dank." emitted at times where one speaker's track was 100% digital silence).
# Set STT_VAD_FILTER=0 to transcribe every sample verbatim.
STT_VAD_FILTER: bool = _safe_bool(os.getenv("STT_VAD_FILTER"), True)
# Save the finished transcript for the ACTIVE session automatically when
# transcription completes (see bot_core.sessions.add_transcript).  The full
# transcript is stored as its own .md file under the session's transcripts/
# folder, so it shows up in /session_notes and stays RAG-searchable (chunked
# by the KB indexer). Set STT_ADD_TO_SESSION=0 to keep transcripts out of
# the sessions.
STT_ADD_TO_SESSION: bool = _safe_bool(os.getenv("STT_ADD_TO_SESSION"), True)

CHUNK_TARGET: int = _safe_int(os.getenv("CHUNK_SIZE"), 2000)
RAG_MAX_DOCS: int = _safe_int(os.getenv("RAG_MAX_DOCS"), 4)
RAG_RETRIEVAL_METHOD: str = os.getenv("RAG_RETRIEVAL_METHOD", "vector").lower()
RAG_MAX_CHARS: int = _safe_int(os.getenv("RAG_MAX_CHARS"), 24000)
RAG_WINDOW_LINES: int = _safe_int(os.getenv("RAG_WINDOW_LINES"), 80)

# ── Candidate pool sizing (retrieval recall) ───────────────────────────────
# How many candidate chunks each retrieval leg pulls before RRF fusion. Larger
# pools improve RECALL for hard lookups (exact names, rare terms): the right
# chunk is more likely to land in the candidate set so it can win the ranking.
# The final attached context is still bounded by RAG_MAX_DOCS / RAG_MAX_CHARS /
# RAG_ATTACH_FLOOR, so a bigger pool does NOT bloat the prompt. Cost is
# negligible while the cross-encoder reranker is disabled (no per-candidate
# scoring). Defaults raised from the old hardcoded 16 (dense) / 32 (lexical).
RAG_DENSE_TOP_K: int = _safe_int(os.getenv("RAG_DENSE_TOP_K"), 48)
RAG_LEXICAL_TOP_K: int = _safe_int(os.getenv("RAG_LEXICAL_TOP_K"), 48)

# Embedding batch size (documents per /embeddings call) for index builds. Bigger
# batches amortize per-request overhead, which helps SHORT/uniform chunks — but on
# long or mixed-length content they can be SLOWER, because the backend pads every
# sequence in a batch to the longest one (wasted compute). A real KB with long
# sections measured ~1.4 chunks/s at 8 vs ~0.2 chunks/s at 32-longest, so 8 is a good
# default; only raise it if your content is short and you've measured a win.
# Queries are single-text, so this never affects them.
RAG_EMBED_BATCH_SIZE: int = _safe_int(os.getenv("RAG_EMBED_BATCH_SIZE"), 8)

# ── Last-session context (continuity) ────────────────────────────────────
# Attach the previous (last ended) session as a compact context block so the AI
# has continuity across sessions. Uses the stored overview when present, else a
# tail of notes; size-capped by LAST_SESSION_MAX_CHARS. Only a genuinely
# *previous* (ended) session is attached — never the still-active one (already in
# history). Set LAST_SESSION_CONTEXT_ENABLED=0 to disable.
LAST_SESSION_CONTEXT_ENABLED: bool = _safe_bool(os.getenv("LAST_SESSION_CONTEXT_ENABLED"), True)
LAST_SESSION_MAX_CHARS: int = _safe_int(os.getenv("LAST_SESSION_MAX_CHARS"), 4000)

# ── Low-confidence query rewriting (vector path only) ─────────────────────
# When the top vector similarity score for a query is below RAG_REWRITE_MIN_SCORE,
# the rewriter asks the LLM for up to RAG_QUERY_MAX_EXPANSIONS alternative phrasings,
# embeds them in one batch, and merges the rankings (reciprocal rank fusion).
# Confident queries pay nothing extra.  Set RAG_QUERY_REWRITER=0 to disable entirely.
RAG_QUERY_REWRITER: bool = _safe_bool(os.getenv("RAG_QUERY_REWRITER"), True)
# P1 #10: route through _safe_float so a garbage value (e.g. RAG_REWRITE_MIN_SCORE=abc)
# falls back to the 0.35 default instead of raising ValueError at *import* — the
# same treatment every other numeric env var above gets. (The old
# ``float(...) or 0.35`` never guarded against an unparseable string.)
RAG_REWRITE_MIN_SCORE: float = _safe_float(os.getenv("RAG_REWRITE_MIN_SCORE"), 0.35)
RAG_QUERY_MAX_EXPANSIONS: int = _safe_int(os.getenv("RAG_QUERY_MAX_EXPANSIONS"), 3)

# ── Min-attachment relevance floor (opt-in) ───────────────────────────────
# When > 0, dense chunks scoring below this cosine-similarity floor are dropped
# BEFORE hybrid fusion so low-relevance hits don't crowd out good matches. Only
# the dense leg is filtered — BM25-only (lexical) matches are preserved. Default
# 0 = OFF (keep everything). Tune from the "Vector scores for ..." log lines.
RAG_MIN_ATTACH_SCORE: float = _safe_float(os.getenv("RAG_MIN_ATTACH_SCORE"), 0.0)

# ── Per-file chunk budget + attach-time relevance floor ───────────────────
# RAG_MAX_CHUNKS_PER_FILE caps how many of a file's ranked chunks are attached
# (they're joined into one per-file entry). Lower = tighter context, less
# "whole file" bloat; higher = more surrounding sections. Default 3.
RAG_MAX_CHUNKS_PER_FILE: int = _safe_int(os.getenv("RAG_MAX_CHUNKS_PER_FILE"), 3)
# RAG_ATTACH_FLOOR is a per-chunk relevance gate applied at ATTACH time (after
# hybrid fusion / rerank). A ranked chunk whose DENSE cosine similarity is below
# this floor is skipped, so only genuinely relevant sections are attached. This
# uses the original dense score (not the rank-based RRF score), and lexical-only
# chunks (exact-term BM25 hits with no dense score) are always kept. Default 0.50;
# set to 0 to disable. A safety net in the retriever guarantees an over-aggressive
# floor can never yield an empty context (it falls back to unfiltered selection).
RAG_ATTACH_FLOOR: float = _safe_float(os.getenv("RAG_ATTACH_FLOOR"), 0.50)

# ── Streaming AI responses (P3 #24) ────────────────────────────────────────
# When on, /ai streams the completion token-by-token and progressively edits a
# single Discord message as the text grows (instead of the user staring at a
# typing indicator until the full reply is ready). Set AI_STREAM=0 to restore
# the classic "wait, then deliver the whole reply" behaviour. This is an
# OPT-IN: the reply is built up by editing one message, which suits most
# setups, but a couple of Discord/local-backend edge cases (and the very long
# reply paths that re-deliver as multi-message chunks) are simpler with the
# proven non-streaming path, so it defaults to off.
AI_STREAM: bool = _safe_bool(os.getenv("AI_STREAM"), True)
# Minimum wall-clock seconds between successive message edits while streaming.
# Discord rate-limits message edits; keeping a small floor avoids hammering the
# API for very fast (short) generations while still giving live feedback.
AI_STREAM_EDIT_INTERVAL_S: float = _safe_float(os.getenv("AI_STREAM_EDIT_INTERVAL_S"), 2.5)
# ── Streaming response time budgets (seconds) ───────────────────────────
# A streamed response has three distinct clock phases, each with its own
# configurable budget:
#   1. AI_STREAM_INITIAL_TIMEOUT_S — how long the backend may take BEFORE
#      the first chunk arrives (prompt processing + first token; for
#      thinking models this includes the whole reasoning phase, so it is
#      deliberately a bit larger than the per-chunk timeout).
#   2. AI_STREAM_CHUNK_TIMEOUT_S — maximum gap between two consecutive
#      chunks once generation has started (the SSE read timeout).
#   3. AI_STREAM_TOTAL_TIMEOUT_S — hard cap for the WHOLE response, from
#      request start to the final chunk (guards against a slow backend
#      dribbling out chunks forever).
AI_STREAM_INITIAL_TIMEOUT_S: float = _safe_float(os.getenv("AI_STREAM_INITIAL_TIMEOUT_S"), 90.0)
AI_STREAM_CHUNK_TIMEOUT_S: float = _safe_float(os.getenv("AI_STREAM_CHUNK_TIMEOUT_S"), 60.0)
AI_STREAM_TOTAL_TIMEOUT_S: float = _safe_float(os.getenv("AI_STREAM_TOTAL_TIMEOUT_S"), 300.0)
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


