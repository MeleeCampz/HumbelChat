"""Application settings — all environment variables with defaults, typed."""
from __future__ import annotations

import logging
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
# Global response-format appendix, added to EVERY character's system prompt
# (kept out of characters.json so persona files stay clean).  Fixes the model
# compressing D&D stat blocks into one semicolon-separated run-on sentence.
STAT_BLOCK_FORMAT_RULES: str = os.getenv(
    "STAT_BLOCK_FORMAT_RULES",
    (
        "\n\n<stat-block-format>\n"
        "Discord wraps text at ~40 monospace chars on mobile and ~75 on "
        "desktop, so column alignment only survives when NO line wraps. "
        "Format D&D content accordingly: never one long semicolon-separated "
        "sentence, never a wide table, and NEVER dash/em-dash placeholders "
        "for missing values (omit them instead).\n"
        "1) STAT BLOCKS: name line in normal markdown, then a NARROW fenced "
        "code block (``` ... ```), max ~40 chars per line, at most 3 short "
        "columns, e.g. for an Awakened Tree:\n"
        "**Awakened Tree** *(Huge Plant, Neutral)* · CR 2\n"
        "```\n"
        "AC 13    HP 59 (7d12+14)   Speed 20 ft.\n"
        "STR 19 (+4)  DEX 6 (-2)   CON 15 (+2)\n"
        "INT 10 (+0)  WIS 10 (+0)  CHA 7 (-2)\n"
        "Vuln: Fire · Res: Bludgeoning, Piercing\n"
        "```\n"
        "Align columns with spaces; last line(s): only notable traits. "
        "Always close the fence.\n"
        "2) COMPARISONS / MULTI-COLUMN DATA (armor, weapons, items, class "
        "tables): NO code fence, NO pipe tables (| ... |), NEVER HTML "
        "(<table>, <tr>, <td> — Discord renders it as raw text). One entry "
        "per line: bold name, a colon, then labeled values separated by "
        "\" · \". Keep EVERY line under ~60 characters — label every value so "
        "NO header row is needed, OMIT empty or not-yet-applicable values "
        "entirely (no dash placeholders), and give long text values (feature "
        "names, descriptions) their own labeled line directly under the "
        "entry, e.g.:\n"
        "**Hide Armor**: AC 12 + Dex (max 2) · Weight 12 lb. · Cost 10 GP\n"
        "**Wizard, Level 1**: Prof +2 · Cantrips 3 · Prepared 4 · Slots 2\n"
        "Features: Spellcasting, Ritual Adept, Arcane Recovery\n"
        "**Wizard, Level 5**: Prof +3 · Cantrips 4 · Prepared 9 · Slots 4/3/2\n"
        "Feature: Memorize Spell\n"
        "Short lines wrap gracefully on any device.\n"
        "</stat-block-format>"
    ),
)
CONTEXT_WINDOW: int = _safe_int(os.getenv("CONTEXT_WINDOW"), 10)
# ── AI timeout (single knob, idle-based) ───────────────────────────────
# Maximum SILENCE from the backend before a request is declared failed.
# Passed to the openai SDK as ``timeout=``: for LM Studio's single-shot
# responses that bounds the total wait; on streams httpx applies it per read,
# so any arriving chunk resets it (idle watchdog). There is deliberately NO
# total wall-clock cap — a slow-but-alive generation may run as long as it
# keeps producing data; /ai stop is the escape hatch. Raise this for models
# with long silent thinking phases (e.g. 300).
# ``AI_REQUEST_TIMEOUT`` is kept as a fallback alias for existing .env files.
AI_TIMEOUT_S: float = _safe_float(
    os.getenv("AI_TIMEOUT_S") or os.getenv("AI_REQUEST_TIMEOUT"), 120.0
)
# §3.9: lightweight backend liveness probe. 0 = only probe once on startup;
# a positive value starts a periodic background check at that interval.
AI_HEALTH_CHECK_INTERVAL: int = _safe_int(os.getenv("AI_HEALTH_CHECK_INTERVAL"), 0)
AI_HEALTH_CHECK_TIMEOUT: int = _safe_int(os.getenv("AI_HEALTH_CHECK_TIMEOUT"), 5)
# P1 #6: fallback (seconds) used when a 429 carries no parseable Retry-After
# header. The header value itself is clamped to [5, 120] s (see
# bot_core.errors._parse_retry_after); this is only the no-header fallback.
AI_RETRY_AFTER_FALLBACK_S: int = _safe_int(os.getenv("AI_RETRY_AFTER_FALLBACK_S"), 30)
# ── Reasoning effort (per task family, #20) ───────────────────────
# Qwen3-style hybrid-thinking models spend a hidden reasoning phase before
# answering; at the model default (xhigh on Qwen3.8) even simple questions pay
# for maximal reasoning. These knobs dial that per TASK FAMILY so latency-
# sensitive calls can run shallow while long-form quality calls stay deep.
# Values: off | low | medium | high | xhigh  ("off" = enable_thinking=False).
#: /ai chat (ask_ai / ask_ai_stream). None = omit the param entirely →
#: current behaviour (thinking on, model default effort).
AI_REASONING_EFFORT: str | None
#: Short mechanical calls: /summarize, /translate, /ocr. Effective default
#: "off" — a short factual answer gains nothing from the reasoning phase.
#: Note: /translate and /ocr previously had NO thinking control at all (ran at
#: the model default), so this is a deliberate latency fix.
SUMMARY_REASONING_EFFORT: str | None
#: /end_session recap. None = omit the param → current behaviour (thinking on,
#: model default effort = max). Accuracy of the stored session digest matters
#: more than its latency, so it stays at the model default unless lowered.
RECAP_REASONING_EFFORT: str | None
#: /end_session merged log — long-form narrative merge where quality matters
#: more than latency (runs deferred). None = omit the param → current
#: behaviour (thinking on, model default effort).
MERGE_LOG_REASONING_EFFORT: str | None

def _reasoning_effort_env(name: str, raw: str | None) -> str | None:
    """Parse a reasoning-effort env var against the allowlist.

    Case-insensitive; empty/unset → ``None``. An unrecognised value logs a
    warning and yields ``None`` (same fail-soft spirit as ``_safe_int``) so a
    typo in .env can never break startup or requests.
    """
    if raw is None:
        return None
    v = raw.strip().lower()
    if not v:
        return None
    allowed = ("off", "low", "medium", "high", "xhigh")
    if v in allowed:
        return v
    logging.getLogger("bot.config.settings").warning(
        "%s=%r is not a valid reasoning effort (allowed: %s); ignoring it.",
        name, raw, "/".join(allowed),
    )
    return None

AI_REASONING_EFFORT = _reasoning_effort_env("AI_REASONING_EFFORT", os.getenv("AI_REASONING_EFFORT"))
SUMMARY_REASONING_EFFORT = _reasoning_effort_env("SUMMARY_REASONING_EFFORT", os.getenv("SUMMARY_REASONING_EFFORT"))
RECAP_REASONING_EFFORT = _reasoning_effort_env("RECAP_REASONING_EFFORT", os.getenv("RECAP_REASONING_EFFORT"))
MERGE_LOG_REASONING_EFFORT = _reasoning_effort_env("MERGE_LOG_REASONING_EFFORT", os.getenv("MERGE_LOG_REASONING_EFFORT"))

# Retained for reference / backward compatibility: chat no longer applies a
# global output cap — requests omit max_tokens so the backend (LM Studio)
# lets the model use its maximum output length. A per-character ``max_tokens``
# in characters.json is still honoured when explicitly set.
MAX_TOKENS: int = _safe_int(os.getenv("MAX_TOKENS"), 8000)
MAX_TOKENS_HARD_CAP: int = _safe_int(os.getenv("MAX_TOKENS_HARD_CAP"), 16384)
#: Output cap for summary calls (/summarize, /end_session recap).
#: 0 (default) = do NOT send max_tokens → model maximum output length.
#: Set a positive value to cap those calls.
SUMMARY_CALL_MAX_TOKENS: int = _safe_int(os.getenv("SUMMARY_CALL_MAX_TOKENS"), 0)

# Fallback models tried (in order) when DEFAULT_MODEL fails during summarize/translate.
# Comma-separated list of model slugs.
FALLBACK_MODELS: list[str] = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()
]

# ── Session recap prompt (customizable) ──────────────────────────────────
#: Default system prompt used by /end_session to write the session recap — a
#: very brief digest (what happened + suggested next steps) that is stored
#: with the session and delivered when the NEXT session starts.  Strict by
#: design: it must (1) only use THIS session's own sources, (2) never invent
#: or pull in events from other sessions, and (3) itself determine the
#: language of the provided sources and write the recap in that language
#: (no hard-coded language detection on the bot side).
DEFAULT_SESSION_MERGE_PROMPT: str = (
    "You are merging the individual player logs of exactly ONE session into a single, "
    "complete, well-formatted session log.\n"
    "The user message contains the ONLY sources: the session's own documents — each "
    "player's own log/notes for the SAME session. They cover the same chronological "
    "events from different perspectives and overlap heavily.\n"
    "Produce ONE combined session log that:\n"
    "  - Follows the chronological order of the session, organised into Markdown sections\n"
    "    (one per location / scene / major event).\n"
    "  - Merges duplicate information: when several logs describe the same event, combine\n"
    "    them into one passage that keeps every detail any of them contain.\n"
    "  - Keeps ALL concrete details from every source: names, places, numbers, amounts,\n"
    "    loot, dice results, decisions and open threads — if a detail appears in any log,\n"
    "    it must appear in the merged log.\n"
    "  - Reads as one coherent narrative record, not as a list of quoted sources.\n"
    "Return ONLY the merged session log in Markdown.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the provided documents. Add NOTHING, infer nothing, and do not recall any "
    "events, names or places that do not appear in them.\n"
    "- Never describe events from other/previous sessions and never fill gaps with generic "
    "filler or action from the larger campaign frame.\n"
    "- If a detail is ambiguous or contradicted between logs, keep it as the sources state "
    "it — do not resolve contradictions by inventing facts.\n"
    "- LANGUAGE: Determine the language of the provided sources yourself and write the "
    "ENTIRE merged log in that language (German sources → German log, English sources → "
    "English log). Do not mix languages and do not translate proper names (characters, "
    "places, items).\n"
    "- Keep proper names, numbers, amounts and item names exactly as they appear in the sources."
)

#: Override for SESSION_MERGE_PROMPT — set in .env to customize how the
#: /end_session AI session-log merge is written. Empty = use DEFAULT_SESSION_MERGE_PROMPT.
SESSION_MERGE_PROMPT: str = os.getenv("SESSION_MERGE_PROMPT", "")

DEFAULT_SESSION_RECAP_PROMPT: str = (
    "You are writing a brief recap of exactly ONE session. It is stored with the "
    "session and shown when the user starts the NEXT session.\n"
    "The user message contains the ONLY sources for this session:\n"
    "  1) Session files (attachments/ and transcripts/) — the complete, verbatim session record.\n"
    "  2) Session notes — short, timestamped bullet points created during the session.\n"
    "Write a brief recap (max. ~150 words) with exactly two parts:\n"
    "  - **What happened:** 6–8 SHORT bullets in order — one event per bullet, each at most "
    "one line; pick the most important events, no scene-setting or background.\n"
    "  - **Next steps:** 1–3 short bullets — open points / hooks / consequences for the "
    "next session (only if the sources explicitly support them; otherwise write a single "
    "bullet saying there are no open points).\n"
    "Use Markdown. Return ONLY the recap.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the sources in the user message. Add NOTHING, infer nothing, and do not "
    "recall any events, names or places that do not appear in them.\n"
    "- The session files are authoritative; the notes serve only as a supplement. On "
    "conflict, the session files take precedence.\n"
    "- Never narrate what happened before this session and never fill gaps with generic "
    "filler from the larger campaign frame. If the sources are thin, keep the recap "
    "short and factual; do not pad it.\n"
    "- Next steps must be open points the sources themselves name (an announced task, a "
    "threat, an unresolved thread) — never speculation about what characters know, think "
    "or will do.\n"
    "- LANGUAGE: Determine the language of the provided sources yourself and write the "
    "ENTIRE recap in that language (German sources → German recap, English sources → "
    "English recap). Do not mix languages and do not translate proper names (characters, "
    "places, items).\n"
    "- Keep proper names, numbers, amounts and item names exactly as they appear in the sources."
)

#: Override for SESSION_RECAP_PROMPT — set in .env to customize how the
#: /end_session recap of the session is written. Empty = use DEFAULT_SESSION_RECAP_PROMPT.
SESSION_RECAP_PROMPT: str = os.getenv("SESSION_RECAP_PROMPT", "")

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
# The correctly-named env var RERANK_TOP_K wins; the historical (misnamed)
# RAG_VECTOR_TOP_K stays supported as an alias for existing deployments.
RERANK_TOP_K: int = _safe_int(
    os.getenv("RERANK_TOP_K", os.getenv("RAG_VECTOR_TOP_K")), 15
)
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

# ── Session notes location ────────────────────────────────────────────────
# Where per-session folders live. Inside the KB by default so each session's
# combined notes.md is automatically RAG-indexed (and shows up in
# /list_kb_docs). Override to point inside an Obsidian campaign vault, e.g.
# <KB_PATH>/humblewood/session_notes — see docs/obsidian-vault.md (#19).
SESSIONS_NOTES_DIR: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("SESSIONS_NOTES_DIR"), str(KB_PATH / "session_notes"))
)

# ── Obsidian vault sync (#19) ─────────────────────────────────────────────
# Semicolon-separated `name=git_url` entries, e.g.
#   humblewood=https://…;dnd_handbook=https://…
# Each vault is cloned into <KB_PATH>/<name>/ and pulled + incrementally
# re-indexed on a periodic background loop. Empty = feature off. Private repos
# use a token-embedded URL: https://<user>:<token>@github.com/<user>/<repo>.git
OBSIDIAN_VAULTS: str = os.getenv("OBSIDIAN_VAULTS", "")
# Name of the vault that is TWO-WAY and holds the session notes (its URL needs
# write access). Must be one of the names in OBSIDIAN_VAULTS; empty = all
# vaults are read-only mirrors. See docs/obsidian-vault.md.
OBSIDIAN_CAMPAIGN_VAULT: str = os.getenv("OBSIDIAN_CAMPAIGN_VAULT", "")
# Pull/commit period in seconds; 0 = one startup pass, no periodic loop.
OBSIDIAN_VAULT_PULL_INTERVAL: int = _safe_int(
    os.getenv("OBSIDIAN_VAULT_PULL_INTERVAL"), 300
)

# ── Item tables (/roll_items) ───────────────────────────────────────────────
# One CSV per rollable table, plus CR tier files (cr_tiers.csv and optional
# per-table <table>_tiers.csv) for CR-scaled rolls.
# Defaults to <repo_root>/data/items.
ITEMS_DIR: pathlib.Path = pathlib.Path(
    _or_default(os.getenv("ITEMS_DIR"), str(_REPO_ROOT / "data" / "items"))
)

#: Table used by /roll_items when no table is named — a CSV filename in
#: ITEMS_DIR without the .csv suffix (case-insensitive).
DEFAULT_ITEM_TABLE: str = _or_default(os.getenv("DEFAULT_ITEM_TABLE"), "magic_items")

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

# ── Prompt budget (Phase 3 — decision Q3a, 2026-09-23) ───────────────────
# Soft cap on total prompt size, in chars (≈ tokens × 4). When the estimate
# exceeds it, oldest history is trimmed first, then lowest-ranked RAG docs
# (see ai_client._apply_prompt_budget). Tune to the model's real context window.
PROMPT_BUDGET_CHARS: int = _safe_int(os.getenv("PROMPT_BUDGET_CHARS"), 100000)

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
# Rewrite/expand EVERY query, not just German or low-confidence ones. With
# thinking disabled on the rewrite call (see kb/query_rewriter.py) the extra LLM
# call costs ~0.5-1s — negligible next to answer generation — and removes the
# language-detection edge cases (German without umlauts slips past detection;
# mangled English like "wizzard" or "armor stat block" gets no help otherwise).
# Set RAG_REWRITE_ALL_QUERIES=0 to fall back to German/low-confidence-only.
RAG_REWRITE_ALL_QUERIES: bool = _safe_bool(os.getenv("RAG_REWRITE_ALL_QUERIES"), True)

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


