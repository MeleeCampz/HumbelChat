"""Provider-agnostic AI client + RAG orchestration.

P2 additions:
  - Input length cap (MAX_INPUT_CHARS)
  - Per-user sliding-window rate limiting
  - RAG context injected into user message (not system prompt)

P3 additions:
  - Streaming completions (:func:`ask_ai_stream`) for live progress display
  - Live queue-depth tracking on the global AI slot (:func:`ai_queue_depth`)

P5 additions:
  - Streaming is the default delivery path for /ai (``AI_STREAM=1``)
  - A single timeout knob ``AI_TIMEOUT_S`` bounds how long the backend may
    stay silent: total-wait bound for normal responses, per-read idle
    watchdog for streams (any chunk resets it; no total wall-clock cap)
  - The shared client is created with ``max_retries=0`` so the bot's own
    policy (exactly ONE retry on transient failure) is the only retry layer
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI, APIStatusError
from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.characters import Character

from bot_core.history import ensure_history, get_history, set_history
from config.settings import (
    INFER_URL,
    INFER_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    STAT_BLOCK_FORMAT_RULES,
    AI_TIMEOUT_S,
    CONTEXT_WINDOW,
    KB_PATH,
    RAG_MAX_DOCS,
    RAG_MAX_CHARS,
    RAG_WINDOW_LINES,
    LAST_SESSION_CONTEXT_ENABLED,
    LAST_SESSION_MAX_CHARS,
    RAG_RETRIEVAL_METHOD,
    MAX_INPUT_CHARS,
    AI_RATE_LIMIT_MAX,
    AI_RATE_LIMIT_WINDOW,
    PROMPT_BUDGET_CHARS,
)

log = logging.getLogger("bot.bot_core")

# ── Shared client (singleton) ─────────────────────────────────────────
_shared_client: AsyncOpenAI | None = None


def _make_client() -> AsyncOpenAI:
    global _shared_client
    if _shared_client is None:
        # max_retries=0: the bot owns its retry policy —
        # ``_call_completion_with_retry`` does exactly ONE bounded retry.
        # With the SDK default (3 retries) every timed-out request was
        # actually 3×145s of SDK-level retries inside a single "attempt",
        # tripling the worst-case latency before the bot's own retry logic
        # was even asked to step in.
        _shared_client = AsyncOpenAI(
            api_key=INFER_API_KEY, base_url=INFER_URL, max_retries=0
        )
    return _shared_client


# ─────────────────────────────────────────────────────────────────────────
#  P2-2: Per-user sliding-window rate limiter
# ─────────────────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when a user exceeds the per-user AI request limit."""

    def __init__(self, user_id: str, retry_after: int) -> None:
        self.user_id = user_id
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Try again in {retry_after}s.")


class _SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by user identifier."""

    def __init__(self, max_requests: int, window_sec: int) -> None:
        self._max = max_requests
        self._window = window_sec
        self._hits: dict[str, list[float]] = {}

    def check(self, key: str) -> None:
        """Raise RateLimitError if the key has exceeded the limit."""
        now = time.monotonic()
        cutoff = now - self._window
        timestamps = [t for t in self._hits.get(key, []) if t > cutoff]
        if len(timestamps) >= self._max:
            earliest = min(timestamps)
            retry_after = max(1, int(earliest + self._window - now))
            # still prune and store
            self._hits[key] = timestamps
            raise RateLimitError(key, retry_after)
        timestamps.append(now)
        self._hits[key] = timestamps
        # Opportunistic GC of stale keys
        if len(self._hits) > 500:
            self._hits = {k: v for k, v in self._hits.items() if any(t > cutoff for t in v)}


_rate_limiter = _SlidingWindowRateLimiter(AI_RATE_LIMIT_MAX, AI_RATE_LIMIT_WINDOW)


def check_rate_limit(user_id: str) -> None:
    """Public helper — raises RateLimitError if the user is over limit."""
    _rate_limiter.check(user_id)


# ─────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────

# Cache of the backend's model list. _validate_model used to call
# ``models.list()`` on *every single request* — one extra round-trip per
# message. The list is cached for a short TTL instead; backends rarely add or
# remove models mid-session, and a stale cache entry only costs one fallback.
_MODEL_LIST_TTL_SEC = 300.0
_model_list_cache: dict[str, tuple[float, set[str]]] = {}


def _clear_model_list_cache() -> None:
    """Drop cached model lists (used by tests)."""
    _model_list_cache.clear()


async def _validate_model(client: AsyncOpenAI, effective_model: str) -> str:
    """Guard against stale character models: fall back to the .env default.

    The backend's model list is cached for ``_MODEL_LIST_TTL_SEC`` seconds so
    each request doesn't pay an extra ``models.list()`` round-trip.
    """
    if not effective_model:
        return DEFAULT_MODEL or ""
    now = time.monotonic()
    available: set[str] | None = None
    cached = _model_list_cache.get(INFER_URL)
    if cached is not None and now - cached[0] < _MODEL_LIST_TTL_SEC:
        available = cached[1]
    else:
        try:
            models_resp = await client.models.list()
            available = {m.id for m in models_resp.data}
            _model_list_cache[INFER_URL] = (now, available)
        except Exception as e:
            log.warning("Could not list backend models at %s: %s", INFER_URL, e)
    if available and effective_model not in available and effective_model != DEFAULT_MODEL:
        log.warning("Model '%s' not found on backend; falling back to '%s'", effective_model, DEFAULT_MODEL)
        return DEFAULT_MODEL or ""
    return effective_model


def _build_rag_context(kb_docs: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Build the RAG context string from retrieved docs.

    Returns (rag_context_str, list_of_doc_names_included).
    """
    if not kb_docs:
        return "", []

    limit = RAG_MAX_DOCS
    doc_names = [name for name, _ in kb_docs[:limit]]
    log.info("RAG: Attaching %d KB document(s) to context: [%s]",
             len(doc_names), ", ".join(f'"{n}"' for n in doc_names))

    max_chars = RAG_MAX_CHARS
    parts = ["=== Knowledge Base ===\n"]
    chars_used = len(parts[-1])
    docs_added = 0
    included_names: list[str] = []

    for display_name, content in kb_docs[:limit]:
        doc_block = f"\n--- {display_name} ---\n{content}"
        if chars_used + len(doc_block) > max_chars:
            log.info("RAG: skipped doc '%s' (%d chars) — remaining budget %d chars",
                     display_name, len(doc_block), max_chars - chars_used)
            continue
        parts.append(doc_block)
        chars_used += len(doc_block)
        docs_added += 1
        included_names.append(display_name)

    rag_context = "\n".join(parts) if docs_added else ""
    if docs_added < len(kb_docs):
        log.info("RAG: included %d/%d documents (~%.0fK chars) — budget cap reached",
                 docs_added, len(kb_docs), chars_used / 1024)
    return rag_context, included_names



# ─────────────────────────────────────────────────────────────────────────
#  Phase 3: prompt budget (decision Q3a, 2026-09-23)
# ─────────────────────────────────────────────────────────────────────────
# Keeps the *estimated* prompt size (chars/4 ≈ tokens) under
# PROMPT_BUDGET_CHARS so large RAG contexts can't push the whole prompt past
# the model's context window. Trim policy (in order):
#   1. Drop oldest history messages first (from the front of the history;
#      the last user turn + RAG context are NEVER trimmed).
#   2. Only if history is exhausted: drop lowest-ranked RAG docs from the
#      END of the list (keep the highest-ranked ones — answer quality beats
#      history).
# RAG context is therefore always preserved over history (the answer
# quality target). Logs one line when any trimming actually happens so the
# budget can be tuned from real traffic.


def _apply_prompt_budget(
    recent_history: list[ChatCompletionMessageParam],
    kb_docs: list[tuple[str, str]],
    rag_context: str,
    included_names: list[str],
    system_p: str,
    username: str,
    user_message: str,
) -> tuple[list[ChatCompletionMessageParam], list[tuple[str, str]], str, list[str]]:
    """Trim *recent_history* / *kb_docs* until the estimated prompt fits.

    The estimate uses the *exact* same message shapes that
    ``_build_ai_request`` assembles (system prompt + history + the final user
    message with RAG), so no part of the prompt is under-counted.  Returns
    the possibly-reduced ``(history, kb_docs, rag_context, included_names)``.
    The last user turn + RAG context are never trimmed; history is dropped
    oldest-first, then lowest-ranked RAG docs (see the policy note above).
    """
    history = list(recent_history)
    docs = list(kb_docs)
    context = rag_context
    names = list(included_names)

    def _est(hist: list[ChatCompletionMessageParam], ctx: str) -> int:
        total = (len(system_p) if system_p else 0)
        # History content is always a plain str at runtime (see _persist_turn);
        # the openai TypedDict union also allows part-lists, which we never store.
        total += sum(len(cast(str, m.get("content", ""))) for m in hist)
        total += len(_append_user_message(username, user_message, ctx))
        return total

    if _est(history, context) <= PROMPT_BUDGET_CHARS:
        return history, docs, context, names  # fits — no trimming

    # Step 1: drop oldest history messages (from the front) until it fits.
    over = _est(history, context) - PROMPT_BUDGET_CHARS
    trimmed = 0
    while over > 0 and history:
        history = history[1:]
        trimmed += 1
        over = _est(history, context) - PROMPT_BUDGET_CHARS
    if trimmed:
        log.info(
            "Prompt budget: trimmed %d oldest history message(s) (history now %d msg(s)).",
            trimmed, len(history),
        )
    if over <= 0:
        return history, docs, context, names  # fits now

    # Step 2: still over (history exhausted or empty) — drop lowest-ranked
    # RAG docs from the END of the ranked list, rebuilding the context.
    if not docs:
        log.warning(
            "Prompt budget: prompt is %.1fK chars over the %d-char budget with no "
            "history or RAG docs left to trim — sending as-is.",
            _est(history, context) / 1024, PROMPT_BUDGET_CHARS,
        )
        return history, docs, context, names

    dropped = 0
    while over > 0 and docs:
        docs = docs[:-1]
        if names:
            names = names[:-1]
        dropped += 1
        context, _ = _build_rag_context(docs)
        over = _est(history, context) - PROMPT_BUDGET_CHARS
    log.info(
        "Prompt budget: dropped %d lowest-ranked RAG doc(s); %d doc(s) remain "
        "(~%.1fK chars total, budget %d chars).",
        dropped, len(docs), _est(history, context) / 1024, PROMPT_BUDGET_CHARS,
    )
    if over > 0:
        # Even with no docs the base prompt alone exceeds the budget.
        log.warning(
            "Prompt budget: base prompt alone is %.1fK chars over the %d-char budget.",
            over / 1024, PROMPT_BUDGET_CHARS,
        )
    return history, docs, context, names


# ─────────────────────────────────────────────────────────────────────────
#  P1 #7: bounded retry of the *completion* call on transient failures
# ─────────────────────────────────────────────────────────────────────────
# Total attempts = 2 (one retry) to keep worst-case latency bounded; the
# backoff delay is overridable so tests don't actually sleep.
_COMPLETION_MAX_ATTEMPTS = 2
_COMPLETION_RETRY_BACKOFF_S = 1.5


def _is_transient_api_failure(exc: BaseException) -> bool:
    """True for failures a retry is worth: 5xx, 429, timeouts, connections.

    Never retries 4xx auth/validation/model-not-found (400/401/403/404 etc.).
    429 is included: the backend said "slow down" — one retry after a short
    backoff is worth trying (the retry-then-raise path still surfaces the
    backend's Retry-After message if it fails again).
    """
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    name = type(exc).__name__
    return name in (
        "APIConnectionError", "APITimeoutError",
        "ConnectError", "ConnectTimeout", "ReadTimeout", "PoolTimeout",
        "ConnectionError", "RemoteProtocolError",
        "TimeoutError",
    ) or "timed out" in str(exc).lower()


async def _call_completion_with_retry(client: AsyncOpenAI, **kwargs: Any) -> Any:
    """Invoke ``chat.completions.create`` with one retry on transient errors.

    P1 #7: a transient 5xx / timeout / connection blip used to fail the whole
    turn on the first attempt. Retried once after a short exponential
    backoff; anything non-transient (4xx, malformed request) re-raises
    immediately. NOTE: this wraps ONLY the completion call — the RAG query
    rewrite sub-call in ``kb.retrievers`` keeps its own budget and is never
    retried here.
    """
    last_exc: BaseException | None = None
    for attempt in range(_COMPLETION_MAX_ATTEMPTS):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — classified by the caller
            if attempt + 1 >= _COMPLETION_MAX_ATTEMPTS or not _is_transient_api_failure(e):
                raise
            last_exc = e
            delay = _COMPLETION_RETRY_BACKOFF_S * (2 ** attempt)
            log.warning(
                "Transient AI failure (%s: %s); retrying completion call in %.1fs (attempt %d/%d)",
                type(e).__name__, e, delay, attempt + 1, _COMPLETION_MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay)
    # Unreachable: the loop above always returns or raises on every attempt.
    raise last_exc  # type: ignore[misc]  # pragma: no cover


def _resolve_request_params(char_obj: Character | None) -> tuple[int | None, float]:
    """Resolve max_tokens and temperature from character config.

    No global output cap: when the character does not set an explicit
    ``max_tokens`` this returns ``None`` so callers OMIT the parameter and
    the backend (LM Studio) lets the model use its maximum output length.
    """
    _request_max_tokens: int | None = \
        char_obj.max_tokens if (char_obj and char_obj.max_tokens) else None

    _char_temp = getattr(char_obj, "temperature", None)
    _request_temp: float = float(_char_temp) if isinstance(_char_temp, (int, float)) else 0.7

    return _request_max_tokens, _request_temp


# ─────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────

# ── Global AI serialization (single local backend) ────────────────────────
# The bot runs against one local inference backend that serves a single model.
# Two concurrent /ai requests from *different* channels would issue parallel
# LLM calls, which a single-model local server cannot serve within the time
# budget.  Requests are already serialized per channel (utils.channel_queue)
# for message ordering; this process-wide slot extends that so at most ONE AI
# request (RAG retrieval + completion) is in flight at any time.  asyncio.Lock
# wakes waiters FIFO, matching the per-channel queue's fairness.
_global_ai_lock: asyncio.Lock | None = None


def _get_global_ai_lock() -> asyncio.Lock:
    global _global_ai_lock
    if _global_ai_lock is None:
        _global_ai_lock = asyncio.Lock()
    return _global_ai_lock


# Live queue state (P3 #26). asyncio.Lock doesn't expose how many coroutines
# are waiting, so we track it by hand to be able to (a) tell a user how many
# requests are ahead of them and (b) log long waits. These are mutated only on
# the event-loop thread, so no extra locking is needed.
_ai_active: bool = False    # True while some request holds the slot
_ai_waiters: int = 0        # requests queued behind the active one


@asynccontextmanager
async def _ai_slot() -> AsyncIterator[None]:
    """Hold the process-wide AI slot for the duration of one request.

    P3 #26: maintains the live queue counters so callers can surface
    "queued behind N requests" feedback instead of waiting silently.
    """
    global _ai_active, _ai_waiters
    _ai_waiters += 1
    got_slot = False
    try:
        async with _get_global_ai_lock():
            got_slot = True
            _ai_waiters -= 1
            _ai_active = True
            try:
                yield
            finally:
                _ai_active = False
    finally:
        # We never acquired the slot (e.g. cancelled while waiting) — drop our
        # waiter count so the queue depth stays accurate.
        if not got_slot:
            _ai_waiters = max(0, _ai_waiters - 1)


def ai_slot_busy() -> bool:
    """True if an AI request is currently in flight (holding the slot)."""
    return _ai_active


def ai_queue_depth() -> int:
    """Number of AI requests queued behind the one currently in flight."""
    return _ai_waiters


def _reset_ai_slot_state() -> None:
    """Test helper: clear the queue counters + the lock (fresh event loop)."""
    global _ai_active, _ai_waiters, _global_ai_lock
    _ai_active = False
    _ai_waiters = 0
    _global_ai_lock = None


class _AIRequestContext:
    """Everything one AI turn needs, computed once (P3 #24/#26).

    Built a single place (:func:`_build_ai_request`) and shared between the
    non-streaming :func:`ask_ai` and the streaming :func:`ask_ai_stream`, so
    the two paths can never diverge in how they resolve the model, RAG
    context, messages, and timeout.
    """

    def __init__(
        self,
        *,
        effective_model: str,
        client: AsyncOpenAI,
        messages: list[ChatCompletionMessageParam],
        user_message: str,
        guild_id: int,
        channel_id: int | None,
        username: str,
        system_p: str,
        rag_context: str,
        included_names: list[str],
        recent_history: list[ChatCompletionMessageParam],
        total_chars: int,
        max_tokens: int | None,
        temperature: float,
        timeout_sec: float,
    ) -> None:
        self.effective_model = effective_model
        self.client = client
        self.messages = messages
        self.user_message = user_message
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.username = username
        self.system_p = system_p
        self.rag_context = rag_context
        self.included_names = included_names
        self.recent_history = recent_history
        self.total_chars = total_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec


def _compose_system_prompt(char_obj: Character | None) -> str:
    """Persona prompt + the global response-format appendix (stat blocks).

    The appendix lives in settings (env-overridable via ``STAT_BLOCK_FORMAT_RULES``)
    so characters.json stays pure persona; every character gets the same
    formatting rules.
    """
    base = getattr(char_obj, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT \
        or "You are a helpful AI assistant."
    return base + STAT_BLOCK_FORMAT_RULES


def _session_field(session: object, key: str, default: Any = None) -> Any:
    """Read *key* from a session that may be a dict (live) or an object (tests)."""
    if isinstance(session, dict):
        return session.get(key, default)
    return getattr(session, key, default)


def _build_last_session_context(session: object) -> str:
    """Format the previous session as a BRIEF, size-capped context block.

    Overview only — deliberately NO notes / merged log: detailed session
    content is retrieved on demand by RAG from the indexed ``notes.md``
    (one file per session), so the proactive block only orients the model.
    The block is a pure function of the session data: no per-turn timestamps,
    no randomness — the same ended session renders to BYTE-IDENTICAL output
    on every turn ("smart context": stable across the conversation, unlike
    query-dependent RAG chunks).  If the overview alone exceeds
    ``LAST_SESSION_MAX_CHARS`` it is truncated from the tail.

    Returns "" when there is no stored overview or the feature is disabled.
    Never raises — continuity must never break a turn.
    """
    if not LAST_SESSION_CONTEXT_ENABLED:
        return ""
    try:
        if session is None:
            return ""
        name = _session_field(session, "name") or "the previous session"
        started = _session_field(session, "started_at")
        ended = _session_field(session, "ended_at")

        def _fmt_ts(ts: Any) -> str:
            try:
                return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError):
                return ""

        when = " → ".join(t for t in (_fmt_ts(started), _fmt_ts(ended)) if t)
        header = f"[Previous session — {name}" + (f" ({when})" if when else "") + "]"

        overview = str(_session_field(session, "overview") or "").strip()
        if not overview:
            return ""

        block = f"{header}\nOverview:\n{overview}"
        max_chars = LAST_SESSION_MAX_CHARS
        if len(block) <= max_chars:
            return block
        suffix = " …(truncated)"
        # +1: the newline that joins the header and the Overview part.
        take = max_chars - len(header) - 1 - len("Overview:\n") - len(suffix)
        if take <= 0:
            return header[:max_chars]  # pathological: name alone fills budget
        return f"{header}\nOverview:\n{overview[:take].rstrip()}{suffix}"
    except Exception as e:
        log.warning("last-session context build failed: %s", e)
        return ""


def _append_user_message(username: str, user_message: str, rag_context: str,
                         session_context: str = "") -> str:
    """Build the *final* user-turn content: username decoration + optional
    previous-session + RAG context blocks.

    P3 #37: the username is stripped of Discord markdown emphasis characters
    so a display name like ``**bold**`` can't break the ``**{username}:**``
    prompt-prefix formatting (it would otherwise swallow a following token).
    """
    safe_username = (username or "").replace("*", "")
    base = f"**{safe_username}:** {user_message}" if safe_username else user_message
    blocks: list[str] = []
    if session_context:
        blocks.append(f"[Previous session context]\n{session_context}")
    if rag_context:
        blocks.append(f"[Relevant knowledge-base context]\n{rag_context}")
    if blocks:
        base = "---\n\n".join(blocks) + "\n\n---\n\n" + base
    return base


async def _build_ai_request(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int | None,
    username: str = "",
    user_id: str | int | None = None,
    char_key: str | None = None,
) -> _AIRequestContext:
    """Resolve model + RAG + messages for one turn, with validation.

    Raises :class:`RateLimitError` (re-raised as-is) or ``ValueError`` (input
    too long / no model) before any completion work. All shared state (history
    read, RAG retrieval, model validation) lives here so both delivery paths
    behave identically.
    """
    # ── P2-1: Input length cap ──────────────────────────────────────────
    if len(user_message) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Input too long: {len(user_message)} chars exceeds the "
            f"{MAX_INPUT_CHARS}-character limit. Please shorten your message."
        )

    # ── P2-2: Rate limiting ─────────────────────────────────────────────
    if user_id is not None:
        check_rate_limit(str(user_id))

    effective_model = (model_slug or "").strip() or DEFAULT_MODEL
    if not effective_model:
        raise ValueError(
            f"No model configured for this request. Character model='{model_slug}' is empty "
            f"and DEFAULT_MODEL is not set. Set MODEL_NAME in .env or add a model to the character."
        )
    log.debug("Using model '%s' for this request.", effective_model)

    client = _make_client()
    effective_model = await _validate_model(client, effective_model)

    ensure_history(guild_id, channel_id)
    history = get_history(guild_id, channel_id)
    max_messages = CONTEXT_WINDOW

    from config.characters import get_character, default_character
    from bot_core.history import get_active_char_key

    # An explicit /ai character (or the prefix path's resolved active key)
    # must drive the persona, not only the model slug. Fall back to the
    # channel's active character when the caller did not pass one.
    lookup_key = char_key or get_active_char_key(guild_id, channel_id)
    char_obj = get_character(lookup_key) or default_character()
    system_p = _compose_system_prompt(char_obj)

    # ── RAG context ──────────────────────────────────────────────────
    rag_context = ""
    included_names: list[str] = []
    from kb.retrievers import retrieve_kb_documents
    kb_docs = await retrieve_kb_documents(
        query=user_message,
        kb_path=KB_PATH,
        strategy=RAG_RETRIEVAL_METHOD,
        top_n=RAG_MAX_DOCS,
        window_lines=RAG_WINDOW_LINES,
        rewrite_model=effective_model,  # same model as the completion call
    )
    if kb_docs:
        rag_context, included_names = _build_rag_context(kb_docs)

    # ── Last-session context (continuity) ────────────────────────────────
    session_ctx = ""
    if LAST_SESSION_CONTEXT_ENABLED:
        from bot_core import sessions as _sessions
        try:
            last = _sessions.get_last_session()
            # Only attach a genuinely *previous* (ended) session; get_last_session
            # returns the still-active one when it exists (already in history) → skip.
            if last is not None and _session_field(last, "ended_at") is not None:
                session_ctx = _build_last_session_context(last)
        except Exception as e:
            log.warning("last-session context lookup failed: %s", e)

    # ── Build messages ─────────────────────────────────────────────────
    # Stored chat history is plain {"role", "content"} dicts — exactly the
    # openai user/assistant message shape (cast at the storage boundary).
    recent_history: list[ChatCompletionMessageParam] = \
        cast(list[ChatCompletionMessageParam],
             history[-(2 * max_messages):] if max_messages else [])

    # ── Phase 3: prompt budget (decision Q3a) ────────────────────────────
    # If the estimated prompt exceeds PROMPT_BUDGET_CHARS, trim oldest
    # history first, then lowest-ranked RAG docs (see _apply_prompt_budget).
    if kb_docs or recent_history:
        recent_history, _kept_docs, rag_context, included_names = _apply_prompt_budget(
            recent_history, kb_docs, rag_context, included_names,
            system_p, username, user_message,
        )

    messages: list[ChatCompletionMessageParam] = []
    # P2-3: system prompt is persona-only (no RAG)
    if system_p:
        messages.append({"role": "system", "content": system_p})
    messages.extend(recent_history)

    # P2-3: RAG context injected into the user message, right before the question
    messages.append({"role": "user", "content": _append_user_message(username, user_message, rag_context, session_ctx)})

    _total_chars = sum(len(cast(str, m.get("content", ""))) for m in messages)
    _approx_tokens = int(_total_chars / 4)
    log.info(
        "ask_ai → model=%s messages_in_prompt=%d KB_files=%d system_chars=%d rag_chars=%d history_msgs=%d total_chars=%.1fK estimated_tokens=%d",
        effective_model, len(messages), len(included_names),
        len(system_p), len(rag_context) if rag_context else 0,
        len(recent_history), _total_chars / 1024, _approx_tokens,
    )
    if included_names:
        for display_name in included_names:
            log.debug("RAG doc included: %s", display_name)

    _request_max_tokens, _request_temp = _resolve_request_params(char_obj)

    # Idle-based timeout: AI_TIMEOUT_S bounds how long the backend may stay
    # SILENT (single knob — see config/settings.py). No total wall-clock cap.
    timeout_sec = AI_TIMEOUT_S

    return _AIRequestContext(
        effective_model=effective_model,
        client=client,
        messages=messages,
        user_message=user_message,
        guild_id=guild_id,
        channel_id=channel_id,
        username=username,
        system_p=system_p,
        rag_context=rag_context,
        included_names=included_names,
        recent_history=recent_history,
        total_chars=_total_chars,
        max_tokens=_request_max_tokens,
        temperature=_request_temp,
        timeout_sec=timeout_sec,
    )


def _persist_turn(guild_id: int, channel_id: int | None, user_message: str,
                  reply_text: str) -> None:
    """Append a (user, assistant) pair to history and persist it (P0 #2).

    Store the *clean* user message (not the RAG-inflated ``user_content``),
    then write the trimmed tail to disk so a restart can't lose recent turns.
    """
    history = get_history(guild_id, channel_id)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply_text})
    max_entries = 2 * CONTEXT_WINDOW if CONTEXT_WINDOW else 50
    set_history(guild_id, channel_id, history[-max_entries:])


def _friendly_ai_error(e: BaseException, *, model: str, backend_url: str) -> None:
    """Raise the right, user-facing exception for a failed completion (P1 #7/#8).

    §3.7: surface a user-friendly message instead of a raw SDK traceback. A
    classified ``RateLimitError`` is re-raised as-is (handlers show the
    retry-after message); known categories become a ``ValueError`` carrying the
    friendly message; anything else propagates.
    """
    from bot_core.errors import classify_ai_error
    classified = classify_ai_error(e, model=model, backend_url=backend_url)
    log.error("AI request failed (%s): %s", getattr(classified, "category", "unknown"), e)
    if isinstance(classified, RateLimitError):
        raise classified from e
    if isinstance(classified, ValueError) or getattr(classified, "category", "") in (
        "timeout", "model_not_found", "backend_down", "backend_error",
        "truncated_response",
    ):
        raise ValueError(classified.user_message) from e
    raise e


async def complete_text(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int | None = None,
    timeout: float | None = None,
    disable_thinking: bool = False,
) -> str:
    """Single-shot completion returning the visible reply text.

    Shared by the summary-type call sites (/summarize, /end_session overview
    and merged log):

    * ``max_tokens=None`` (default) → the parameter is OMITTED so the backend
      (LM Studio) lets the model use its maximum output length.
    * ``timeout`` defaults to ``AI_TIMEOUT_S`` — maximum silence from the
      backend before the request fails.
    * ``disable_thinking=True`` sends ``extra_body={"enable_thinking": False}``
      (Qwen3-style hybrid-thinking models); if the backend rejects the param,
      one plain retry follows (same pattern as kb/query_rewriter.py).
    * A response cut off at the model's output limit with NO visible content
      raises ``AIResponseTruncatedError``; a PARTIAL answer is returned with a
      warning log. Callers keep their own fallback logic.
    """
    from bot_core.errors import extract_reply_text

    _timeout = timeout if timeout is not None else AI_TIMEOUT_S

    def _kwargs(disable_thinking: bool) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            timeout=_timeout,
        )
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        if disable_thinking:
            kw["extra_body"] = {"enable_thinking": False}
        return kw

    try:
        if disable_thinking:
            try:
                resp = await client.chat.completions.create(**_kwargs(True))
            except APIStatusError as e:
                # Only a 4xx rejection (e.g. the backend does not know the
                # ``enable_thinking`` param) justifies a plain retry. Timeouts
                # and connection/5xx failures must propagate immediately —
                # retrying them would double the silence before surfacing.
                if e.status_code is None or e.status_code >= 500:
                    raise
                resp = await client.chat.completions.create(**_kwargs(False))
        else:
            resp = await client.chat.completions.create(**_kwargs(False))
    except Exception as e:  # noqa: BLE001 — classified below
        _friendly_ai_error(e, model=model, backend_url=INFER_URL)

    try:
        text = extract_reply_text(resp, default="")
    except Exception as e:  # noqa: BLE001
        # AIResponseTruncatedError (empty answer at the model's output limit)
        # and AIBackendError (empty choices) both propagate — callers keep
        # their own fallback logic.
        if isinstance(e, ValueError):
            raise
        _friendly_ai_error(e, model=model, backend_url=INFER_URL)

    choices = getattr(resp, "choices", None)
    if (
        choices
        and getattr(choices[0], "finish_reason", None) == "length"
        and text.strip()
    ):
        log.warning(
            "complete_text(%s): response hit the model's output limit "
            "(finish_reason='length'); the answer may be cut off (%d chars)",
            model, len(text),
        )
    return text


async def ask_ai(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int | None,
    username: str = "",
    user_id: str | int | None = None,
    char_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """AI request with RAG, rate limiting, and input validation.

    Returns (reply_text, extra_info_dict).
    """
    async with _ai_slot():
        ctx = await _build_ai_request(
            user_message, model_slug, guild_id, channel_id,
            username=username, user_id=user_id, char_key=char_key,
        )

        from bot_core.errors import extract_reply_text

        # Omit max_tokens when unset so the backend (LM Studio) lets the model
        # use its maximum output length.
        create_kwargs: dict[str, Any] = dict(
            model=ctx.effective_model,
            messages=ctx.messages,
            temperature=ctx.temperature,
            stream=False,
            timeout=ctx.timeout_sec,
        )
        if ctx.max_tokens is not None:
            create_kwargs["max_tokens"] = ctx.max_tokens

        try:
            # P1 #7: one bounded retry on transient 5xx/timeout/connection
            # failures before we give up on the turn.
            resp = await _call_completion_with_retry(ctx.client, **create_kwargs)
            # P1 #8: extract the reply safely — an empty `choices` array is a
            # malformed backend response (AIBackendError), and a response cut
            # off at the model's output limit with no visible answer raises
            # AIResponseTruncatedError. Both fall through to _friendly_ai_error
            # below as real errors — there is no bigger budget to retry with
            # when max_tokens is omitted (model max).
            reply_text = extract_reply_text(resp)
        except Exception as e:  # noqa: BLE001 — classified below
            _friendly_ai_error(e, model=ctx.effective_model, backend_url=INFER_URL)

        log.debug("RAW_AI_RESPONSE_START\n%s\nRAW_AI_RESPONSE_END", reply_text)

        # ── Update history (P0 #2) ──────────────────────────────────────
        _persist_turn(guild_id, channel_id, user_message, reply_text)

        approx_tokens = max(1, len(reply_text) // 4)  # §4.3: char-based estimate, not word count
        return reply_text, {"model_used": ctx.effective_model, "tokens_approx": approx_tokens}


async def ask_ai_stream(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int | None,
    username: str = "",
    user_id: str | int | None = None,
    char_key: str | None = None,
) -> AsyncIterator[str]:
    """Streaming variant of :func:`ask_ai` (P3 #24).

    An ``async generator`` that yields progressively-grown text as the
    completion is produced, and yields the *final* full reply as its last
    value. It holds the same process-wide AI slot for the whole turn and
    persists history exactly once, at the end, so a mid-stream failure (or a
    cancelled consumer) leaves no partial turn in the history.

    Timeout (single knob, ``AI_TIMEOUT_S``, overridable in .env): maximum
    SILENCE from the backend. The SDK passes it to httpx as a per-read
    timeout, which on a stream is an idle watchdog — any arriving chunk
    resets it, so a slow-but-alive generation may run indefinitely while a
    stalled/dead backend fails within ``AI_TIMEOUT_S`` of silence. There is
    deliberately no total wall-clock cap; /ai stop (cancellation) is the
    escape hatch.

    Yields
    ------
    str
        Each yield is the reply text so far (monotonically growing). The
        final yield is the complete reply.

    Raises
    ------
    asyncio.CancelledError
        Propagates cleanly so a consumer can abandon the turn (P3 #25).
    RateLimitError
        Re-raised as-is (see :func:`_friendly_ai_error`).
    ValueError
        Input-too-long / no-model / classified friendly error (P1 #7/#8).
    """
    async with _ai_slot():
        ctx = await _build_ai_request(
            user_message, model_slug, guild_id, channel_id,
            username=username, user_id=user_id, char_key=char_key,
        )

        collected: list[str] = []
        try:
            # The streaming call is NOT retried: a mid-stream transient
            # failure is ambiguous (we may have already emitted tokens), so
            # we surface it immediately rather than risk a duplicated partial
            # reply. The non-streaming path keeps its one bounded retry.
            #
            # Timeout: AI_TIMEOUT_S is an idle watchdog — httpx applies the
            # SDK timeout per read, so a stalled stream (no chunk within
            # AI_TIMEOUT_S) raises ReadTimeout while an active generation may
            # run indefinitely. No total wall-clock cap; /ai stop is the
            # escape hatch.
            from openai import AsyncStream
            from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

            create_kwargs: dict[str, Any] = dict(
                model=ctx.effective_model,
                messages=ctx.messages,
                temperature=ctx.temperature,
                stream=True,
                timeout=AI_TIMEOUT_S,
            )
            if ctx.max_tokens is not None:
                create_kwargs["max_tokens"] = ctx.max_tokens

            stream: AsyncStream[ChatCompletionChunk] = \
                await ctx.client.chat.completions.create(**create_kwargs)
            try:
                while True:
                    try:
                        chunk = await stream.__anext__()
                    except StopAsyncIteration:
                        break

                    # Track the server-side finish reason (set on the final
                    # chunk) so a truncated-but-silent stream can be told
                    # apart from an ordinary empty answer (P4).
                    if getattr(chunk, "choices", None):
                        fr = getattr(chunk.choices[0], "finish_reason", None)
                        if fr:
                            setattr(stream, "_final_finish_reason", fr)
                    try:
                        delta = chunk.choices[0].delta.content
                    except (IndexError, AttributeError, KeyError, TypeError):
                        continue
                    if delta:
                        collected.append(delta)
                        yield "".join(collected)
            finally:
                # Release the HTTP connection when the loop ends for any
                # reason (success, timeout, cancellation); idempotent in the
                # openai SDK.
                try:
                    # openai's AsyncStream.close() is async — awaiting it is
                    # what actually releases the HTTP connection (P3 #3 fix).
                    await stream.close()
                except Exception:
                    pass
        except asyncio.CancelledError:
            # Abandoned (e.g. /ai stop, P3 #25). Do not persist a partial turn.
            log.info(
                "ask_ai_stream: cancelled — %d chars accumulated, not persisted",
                len("".join(collected)),
            )
            raise
        except Exception as e:  # noqa: BLE001 — classified below
            _friendly_ai_error(e, model=ctx.effective_model, backend_url=INFER_URL)

        reply_text = "".join(collected).strip()
        if not reply_text:
            # P4: an empty stream whose final chunk reported
            # finish_reason "length" means the thinking model spent its whole
            # max_tokens budget with nothing left for the answer — report that
            # honestly instead of the generic empty-response message.
            from bot_core.errors import AIBackendError, AIResponseTruncatedError
            if getattr(stream, "_final_finish_reason", None) == "length":
                limit_desc = (
                    f"max_tokens={ctx.max_tokens}"
                    if ctx.max_tokens is not None
                    else "the model's maximum output length"
                )
                _friendly_ai_error(
                    AIResponseTruncatedError(
                        f"Streamed response truncated at {limit_desc} "
                        f"with no visible answer.",
                        max_tokens=ctx.max_tokens,
                    ),
                    model=ctx.effective_model, backend_url=INFER_URL,
                )
            _friendly_ai_error(
                AIBackendError("The AI backend returned an empty response."),
                model=ctx.effective_model, backend_url=INFER_URL,
            )

        log.debug("RAW_AI_RESPONSE_START\n%s\nRAW_AI_RESPONSE_END", reply_text)
        _persist_turn(guild_id, channel_id, user_message, reply_text)
        # Final value: the complete reply (also the last delta).
        yield reply_text
