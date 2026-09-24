"""Provider-agnostic AI client + RAG orchestration.

P2 additions:
  - Input length cap (MAX_INPUT_CHARS)
  - Per-user sliding-window rate limiting
  - RAG context injected into user message (not system prompt)

P3 additions:
  - Streaming completions (:func:`ask_ai_stream`) for live progress display
  - Live queue-depth tracking on the global AI slot (:func:`ai_queue_depth`)
"""
from __future__ import annotations

import logging
import time

import asyncio
from contextlib import asynccontextmanager
from openai import AsyncOpenAI, APIStatusError

from bot_core.history import ensure_history, get_history, set_history
from config.settings import (
    INFER_URL,
    INFER_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    CONTEXT_WINDOW,
    REQUEST_TIMEOUT,
    MAX_TOKENS,
    MAX_TOKENS_HARD_CAP,
    KB_PATH,
    RAG_MAX_DOCS,
    RAG_MAX_CHARS,
    RAG_WINDOW_LINES,
    RAG_RETRIEVAL_METHOD,
    MAX_INPUT_CHARS,
    AI_RATE_LIMIT_MAX,
    AI_RATE_LIMIT_WINDOW,
)

log = logging.getLogger("bot.bot_core")

# ── Shared client (singleton) ─────────────────────────────────────────
_shared_client: AsyncOpenAI | None = None


def _make_client() -> AsyncOpenAI:
    global _shared_client
    if _shared_client is None:
        _shared_client = AsyncOpenAI(api_key=INFER_API_KEY, base_url=INFER_URL)
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
        return DEFAULT_MODEL
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


def _scaled_timeout(total_chars: int, max_tokens: int | None = None) -> float:
    """§2.3: derive a request timeout that scales with prompt size.

    Base timeout (``REQUEST_TIMEOUT``) covers a small prompt. For every
    additional 1000 chars of prompt (≈250 tokens) we allow 0.5 s more.

    Large *output* budgets (e.g. thinking models that may produce up to
    ``max_tokens`` tokens) are scaled the same way per 1000 max_tokens,
    because a full-budget generation at low local inference speed needs
    roughly the same wall time as a same-size prompt. The cap is
    ``REQUEST_TIMEOUT * 8`` so a generous budget (e.g. 16 K tokens) cannot
    time out while the backend is legitimately still generating — at
    ``AI_REQUEST_TIMEOUT=120`` that is 960 s worst case, still finite.
    """
    extra = (total_chars / 1000) * 0.5
    if max_tokens:
        extra += (max_tokens / 1000) * 0.5
    return min(REQUEST_TIMEOUT + extra, REQUEST_TIMEOUT * 8)


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


async def _call_completion_with_retry(client: AsyncOpenAI, **kwargs):
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
    raise last_exc  # pragma: no cover — unreachable, loop always returns/raises


def _resolve_request_params(char_obj) -> tuple[int, float]:
    """Resolve max_tokens and temperature from character config."""
    _char_max = char_obj.max_tokens if (char_obj and char_obj.max_tokens) else None
    _request_max_tokens: int = _char_max if _char_max else MAX_TOKENS
    _request_max_tokens = min(_request_max_tokens, MAX_TOKENS_HARD_CAP)

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
async def _ai_slot():
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
        messages: list[dict],
        user_message: str,
        guild_id: int,
        channel_id: int,
        username: str,
        system_p: str,
        rag_context: str,
        included_names: list[str],
        recent_history: list[dict],
        total_chars: int,
        max_tokens: int,
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


def _append_user_message(username: str, user_message: str, rag_context: str) -> str:
    """Build the *final* user-turn content: username decoration + RAG context.

    P3 #37: the username is stripped of Discord markdown emphasis characters
    so a display name like ``**bold**`` can't break the ``**{username}:**``
    prompt-prefix formatting (it would otherwise swallow a following token).
    """
    safe_username = (username or "").replace("*", "")
    base = f"**{safe_username}:** {user_message}" if safe_username else user_message
    if rag_context:
        base = (
            f"[Relevant knowledge-base context]\n{rag_context}\n\n"
            f"---\n\n"
            f"{base}"
        )
    return base


async def _build_ai_request(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int,
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
    system_p = getattr(char_obj, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT or "You are a helpful AI assistant."

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

    # ── Build messages ─────────────────────────────────────────────────
    messages: list[dict] = []
    # P2-3: system prompt is persona-only (no RAG)
    if system_p:
        messages.append({"role": "system", "content": system_p})

    recent_history = history[-(2 * max_messages):] if max_messages else []
    messages.extend(recent_history)

    # P2-3: RAG context injected into the user message, right before the question
    messages.append({"role": "user", "content": _append_user_message(username, user_message, rag_context)})

    _total_chars = sum(len(m.get("content", "")) for m in messages)
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

    # §2.3: scale the timeout with prompt size — large RAG contexts take the
    # backend much longer, and a flat 120 s cap caused frequent timeouts on
    # 60 K+ token prompts. Add 0.5 s per 1000 chars of prompt (capped).
    # Timeout scales with prompt size AND the output budget (a thinking
    # model generating up to max_tokens needs wall time for the output too).
    timeout_sec = _scaled_timeout(_total_chars, _request_max_tokens)

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


def _persist_turn(guild_id: int, channel_id: int, user_message: str, reply_text: str) -> None:
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


async def ask_ai(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int,
    username: str = "",
    user_id: str | int | None = None,
    char_key: str | None = None,
) -> tuple[str, dict]:
    """AI request with RAG, rate limiting, and input validation.

    Returns (reply_text, extra_info_dict).
    """
    async with _ai_slot():
        ctx = await _build_ai_request(
            user_message, model_slug, guild_id, channel_id,
            username=username, user_id=user_id, char_key=char_key,
        )

        from bot_core.errors import AIResponseTruncatedError, extract_reply_text

        try:
            # P1 #7: one bounded retry on transient 5xx/timeout/connection
            # failures before we give up on the turn.
            resp = await _call_completion_with_retry(
                ctx.client,
                model=ctx.effective_model,
                messages=ctx.messages,
                temperature=ctx.temperature,
                max_tokens=ctx.max_tokens,
                stream=False,
                timeout=ctx.timeout_sec,
            )
            # P1 #8: extract the reply safely — an empty `choices` array is a
            # malformed backend response, not a crash. `extract_reply_text`
            # returns a placeholder for empty *content* and raises AIBackendError
            # for empty *choices*, which the except below turns into a
            # user-friendly message.
            reply_text = extract_reply_text(resp)
        except AIResponseTruncatedError:
            # P4: the model spent its ENTIRE max_tokens budget on thinking
            # and produced no answer (finish_reason "length", content empty).
            # Retry once with a bigger budget (4×, capped at the hard cap).
            # If the budget was already the cap, there is nothing bigger to
            # try — surface the real error instead of an empty reply.
            retry_budget = min(MAX_TOKENS_HARD_CAP, ctx.max_tokens * 4)
            if retry_budget <= ctx.max_tokens:
                log.error(
                    "Response truncated at max_tokens=%d and the budget is "
                    "already the hard cap — surfacing a real error",
                    ctx.max_tokens,
                )
                _friendly_ai_error(
                    AIResponseTruncatedError(max_tokens=ctx.max_tokens),
                    model=ctx.effective_model, backend_url=INFER_URL,
                )
            log.warning(
                "Response truncated (empty answer at max_tokens=%d); "
                "retrying once with max_tokens=%d",
                ctx.max_tokens, retry_budget,
            )
            try:
                resp = await _call_completion_with_retry(
                    ctx.client,
                    model=ctx.effective_model,
                    messages=ctx.messages,
                    temperature=ctx.temperature,
                    max_tokens=retry_budget,
                    stream=False,
                    timeout=ctx.timeout_sec,
                )
                reply_text = extract_reply_text(resp)
            except Exception as e:  # noqa: BLE001 — classified below
                # Covers the retry call's own transient failures AND a second
                # truncation → friendly error, never an empty reply.
                _friendly_ai_error(e, model=ctx.effective_model, backend_url=INFER_URL)
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
    channel_id: int,
    username: str = "",
    user_id: str | int | None = None,
    char_key: str | None = None,
):
    """Streaming variant of :func:`ask_ai` (P3 #24).

    An ``async generator`` that yields progressively-grown text as the
    completion is produced, and yields the *final* full reply as its last
    value. It holds the same process-wide AI slot for the whole turn and
    persists history exactly once, at the end, so a mid-stream failure (or a
    cancelled consumer) leaves no partial turn in the history.

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
            # The streaming call is NOT retried: a mid-stream transient failure
            # is ambiguous (we may have already emitted tokens), so we surface
            # it immediately rather than risk a duplicated partial reply. The
            # non-streaming path keeps its one bounded retry.
            from openai import AsyncStream
            from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

            stream: AsyncStream[ChatCompletionChunk] = \
                await ctx.client.chat.completions.create(
                    model=ctx.effective_model,
                    messages=ctx.messages,  # type: ignore[arg-type]
                    temperature=ctx.temperature,
                    max_tokens=ctx.max_tokens,
                    stream=True,
                    timeout=ctx.timeout_sec,
                )
            async for chunk in stream:
                # Track the server-side finish reason (set on the final
                # chunk) so a truncated-but-silent stream can be told apart
                # from an ordinary empty answer (P4).
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
                _friendly_ai_error(
                    AIResponseTruncatedError(
                        f"Streamed response truncated at max_tokens="
                        f"{ctx.max_tokens} with no visible answer.",
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
