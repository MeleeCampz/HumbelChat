"""Provider-agnostic AI client + RAG orchestration.

P2 additions:
  - Input length cap (MAX_INPUT_CHARS)
  - Per-user sliding-window rate limiting
  - RAG context injected into user message (not system prompt)
  - Streaming responses (ask_ai_stream)
"""
from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

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


def _scaled_timeout(total_chars: int) -> float:
    """§2.3: derive a request timeout that scales with prompt size.

    Base timeout (``REQUEST_TIMEOUT``) covers a small prompt. For every
    additional 1000 chars of prompt (≈250 tokens) we allow 0.5 s more,
    capped at ``REQUEST_TIMEOUT * 4`` to avoid unbounded waits.
    """
    extra = (total_chars / 1000) * 0.5
    return min(REQUEST_TIMEOUT + extra, REQUEST_TIMEOUT * 4)


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

async def ask_ai(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int,
    username: str = "",
    user_id: str | int | None = None,
) -> tuple[str, dict]:
    """Non-streaming AI request with RAG, rate limiting, and input validation.

    Returns (reply_text, extra_info_dict).
    """
    # ── P2-1: Input length cap ──────────────────────────────────────
    if len(user_message) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Input too long: {len(user_message)} chars exceeds the "
            f"{MAX_INPUT_CHARS}-character limit. Please shorten your message."
        )

    # ── P2-2: Rate limiting ─────────────────────────────────────────
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

    active_key = get_active_char_key(guild_id, channel_id)
    char_obj = get_character(active_key) or default_character()
    system_p = getattr(char_obj, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT or "You are a helpful AI assistant."

    # ── RAG context ────────────────────────────────────────────────
    rag_context = ""
    included_names: list[str] = []
    from kb.retrievers import retrieve_kb_documents
    kb_docs = await retrieve_kb_documents(
        query=user_message,
        kb_path=KB_PATH,
        strategy=RAG_RETRIEVAL_METHOD,
        top_n=RAG_MAX_DOCS,
        window_lines=RAG_WINDOW_LINES,
    )
    if kb_docs:
        rag_context, included_names = _build_rag_context(kb_docs)

    # ── Build messages ─────────────────────────────────────────────
    messages: list[dict] = []
    # P2-3: system prompt is persona-only (no RAG)
    if system_p:
        messages.append({"role": "system", "content": system_p})

    recent_history = history[-(2 * max_messages):] if max_messages else []
    messages.extend(recent_history)

    # P2-3: RAG context injected into the user message, right before the question
    user_content = f"**{username}:** {user_message}" if username else user_message
    if rag_context:
        user_content = (
            f"[Relevant knowledge-base context]\n{rag_context}\n\n"
            f"---\n\n"
            f"{user_content}"
        )
    messages.append({"role": "user", "content": user_content})

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

    timeout_sec = REQUEST_TIMEOUT
    _request_max_tokens, _request_temp = _resolve_request_params(char_obj)

    # §2.3: scale the timeout with prompt size — large RAG contexts take the
    # backend much longer, and a flat 120 s cap caused frequent timeouts on
    # 60 K+ token prompts. Add 0.5 s per 1000 chars of prompt (capped).
    timeout_sec = _scaled_timeout(_total_chars)

    try:
        resp = await client.chat.completions.create(
            model=effective_model,
            messages=messages,
            temperature=_request_temp,
            max_tokens=_request_max_tokens,
            stream=False,
            timeout=timeout_sec,
        )
    except Exception as e:
        # §3.7: structured error taxonomy — surface a user-friendly message
        # instead of a raw SDK traceback.
        from bot_core.errors import classify_ai_error
        classified = classify_ai_error(e, model=effective_model, backend_url=INFER_URL)
        log.error("AI request failed (%s): %s", getattr(classified, "category", "unknown"), e)
        if isinstance(classified, RateLimitError):
            # Classifier is pure (returns, never raises) — re-raise the
            # classified error so handlers can show the retry-after message.
            raise classified from e
        if isinstance(classified, ValueError) or getattr(classified, "category", "") in ("timeout", "model_not_found", "backend_down"):
            raise ValueError(classified.user_message) from e
        raise

    reply_text = resp.choices[0].message.content or "(empty response)"
    log.info("RAW_AI_RESPONSE_START\n%s\nRAW_AI_RESPONSE_END", reply_text)

    # ── Update history ─────────────────────────────────────────────
    # Store the *clean* user message, not `user_content` (which carries the
    # RAG context blob + username decoration). Persisting the inflated form
    # would re-inject stale KB context into every subsequent turn and bloat
    # prompts by ~RAG_MAX_CHARS per past turn.
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply_text})
    max_entries = 2 * CONTEXT_WINDOW if CONTEXT_WINDOW else 50
    if len(history) > max_entries:
        set_history(guild_id, channel_id, history[-max_entries:])

    approx_tokens = max(1, len(reply_text) // 4)  # §4.3: char-based estimate, not word count
    return reply_text, {"model_used": effective_model, "tokens_approx": approx_tokens}


async def ask_ai_stream(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int,
    username: str = "",
    user_id: str | int | None = None,
):
    """Streaming AI request. Yields text chunks as they arrive.

    Same validation, rate limiting, RAG, and history handling as ask_ai,
    but uses ``stream=True`` and yields chunks one at a time.

    Usage::

        chunks: list[str] = []
        async for chunk in ask_ai_stream(...):
            chunks.append(chunk)
        full_text = "".join(chunks)

    After iteration completes, conversation history is updated.
    """
    # ── P2-1: Input length cap ──────────────────────────────────────
    if len(user_message) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Input too long: {len(user_message)} chars exceeds the "
            f"{MAX_INPUT_CHARS}-character limit. Please shorten your message."
        )

    # ── P2-2: Rate limiting ─────────────────────────────────────────
    if user_id is not None:
        check_rate_limit(str(user_id))

    effective_model = (model_slug or "").strip() or DEFAULT_MODEL
    if not effective_model:
        raise ValueError(
            f"No model configured for this request. Character model='{model_slug}' is empty "
            f"and DEFAULT_MODEL is not set. Set MODEL_NAME in .env or add a model to the character."
        )

    client = _make_client()
    effective_model = await _validate_model(client, effective_model)

    ensure_history(guild_id, channel_id)
    history = get_history(guild_id, channel_id)
    max_messages = CONTEXT_WINDOW

    from config.characters import get_character, default_character
    from bot_core.history import get_active_char_key

    active_key = get_active_char_key(guild_id, channel_id)
    char_obj = get_character(active_key) or default_character()
    system_p = getattr(char_obj, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT or "You are a helpful AI assistant."

    # ── RAG context ────────────────────────────────────────────────
    rag_context = ""
    included_names: list[str] = []
    from kb.retrievers import retrieve_kb_documents
    kb_docs = await retrieve_kb_documents(
        query=user_message,
        kb_path=KB_PATH,
        strategy=RAG_RETRIEVAL_METHOD,
        top_n=RAG_MAX_DOCS,
        window_lines=RAG_WINDOW_LINES,
    )
    if kb_docs:
        rag_context, included_names = _build_rag_context(kb_docs)

    # ── Build messages ─────────────────────────────────────────────
    messages: list[dict] = []
    if system_p:
        messages.append({"role": "system", "content": system_p})

    recent_history = history[-(2 * max_messages):] if max_messages else []
    messages.extend(recent_history)

    user_content = f"**{username}:** {user_message}" if username else user_message
    if rag_context:
        user_content = (
            f"[Relevant knowledge-base context]\n{rag_context}\n\n"
            f"---\n\n"
            f"{user_content}"
        )
    messages.append({"role": "user", "content": user_content})

    log.info(
        "ask_ai_stream → model=%s channel=%s messages=%d rag_docs=%d total_chars=%.1fK",
        effective_model, channel_id, len(messages), len(included_names),
        sum(len(m.get("content", "")) for m in messages) / 1024,
    )

    _request_max_tokens, _request_temp = _resolve_request_params(char_obj)
    # §2.3: apply the same prompt-size-aware timeout to the stream setup path.
    _stream_total_chars = sum(len(m.get("content", "")) for m in messages)
    _stream_timeout_sec = _scaled_timeout(_stream_total_chars)

    # ── Stream ─────────────────────────────────────────────────────
    try:
        stream = await client.chat.completions.create(
            model=effective_model,
            messages=messages,
            temperature=_request_temp,
            max_tokens=_request_max_tokens,
            stream=True,
            timeout=_stream_timeout_sec,
        )

        full_text_parts: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta is not None:
                full_text_parts.append(delta)
                yield delta

        reply_text = "".join(full_text_parts) or "(empty response)"
        log.info("STREAM_AI_RESPONSE channel=%s len=%d chars", channel_id, len(reply_text))

        # ── Update history after full stream ──────────────────────
        # Clean message only — see the note in ask_ai() above.
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply_text})
        max_entries = 2 * CONTEXT_WINDOW if CONTEXT_WINDOW else 50
        if len(history) > max_entries:
            set_history(guild_id, channel_id, history[-max_entries:])

    except Exception as e:
        from bot_core.errors import classify_ai_error
        classified = classify_ai_error(e, model=effective_model, backend_url=INFER_URL)
        log.error("AI stream failed (%s): %s", getattr(classified, "category", "unknown"), e)
        if isinstance(classified, RateLimitError):
            raise classified from e
        if getattr(classified, "category", "") in ("timeout", "model_not_found", "backend_down"):
            raise ValueError(classified.user_message) from e
        raise

