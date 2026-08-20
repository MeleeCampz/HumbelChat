"""Provider-agnostic AI client + RAG orchestration."""
from __future__ import annotations

import logging
import math

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
)

log = logging.getLogger("bot.bot_core")

# ── Shared client (singleton) ─────────────────────────────────────────
# Creating an AsyncOpenAI per request opened a new connection pool each
# time; reuse one client (and its httpx pool) for the process lifetime.
_shared_client: AsyncOpenAI | None = None


def _make_client() -> AsyncOpenAI:
    global _shared_client
    if _shared_client is None:
        _shared_client = AsyncOpenAI(api_key=INFER_API_KEY, base_url=INFER_URL)
    return _shared_client


async def ask_ai(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int,
    username: str = "",
) -> tuple[str, dict]:
    effective_model = (model_slug or "").strip() or DEFAULT_MODEL
    if not effective_model:
        raise ValueError(
            f"No model configured for this request. Character model='{model_slug}' is empty "
            f"and DEFAULT_MODEL is not set. Set MODEL_NAME in .env or add a model to the character."
        )
    log.debug("Using model '%s' for this request.", effective_model)

    client = _make_client()

    # Guard against stale character models: fall back to the .env default
    # instead of failing with a 400 mid-conversation.
    try:
        models_resp = await client.models.list()
        available = {m.id for m in models_resp.data}
        if available and effective_model not in available and effective_model != DEFAULT_MODEL:
            log.warning("Model '%s' not found on backend; falling back to '%s'", effective_model, DEFAULT_MODEL)
            effective_model = DEFAULT_MODEL
    except Exception as e:
        log.warning("Could not list backend models at %s: %s", INFER_URL, e)

    ensure_history(guild_id, channel_id)
    history = get_history(guild_id, channel_id)
    max_messages = CONTEXT_WINDOW

    from config.characters import get_character, default_character
    from bot_core.history import get_active_char_key

    active_key = get_active_char_key(guild_id, channel_id)
    char_obj = get_character(active_key) or default_character()
    system_p = getattr(char_obj, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT or "You are a helpful AI assistant."

    # RAG context injection
    rag_context = ""
    from kb.retrievers import retrieve_kb_documents
    kb_docs = await retrieve_kb_documents(
        query=user_message,
        kb_path=KB_PATH,
        strategy=RAG_RETRIEVAL_METHOD,
        top_n=RAG_MAX_DOCS,
        window_lines=RAG_WINDOW_LINES,
    )
    if kb_docs:
        limit = RAG_MAX_DOCS
        doc_names = [name for name, _ in kb_docs[:limit]]
        log.info("RAG: Attaching %d KB document(s) to context: [%s]",
                 len(doc_names), ", ".join(f'"{n}"' for n in doc_names))
        max_chars = RAG_MAX_CHARS
        parts = ["=== Knowledge Base ===\n"]
        chars_used = len(parts[-1])
        docs_added = 0
        for display_name, content in kb_docs[:limit]:
            doc_block = f"\n--- {display_name} ---\n{content}"
            if chars_used + len(doc_block) > max_chars:
                log.info("RAG: skipped doc '%s' (%d chars) — remaining budget %d chars",
                         display_name, len(doc_block), max_chars - chars_used)
                # Continue (not break): an oversized top-ranked doc must not
                # block smaller, still-fitting documents from being attached.
                continue
            parts.append(doc_block)
            chars_used += len(doc_block)
            docs_added += 1
        rag_context = "\n".join(parts) if docs_added else ""
        if docs_added < len(kb_docs):
            log.info("RAG: included %d/%d documents (~%.0fK chars) — budget cap reached",
                     docs_added, len(kb_docs), chars_used / 1024)

    messages: list[dict] = []
    if rag_context:
        messages.append({"role": "system", "content": f"{system_p}\n\nRelevant knowledge-base context:\n\n{rag_context}"})
    elif system_p:
        messages.append({"role": "system", "content": system_p})

    recent_history = history[-(2 * max_messages):] if max_messages else []
    messages.extend(recent_history)

    user_content = f"**{username}:** {user_message}" if username else user_message
    messages.append({"role": "user", "content": user_content})

    _total_chars = sum(len(m.get("content", "")) for m in messages)
    _approx_tokens = int(_total_chars / 4)
    log.info(
        "ask_ai → model=%s messages_in_prompt=%d KB_files=%d system_chars=%d rag_chars=%d history_msgs=%d total_chars=%.1fK estimated_tokens=%d",
        model_slug, len(messages), len(kb_docs),
        len(system_p), len(rag_context) if rag_context else 0,
        len(recent_history), _total_chars / 1024, _approx_tokens,
    )
    if kb_docs:
        for display_name, content in kb_docs[:limit]:
            log.info("RAG doc: %s — %.1fK chars (%d estimated tokens)",
                     display_name, len(content) / 1024, int(len(content) / 4))

    timeout_sec = REQUEST_TIMEOUT

    _char_max = char_obj.max_tokens if (char_obj and char_obj.max_tokens) else None
    _request_max_tokens: int = _char_max if _char_max else MAX_TOKENS

    _request_max_tokens = min(_request_max_tokens, MAX_TOKENS_HARD_CAP)

    # Per-character temperature (characters.json), falling back to a sane default.
    _char_temp = getattr(char_obj, "temperature", None)
    _request_temp: float = float(_char_temp) if isinstance(_char_temp, (int, float)) else 0.7

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
        error_msg = str(e)
        status_code = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
        if status_code == 400 or "model not found" in error_msg.lower():
            raise ValueError(
                f"Model '{effective_model}' not found on the AI backend at {INFER_URL}. "
                f"Requested character model: '{model_slug}'. Make sure this model exists and is available.\n"
                f"Full error: {error_msg}"
            ) from e
        raise

    reply_text = resp.choices[0].message.content or "(empty response)"
    log.info("RAW_AI_RESPONSE_START\n%s\nRAW_AI_RESPONSE_END", reply_text)

    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": reply_text})
    max_entries = 2 * CONTEXT_WINDOW if CONTEXT_WINDOW else 50
    if len(history) > max_entries:
        set_history(guild_id, channel_id, history[-max_entries:])

    approx_tokens = len(reply_text.split())
    return reply_text, {"model_used": effective_model, "tokens_approx": approx_tokens}


def get_current_message_count(guild_id: int, channel_id: int) -> int:
    return len(get_history(guild_id, channel_id))
