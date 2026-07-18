"""Provider-agnostic AI client + per-channel conversation history."""
from __future__ import annotations

import logging
import pathlib
from openai import AsyncOpenAI
from config.settings import settings
from kb.retrievers import retrieve_kb_documents, get_available_strategies

log = logging.getLogger("bot.bot_core")


# ── Conversation history (shared across all modules) ─────────────────
# guild_id -> channel_id -> [messages]
_chat_history: dict[int, dict[int, list[dict]]] = {}

# Per-guild / per-channel active character map
_ACTIVE_CHARACTERS: dict[tuple[int, int], str] = {}

def get_active_char_key(guild_id: int | None, channel_id: int) -> str:
    """Return the active character key for this guild/channel, or default."""
    from config.characters import default_character
    if guild_id is not None and (guild_id, channel_id) in _ACTIVE_CHARACTERS:
        return _ACTIVE_CHARACTERS[(guild_id, channel_id)]
    return default_character().key

def set_active_char_key(guild_id: int | None, channel_id: int, char_key: str) -> None:
    """Set the active character for this guild/channel."""
    if guild_id is not None:
        _ACTIVE_CHARACTERS[(guild_id, channel_id)] = char_key

# KB state — populated at startup
_kb_kb_name: str = "humblewood"       # from settings.DEFAULT_KB_NAME
_kb_path_root: pathlib.Path = settings.KB_PATH


def ensure_history(guild_id: int, channel_id: int) -> None:
    """Ensure history lists exist for this guild+channel."""
    _chat_history.setdefault(guild_id, {})
    _chat_history[guild_id].setdefault(channel_id, [])


# ── Public API ───────────────────────────────────────────────────────

async def ask_ai(
    user_message: str,
    model_slug: str,
    guild_id: int,
    channel_id: int,
    username: str = "",
) -> tuple[str, dict]:
    """Send *user_message* to the AI using *model_slug* and return (reply_text, extra_info).

    Flow:
      1. Build message buffer: system prompt + bounded history + RAG context
      2. Call the AI provider via AsyncOpenAI client
      3. Append user+assistant to history, truncate to CONTEXT_WINDOW
      4. Return reply and metadata (model_used, tokens_approx)
    """

    # ── 1. Prepare client ────────────────────────────────────────────────
    effective_model = model_slug or settings.DEFAULT_MODEL
    if not effective_model:
        raise ValueError(
            f"No model configured for this request. Character model='{model_slug}' is empty "
            f"and DEFAULT_MODEL (from env MODEL_NAME) is not set either. "
            f"Set MODEL_NAME in your .env file, or add a non-empty 'model' field to the "
            f"character in characters.json (e.g. 'qwen3.5:9b')."
        )
    log.debug("Using model '%s' for this request.", effective_model)

    client = AsyncOpenAI(
        api_key=settings.INFER_API_KEY or "local-model-key",
        base_url=settings.INFER_URL,
    )

    ensure_history(guild_id, channel_id)
    history: list[dict] = _chat_history[guild_id][channel_id]
    max_messages = settings.CONTEXT_WINDOW  # number of (user+assistant) pairs to keep

    # ── 2. System prompt ─────────────────────────────────────────────────
    from config.characters import get_character, default_character
    active_key = get_active_char_key(guild_id, channel_id)
    char_obj = get_character(active_key) or default_character()
    system_p = char_obj.system_prompt if char_obj and hasattr(char_obj, 'system_prompt') and char_obj.system_prompt else settings.DEFAULT_SYSTEM_PROMPT
    if not system_p:
        system_p = settings.DEFAULT_SYSTEM_PROMPT or "You are a helpful AI assistant."

    # ── 3. RAG context injection (query-aware relevance ranking) ───────
    rag_context = ""
    kb_docs = retrieve_kb_documents(
        query=user_message,
        kb_path=settings.KB_PATH,
        strategy=settings.RAG_RETRIEVAL_METHOD,
        top_n=settings.RAG_MAX_DOCS,
    )

    if kb_docs:
        limit = settings.RAG_MAX_DOCS
        doc_names = [name for name, _ in kb_docs[:limit]]
        log.info("RAG: Attaching %d KB document(s) to context: [%s]",
                 len(doc_names), ', '.join(f'"{n}"' for n in doc_names))
        parts = [f"=== Knowledge Base: {_kb_kb_name} ===\n"]
        for display_name, content in kb_docs[:limit]:  # top N files (configurable via RAG_MAX_DOCS)
            parts.append(f"\n--- {display_name} ---")
            parts.append(content)
        rag_context = "\n".join(parts)

    messages: list[dict] = []
    if rag_context:
        messages.append({
            "role": "system",
            "content": f"{system_p}\n\nRelevant knowledge-base context:\n\n{rag_context}",
        })
    elif system_p:
        messages.append({"role": "system", "content": system_p})

    # ── 4. Add bounded history ───────────────────────────────────────────
    # Keep last N messages (N = CONTEXT_WINDOW pairs, each pair is 2 messages)
    recent_history = history[-(2 * max_messages):] if max_messages else []
    messages.extend(recent_history)

    # ── 5. Append caller's message ───────────────────────────────────────
    if username:
        user_content = f"**{username}:** {user_message}"
    else:
        user_content = user_message
    messages.append({"role": "user", "content": user_content})

    # ── 5b. Context size logging ─────────────────────────────────────────
    _total_chars = sum(len(m.get("content", "")) for m in messages)
    _approx_tokens = int(_total_chars / 4)  # rough estimate: ~4 chars per token
    log.info("ask_ai → model=%s messages_in_prompt=%d KB_files=%d system_chars=%d rag_chars=%d history_msgs=%d total_chars=%.1fK estimated_tokens=%d",
             model_slug,
             len(messages),
             len(kb_docs),
             len(system_p),
             len(rag_context) if rag_context else 0,
             len(recent_history),
             _total_chars / 1024,
             _approx_tokens)

    # Per-file RAG breakdown (for monitoring context bloat)
    if kb_docs:
        for display_name, content in kb_docs[:limit]:
            log.info("RAG doc: %s — %.1fK chars (%d estimated tokens)",
                     display_name,
                     len(content) / 1024,
                     int(len(content) / 4))

    # ── 6. Call provider ─────────────────────────────────────────────────
    timeout_sec = settings.REQUEST_TIMEOUT
    try:
        resp = await client.chat.completions.create(
            model=effective_model,
            messages=messages,
            temperature=0.7,
            max_tokens=settings.MAX_TOKENS,
            stream=False,
            timeout=timeout_sec,
        )
    except Exception as e:
        import traceback
        error_msg = str(e)
        status_code = getattr(getattr(e, 'response', None), 'status_code', None) or (getattr(e, 'status_code', None))
        if status_code == 400 or 'model not found' in error_msg.lower():
            raise ValueError(
                f"Model '{effective_model}' not found on the AI backend at {settings.INFER_URL}. "
                f"Requested character model: '{model_slug}'. Make sure this model exists and is available.\n"
                f"Full error: {error_msg}"
            ) from e
        raise

    reply_text = resp.choices[0].message.content or "(empty response)"
    log.info("RAW_AI_RESPONSE_START\n%s\nRAW_AI_RESPONSE_END", reply_text)

    # ── 7. Update history (bounded) ──────────────────────────────────────
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": reply_text})
    max_entries = 2 * settings.CONTEXT_WINDOW if settings.CONTEXT_WINDOW else 50
    if len(history) > max_entries:
        _chat_history[guild_id][channel_id] = history[-max_entries:]

    approx_tokens = len(reply_text.split())
    return reply_text, {
        "model_used": effective_model,
        "tokens_approx": approx_tokens,
    }


# ── History helpers (needed by other modules) ────────────────────────

async def clear_history(guild_id: int, channel_s: int) -> str:
    """Clear chat history for this guild+channel. Returns confirmation message."""
    _chat_history.pop(guild_id, {}).pop(channel_s, None)
    return "Chat history cleared."


def get_current_message_count(guild_id: int, channel_id: int) -> int:
    """Return the number of messages currently in history for this channel."""
    return len(_chat_history.get(guild_id, {}).get(channel_id, []))