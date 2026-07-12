"""Provider-agnostic AI client + per-channel conversation history."""
from __future__ import annotations

import logging
import pathlib
from openai import AsyncOpenAI
from config.settings import settings
from kb.reader import read_kb_files

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
    client = AsyncOpenAI(
        api_key=settings.INFER_API_KEY or "local-model-key",
        base_url=settings.INFER_URL,
    )

    ensure_history(guild_id, channel_id)
    history: list[dict] = _chat_history[guild_id][channel_id]
    max_messages = settings.CONTEXT_WINDOW  # number of (user+assistant) pairs to keep

    # ── 2. System prompt ─────────────────────────────────────────────────
    system_p = settings.DEFAULT_SYSTEM_PROMPT or "You are a helpful AI assistant."

    # ── 3. RAG context injection ──────────────────────────────────────────
    rag_context = ""
    kb_docs = read_kb_files(settings.KB_PATH)
    if kb_docs:
        parts = [f"=== Knowledge Base: {_kb_kb_name} ===\n"]
        for display_name, content in kb_docs[:5]:  # top 5 files
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

    log.info("ask_ai → model=%s messages_in_prompt=%d KB_files=%d",
             model_slug, len(messages), len(kb_docs))

    # ── 6. Call provider ─────────────────────────────────────────────────
    timeout_sec = settings.REQUEST_TIMEOUT
    resp = await client.chat.completions.create(
        model=model_slug,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=False,
        timeout=timeout_sec,
    )

    reply_text = resp.choices[0].message.content or "(empty response)"

    # ── 7. Update history (bounded) ──────────────────────────────────────
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": reply_text})
    max_entries = 2 * settings.CONTEXT_WINDOW if settings.CONTEXT_WINDOW else 50
    if len(history) > max_entries:
        _chat_history[guild_id][channel_id] = history[-max_entries:]

    approx_tokens = len(reply_text.split())
    return reply_text, {
        "model_used": model_slug,
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
