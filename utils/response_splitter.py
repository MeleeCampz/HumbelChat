"""Paragraph-aware response splitting for long Discord messages.

Discord hard-limits messages at 2000 characters. This module keeps every
emitted chunk below a configurable safe budget so there is still room for the
`[N/M]` metadata prefix added by :func:`send_long_response`, and it accounts
for the optional character header so header + body never exceeds that budget
(code review §3.8).
"""
from __future__ import annotations

DISCORD_MSG_LIMIT: int = 2000
# Leave room for the `[N/M] ` metadata prefix.  Ten characters covers a
# realistic chunk count while still staying under Discord's 2000-char limit.
DISCORD_SAFE_CHUNK: int = 1990


def _split_long_paragraph(text: str, limit: int) -> list[str]:
    """Split a single paragraph so each piece is at most *limit* chars."""
    pieces: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(line), limit):
                pieces.append(line[i : i + limit])
            continue

        if not current:
            current = line
            continue

        candidate = current + line
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = line

    if current:
        pieces.append(current)

    return pieces


def _split_long_message(text: str, header_text: str = "") -> list[str]:
    """Split *text* into chunks that respect Discord's message limit.

    The header is included in every returned chunk and counts against the
    safe chunk budget. The function prefers paragraph boundaries, then line
    boundaries, then hard line cuts for extremely long tokens or URLs.

    Args:
        text: The full reply text.
        header_text: Optional prefix (e.g. ``"--- Character ---"``) prepended
            to every chunk.

    Returns:
        A list of chunks ready to send as separate Discord messages.
    """
    if not text:
        return [""]

    header_prefix = f"{header_text}\n" if header_text else ""
    max_chunk = min(DISCORD_SAFE_CHUNK, DISCORD_MSG_LIMIT)
    body_limit = max(1, max_chunk - len(header_prefix))

    if len(header_prefix + text) <= max_chunk:
        return [text]

    chunks: list[str] = []
    current_chunk_body = ""

    for paragraph in text.split("\n\n"):
        separator = "\n\n" if current_chunk_body else ""

        if len(current_chunk_body + separator + paragraph) <= body_limit:
            current_chunk_body += separator + paragraph
            continue

        if current_chunk_body:
            chunks.append(current_chunk_body)
            current_chunk_body = ""

        if len(paragraph) <= body_limit:
            current_chunk_body = paragraph
            continue

        paragraph_pieces = _split_long_paragraph(paragraph, body_limit)
        current_chunk_body = paragraph_pieces[0]
        for piece in paragraph_pieces[1:]:
            chunks.append(current_chunk_body)
            current_chunk_body = piece

    if current_chunk_body:
        chunks.append(current_chunk_body)

    return [f"{header_prefix}{chunk}".strip() for chunk in chunks if chunk.strip()]


async def send_long_response(source, reply_text: str, char_name: str = "") -> None:
    """Send *reply_text* to a Discord channel following up on *source*.

    Works with both Slash Commands (``followup.send``) and prefix commands
    (``reply``). Chunks long responses into paragraph-aware pieces.

    Args:
        source: The Discord interaction/message to follow up from.
        reply_text: The full reply content.
        char_name: Display name for header metadata.
    """
    chunks = _split_long_message(reply_text, f"--- {char_name} ---" if char_name else "")

    for idx, chunk in enumerate(chunks, 1):
        meta = f"[{idx}/{len(chunks)}] "
        full_msg = f"{meta}{chunk}".strip()

        if hasattr(source, "followup"):
            await source.followup.send(full_msg)
        else:
            await source.reply(full_msg)
