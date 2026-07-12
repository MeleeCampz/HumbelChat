"""Paragraph-aware response splitting for long Discord messages.

Keeps code blocks, lists, and sections intact while respecting Discord's 2000-char limit
(uses 1900 to leave room for metadata).
"""
from __future__ import annotations


def _split_long_message(text: str, header_text: str = "") -> list[str]:
    """Split *text* into multiple chunks if it exceeds Discord's ~1900-char limit.

    Uses paragraph-aware splitting so code blocks and lists stay unbroken.
    
    Args:
        text: The full reply text.
        header_text: Optional prefix (e.g., "--- Character ---") prepended to every chunk.
        
    Returns:
        A list of chunks ready to send as separate Discord messages.
    """
    MAX = 1900          # leave room for "[N/M] " metadata prefix
    meta_prefix = f"{header_text}\n" if header_text else ""

    if len(meta_prefix + text) <= MAX:
        return [text]

    # Split by double-newlines (paragraphs), then force-split oversized paragraphs
    paragraphs = text.split('\n\n')
    chunks, current = [], ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(meta_prefix) + len(candidate) <= MAX:
            current = candidate
        elif current:
            # Current chunk is full — emit it, start a new one with this paragraph
            chunks.append(current.strip())
            # If the single paragraph itself is too long → force-split by words
            if len(meta_prefix) + len(para) > MAX:
                words = para.split()
                word_chunks, sub = [], ""
                for w in words:
                    test = f"{sub} {w}".strip() if sub else w
                    if len(test) > MAX:
                        chunks.append((meta_prefix + sub).strip())
                        sub = w
                    else:
                        sub = test
                current = sub
            else:
                current = para
        else:
            # First paragraph was also too long → force-split
            words = para.split()
            sub, started = "", False
            for w in words:
                test = f"{sub} {w}".strip() if sub else w
                if len(test) > MAX:
                    chunks.append((meta_prefix + sub).strip())
                    sub = w
                    started = True
                else:
                    sub = test
                    started = True
            current = sub

    if current:
        chunks.append(current.strip())

    return chunks


async def send_long_response(source, reply_text: str, char_name: str = "") -> None:
    """Send *reply_text* to a Discord channel following up on *source*.
    
    Works with both Slash Commands (followup.send) and prefix commands (reply).
    Chunks long responses into paragraph-aware pieces.
    
    Args:
        source: The Discord interaction/message to follow up from.
        reply_text: The full reply content.
        char_name: Display name for header metadata.
    """
    # Determine if we need chunking
    chunks = _split_long_message(reply_text, f"--- {char_name} ---")

    for idx, chunk in enumerate(chunks, 1):
        meta = f"[{idx}/{len(chunks)}] "
        full_msg = f"{meta}{chunk}".strip()

        # Follow-up for slash commands; reply for prefix commands
        if hasattr(source, 'followup'):
            await source.followup.send(full_msg)
        else:
            await source.reply(full_msg)
