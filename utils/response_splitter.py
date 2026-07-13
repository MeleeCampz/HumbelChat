"""Paragraph-aware response splitting for long Discord messages.

Keeps code blocks, lists, and sections intact while respecting Discord's 2000-char limit
(uses 1900 to leave room for metadata).
"""
from __future__ import annotations


def _split_long_message(text: str, header_text: str = "") -> list[str]:
    """Split *text* into multiple chunks if it exceeds Discord's ~1900-char limit.

    Uses paragraph-aware splitting so code blocks and lists stay unbroken.
    Every chunk will include the header_text if provided.
    
    Args:
        text: The full reply text.
        header_text: Optional prefix (e.g., "--- Character ---") prepended to every chunk.
        
    Returns:
        A list of chunks ready to send as separate Discord messages.
    """
    MAX = 1900          # leave room for "[N/M] " metadata prefix
    header_prefix = f"{header_text}\n" if header_text else ""

    if len(header_prefix + text) <= MAX:
        return [text]

    chunks: list[str] = []
    current_chunk_body = ""

    # Split by double-newlines (paragraphs)
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        separator = "\n\n" if current_chunk_body else ""
        
        # Check if the paragraph itself is too large to fit even in a new chunk with header
        if len(header_pseudo := (header_prefix + para)) > MAX:
            # Flush existing body before handling the massive paragraph
            if current_chunk_body:
                chunks.append(current_chunk_body)
                current_chunk_body = ""
            
            # Split this oversized paragraph by words
            words = para.split()
            sub_para_body = ""
            for word in words:
                word_sep = " " if sub_para_body else ""
                if len(header_prefix) + len(sub_para_body + word_sep + word) <= MAX:
                    sub_para_body += (word_sep + word)
                else:
                    # Sub-paragraph is full, flush it
                    if sub_para_body:
                        chunks.append(sub_para_body)
                    sub_para_body = word
            current_chunk_body = sub_para_body
        
        else:
            # Check if adding this paragraph to the current body exceeds MAX (with header)
            if len(header_prefix) + len(current_chunk_body + separator + para) <= MAX:
                current_chunk_body += (separator + para)
            else:
                # Flush existing and start new with this paragraph
                if current_chunk_body:
                    chunks.append(current_chunk_body)
                current_chunk_body = para

    if current_chunk_body:
        chunks.append(current_chunk_body)

    # Prepend header to every chunk so context is preserved
    return [f"{header_prefix}{c}".strip() for c in chunks if c.strip()]


async def send_long_response(source, reply_text: str, char_name: str = "") -> None:
    """Send *reply_text* to a Discord channel following up on *source*.
    
    Works with both Slash Commands (followup.send) and prefix commands (reply).
    Chunks long responses into paragraph-aware pieces.
    
    Args:
        source: The Discord interaction/message to follow up from.
        reply_text: The full reply content.
        char_name: Display name for header metadata.
    """
    # Determine if we need chunking. Each chunk will already have the header text.
    chunks = _split_long_message(reply_text, f"--- {char_name} ---" if char_name else "")

    for idx, chunk in enumerate(chunks, 1):
        meta = f"[{idx}/{len(chunks)}] "
        full_msg = f"{meta}{chunk}".strip()

        # Follow-up for slash commands; reply for prefix commands
        if hasattr(source, 'followup'):
            await source.followup.send(full_msg)
        else:
            await source.reply(full_msg)
