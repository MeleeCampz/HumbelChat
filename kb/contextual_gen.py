"""Contextual Retrieval generation — shared core (#52).

Generates the one-line, chunk-specific context descriptions (the #39 item 7
sidecar) for KB chunks that lack them. Two consumers:

* the optional **idle-time background fill** at bot startup (``main.py``,
  gated by ``CONTEXTUAL_BACKGROUND_FILL=1``) — this module's
  :func:`generate_missing_context_lines`, driven by the bot's own AI client;
* the offline script ``scripts/contextual_reindex.py`` — reuses
  :data:`BATCH_PROMPT` and :func:`parse_json_array` (single source of truth).

Design constraints:

* **Resumable** — chunks whose content hash already has a line are skipped,
  and the sidecar is persisted (atomic replace) after EVERY batch, so an
  interrupted fill never loses finished work.
* **Gentle** — batches are separated by a pause (default 60 s) so the shared
  chat model keeps serving live /ai traffic; callers may abort at any batch
  boundary via *should_stop* (e.g. bot shutdown).
* **Live-safe** — after each batch the sidecar is reloaded into the index,
  so in-flight retrieval immediately benefits from the new lines.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger("bot.contextual_gen")

# Prompt for one batch of chunks from the same file (moved verbatim from
# scripts/contextual_reindex.py — keep both callers in sync via this module).
BATCH_PROMPT = """You are building a search index for a D&D knowledge base. For each chunk below, write ONE line of context (max ~140 characters) that a retrieval system can match against user questions. The line must state what the chunk is about INDEPENDENTLY of its own text: include the file/section it comes from, and any key proper names, numbers, or dates it contains. For campaign session-log chunks, say which session it is (ordinal + date + title when visible in the path or headings) and what happens in this excerpt. Do not quote the chunk verbatim; do not add commentary about your task.

File: {file_path}
{chunks_block}

Respond with a JSON array of exactly {n} strings, in the same order as the chunks."""


def parse_json_array(text: str, expected: int) -> list[str] | None:
    """Extract the JSON array from a model reply (tolerates prose fences).

    Single-chunk batches often get a BARE string back (no brackets) — accept
    that shape when exactly one line is expected.
    """
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [x.strip() for x in data]
        except json.JSONDecodeError:
            pass
    if expected == 1:
        line = text.strip().strip('"').strip()
        if 0 < len(line) <= 300 and "[" not in line and not line.lower().startswith(("here", "sure", "ok")):
            return [line]
    return None


async def generate_missing_context_lines(
    index: Any,
    *,
    batch_size: int = 12,
    batch_pause: float = 60.0,
    max_chars: int = 1500,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Fill missing context lines for *index*'s chunks; return lines generated.

    *index* is a ``KBIndex`` (duck-typed to avoid an import cycle at module
    load). Chunks are grouped by source file and batched; each batch is one
    chat completion on the bot's default model. After every successful batch
    the sidecar is saved through ``index.save_context_lines`` and reloaded,
    so live retrieval sees new lines immediately. A batch that fails three
    times is skipped (resumable on a later run) — generation errors are data,
    never fatal to the bot.
    """
    from bot_core.ai_client import _make_client, _validate_model, complete_text
    from config.settings import DEFAULT_MODEL

    inner = index._index
    if inner is None:
        return 0
    docs = list(inner._docs)
    missing = [d for d in docs if not d.context_line and d.content.strip()]
    if not missing:
        log.info("Contextual fill: all %d chunk(s) already have context lines.", len(docs))
        return 0

    # Group consecutive chunks by source file so each prompt stays on one file.
    batches: list[list[Any]] = []
    cur: list[Any] = []
    cur_file: str | None = None
    for d in missing:
        if cur and d.source() != cur_file:
            batches.append(cur)
            cur, cur_file = [], None
        cur.append(d)
        cur_file = d.source()
        if len(cur) >= batch_size:
            batches.append(cur)
            cur, cur_file = [], None
    if cur:
        batches.append(cur)

    log.info("Contextual fill: %d chunk(s) missing lines in %d batch(es) of up to %d.",
             len(missing), len(batches), batch_size)

    client = _make_client()
    model = await _validate_model(client, DEFAULT_MODEL)
    generated = 0
    for bi, batch in enumerate(batches, 1):
        if should_stop is not None and should_stop():
            log.info("Contextual fill stopped by caller after %d line(s).", generated)
            return generated
        file_path = batch[0].source()
        chunks_block = "\n".join(
            f"[{i}] section={c.display_name.split(' [', 1)[-1] if ' [' in c.display_name else '(full document)'}\n"
            f"{c.content[:max_chars]}"
            for i, c in enumerate(batch, 1)
        )
        prompt = BATCH_PROMPT.format(file_path=file_path, chunks_block=chunks_block, n=len(batch))
        ok = False
        for attempt in range(3):
            if should_stop is not None and should_stop():
                return generated
            try:
                reply = await complete_text(
                    client, model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=900, reasoning_effort="none",
                )
                lines = parse_json_array(reply, expected=len(batch))
                if lines is not None and len(lines) == len(batch):
                    for c, line in zip(batch, lines):
                        if line:
                            index._context_lines[_content_hash_of(c)] = line[:300]
                            c.context_line = line[:300]
                    index.save_context_lines(index._context_lines)
                    index.reload_context_lines()
                    generated += len(batch)
                    ok = True
                    break
                log.warning("Contextual fill batch %d: bad reply shape (%s vs %d) — retrying",
                           bi, f"{len(lines)}" if lines else "none", len(batch))
            except Exception as exc:  # noqa: BLE001 — data, never fatal
                log.warning("Contextual fill batch %d: attempt %d failed: %r — retrying",
                            bi, attempt + 1, exc)
                await asyncio.sleep(3 * (attempt + 1))
        if not ok:
            log.warning("Contextual fill batch %d: giving up on %d chunk(s) (resumable later)",
                        bi, len(batch))
        if bi % 5 == 0 or bi == len(batches):
            log.info("Contextual fill progress: batch %d/%d  generated=%d  sidecar size=%d",
                     bi, len(batches), generated, len(index._context_lines))
        if bi < len(batches) and batch_pause > 0:
            await asyncio.sleep(batch_pause)
    return generated


def _content_hash_of(doc: Any) -> str:
    """Content hash of a chunk's ORIGINAL text — the sidecar key."""
    from kb.vector_db import _content_hash
    return _content_hash(doc.content)
