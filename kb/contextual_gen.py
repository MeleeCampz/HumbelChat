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
* **Vector-consistent (#77)** — each batch is followed by a targeted
  ``index.reembed_contextualized(hashes)`` so the chunk's dense vector is built
  from the contextual passage, not just the raw content. Generation without it
  leaves half-contextualized chunks (BM25 framed, dense leg blind).
"""
from __future__ import annotations

import asyncio
import inspect
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


def make_fill_client():
    """The chat client the fill talks to (same endpoint/credentials as /ai).

    A thin indirection over ``bot_core.ai_client._make_client`` so the readiness
    check in main.py and the generation loop below share one construction site —
    and so tests can substitute a stub client without patching a private name.
    """
    from bot_core.ai_client import _make_client
    return _make_client()


# How many deferral notices this process has logged (see _defer_log).
_defer_notices = 0


def _defer_log(msg: str, *args) -> None:
    """Loud the first time, quiet afterwards.

    Both deferral reasons (embedder offline because the GPU is painting, a model
    id that does not match the backend) are steady states, not incidents: at
    CONTEXTUAL_FILL_RECHECK_S=1800 an unthrottled WARNING would be 48 lines a day
    saying the same thing until somebody fixes the cause.
    """
    global _defer_notices
    _defer_notices += 1
    if _defer_notices <= 2 or _defer_notices % 10 == 0:
        log.warning(msg, *args)
    else:
        log.info(msg + " (notice #%d)", *args, _defer_notices)


async def resolve_fill_model(client: Any, timeout: float | None = None) -> tuple[bool, str | None]:
    """Decide which chat model the fill may use, before spending a whole run on it.

    Returns ``(usable, model)``: ``(True, None)`` = use ``DEFAULT_MODEL``;
    ``(True, name)`` = a hosted model was picked instead; ``(False, None)`` =
    don't run this round.

    Why this exists: the generation prompt does NOT need the campaign's persona
    model, but it does need *a* model the backend actually hosts. Asking for an
    id the backend does not serve fails every batch with an opaque 400 — and a
    fill that hangs on the first batch is worse than one that defers.
    ``timeout`` bounds the ``models.list`` call (the shared OpenAI client has no
    timeout override, so an unreachable backend would otherwise sit on its
    default for minutes).
    """
    from bot_core.ai_client import DEFAULT_MODEL
    from config.settings import (
        CONTEXTUAL_FILL_ALLOW_MODEL_FALLBACK,
        CONTEXTUAL_FILL_MODEL,
        CONTEXTUAL_FILL_LLM_TIMEOUT_S,
    )

    want = CONTEXTUAL_FILL_MODEL or DEFAULT_MODEL or ""
    if not want:
        return True, None  # nothing configured — let the backend decide
    from bot_core.ai_client import _backend_model_ids
    ids: list[str] = []
    try:
        # Cache-backed (300 s TTL, shared with every /ai request) and bounded: the
        # OpenAI client sets no timeout of its own, so an unreachable backend
        # would otherwise sit on its library default for minutes.
        avail = await asyncio.wait_for(
            _backend_model_ids(client), timeout=timeout or CONTEXTUAL_FILL_LLM_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001 — reachability, not correctness
        log.warning(
            "Contextual fill: could not list backend models (%s: %s) — proceeding with '%s'",
            type(exc).__name__, exc, want,
        )
        return True, None
    if avail is None:
        # The listing FAILED (endpoint unsupported, stub client, network error):
        # we know nothing, so we must not defer a round on guesswork.
        return True, None
    ids = sorted(avail)
    if not ids:
        return True, None
    if want in ids:
        return True, None
    if CONTEXTUAL_FILL_MODEL:
        # The operator pinned a model that the backend does not host — that is a
        # config error, not something to paper over with an arbitrary pick.
        _defer_log(
            "Contextual fill: CONTEXTUAL_FILL_MODEL '%s' is not hosted by this "
            "backend (serving: %s) — round deferred.", want, ", ".join(ids[:5]),
        )
        return False, None
    if not CONTEXTUAL_FILL_ALLOW_MODEL_FALLBACK:
        _defer_log(
            "Contextual fill: DEFAULT_MODEL '%s' is not hosted by this backend "
            "(serving: %s) — round deferred. Point MODEL_NAME at a hosted model, "
            "set CONTEXTUAL_FILL_MODEL, or CONTEXTUAL_FILL_ALLOW_MODEL_FALLBACK=1.",
            want, ", ".join(ids[:5]),
        )
        return False, None
    # Fallback: a real hosted id beats a guessed one. Prefer the shortest name —
    # long ids are usually quantization variants of the same deployment.
    return True, ids[0]


async def generate_missing_context_lines(
    index: Any,
    *,
    batch_size: int = 12,
    batch_pause: float = 60.0,
    max_chars: int = 1500,
    should_stop: Callable[[], bool] | None = None,
    reembed: bool = True,
    timeout: float | None = None,
    model: str | None = None,
) -> int:
    """Fill missing context lines for *index*'s chunks; return lines generated.

    *index* is a ``KBIndex`` (duck-typed to avoid an import cycle at module
    load). Chunks are grouped by source file and batched; each batch is one
    chat completion on the bot's default model. After every successful batch
    the sidecar is saved through ``index.save_context_lines`` and reloaded,
    so live retrieval sees new lines immediately. A batch that fails three
    times is skipped (resumable on a later run) — generation errors are data,
    never fatal to the bot.

    #77 — ``reembed`` (default ON): every batch's freshly-lined chunks are handed
    to ``index.reembed_contextualized(hashes)`` so their VECTORS are rebuilt from
    ``context_line + content`` too. Without that step a chunk gains BM25/prompt
    framing only, while its dense vector still describes the raw text. A re-embed
    failure never aborts the fill (the lines are useful on their own); failed
    hashes are retried once after the last batch, and any that remain are named
    in a warning pointing at the repair flag.

    #80 — ``timeout`` bounds every chat call (the shared OpenAI client sets no
    timeout of its own, so an unreachable backend would otherwise block the fill
    on its library default). ``model`` lets a caller that already resolved a
    hosted model (:func:`resolve_fill_model`) pass it in instead of re-listing.
    """
    from bot_core.ai_client import _validate_model, complete_text
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

    client = make_fill_client()
    if model is None:
        ok, resolved = await resolve_fill_model(client)
        if not ok:
            return 0
        model = resolved or DEFAULT_MODEL
    model = await _validate_model(client, model)
    generated = 0
    # #77: per-batch re-embed of exactly the chunks just contextualized. Looked
    # up dynamically so duck-typed stores without the method (tests, third-party
    # callers) keep working — they just skip the vector refresh.
    reembed_fn = getattr(index, "reembed_contextualized", None) if reembed else None
    if reembed and reembed_fn is None:
        log.debug("Contextual fill: store has no reembed_contextualized(); vectors left unchanged")
    reembedded = 0
    pending_reembed: set[str] = set()

    async def _reembed(hashes: set[str]) -> int:
        """Call the store's targeted re-embed (sync or async); never raises."""
        try:
            result = reembed_fn(hashes)
            if inspect.isawaitable(result):
                result = await result
            return int(result or 0)
        except Exception as exc:  # noqa: BLE001 — vectors lag, the fill continues
            log.warning("Contextual fill re-embed failed for %d chunk(s): %r", len(hashes), exc)
            return 0

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
                    timeout=timeout,
                )
                lines = parse_json_array(reply, expected=len(batch))
                if lines is not None and len(lines) == len(batch):
                    batch_hashes: set[str] = set()
                    for c, line in zip(batch, lines):
                        if line:
                            h = _content_hash_of(c)
                            index._context_lines[h] = line[:300]
                            c.context_line = line[:300]
                            batch_hashes.add(h)
                    # Plain single-arg save: stamp-aware stores carry their own
                    # re-embed stamp through the write (KBIndexStore does), and
                    # duck-typed stores (tests, embedder-free runners) stay valid
                    # call sites. Generation never stamps — only a successful
                    # re-embed does (stamp_context_vectors).
                    index.save_context_lines(index._context_lines)
                    index.reload_context_lines()
                    if reembed_fn is not None and batch_hashes:
                        n_re = await _reembed(batch_hashes)
                        reembedded += n_re
                        if n_re < len(batch_hashes):
                            # Retry these once after the loop (downed embedder,
                            # GPU flipped to image mode mid-fill, …).
                            pending_reembed |= batch_hashes
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
    if reembed_fn is not None and pending_reembed:
        n_retry = await _reembed(set(pending_reembed))
        reembedded += n_retry
        if n_retry < len(pending_reembed):
            log.warning(
                "Contextual fill: %d of %d chunk(s) still carry pre-context vectors — "
                "repair with scripts/contextual_reindex.py --reembed-contextualized",
                len(pending_reembed) - n_retry, len(pending_reembed),
            )
    if reembed_fn is not None:
        log.info("Contextual fill: %d chunk(s) re-embedded with their new context line.", reembedded)
    return generated


def _content_hash_of(doc: Any) -> str:
    """Content hash of a chunk's ORIGINAL text — the sidecar key."""
    from kb.vector_db import _content_hash
    return _content_hash(doc.content)
