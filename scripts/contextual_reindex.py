#!/usr/bin/env python3
r"""Tier 2 (#39 item 7): generate Contextual Retrieval lines for every KB chunk.

One-time offline LLM job (research doc §2.2): each chunk gets a one-line,
chunk-specific context description that is PREPENDED to the text both when it
is embedded and when it is attached to a prompt. Anthropic's study measured
−49% retrieval failure from this (−67% with rerank); our decontextualized
table rows and undated session-log excerpts are the ideal targets.

Pipeline:
  1. Read every chunk (content + hash) from the live vector DB.
  2. Batch chunks by file (default 12/call) and ask the chat model for a JSON
     array of one-line context descriptions, keyed to the chunk order.
  3. Persist ``<persist_dir>/context_lines.json`` (``source_file\0content_hash``
     → line, R5; legacy bare-hash files are migrated in place once) after every
     batch — resumable: keys already present are skipped.
  4. Re-embed the chunks just contextualized (``--reembed-missing``): a chunk that
     got its line AFTER being embedded still carries a vector of the RAW text, so
     the dense leg never sees the framing (#77). Targeted, seconds. The bot's own
     startup fill does generate + re-embed in one process; this script is the
     offline/repair twin.

Safe by default: without the sidecar file every code path behaves exactly as
before (passage_text() == content). Run with --dry-run to see the plan first.

⚠ Anything that writes the vector cache (--reembed-*, --rebuild) must run with
the BOT STOPPED: a live bot holds the pre-context vectors in memory and re-
persists them on its next incremental save or shutdown, overwriting this script's
work. ``docker compose stop bot`` → run → ``docker compose start bot``.
While the script runs it refreshes ``fill_heartbeat`` in the cache dir, so the
bot's own fill loop defers instead of racing it — that protects the sidecar and
the loop's rounds, not the bot's in-memory copy. Stopping the bot is still the
correct thing to do.

Usage (from the repo root, real backends — on Windows use .venv\Scripts\python.exe,
a bare invocation picks up a system Python that has none of this project's deps):
  # read-only state report (safe with the bot running):
  .venv/bin/python scripts/contextual_reindex.py --status
  EMBED_URL=http://host.docker.internal:11435/v1 \
  .venv/bin/python scripts/contextual_reindex.py --batch-size 12 --reembed-missing
  # repair vectors that lag their context line, without generating anything:
  .venv/bin/python scripts/contextual_reindex.py --reembed-only
  # full re-embed of everything (model or schema change):
  .venv/bin/python scripts/contextual_reindex.py --rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sqlite3
import sys
import time
from typing import Callable

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_KB_PATH = REPO_ROOT / "data" / "knowledge"
# Mirrors kb.index.DEFAULT_PERSIST_DIR / CACHE_DIRNAME (kept as a literal here
# because this script sets sys.path before importing kb).*
DEFAULT_PERSIST_DIR = DEFAULT_KB_PATH / ".vector_index_cache"

# Prompt + reply parsing live in kb.contextual_gen (shared with the #52
# background fill) — single source of truth for both generation paths.
from kb.contextual_gen import BATCH_PROMPT as _BATCH_PROMPT  # noqa: E402
from kb.contextual_gen import parse_json_array as _parse_json_array  # noqa: E402
from kb.vector_db import sidecar_key as _sidecar_key  # noqa: E402


def _chunk_key(c: dict) -> str:
    """The sidecar key of a DB chunk row (R5: file-scoped composite)."""
    return _sidecar_key(c["source"], c["content"])


def _load_chunks(db_path: pathlib.Path) -> list[dict]:
    """All chunk rows from the live vector DB (content + hash + names)."""
    if not db_path.exists():
        sys.exit(f"vector DB not found at {db_path} — build the index first")
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(document_index)")}
        if "content_hash" not in cols or "content" not in cols:
            sys.exit("vector DB has a pre-v3 schema — rebuild the index first")
        src_col = "source_file" if "source_file" in cols else "doc_name"
        rows = conn.execute(
            f"SELECT {src_col} AS s, doc_name AS d, content AS c, content_hash AS h "
            "FROM document_index ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"source": s, "doc_name": d, "content": c, "hash": h}
        for s, d, c, h in rows if c and c.strip()
    ]


def _load_sidecar(persist_dir: pathlib.Path) -> tuple[dict[str, str], bool]:
    """Read the sidecar → (entries, legacy_keyed).

    ``legacy_keyed`` is True for a pre-R5 file (entries keyed by bare content
    hash); main() migrates such a file before using it for anything else.
    """
    path = persist_dir / "context_lines.json"
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"existing sidecar unreadable ({path}): {exc}")
    if not isinstance(data, dict):
        return {}, False
    lines = {k: v for k, v in data.items()
             if isinstance(k, str) and isinstance(v, str) and v and not k.startswith("__")}
    legacy = not bool(data.get("__v3__"))
    # A pre-stamp file (no __v__, no __v2__ marker) is migrated here exactly like
    # KBIndexStore does at load: every existing line is recorded as already
    # embedded. If the script instead wrote the marker with an EMPTY stamp, the
    # bot would later treat all those historical lines as un-embedded and re-embed
    # the entire index once "for free".
    if lines and not data.get("__v__") and not data.get("__v2__"):
        _save_sidecar(persist_dir, lines, stamped=set(lines), keyed_v3=False)
        print(f"sidecar: {len(lines)} pre-stamp entr(ies) recorded as already embedded "
              "(stamp format written)")
    return lines, legacy


def _load_stamp(persist_dir: pathlib.Path) -> set[str]:
    """The ``__v__`` re-embed stamp recorded in the sidecar (#80)."""
    path = persist_dir / "context_lines.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    stamp = data.get("__v__") if isinstance(data, dict) else None
    return {h for h in stamp if isinstance(h, str)} if isinstance(stamp, list) else set()


def _migrate_sidecar_v3(
    persist_dir: pathlib.Path, chunks: list[dict], sidecar: dict[str, str]
) -> dict[str, str]:
    """R5: rewrite a bare-hash sidecar as per-file ``source\0hash`` entries.

    Same rules as KBIndexStore._migrate_sidecar_v3: unique-hash lines carry
    over (stamp included), lines shared by several chunks are DROPPED for
    per-file regeneration, entries for gone chunks (orphans) disappear.
    Returns the migrated map (writes the file once).
    """
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["hash"]] = counts.get(c["hash"], 0) + 1
    stamped = _load_stamp(persist_dir)
    new_lines: dict[str, str] = {}
    new_stamp: set[str] = set()
    dropped = 0
    for c in chunks:
        h = c["hash"]
        line = sidecar.get(h)
        if not line:
            continue
        if counts[h] > 1:
            dropped += 1
            continue
        key = _chunk_key(c)
        if key in new_lines:
            continue
        new_lines[key] = line
        if h in stamped:
            new_stamp.add(key)
    # Legacy entries no chunk claims stay dropped (they were orphans anyway).
    orphans = len([h for h in sidecar if h not in counts])
    _save_sidecar(persist_dir, new_lines, stamped=new_stamp)
    print(f"sidecar re-keyed (R5): {len(new_lines)}/{len(sidecar)} entr(ies) moved to "
          f"per-file keys, {dropped} collision-sharing chunk(s) freed for regeneration, "
          f"{orphans} orphan entr(ies) dropped")
    return new_lines


def _save_sidecar(
    persist_dir: pathlib.Path,
    lines: dict[str, str],
    *,
    stamped: set[str] | None = None,
    keyed_v3: bool = True,
) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / "context_lines.json"
    payload: dict[str, object] = dict(lines)
    # "__v2__" marks a stamp-aware file: a file WITHOUT it predates the stamp and the
    # store migrates it (stamp everything once). Without the marker, a file written
    # mid-generation would be migrated too — silently claiming fresh lines were
    # already embedded.
    payload["__v2__"] = 1
    # R5: "__v3__" claims the keys are file-scoped composites — only set it when
    # that is actually true of *lines* (the pre-stamp restamp write passes
    # keyed_v3=False while the entries are still bare hashes).
    if keyed_v3:
        payload["__v3__"] = 1
    if stamped is not None:
        payload["__v__"] = sorted(stamped)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(path)  # atomic: readers never see a partial file


async def _chat(client, base_url: str, model: str, prompt: str, max_tokens: int = 900) -> str:
    """One non-streaming completion (thinking off where the backend allows it)."""
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "reasoning_effort": "none",
    }
    resp = await client.post(url, json=body, timeout=180)
    if resp.status_code in (400, 404) and "reasoning_effort" in resp.text:
        # Backend without the knob — retry plain.
        body.pop("reasoning_effort")
        resp = await client.post(url, json=body, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


async def _generate(
    chunks: list[dict],
    sidecar: dict[str, str],
    persist_dir: pathlib.Path,
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int,
    max_chars: int,
    limit: int | None,
    dry_run: bool,
    priority_substring: str = "",
    path_substring: str = "",
    reference_block: str = "",
    beat: Callable[[], None] | None = None,
) -> tuple[int, set[str]]:
    """Generate missing lines → (failed_chunk_count, generated_hashes)."""
    import httpx

    todo = [c for c in chunks if _chunk_key(c) not in sidecar]
    # Scope filter: only files whose path contains the substring.
    if path_substring:
        todo = [c for c in todo if path_substring in c["source"]]
    # Priority: files whose path contain the substring first (stable within
    # each group) — e.g. "SessionLogs" so campaign-log context lands first.
    if priority_substring:
        todo.sort(key=lambda c: 0 if priority_substring in c["source"] else 1)
    if limit is not None:
        todo = todo[:limit]
    print(f"chunks total={len(chunks)}  already contextualized={len(chunks) - len([c for c in chunks if _chunk_key(c) not in sidecar])}  to do={len(todo)}")

    # Group consecutive chunks by source file so each prompt stays on one file.
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_file: str | None = None
    for c in todo:
        if cur and c["source"] != cur_file:
            batches.append(cur)
            cur, cur_file = [], None
        cur.append(c)
        cur_file = c["source"]
        if len(cur) >= batch_size:
            batches.append(cur)
            cur, cur_file = [], None
    if cur:
        batches.append(cur)

    if dry_run:
        print(f"dry-run: {len(batches)} LLM call(s) of up to {batch_size} chunks; no writes.")
        return 0

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    done = failed = 0
    generated_hashes: set[str] = set()
    t0 = time.time()
    # trust_env=False: bypass the sandbox proxy env entirely — the inference
    # backend is a direct host connection (host.docker.internal), not egress.
    async with httpx.AsyncClient(headers=headers, trust_env=False) as client:
        for bi, batch in enumerate(batches, 1):
            file_path = batch[0]["source"]
            chunks_block = "\n".join(
                f"[{i}] section={c['doc_name'].split(' [', 1)[-1] if ' [' in c['doc_name'] else '(full document)'}\n"
                f"{c['content'][:max_chars]}"
                for i, c in enumerate(batch, 1)
            )
            prompt = (
                (reference_block + "\n")
                if reference_block
                else ""
            ) + _BATCH_PROMPT.format(file_path=file_path, chunks_block=chunks_block, n=len(batch))
            ok = False
            for attempt in range(3):
                try:
                    reply = await _chat(client, base_url, model, prompt)
                    lines = _parse_json_array(reply, expected=len(batch))
                    if lines is not None and len(lines) == len(batch):
                        for c, line in zip(batch, lines):
                            if line:
                                key = _chunk_key(c)
                                sidecar[key] = line[:300]
                                generated_hashes.add(key)
                        # Stamp is carried through untouched — a generated line is
                        # not an embedded vector; _reembed does the stamping.
                        _save_sidecar(persist_dir, sidecar, stamped=_load_stamp(persist_dir))
                        done += len(batch)
                        ok = True
                        break
                    print(f"  batch {bi}: bad reply shape ({len(lines) if lines else 0} vs {len(batch)}) — retrying")
                except Exception as exc:  # noqa: BLE001 — data, never fatal
                    print(f"  batch {bi}: attempt {attempt + 1} failed: {exc!r} — retrying")
                    await asyncio.sleep(3 * (attempt + 1))
            if not ok:
                failed += len(batch)
                print(f"  batch {bi}: giving up on {len(batch)} chunk(s)")
            if bi % 10 == 0 or bi == len(batches):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                print(f"progress: batch {bi}/{len(batches)}  chunks ok={done} failed={failed}  "
                      f"{rate:.1f} chunk/s  sidecar size={len(sidecar)}")
            if beat is not None:
                beat()
    return failed, generated_hashes


async def _rebuild(kb_path: pathlib.Path, persist_dir: pathlib.Path) -> None:
    from kb.index import KBIndexStore

    manager = KBIndexStore(kb_path, persist_dir=persist_dir)
    n_ctx = manager.reload_context_lines()
    print(f"sidecar loaded: {len(n_ctx)} context line(s); forcing full re-embed ...")
    idx = await manager.rebuild()
    print(f"index rebuilt: {idx.count()} chunk(s) → {manager.db_path}")
    await manager.shutdown()


async def _reembed(kb_path: pathlib.Path, persist_dir: pathlib.Path,
                   hashes: set[str] | None) -> None:
    """#77: re-embed chunks whose vector predates their context line.

    *hashes*: chunks to refresh (R5 composite sidecar keys); ``None`` = every
    chunk that HAS a context line (the repair path). Runs through the same
    store code the bot uses: contextual passage → index embedder → incremental
    upsert.
    """
    from kb.index import KBIndexStore

    manager = KBIndexStore(kb_path, persist_dir=persist_dir)
    try:
        n = await manager.reembed_contextualized(hashes)
        scope = ("every un-embedded contextualized chunk" if hashes is None
                 else f"{len(hashes)} chunk(s) generated this run")
        print(f"re-embed ({scope}): {n} chunk(s) now embedded as context + content → {manager.db_path}")
        lag = manager.contextual_vector_lag()
        if lag:
            print(f"still lagging: {lag} chunk(s) — their vectors still describe the raw "
                  "chunk (backend failures). Re-run to resume.")
    except Exception as exc:  # noqa: BLE001 — data job: report, never traceback
        print(f"re-embed failed: {exc!r}")
    finally:
        await manager.shutdown()


def _prune_orphans(persist_dir: pathlib.Path, chunks: list[dict], sidecar: dict[str, str]) -> int:
    """Drop sidecar entries whose chunk no longer exists (tidiness, #77 item 3)."""
    known = {_chunk_key(c) for c in chunks}
    orphans = [h for h in sidecar if h not in known]
    if not orphans:
        print("orphans: none — every sidecar entry matches a live chunk")
        return 0
    for h in orphans:
        sidecar.pop(h)
    # The re-embed stamp is pruned with them — an orphan hash in __v__ is pure
    # cruft and a growing stamp slows every stamp write.
    _save_sidecar(persist_dir, sidecar, stamped={h for h in _load_stamp(persist_dir) if h in known})
    print(f"orphans: removed {len(orphans)} stale sidecar entry(ies); {len(sidecar)} remain")
    return len(orphans)


_HEARTBEAT_HITS = 0


def _beat(persist_dir: pathlib.Path) -> None:
    """Heartbeat for the fill loop (main.py::_contextual_sidecar_worker).

    The bot defers its own fill round while this file is fresh, which is what
    makes "the bot and the script both wrote the cache" a race we do not have.
    Refreshed while generating; removed on exit. A crash leaves a file that ages
    out after 120 s and is ignored (and reported) by the loop.
    """
    global _HEARTBEAT_HITS
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / "fill_heartbeat").write_text(
            f"{time.time()} pid={os.getpid()} hits={_HEARTBEAT_HITS}\n", encoding="utf-8"
        )
        _HEARTBEAT_HITS += 1
    except OSError:
        pass


def main() -> None:
    # settings.py does NOT load .env itself — callers must (see main.py /
    # rag_agent_probe.py). Without this, INFER_URL falls back to the Ollama
    # default and every call in this script hits 127.0.0.1:11434.
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    from config.settings import DEFAULT_MODEL, INFER_API_KEY, INFER_URL

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kb-path", default=str(DEFAULT_KB_PATH))
    ap.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL or "")
    ap.add_argument("--base-url", default=INFER_URL, help="OpenAI-compatible base URL")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=250, help="chunk preview length per prompt")
    ap.add_argument("--priority", default="",
                    help="process files whose path contains this substring first")
    ap.add_argument("--path-substring", default="",
                    help="only process chunks from files whose path contains this substring")
    ap.add_argument("--reference-file", default="",
                    help="text file with reference material appended to every prompt "
                        "(e.g. the campaign session-order table)")
    ap.add_argument("--limit", type=int, default=None, help="only process this many chunks (testing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="force a FULL index rebuild with contextual passages (atomic cache swap)")
    ap.add_argument("--reembed-missing", action="store_true",
                    help="#77: after generation, re-embed exactly the chunks generated in this run")
    ap.add_argument("--reembed-contextualized", action="store_true",
                    help="#77 repair: re-embed EVERY chunk that has a context line (no cache drop)")
    ap.add_argument("--reembed-only", action="store_true",
                    help="skip generation and re-embed every contextualized chunk "
                         "(shorthand for --reembed-contextualized)")
    ap.add_argument("--prune-orphans", action="store_true",
                    help="remove sidecar entries whose chunk no longer exists")
    ap.add_argument("--status", action="store_true",
                    help="read-only report: line coverage, re-embed coverage, orphans")
    ap.add_argument("--stamp-all", action="store_true",
                    help="declare every existing context line as already embedded "
                         "(trust the historical invariant; skips the re-embed pass)")
    args = ap.parse_args()

    persist_dir = pathlib.Path(args.persist_dir)
    kb_path = pathlib.Path(args.kb_path)
    reembed_all = args.reembed_contextualized or args.reembed_only
    chunks = _load_chunks(persist_dir / "vector_index.db")
    sidecar, legacy = _load_sidecar(persist_dir)
    if legacy and sidecar:
        # R5: a pre-v3 file is re-keyed against the live chunk rows before any
        # of the modes below inspect it (idempotent — v3 files skip this).
        sidecar = _migrate_sidecar_v3(persist_dir, chunks, sidecar)

    if args.stamp_all:
        _save_sidecar(persist_dir, sidecar, stamped=set(sidecar))
        print(f"stamp: {len(sidecar)} context line(s) recorded as already embedded — "
              "no vectors touched. New or re-synced chunks will still be refreshed "
              "by the bot's fill loop.")
        return

    if args.status:
        stamped = _load_stamp(persist_dir)
        keyed = {_chunk_key(c): c for c in chunks}
        with_line = [k for k in sidecar if k in keyed]
        lag = [k for k in with_line if k not in stamped]
        print(f"chunks={len(chunks)}  lines={len(with_line)}  missing={len(keyed) - len(with_line)}")
        print(f"vectors built WITH their line={len(with_line) - len(lag)}  "
              f"lagging={len(lag)}  "
              f"(stamp: {'present' if stamped else 'written before the stamp existed'})")
        print(f"orphan sidecar entries={sum(1 for k in sidecar if k not in keyed)}")
        if lag:
            print("next: the bot's fill loop repairs these on its next round, or run "
                  "--reembed-contextualized with the bot stopped (it rewrites the cache "
                  "in place); --stamp-all declares them embedded without touching vectors.")
        return

    if args.dry_run:
        if not args.reembed_only:
            asyncio.run(_generate(
                chunks, sidecar, persist_dir, args.model, args.base_url.rstrip("/"),
                INFER_API_KEY, args.batch_size, args.max_chars, args.limit, True,
                args.priority,
                args.path_substring,
                open(args.reference_file, encoding="utf-8").read() if args.reference_file else "",
            ))
        if reembed_all:
            print(f"dry-run: would re-embed "
                  f"{sum(1 for c in chunks if _chunk_key(c) in sidecar)} contextualized chunk(s).")
        if args.prune_orphans:
            known = {_chunk_key(c) for c in chunks}
            print(f"dry-run: would prune {sum(1 for h in sidecar if h not in known)} "
                  f"orphan sidecar entry(ies).")
        return

    if args.reembed_missing or reembed_all or args.rebuild or args.prune_orphans:
        print("WARNING: writing the live vector cache/sidecar. Stop the bot first "
              "(a running bot keeps the pre-context vectors in memory and re-persists "
              "them on its next save/shutdown, overwriting this run).")
    # Heartbeat while we work: the bot's fill loop defers while this is fresh.
    _beat(persist_dir)
    try:
        if args.reembed_only:
            failed, generated = 0, set()
        else:
            failed, generated = asyncio.run(_generate(
                chunks, sidecar, persist_dir, args.model, args.base_url.rstrip("/"),
                INFER_API_KEY, args.batch_size, args.max_chars, args.limit, False,
                args.priority,
                args.path_substring,
                open(args.reference_file, encoding="utf-8").read() if args.reference_file else "",
                beat=lambda: _beat(persist_dir),
            ))

        if args.prune_orphans:
            _prune_orphans(persist_dir, chunks, sidecar)
        if args.rebuild:
            asyncio.run(_rebuild(kb_path, persist_dir))
        elif reembed_all:
            asyncio.run(_reembed(kb_path, persist_dir, None))
        elif args.reembed_missing:
            if generated:
                asyncio.run(_reembed(kb_path, persist_dir, generated))
            else:
                print("--reembed-missing: nothing was generated this run — nothing to re-embed. "
                      "To refresh context lines that already exist: --reembed-contextualized.")
    finally:
        try:
            (persist_dir / "fill_heartbeat").unlink()
        except OSError:
            pass

    if failed and not args.rebuild:
        print(f"done with {failed} uncontextualized chunk(s) — re-run to resume.")


if __name__ == "__main__":
    main()
