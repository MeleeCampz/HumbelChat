#!/usr/bin/env python3
"""Tier 2 (#39 item 7): generate Contextual Retrieval lines for every KB chunk.

One-time offline LLM job (research doc §2.2): each chunk gets a one-line,
chunk-specific context description that is PREPENDED to the text both when it
is embedded and when it is attached to a prompt. Anthropic's study measured
−49% retrieval failure from this (−67% with rerank); our decontextualized
table rows and undated session-log excerpts are the ideal targets.

Pipeline:
  1. Read every chunk (content + hash) from the live vector DB.
  2. Batch chunks by file (default 12/call) and ask the chat model for a JSON
     array of one-line context descriptions, keyed to the chunk order.
  3. Persist ``<persist_dir>/context_lines.json`` (content_hash → line) after
     every batch — resumable: hashes already present are skipped.
  4. Optionally re-embed the whole index with the contextual passages
     (``--rebuild``): forces a full rebuild via the normal atomic-swap path,
     so an interrupted run never leaves the bot without its old cache.

Safe by default: without the sidecar file every code path behaves exactly as
before (passage_text() == content). Run with --dry-run to see the plan first.

Usage (from the repo root, real backends):
  EMBED_URL=http://host.docker.internal:11435/v1 \
  .venv/bin/python scripts/contextual_reindex.py --batch-size 12
  # then, once the sidecar looks good:
  .venv/bin/python scripts/contextual_reindex.py --rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sqlite3
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_KB_PATH = REPO_ROOT / "data" / "knowledge"
DEFAULT_PERSIST_DIR = DEFAULT_KB_PATH / ".vector_index_cache"

# Prompt for one batch of chunks from the same file.
_BATCH_PROMPT = """You are building a search index for a D&D knowledge base. For each chunk below, write ONE line of context (max ~140 characters) that a retrieval system can match against user questions. The line must state what the chunk is about INDEPENDENTLY of its own text: include the file/section it comes from, and any key proper names, numbers, or dates it contains. For campaign session-log chunks, say which session it is (ordinal + date + title when visible in the path or headings) and what happens in this excerpt. Do not quote the chunk verbatim; do not add commentary about your task.

File: {file_path}
{chunks_block}

Respond with a JSON array of exactly {n} strings, in the same order as the chunks."""


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


def _load_sidecar(persist_dir: pathlib.Path) -> dict[str, str]:
    path = persist_dir / "context_lines.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"existing sidecar unreadable ({path}): {exc}")


def _save_sidecar(persist_dir: pathlib.Path, lines: dict[str, str]) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = persist_dir / "context_lines.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lines, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(path)  # atomic: readers never see a partial file


async def _chat(client, base_url: str, model: str, prompt: str, max_tokens: int = 2048) -> str:
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


def _parse_json_array(text: str) -> list[str] | None:
    """Extract the JSON array from a model reply (tolerates prose fences)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data]
    except json.JSONDecodeError:
        pass
    return None


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
) -> int:
    import httpx

    todo = [c for c in chunks if c["hash"] not in sidecar]
    if limit is not None:
        todo = todo[:limit]
    print(f"chunks total={len(chunks)}  already contextualized={len(chunks) - len([c for c in chunks if c['hash'] not in sidecar])}  to do={len(todo)}")

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
            prompt = _BATCH_PROMPT.format(file_path=file_path, chunks_block=chunks_block, n=len(batch))
            ok = False
            for attempt in range(3):
                try:
                    reply = await _chat(client, base_url, model, prompt)
                    lines = _parse_json_array(reply)
                    if lines is not None and len(lines) == len(batch):
                        for c, line in zip(batch, lines):
                            if line:
                                sidecar[c["hash"]] = line[:300]
                        _save_sidecar(persist_dir, sidecar)
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
    return failed


async def _rebuild(persist_dir: pathlib.Path) -> None:
    from kb.index import KBIndexStore

    manager = KBIndexStore(DEFAULT_KB_PATH, persist_dir=persist_dir)
    n_ctx = manager.reload_context_lines()
    print(f"sidecar loaded: {len(n_ctx)} context line(s); forcing full re-embed ...")
    idx = await manager.rebuild(force_rebuild=True)
    print(f"index rebuilt: {idx.count()} chunk(s) → {manager.db_path}")
    await manager.shutdown()


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
    ap.add_argument("--max-chars", type=int, default=400, help="chunk preview length per prompt")
    ap.add_argument("--limit", type=int, default=None, help="only process this many chunks (testing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="after generation: force a full re-embed with contextual passages")
    args = ap.parse_args()

    persist_dir = pathlib.Path(args.persist_dir)
    chunks = _load_chunks(persist_dir / "vector_index.db")
    sidecar = _load_sidecar(persist_dir)
    failed = asyncio.run(_generate(
        chunks, sidecar, persist_dir, args.model, args.base_url.rstrip("/"),
        INFER_API_KEY, args.batch_size, args.max_chars, args.limit, args.dry_run,
    ))
    if args.rebuild and not args.dry_run:
        asyncio.run(_rebuild(persist_dir))
    if failed and not args.rebuild:
        print(f"done with {failed} uncontextualized chunk(s) — re-run to resume.")


if __name__ == "__main__":
    main()
