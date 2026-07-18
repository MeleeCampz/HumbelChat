"""Unified KB document retriever — keyword or vector similarity strategies.

Provides a single entry point ``retrieve_kb_documents()`` that bot_core.py
calls for RAG context injection.  The active strategy is controlled by the
environment variable **RAG_RETRIEVAL_METHOD** (default: ``keyword``).

Available strategies
--------------------
keyword — heuristic scoring of filenames, headers, and body overlap
          (existing engine in kb.reader).

vector  — cosine-similarity embedding search via the OpenWebUI backend
          using model ``nomic-embed-text:latest``.  Documents are chunked
          semantically before embedding and indexed on first use with
          SQLite persistence for fast bot restarts.

Usage
-----
    from kb.retrievers import retrieve_kb_documents, KB_STRATEGIES

    results = retrieve_kb_documents(
        query="Tell me about the unique time system in humblewood.",
        kb_path="/path/to/knowledge",
        strategy="vector",       # or "keyword" (default)
        top_n=5,
    )

Both strategies return ``list[tuple[str, str]]`` of ``(display_name, content)``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import Optional

logger = logging.getLogger("kb.retrievers")

# ───────────────────────────── Constants ──────────────────────────────

KB_STRATEGIES = frozenset({"keyword", "vector"})
DEFAULT_METHOD = os.getenv("RAG_RETRIEVAL_METHOD", "keyword").lower()

# Singleton index store — lazily initialized
_index_store: Optional["kb.index.KBIndexStore"] = None
_kb_path_for_store: str | pathlib.Path | None = None


async def _ensure_index_store(kb_path: str | pathlib.Path) -> Optional["kb.index.KBIndexStore"]:
    """Create or return the cached index store, loading/building the index."""
    global _index_store, _kb_path_for_store

    if _index_store is not None:
        return _index_store  # already built

    if _kb_path_for_store == kb_path and os.path.exists(str(kb_path)):
        return _index_store  # same path, reuse

    from kb.index import KBIndexStore

    store = KBIndexStore(kb_path)
    await store.load()
    _index_store = store
    _kb_path_for_store = kb_path
    logger.info("Vector index store ready (%d chunks)", store.get_index().count() if store.get_index() else 0)
    return _index_store


# ───────────────────────────── Strategies ─────────────────────────────

def _retrieve_keyword(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Keyword-based retrieval using the existing heuristic engine.

    Phase 1: Score all files against *query* terms using a lightweight scan
             (first 300 lines is sufficient for relevance ranking).
    Phase 2: Extract only relevant line-windows from the top-N documents
             via get_relevant_chunks(), avoiding full-file dump in context.
    """
    from kb.reader import read_kb_files, get_relevant_chunks

    # Phase 1 — quick scoring pass (300 lines is plenty for keyword overlap)
    scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300)
    if not scored:
        return []

    # Phase 2 — extract only matched windows from top documents
    doc_names = [name for name, _ in scored[:top_n]]
    chunks = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=5)

    logger.info(
        "Keyword retrieval: %d files ranked → %d relevant chunk(s) with ~%.0f chars",
        len(scored), len(chunks),
        sum(len(c) for _, c in chunks) if chunks else 0,
    )
    return chunks


async def _retrieve_vector(
    query: str,
    kb_path: str | pathlib.Path,
    top_n: int,
) -> list[tuple[str, str]]:
    """Hybrid vector+keyword retrieval using the persist-backed index.

    Falls back to keyword-only if the vector index is unavailable.
    Combines cosine-similarity of chunks with filename/body keyword boosting
    so that files whose names or content match query terms strongly (e.g. a
    file literally named ``Humblewood_Calendar.md`` when searching for the
    "humblewood time system") are not drowned out by general-topic matches.
    """
    try:
        from kb.reader import read_kb_files  # type: ignore[import]
    except ImportError:
        read_kb_files = None  # pragma: nocover

    store = await _ensure_index_store(kb_path)
    if store is None:
        logger.warning("Vector index store unavailable for '%s'; falling back to keyword", kb_path)
        from kb.reader import get_relevant_chunks
        scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300) if read_kb_files else []
        doc_names = [name for name, _ in scored[:top_n]] if scored else []
        chunks = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=5)
        logger.info(
            "Vector→keyword fallback: %d files ranked → %d relevant chunk(s) with ~%.0f chars",
            len(scored), len(chunks),
            sum(len(c) for _, c in chunks) if chunks else 0,
        )
        return chunks

    idx = store.get_index()
    if idx is None or idx.is_empty():
        logger.warning("Vector index is empty for '%s'; falling back to keyword", kb_path)
        from kb.reader import get_relevant_chunks
        scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300) if read_kb_files else []
        doc_names = [name for name, _ in scored[:top_n]] if scored else []
        chunks = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=5)
        logger.info(
            "Vector→keyword fallback (empty index): %d files ranked → %d relevant chunk(s) with ~%.0f chars",
            len(scored), len(chunks),
            sum(len(c) for _, c in chunks) if chunks else 0,
        )
        return chunks

    # ── Query expansion (uses configured LLM backend) ────────────────────
    try:
        from kb.query_rewriter import create_query_rewriter
        rewriter = create_query_rewriter()
        expanded_queries = await rewriter.expand(query)
    except Exception:
        expanded_queries = [query]
        logger.debug("Query expansion unavailable; using original query only")

    # ── Vector search on all expanded queries ────────────────────────────
    import math

    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # Load all docs from the vector index to score against expanded queries
    conn_path = store._db_path  # type: ignore[attr-defined]
    import sqlite3, pickle, json
    if not conn_path.exists():
        # Index is in-memory only — use vector-ranked file names but extract chunks only
        ranked_names = await idx.query(query, top_n=top_n * 20)
        if ranked_names:
            name_set: set[str] = set()
            for name, _ in ranked_names:
                name_set.add(name)
                if " [" in name:
                    name_set.add(name.split(" [")[0])
            # Extract base file names from chunk keys
            doc_stems = sorted({n.split(" [")[0] if " [" in n else n for n in name_set})
            from kb.reader import get_relevant_chunks
            ranked_list = get_relevant_chunks(kb_path, doc_stems[:top_n], query=query, window_lines=5)
        else:
            # No ranked names → full fallback to keyword
            from kb.reader import get_relevant_chunks
            scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300) if read_kb_files else []
            doc_names = [name for name, _ in scored[:top_n]] if scored else []
            ranked_list = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=5)
        logger.info(
            "Vector→keyword fallback (in-memory DB): %d relevant chunk(s) with ~%.0f chars",
            len(ranked_list),
            sum(len(c) for _, c in ranked_list) if ranked_list else 0,
        )
        return ranked_list

    conn = sqlite3.connect(str(conn_path))
    cursor = conn.execute("SELECT doc_name, content, embedding FROM document_index")
    all_chunks: list[tuple[str, str, bytes | str]] = cursor.fetchall()
    conn.close()

    # Embed all expanded queries and collect scores per chunk
    try:
        from kb.embedder_openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(model_name="nomic-embed-text:latest")
        embeddings = await embedder.encode(expanded_queries)
    except Exception as exc:
        logger.warning("Query embedding failed (%s); falling back to keyword", exc)
        from kb.reader import get_relevant_chunks
        scored = read_kb_files(kb_path, query=query, top_n=top_n * 3, max_lines_per_file=300) if read_kb_files else []
        doc_names = [name for name, _ in scored[:top_n]] if scored else []
        ranked_list = get_relevant_chunks(kb_path, doc_names, query=query, window_lines=5)
        logger.info(
            "Vector→keyword fallback (embedding error): %d files ranked → %d relevant chunk(s) with ~%.0f chars",
            len(scored), len(ranked_list),
            sum(len(c) for _, c in ranked_list) if ranked_list else 0,
        )
        return ranked_list

    # Score each chunk: max cosine similarity across expanded queries (clipped to [0,1])
    def _safe_cosine(q_emb: list[float], doc_emb: bytes | str) -> float:
        if isinstance(doc_emb, bytes):
            try:
                emb_data = pickle.loads(doc_emb)
            except Exception:
                emb_data = json.loads(doc_emb)
        else:
            emb_data = doc_emb
        sim = _cosine_sim(q_emb, emb_data)
        return max(0.0, min(1.0, sim))

    chunk_scores: dict[str, float] = {}  # doc_name -> best (clipped) similarity
    for i, q_emb in enumerate(embeddings):
        for name, content, emb_blob in all_chunks:
            raw_sim = _cosine_sim(q_emb, emb_blob)
            clipped = max(0.0, min(1.0, raw_sim))
            if clipped > chunk_scores.get(name, 0):
                chunk_scores[name] = clipped

    # ── Keyword/filename boost ───────────────────────────────────────────
    from kb.reader import _normalize_query as _norm_terms

    query_terms = _norm_terms(query)

    def _filename_boost(display_name: str) -> int:
        """Return a keyword score for filename / header overlap.

        Exact filename match (stem match, case-insensitive) gets +100;
        partial term-in-name match gets +15 per term.
        """
        # Check exact stem match first (strongest signal)
        stem_match = any(
            t.lower() == display_name.lower().replace('.md', '').replace('.txt', '')
            or f"{t}.md" in display_name.lower()
            or f"{t}.txt" in display_name.lower()
            for t in query_terms
        )
        if stem_match:
            return 100
        # Partial match per term
        return sum(15 for t in query_terms if t.lower() in display_name.lower())

    def _header_body_boost(content: str) -> int:
        lines = content.splitlines()[:300]
        score = 0
        for line in lines:
            cl = line.strip().lower()
            for term in query_terms:
                t = term.lower()
                # Header detection (short, starts with uppercase, no spaces)
                is_header = len(line.strip()) <= 40 and line.strip() and line.strip()[0].isupper() and " " not in line.strip()
                if is_header and t in cl:
                    score += 10
                else:
                    hits = cl.count(t)
                    hits = min(hits, 4)
                    if hits > 0:
                        score += hits * 2
        return score

    # Build combined hybrid scores for all chunks
    # all_chunks is [(doc_name(str), content(str), embedding(blob|bytes))] per SQLite query
    hybrid_scores: list[tuple[float, str, str]] = []  # (hybrid_score, name, content)
    for doc_name, chunk_content, emb_blob in all_chunks:
        vector_score = chunk_scores.get(doc_name, 0.0)
        fname_boost = _filename_boost(doc_name) / 100.0  # normalise to [0, ~1]
        body_boost = _header_body_boost(chunk_content) / 500.0  # normalise to [0, ~1]
        hybrid = vector_score + fname_boost + body_boost
        hybrid_scores.append((hybrid, doc_name, chunk_content))

    hybrid_scores.sort(key=lambda t: -t[0])

    # ── Deduplicate to unique files, collecting top chunks per file ────────
    # We need doc_stems (file stems) for get_relevant_chunks
    seen_files: set[str] = set()
    doc_stems: list[str] = []
    for hybrid_score, name, content in hybrid_scores:
        if len(doc_stems) >= top_n * 3:  # gather more candidates for chunk extraction
            break
        base_name = name.split(" [")[0] if " [" in name else name
        if base_name not in seen_files:
            seen_files.add(base_name)
            doc_stems.append(base_name)

    # ── Extract relevant chunks from top-ranked unique files ─────────────
    from kb.reader import get_relevant_chunks
    ranked_list = get_relevant_chunks(kb_path, doc_stems, query=query, window_lines=5)

    logger.info(
        "Vector retrieval: %d files scored → %d relevant chunk(s) with ~%.0f chars",
        len(doc_stems), len(ranked_list),
        sum(len(c) for _, c in ranked_list) if ranked_list else 0,
    )
    return ranked_list


# ───────────────────────────── Public API ──────────────────────────────

def retrieve_kb_documents(
    query: str,
    kb_path: str | pathlib.Path,
    *,
    strategy: str = DEFAULT_METHOD,
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """Retrieve relevant KB documents for *query* using the selected strategy.

    Parameters
    ----------
    query : The user's question / prompt used for retrieval.
    kb_path : Path to the knowledge-base root directory.
    strategy : ``"keyword"`` (default) or ``"vector"``.
    top_n : Number of documents to return.

    Returns
    -------
    list of ``(display_name, content)`` tuples in relevance order.

    Notes
    -----
    When *strategy* is ``"vector"``, this function builds the vector index
    on first use (chunking documents, encoding via OpenWebUI).  On subsequent
    calls the cached SQLite index is loaded directly for fast access.
    If embedding fails at any point, retrieval falls back to keyword search.
    """
    method = strategy.lower() if strategy else DEFAULT_METHOD

    if method == "keyword":
        return _retrieve_keyword(query, kb_path, top_n)

    if method == "vector":
        import nest_asyncio  # type: ignore[import]
        nest_asyncio.apply()
        return asyncio.run(_retrieve_vector(query, kb_path, top_n))  # type: ignore[arg-type]

    # Unknown strategy — fall back to keyword with a warning
    logger.warning(
        "Unknown retrieval strategy '%s'; falling back to keyword", method
    )
    return _retrieve_keyword(query, kb_path, top_n)


async def update_kb_document(file_path: str | pathlib.Path) -> bool:
    """Re-index or add a single KB document. Use after ``!add_kb_file``."""
    global _index_store
    if _index_store is None or _index_store.get_index() is None:
        logger.warning("No index to update; full rebuild needed")
        return False
    return await _index_store.update_single_document(file_path)


async def remove_kb_document(file_path: str | pathlib.Path) -> bool:
    """Remove a KB document from the vector index."""
    global _index_store
    if _index_store is None or _index_store.get_index() is None:
        return False
    return await _index_store.remove_document(file_path)


async def shutdown_vector_store() -> None:
    """Persist index before bot shutdown."""
    global _index_store
    if _index_store is not None:
        await _index_store.shutdown()
        logger.info("Vector index store shut down and persisted")


def get_available_strategies() -> list[str]:
    """Return the list of available retrieval strategies."""
    from config.settings import settings  # type: ignore[attr-defined]

    has_vector = bool(settings.INFER_URL and settings.INFER_API_KEY)
    strategies = ["keyword"]
    if has_vector:
        strategies.append("vector")
    return strategies


def is_vector_available() -> bool:
    """Quick check whether the vector retrieval backend is configured."""
    from config.settings import settings  # type: ignore[attr-defined]

    return bool(settings.INFER_URL and settings.INFER_API_KEY)
