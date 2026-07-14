"""Relevance scoring (TF-IDF) for KB chunk retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher


def tokenize(text: str) -> list[str]:
    """Tokenize and filter stopwords — keeps hyphens for D&D terms like 'Humblefolk'."""
    words = re.findall(r"[a-z0-9]+(?:-\w+)*", text.lower())
    return [w for w in words if w not in _STOPWORDS]


_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "where", "when", "how", "if", "not",
})


def _lower_only(text: str) -> str:
    return text.lower()  # preserve hyphens for compound names 


class ChunkIndex:
    """Lightweight per-file TF-IDF index.

    Each call produces `{ 'offset', 'tokens': [...], 'tfidf': {term: score} }` lists.
    Used by the filesystem reader at query time to rank chunks against a user's prompt.
    """

    @staticmethod
    def from_text(content: str, chunk_size: int = 2000) -> list[dict]:
        raw_chunks = _split_on_heading(content, chunk_size)
        if not raw_chunks:
            return []

        # Build term-doc frequency map across all chunks (sparse)
        term_docs: dict[str, set] = {}
        for idx, chunk in enumerate(raw_chunks):
            tokens = tokenize(chunk)
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    term_docs.setdefault(t, set()).add(idx)
                    seen.add(t)

        n = len(raw_chunks)
        result: list[dict] = []
        for idx, chunk in enumerate(raw_chunks):
            tokens = tokenize(chunk)
            tfidf: dict[str, float] = {}
            for t in set(tokens):  # unique terms only (sparse)
                tf = _tf(t, tokens)
                idf = _idf(t, term_docs, n)
                if tf > 0 and idf > 0:
                    tfidf[t] = tf * idf

            prev_offset = sum(len(raw_chunks[i]) + 1 for i in range(idx))
            result.append({
                "offset": prev_offset,
                "tokens": list(tokens),
                "tfidf": tfidf,
            })
        return result


def _tf(term: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return tokens.count(term) / len(tokens)


def _idf(term: str, term_docs: dict[str, set], n_docs: int) -> float:
    df = sum(1 for s in term_docs.values() if term in s)
    return math.log((n_docs + 1) / (df + 1)) + 1


def _split_on_heading(content: str, chunk_size: int = 2000) -> list[str]:
    """Split on heading markers, force-split oversize chunks."""
    parts = re.split(r"^(#{1,6}\s+.+|={3,}|-{3,})$", content, flags=re.MULTILINE)
    chunks: list[str] = []
    buf, blen = [], 0

    for part in parts:
        if not part.strip():
            continue
        need = blen + len(part)
        if need > chunk_size and buf:
            chunks.append("\n".join(buf))
            buf, blen = [part], len(part)
        else:
            buf.append(part)
            blen += len(part)

    if buf:
        chunks.append("\n".join(buf))

    refined: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            refined.append(c)
        else:
            # Force-split oversize by blank lines
            sub = re.split(r"\n\s*\n", c)
            merged, line_buf = [], ""
            for s in sub:
                if len(line_buf) + len(s) > chunk_size and line_buf:
                    merged.append(line_buf)
                    line_buf = s
                else:
                    line_buf += ("\n\n" + s) if line_buf else s
            if line_buf:
                merged.append(line_buf)
            refined.extend(merged)

    return [c for c in refined if c.strip()]


def relevance_score(query: str, chunks: list[dict]) -> list[tuple[int, float]]:
    """Score *query* against indexed chunks and return sorted (index, score) pairs."""
    if not query or not chunks:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores: list[tuple[int, float]] = []
    for idx, chunk in enumerate(chunks):
        seen_terms = set()
        score = 0.0

        # TF-IDF terms get largest weight (most informative)
        for t, tfidf_score in chunk.get("tfidf", {}).items():
            if t in q_tokens and t not in seen_terms:
                score += tfidf_score * 1.5  # boost for term in index
                seen_terms.add(t)

        # Token overlap (catches non-indexed terms like numbers/symbols)
        chunk_tokens = set(chunk.get("tokens", []))
        for t in q_tokens:
            if t in chunk_tokens and t not in seen_terms:
                score += 0.5

        # Sequence similarity bonus — handles typos and word order
        if score > 0:
            sim = SequenceMatcher(None, q_tokens, chunk.get("tokens", [])).ratio()
            score *= (1 + sim * 0.3)  # up to 30% boost

        if score > 0:
            scores.append((idx, score))

    return sorted(scores, key=lambda x: -x[1])
