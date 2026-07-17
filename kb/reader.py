"""Read chunks from KB files and filter by relevance.

Replaces the in-line filesystem RAG logic that was previously in main.py's
ask_ai_with_model() with a clean, testable module interface.
"""
from __future__ import annotations

import os
import pathlib
import re


# ──────────────────────────── Helpers ────────────────────────────────

def _extract_ext(name: str) -> str:
    """Extract file extension (lowercased), stripping any query-string suffix."""
    base = name.split("?")[0]
    i = base.rfind(".")
    return base[i:].lower() if i > 0 else ""


def _normalize_query(query: str) -> list[str]:
    """Extract meaningful query terms (≥3 alphabetic chars), deduplicated in order."""
    words = re.findall(r"[a-zA-Z_]{3,}", query.lower())
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def _score_file(
    display_name: str,
    content: str,
    query_terms: list[str],
) -> int:
    """Score a KB file for relevance to the given query terms.

    Scoring rules (heuristic, no external libraries needed):
      - File name match (full word):       +15 per term
      - Content title/header line match:   +10 per term per line
      - Content body keyword overlap:        0-8 per term per line (capped at 4 hits)
    """
    score = 0
    name_lower = display_name.lower()

    # --- filename scoring (strongest signal) ---
    for term in query_terms:
        t = term.lower()
        # Check all match patterns for this term
        name_lower_for_t = f" {name_lower} "
        if (
            (f" {t} " in name_lower_for_t)
            or name_lower.startswith(t + "_")
            or name_lower.endswith("_" + t)
            or name_lower == t
        ):
            score += 15

    # --- content scoring (lightweight; scan first 300 lines only) ---
    scan_lines = content.splitlines()[:300]
    for line in scan_lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        cl = cleaned.lower()

        # Detect title/header lines (short + starts with uppercase, no spaces)
        is_header = len(cleaned) <= 40 and cleaned[0].isupper() and " " not in cleaned

        for term in query_terms:
            t = term.lower()
            if is_header and t in cl:
                score += 10
            else:
                hits = cl.count(t)
                hits = min(hits, 4)                       # cap at 4 hits per term
                if hits > 0:
                    score += hits * 2                        # +2 per hit

    return score


# ───────────────────────────── Main API ──────────────────────────────

def read_kb_files(
    kb_path: str | pathlib.Path,
    max_lines_per_file: int = 50,
    max_bytes_per_file: int = 1024 * 1024,   # 1 MB limit
    query: str = "",
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """Read KB files and optionally rank by relevance to *query*.

    Parameters
    ----------
    kb_path : path to the knowledge-base root directory
    max_lines_per_file : cap content lines per file (default 50)
    max_bytes_per_file : cap raw file size in bytes (default 1 MiB)
    query : non-empty string enables relevance ranking via keyword/heading overlap
    top_n : how many files to return after scoring

    Returns
    -------
    list of ``(display_name, truncated_content)`` tuples.

    When *query* is empty the function falls back to alphabetical order (legacy
    behaviour).  When *query* is provided every file is scored and returned in
    descending-score order -- the most contextually relevant documents first.
    """
    kb_root = pathlib.Path(kb_path)
    if not kb_root.exists():
        return []

    raw_files: list[tuple[str, str]] = []
    for p in sorted(kb_root.rglob("*")):
        if not p.is_file() or "?" in p.name:
            continue

        ext = _extract_ext(p.name)
        if ext not in {".txt", ".md"}:
            continue

        content_text = p.read_bytes().decode("utf-8", errors="replace")
        if len(content_text) == 0 or len(content_text) > max_bytes_per_file:
            continue

        lines = content_text.splitlines()[:max_lines_per_file]
        truncated = "\n".join(lines)
        if len(content_text.splitlines()) > max_lines_per_file:
            truncated += "\n... [truncated]"

        base_name = os.path.basename(p.name)
        stem = p.stem
        idx_pos = stem.rfind("_")
        display_name = stem[idx_pos + 1:] if idx_pos > 0 else base_name

        raw_files.append((display_name, truncated))

    # ── Relevance ranking ────────────────────────────────────────────────
    if query:
        query_terms = _normalize_query(query)
        scored: list[tuple[int, str, str]] = []  # (score, name, content)
        for name, content in raw_files:
            score = _score_file(name, content, query_terms) if query_terms else 0
            scored.append((score, name, content))
        scored.sort(key=lambda t: (-t[0], t[1]))  # desc score, alpha tiebreak
        return [(name, content) for _, name, content in scored[:top_n]]

    # No query → legacy alphabetical order
    return raw_files[:top_n]
