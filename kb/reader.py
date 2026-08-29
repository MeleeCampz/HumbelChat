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
    """Extract meaningful query terms (>3 alphabetic chars), deduplicated and de-noised.

    Filters out common English stop words that pollute relevance scoring,
    since they appear in nearly every document and drown out real signals.
    """
    _STOP_WORDS = frozenset({
        # pronouns & determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they',
        'it', 'its', 'my', 'your', 'his', 'her', 'our', 'their',
        # auxiliary / common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'shall', 'should', 'may', 'might', 'must', 'can', 'could',
        # prepositions & conjunctions
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'and', 'but', 'or', 'nor', 'if', 'so', 'as', 'than',
        # question / filler words
        'about', 'tell', 'what', 'who', 'which', 'where', 'when',
        'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'some', 'such', 'no', 'not', 'only',
    })
    words = re.findall(r"[a-zA-Z_]{3,}", query.lower())
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen and w not in _STOP_WORDS:
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
            # Also check individual underscore-separated segments so that
            # e.g. "Humble_World_Overview.txt" matches term "humblewood"
            or any(seg == t for seg in name_lower.replace(".", "_").split("_"))
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
    max_lines_per_file: int = 200,  # Increased: weapon tables are often at line 230+
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
        if ext not in {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}:
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
        # Strip leading numeric prefix (e.g., "70_Species_Equipment" -> "Species_Equipment")
        clean_stem = re.sub(r'^\d+', '', stem)
        display_name = f"{clean_stem}{p.suffix}" if clean_stem else base_name

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

    # No query -> legacy alphabetical order
    return raw_files[:top_n]


def get_relevant_chunks(
    kb_path: str | pathlib.Path,
    doc_names: list[str],
    query: str = "",
    window_lines: int = 5,
) -> list[tuple[str, str]]:
    """Extract only the relevant chunks from the top-ranked documents.

    Instead of dumping entire files into context, this scans each document for
    lines matching *query* terms and returns +/-*window_lines* lines around each match.

    Parameters
    ----------
    kb_path : path to the knowledge-base root directory
    doc_names : list of display names (stemmed) returned from relevance scoring,
                e.g. ['Humblewood_Overview', 'Pike_Mastery']
                Extensions (.txt, .md) are accepted and stripped internally.
    query : original query string used for relevance ranking
    window_lines : number of lines above and below each match to include

    Returns
    -------
    list of ``(display_name, chunk_content)`` tuples,
    where *chunk_content* contains only the relevant line windows.
    """
    kb_root = pathlib.Path(kb_path)
    query_terms = _normalize_query(query) if query else []

    if not doc_names:
        return []

    # Map display name -> actual file path on disk (accept both stem and full name)
    stem_to_file: dict[str, pathlib.Path] = {}
    for p in sorted(kb_root.rglob("*")):
        if not p.is_file() or "?" in p.name:
            continue
        ext = _extract_ext(p.name)
        if ext not in {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}:
            continue
        clean_stem = re.sub(r'^\d+', '', p.stem)
        if clean_stem:
            stem_to_file[clean_stem] = p
        base_name = os.path.basename(p.name).rsplit("?", 1)[0]
        if base_name not in {".", ".."}:
            stem_to_file[base_name] = p

    results: list[tuple[str, str]] = []

    for doc_name in doc_names:
        file_path = stem_to_file.get(doc_name)
        if file_path is None:
            # Try stripping common extensions
            stripped = re.sub(r'\.(txt|md)$', '', doc_name, flags=re.IGNORECASE)
            file_path = stem_to_file.get(stripped)
        if file_path is None:
            continue

        content_text = file_path.read_bytes().decode("utf-8", errors="replace")
        all_lines = content_text.splitlines()

        # Step 1: Find every line matching ANY query term, with hit count.
        any_hits: list[tuple[int, int]] = []  # (hit_count, line_idx)
        for line_idx, line in enumerate(all_lines):
            cleaned = line.strip().lower()
            if any(t in cleaned for t in query_terms):
                hc = sum(1 for t in query_terms if t in cleaned)
                any_hits.append((hc, line_idx))

        if not any_hits:
            continue  # No lines matched this file at all

        # Sort by hit count descending (most relevant first), then by line.
        any_hits.sort(key=lambda x: (-x[0], x[1]))

        def _header_depth(line: str) -> int | None:
            """Markdown header depth of *line* (e.g. '## Rogue' → 2), else None."""
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                return None
            depth = len(stripped) - len(stripped.lstrip("#"))
            # Must be a real ATX header: '#' run followed by whitespace/EOL.
            if depth >= len(stripped) or stripped[depth] not in (" ", "\t"):
                return None
            return depth

        # ---- Step 2a: Guarantee each query term gets a dedicated anchor ----
        # Prefer MARKDOWN HEADERS containing the term over plain body lines.
        # A header anchors the actual section (e.g. "## Rogue"), so its window
        # covers the section's tables; a body line with the same hit count might
        # sit deep in a subclass paragraph and miss them.  Among matching
        # headers, prefer the SHALLOWEST (top-most level) one: it anchors the
        # whole section tree (class features + subclasses), which is what a
        # "what are the X class features" question needs.
        #
        # Two guards keep generic terms from hijacking anchors:
        #  * Depth-1 headers (e.g. "# Classes") are skipped — anchoring on a
        #    file's root heading would pull in unrelated sibling sections.
        #  * Terms present on >15% of the file's lines are treated as noise
        #    ("class", "table", "level"...) and get no guaranteed anchor.
        guaranteed_anchors: list[int] = []
        n_lines = max(len(all_lines), 1)

        for target_term in query_terms:
            term_line_count = sum(1 for li, _hc in any_hits if target_term in all_lines[li].strip().lower())
            if term_line_count > 0.15 * n_lines:
                continue  # too common to be a useful anchor
            header_li: int | None = None
            best_header_depth: int | None = None
            best_body_li = None
            best_hc = -1
            for hc, li in any_hits:
                cleaned = all_lines[li].strip().lower()
                if target_term not in cleaned:
                    continue
                depth = _header_depth(all_lines[li])
                if depth is not None and depth >= 2:
                    if header_li is None or depth < best_header_depth:
                        header_li = li
                        best_header_depth = depth
                elif hc > best_hc:
                    best_body_li = li
                    best_hc = hc
            best_li = header_li if header_li is not None else best_body_li
            if best_li is None:
                continue
            # Only add if not too close to an existing guaranteed anchor
            too_close = any(abs(best_li - a) < (window_lines + 1) for a in guaranteed_anchors)
            if not too_close:
                guaranteed_anchors.append(best_li)

        # ---- Step 2b: Fill remaining anchors from top hits (not yet near guaranteed) ----
        MAX_ANCHORS = max(len(query_terms) + 1, 4)  # At least N+1 anchors for N terms
        final_anchors = list(guaranteed_anchors)

        for hc, li in any_hits:
            if len(final_anchors) >= MAX_ANCHORS:
                break
            # This line must not overlap with any guaranteed anchor's window
            overlaps_guaranteed = any(
                abs(li - a) < (window_lines + 1)
                for a in guaranteed_anchors
            )
            if overlaps_guaranteed:
                continue  # Skip — already covered by guaranteed anchor
            too_close = any(abs(li - a) < (window_lines + 1) for a in final_anchors)
            if not too_close:
                final_anchors.append(li)

        # NOTE: Full-document return was disabled because it defeats RAG_MAX_CHARS budget.
        # A 17KB file would consume ~74% of a 24K char budget, leaving nothing
        # for other relevant documents. Always use targeted line-windows instead.
        
        # Step 4: Build windows around each anchor and merge overlapping ones.
        # A window must never bleed into a *sibling* section.  Without this,
        # an 80-line window on the last table of one class section would pull
        # in the next class's spell tables and the model would mix them
        # together (e.g. inventing spell slots for a non-spellcasting Rogue).
        # Boundaries are defined by header DEPTH: content above the anchor
        # stops at the nearest header of the same or higher level (the top of
        # the anchor's section tree), and content below stops at the next
        # header of the same or higher level.  Sub-sections inside the anchor's
        # own tree (e.g. "#### Level 1: Expertise" under "### Rogue Class
        # Features") are kept — that is where the tables live.
        def _section_top(idx: int) -> int:
            """Index of the nearest header line at or above *idx*."""
            for j in range(idx, -1, -1):
                if _header_depth(all_lines[j]) is not None:
                    return j
            return 0

        def _section_bottom(idx: int) -> int:
            """Index of the line just before the next header at or above the
            same level as the anchor's own section (i.e. end of its tree)."""
            top = _section_top(idx)
            top_depth = _header_depth(all_lines[top]) or 1
            for j in range(idx + 1, len(all_lines)):
                d = _header_depth(all_lines[j])
                if d is not None and d <= top_depth:
                    return j - 1
            return len(all_lines) - 1

        matched_windows: list[tuple[int, int]] = []
        for line_idx in sorted(final_anchors):
            # Anchors that ARE markdown headers get an asymmetric window: a
            # little context above and more below, because the table/feature
            # list usually follows the heading.
            if _header_depth(all_lines[line_idx]) is not None:
                start = max(0, line_idx - min(window_lines // 4, 12))
                end = min(len(all_lines) - 1, line_idx + window_lines * 3)
            else:
                start = max(0, line_idx - window_lines)
                end = min(len(all_lines) - 1, line_idx + window_lines)
            # Clamp to the anchor's own section tree.
            start = max(start, _section_top(line_idx))
            end = min(end, _section_bottom(line_idx))
            matched_windows.append((start, end))

        # Merge overlapping/adjacent pieces so shared lines aren't duplicated
        # in the context (several anchors can cover the same region).
        matched_windows.sort()
        merged: list[tuple[int, int]] = []
        for start, end in matched_windows:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        matched_windows = merged

        if not matched_windows:
            continue

        # Step 5: Build the chunk string.
        chunks: list[str] = []
        for start, end in matched_windows:
            window_text = "\n".join(all_lines[start:end + 1])
            if start > 0 or end < len(all_lines) - 1:
                snippet_header = f"[Lines {start+1}-{end+1} of {doc_name}]"
            else:
                snippet_header = f"[Full content of {doc_name}]"
            chunks.append(f"{snippet_header}\n{window_text}")

        if chunks:
            results.append((f"{doc_name} (relevant chunks)", "\n\n".join(chunks)))

    return results
