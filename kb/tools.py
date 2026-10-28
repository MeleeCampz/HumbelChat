"""In-process KB tools for agent mode (#39, research doc §3.2).

Four tools over the EXISTING vector index + reranker — no new services,
no new VRAM:

  search_kb(query, scope)     → top-k hits as {file, section, snippet, score}
                                (hybrid: dense + BM25 lexical leg, RRF-merged)
  read_section(file, section) → the FULL chunk text (small-to-big); a session
                                FOLDER name also works (reads its notes.md —
                                bare names like 'notes.md' are ambiguous)
  list_sections(file | dir)   → heading-path outline (enumeration support)
  list_sessions()             → chronological campaign session roster with
                                ordinals (folders are date-named, not "session N")

Protocol (backend-agnostic — works with any OpenAI-compatible endpoint):
the model emits ONE JSON object per decision turn, either bare or inside a
```json fenced block::

    {"tool": "search_kb", "args": {"query": "plate armor AC", "scope": "rules"}}

``parse_tool_call`` validates it strictly against ``TOOL_SCHEMAS``; a parse
or validation error is returned to the model as a correction hint (one retry
allowed — see bot_core/agent_loop.py). When the model has nothing left to
look up it answers in plain text instead, which terminates the loop.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── Tool schemas (validation + the prompt-facing description) ─────────────

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_kb": {
        "description": "Search the D&D rules + campaign knowledge base "
                       "(hybrid: semantic + exact keyword). Returns up to %d "
                       "hits with file, section path, a short snippet and a "
                       "relevance score (0..1). Use scope 'rules' for 5e "
                       "rulebook content, 'lore' for campaign/session notes, "
                       "or 'all'.",
        "args": {
            "query": ("str", True, "What to search for — write it as a "
                                   "specific English phrase (names, stats)."),
            "scope": ("str", False, "'rules' | 'lore' | 'all' (default all)"),
        },
    },
    "read_section": {
        "description": "Read the FULL text of one KB chunk (a complete "
                       "section, stat block or item entry) instead of a "
                       "snippet. Give the file from a search_kb hit and "
                       "optionally a section path fragment to disambiguate. "
                       "A bare name without extension works too — 'Player "
                       "Characters' reads that sheet's EVERY section in one "
                       "shot (use a section fragment for just some of them). "
                       "A session FOLDER name from list_sessions also works "
                       "as 'file' (reads that session's full notes).",
        "args": {
            "file": ("str", True, "File (path) from a search_kb hit, or a "
                                 "session folder name from list_sessions."),
            "section": ("str", False, "Section path fragment, e.g. 'Armor'."),
        },
    },
    "list_sections": {
        "description": "List the section outline of one KB file, or the "
                       "files + sections under a directory. Use this for "
                       "enumeration questions ('which backgrounds exist?') "
                       "before reading each entry.",
        "args": {
            "file": ("str", False, "One file to outline."),
            "dir": ("str", False, "A directory prefix to list files of."),
        },
    },
    "find_in_kb": {
        "description": "Find EVERY file/section containing an exact string "
                       "(case-insensitive) — a complete presence list, not a "
                       "relevance ranking. Use for enumeration: 'which "
                       "sessions mention X?', 'where does Y appear?' (search_kb "
                       "only returns its top-k and can silently omit matches; "
                       "this one cannot). Then read_section the hits.",
        "args": {
            "term": ("str", True, "Exact string to find, e.g. a place or "
                               "monster name."),
        },
    },
    "list_sessions": {
        "description": "List all campaign sessions in chronological order with "
                       "their ordinal number ('session 1' = oldest), date, "
                       "title and a teaser line. Session folders are DATE-named, "
                       "so for 'session N' / 'the last session' questions call "
                       "this FIRST to map the ordinal to a folder, then read the "
                       "full notes with read_section(file=\"<folder name>\").",
        "args": {},
    },
}

_SCOPE_VALUES = ("rules", "lore", "all")
_RULES_MARKER = "DnDRules"          # path segment marking rulebook content
_SNIPPET_CHARS = 300                # per-hit snippet size (§3.2)
_TOOL_RESULT_CAP = 4000             # hard cap on one tool result (prompt budget)


class ToolCallParseError(ValueError):
    """The model's turn did not contain a valid tool call."""


# ── Parsing + validation ───────────────────────────────────────────────────

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    """Extract and validate ONE tool call from a model turn.

    Returns ``(tool_name, args)``. Raises :class:`ToolCallParseError` with a
    short, model-readable reason when the turn contains no valid call — the
    agent loop feeds that back as a correction hint (single retry).
    """
    calls = parse_tool_calls(text)
    if not calls:
        raise ToolCallParseError("no valid tool call found")
    return calls[0]


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract and validate the tool call(s) of a model turn.

    Accepts either ONE object ``{"tool": …, "args": …}`` or an ARRAY of
    them (a batch of independent lookups — the agent loop executes a batch
    in parallel). Returns the validated ``[(name, args), …]`` list; EMPTY
    when the turn contains no JSON at all (a plain-text answer). Raises
    :class:`ToolCallParseError` with a model-readable reason as soon as a
    candidate LOOKS like a tool call (carries a 'tool' key, or is an array)
    but does not validate — unrelated JSON in prose (no 'tool' key) is
    skipped so a later real call in the same turn can still be found.
    """
    candidates: list[str] = []
    m = _FENCED_JSON_RE.search(text)
    if m:
        candidates.append(m.group(1))
    # Bare value(s): try every '{' start (objects) and '[' start (arrays).
    for i, ch in enumerate(text):
        if ch in "{[":
            candidates.append(text[i:])

    decoder = json.JSONDecoder()
    for cand in candidates:
        try:
            obj, _end = decoder.raw_decode(cand.lstrip())
        except (json.JSONDecodeError, ValueError):
            continue  # not JSON here — try the next start position
        items = obj if isinstance(obj, list) else [obj]
        # A single object WITHOUT a 'tool' key is unrelated JSON in prose —
        # skip it and try the next start position. Anything that LOOKS like
        # a tool call (has 'tool', or is an array) is validated strictly.
        if len(items) == 1 and isinstance(items[0], dict) \
                and "tool" not in items[0]:
            continue
        calls: list[tuple[str, dict[str, Any]]] = []
        for pos, it in enumerate(items):
            if not isinstance(it, dict):
                raise ToolCallParseError("JSON is not an object (or array "
                                         "of objects)")
            name = it.get("tool")
            if not isinstance(name, str) or name not in TOOL_SCHEMAS:
                known = ", ".join(TOOL_SCHEMAS)
                raise ToolCallParseError(
                    f"call {pos + 1}: unknown or missing tool {name!r} — "
                    f"expected one of: {known}")
            args_raw = it.get("args", {})
            if not isinstance(args_raw, dict):
                raise ToolCallParseError(
                    f"call {pos + 1}: 'args' must be a JSON object")
            calls.append(_validate_args(name, args_raw))
        if calls:
            return calls
    return []


def _validate_args(
    name: str, args: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Type-check *args* against the tool schema; coerce/strip strings."""
    spec = TOOL_SCHEMAS[name]["args"]
    clean: dict[str, Any] = {}
    for key, (typ, required, _desc) in spec.items():
        val = args.get(key)
        if val is None:
            if required:
                raise ToolCallParseError(f"missing required arg '{key}'")
            continue
        if typ == "str":
            if not isinstance(val, str):
                raise ToolCallParseError(f"arg '{key}' must be a string")
            val = val.strip()
            if not val:
                if required:
                    raise ToolCallParseError(f"required arg '{key}' is empty")
                continue
        clean[key] = val
    # Extra args are ignored (models love to add "notes": ...).
    return name, clean


def tool_prompt_block() -> str:
    """The system-prompt addendum describing the protocol + tools."""
    from config.settings import RAG_AGENT_SEARCH_TOP_K

    lines = [
        "You have tools to look things up in the knowledge base. On a turn "
        "where you need more information, reply with a JSON tool call (no prose "
        "around it) — either ONE object:",
        '  {"tool": "<name>", "args": {<arguments>}}',
        "or an ARRAY of independent lookups to run in parallel (faster for "
        "enumeration — e.g. reading several sections at once):",
        '  [{"tool": "read_section", "args": {...}}, {"tool": ...}]',
        "",
    ]
    for name, schema in TOOL_SCHEMAS.items():
        args_desc = "; ".join(
            f"{k}{'*' if req else ''}: {desc}"
            for k, (_t, req, desc) in schema["args"].items()
        )
        desc = schema["description"]
        if "%d" in desc:  # only search_kb carries the top-k placeholder
            desc = desc % RAG_AGENT_SEARCH_TOP_K
        lines.append(f"- {name} — {desc}")
        lines.append(f"  args: {args_desc}   (* = required)")
    lines += [
        "",
        "When NOT to call tools: if the knowledge-base context already attached "
        "to this conversation fully answers the question, answer directly from "
        "it — do not call tools to re-verify what is right there. Call tools "
        "only for what the attached context does not cover: other files or "
        "sections, enumeration ('which/all ...', 'list every ...'), exact names "
        "or terms, or which session something happened in.",
        "Batching: if you need several INDEPENDENT lookups (none needs another's "
        "result), put them all in ONE array so they run in parallel. Only "
        "serialize calls that depend on a previous result.",
        "Answering: after at most a few tool turns, answer in plain text (no "
        "JSON) with the findings, citing file/section per key claim. If nothing "
        "you find is relevant, say plainly that it is not in the rules/campaign "
        "notes instead of guessing.",
    ]
    return "\n".join(lines)


# ── Execution (in-process over the existing index) ────────────────────────

def _scope_filter(scope: str | None) -> Callable[[str], bool] | None:
    if not scope or scope == "all":
        return None
    if scope not in _SCOPE_VALUES:
        raise ToolCallParseError(
            f"scope must be one of {', '.join(_SCOPE_VALUES)} (got {scope!r})")
    if scope == "rules":
        return lambda n: _RULES_MARKER in n
    return lambda n: _RULES_MARKER not in n  # lore: campaign + session notes


async def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a validated tool call; returns model-readable text (capped)."""
    if name == "search_kb":
        return await _tool_search_kb(args["query"], args.get("scope"))
    if name == "read_section":
        return await _tool_read_section(args["file"], args.get("section"))
    if name == "list_sections":
        return await _tool_list_sections(args.get("file"), args.get("dir"))
    if name == "list_sessions":
        return await _tool_list_sessions()
    if name == "find_in_kb":
        return await _tool_find_in_kb(args.get("term", ""))
    raise ToolCallParseError(f"unknown tool {name!r}")  # defensive


def _cap(text: str) -> str:
    if len(text) <= _TOOL_RESULT_CAP:
        return text
    return text[:_TOOL_RESULT_CAP] + "\n…[truncated]"


async def _tool_search_kb(query: str, scope: str | None = None) -> str:
    """Dense top-k → rerank (when available) → compact hit list."""
    from config.settings import RAG_AGENT_SEARCH_TOP_K
    from kb.retrievers import _ensure_index_store
    from kb.reranker import rerank, is_available as _rerank_ok

    store = await _ensure_index_store(_kb_path())
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        return "The knowledge base index is unavailable right now."

    keep = _scope_filter(scope)
    top_k = RAG_AGENT_SEARCH_TOP_K
    from config.settings import RAG_DENSE_TOP_K
    from kb.retrievers import _lexical_ranking, reciprocal_rank_fusion

    # Dense and lexical legs are independent — run them CONCURRENTLY (the
    # embedding call and the CPU BM25 pass overlap; saves one full leg of
    # latency per search). A lexical failure degrades to dense-only.
    async def _dense_leg() -> list[tuple[str, str, float]]:
        ranked, _q_emb = await idx.query_with_embeddings(
            query, top_n=RAG_DENSE_TOP_K)
        return [t for t in ranked if keep(t[0])] if keep else list(ranked)

    def _lex_leg() -> list[tuple[str, str, float]]:
        lex = _lexical_ranking(idx, query, max(top_k * 3, 24))
        return [t for t in lex if keep(t[0])] if keep else list(lex)

    dense_res, lex_res = await asyncio.gather(
        _dense_leg(),
        asyncio.to_thread(_lex_leg),
        return_exceptions=True,
    )
    ranked = (list(dense_res) if isinstance(dense_res, list) else [])
    base_lists: list[list[tuple[str, str, float]]] = [ranked] if ranked else []
    if isinstance(lex_res, list):
        if lex_res:
            base_lists.append(lex_res)
    elif lex_res is not None:  # noqa: BLE001 — pure enhancement
        log.warning("search_kb lexical leg failed (%s); dense order only", lex_res)
    merged = (reciprocal_rank_fusion(base_lists) if len(base_lists) > 1
              else (base_lists[0] if base_lists else []))

    # Rerank the head of the list when the cross-encoder is up; on any
    # failure keep the merged order (degrades exactly like the fast path).
    if _rerank_ok() and merged:
        cands = merged[: max(top_k * 3, 24)]
        try:
            scored = await rerank(query, [c for _, c, _ in cands])
            by_content = {c: s for c, s in scored}
            ranked_sorted = sorted(
                cands, key=lambda t: by_content.get(t[1], 0.0), reverse=True)
        except Exception as exc:  # noqa: BLE001 — pure enhancement
            log.warning("search_kb rerank failed (%s); dense order", exc)
            ranked_sorted = cands
    else:
        ranked_sorted = merged

    hits = []
    for name, content, _score in ranked_sorted[:top_k]:
        file_part = name.split(" [")[0] if " [" in name else name
        section = name[len(file_part):].strip(" []") if " [" in name else ""
        snippet = re.sub(r"\s+", " ", content)[:_SNIPPET_CHARS]
        hits.append({"file": file_part, "section": section, "snippet": snippet})

    if not hits:
        return f"No hits for {query!r}" + (f" in scope '{scope}'" if scope else "")

    # Never emit a truncated JSON array — the model must always get valid
    # JSON. If the full hit list exceeds the tool-output cap, drop hits from
    # the TAIL (least relevant) until it fits.
    def _render(hit_list: list[dict[str, Any]]) -> str:
        return f"{len(hit_list)} hit(s):\n" + json.dumps(hit_list, ensure_ascii=False)

    while hits and len(_render(hits)) > _TOOL_RESULT_CAP:
        hits.pop()
    if not hits:
        return (f"No usable hits for {query!r} — results exceeded the tool "
                f"output limit; try a narrower query or scope.")
    return _render(hits)


def _kb_path() -> pathlib.Path:
    from config.settings import KB_PATH
    return pathlib.Path(KB_PATH)


_READ_FILE_CAP = 40000  # full-file reads (see _tool_read_section)


def _stem(name: str) -> str:
    """Case/extension-insensitive file identity: 'Player Characters' ≡
    'humblewood/HumbleWood/Player Characters.md' (bare stem, no extension)."""
    s = (name or "").strip().lower()
    for ext in (".markdown", ".md"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def _file_matches(file_part: str, file: str) -> bool:
    """Match a chunk's file part against the model's file argument.

    Display names are BARE filenames (no directory prefix), but the model
    may pass a path it inferred from content or search context — so also
    accept a basename match (case-insensitive). Fuzzy fallback (#42,
    2026-10-28): bare stems WITHOUT the extension ('Player Characters')
    resolve case/extension-insensitively — EXACT stem equality only, so
    similarly named files never cross-match.
    """
    f = (file or "").strip()
    if not f:
        return False
    if file_part == f or file_part.endswith(f"/{f}"):
        return True
    if pathlib.Path(file_part).name.lower() == pathlib.Path(f).name.lower():
        return True
    arg_stems = {_stem(f), _stem(pathlib.Path(f).name)}
    return _stem(file_part) in arg_stems


def _single_file_on_disk(file_part: str) -> bool:
    """True when exactly ONE file with this bare name exists under the KB root.

    A full-file read is unambiguous only then: several indexed files can
    share a BARE display name (every session folder's notes.md), and their
    chunks are indistinguishable by name alone — those keep the legacy
    capped path instead of one big mixed blob.
    """
    name = pathlib.Path(file_part).name
    if not name:
        return False
    try:
        hits = [p for p in _kb_path().rglob(name) if p.is_file()]
    except OSError:
        return False
    return len(hits) == 1


async def _tool_read_section(file: str, section: str | None = None) -> str:
    """Full chunk text for (file[, section fragment]).

    A bare stem ('Player Characters') resolves to the whole file when its
    name is unique on disk — EVERY section of it in one shot (#42,
    2026-10-28: the old 3-chunk cap returned only the first sections of a
    multi-section sheet, so group-scene rewrites never saw all anchors).
    Bounded by _READ_FILE_CAP (the backend context absorbs it; worst-case
    rulebook files are ~950 KB and must NOT go through whole).
    """
    from kb.retrievers import _ensure_index_store

    store = await _ensure_index_store(_kb_path())
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        return "The knowledge base index is unavailable right now."

    matches: list[tuple[str, str]] = []
    for name, content in idx.iter_chunks():
        file_part = name.split(" [")[0] if " [" in name else name
        if not _file_matches(file_part, file):
            continue
        if section:
            sec = name[len(file_part):].strip(" []") if " [" in name else ""
            if section.lower() not in sec.lower():
                continue
        matches.append((name, content))

    # Full-file read: the argument names ONE file whose bare name is unique
    # on disk → return ALL its (section-filtered) chunks in document order.
    distinct_parts = {n.split(" [")[0] if " [" in n else n
                      for n, _c in matches}
    if (matches and len(distinct_parts) == 1
            and _single_file_on_disk(next(iter(distinct_parts)))):
        body = "\n\n".join(f"### {n}\n{c}" for n, c in matches)
        if len(body) > _READ_FILE_CAP:
            body = (body[:_READ_FILE_CAP]
                    + f"\n…[truncated at {_READ_FILE_CAP} chars — this file "
                      "has more sections; re-read with a 'section' fragment "
                      "for the rest]")
        return body

    # Every session folder's notes is indexed under the SAME bare name
    # ('notes.md'), so a bare-name hit set is ambiguous across sessions —
    # but a session FOLDER name is unique. If the argument names one, read
    # its notes.md straight from disk (unambiguous, complete).
    if len(matches) != 1:
        folder_text = _session_folder_text(file)
        if folder_text is not None:
            return _cap(folder_text)

    if not matches:
        return f"No chunk found for file={file!r}" + (f" section~{section!r}"
                                                      if section else "")
    parts = []
    for name, content in matches[:3]:
        parts.append(f"### {name}\n{content}")
    return _cap("\n\n".join(parts))


def _session_folder_text(folder_name: str) -> str | None:
    """Full notes.md of a session folder, matched by exact folder name.

    Returns None when no ``SessionLogs/<folder_name>/notes.md`` exists under
    the KB root — callers fall back to chunk matching then.
    """
    target = pathlib.Path(folder_name.strip()).name.lower()
    if not target:
        return None
    root = _kb_path()
    for log_dir in root.rglob("SessionLogs"):
        if not log_dir.is_dir():
            continue
        candidate = log_dir / target
        notes = candidate / "notes.md"
        if candidate.is_dir() and notes.is_file():
            text = notes.read_text(encoding="utf-8", errors="replace")
            return f"### {target}/notes.md\n{text}"
    return None


_FIND_MAX_LOCATIONS = 40           # presence-list cap (§3.2 output budget)


async def _tool_find_in_kb(term: str) -> str:
    """Complete case-insensitive presence list for *term* across all chunks.

    Pure CPU substring scan (no embedding, no rerank — milliseconds even on
    the full 5.7k-chunk index), so it is safe to batch with other calls. A
    term can hit many chunks of one section; locations are deduped to
    file+section with a match count, capped at _FIND_MAX_LOCATIONS.
    """
    if not term:
        raise ToolCallParseError("missing required arg 'term'")
    from kb.retrievers import _ensure_index_store

    store = await _ensure_index_store(_kb_path())
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        return "The knowledge base index is unavailable right now."
    needle = term.lower()

    def _scan() -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, content in idx.iter_chunks():
            n = content.lower().count(needle)
            if n:
                counts[name] = counts.get(name, 0) + n
        return counts

    counts = await asyncio.to_thread(_scan)
    if not counts:
        return f"No chunk contains {term!r}."
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"{len(counts)} location(s) contain {term!r} "
             f"({sum(counts.values())} occurrence(s)):"]
    shown = 0
    for name, n in ranked:
        if shown >= _FIND_MAX_LOCATIONS:
            lines.append(f"…and {len(counts) - shown} more location(s) "
                         f"(narrow the term).")
            break
        lines.append(f"- {name}  ×{n}" if n > 1 else f"- {name}")
        shown += 1
    return _cap("\n".join(lines))


@dataclass(frozen=True)
class _SessionEntry:
    """One campaign session folder (see _session_roster_entries)."""

    ordinal: str
    date: str
    title: str
    folder: str
    path: pathlib.Path
    teaser: str


def _session_roster_entries() -> list[_SessionEntry]:
    """Chronological campaign session roster (oldest first).

    Session folders are date-named (``2026-10-20_02_Title``), so 'session N'
    questions need the ordinal mapping this provides: sort by date, number
    from 1 (oldest). Shared between the ``list_sessions`` tool and the
    /session_file command — one source of truth for the ordinals.
    """
    root = _kb_path()
    folders: list[pathlib.Path] = []
    for log_dir in root.rglob("SessionLogs"):
        if not log_dir.is_dir():
            continue
        folders.extend(d for d in sorted(log_dir.iterdir())
                       if d.is_dir() and (d / "notes.md").is_file())

    parsed: list[tuple[tuple[str, int], _SessionEntry]] = []
    for d in folders:
        # Folder names: 'YYYY-MM-DD_NN' or 'YYYY-MM-DD_NN_Title_With_Gaps'
        # (the ordinal is the first underscore group; everything after it —
        # underscores included — is the title).
        m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d+)(?:[_](.*))?$", d.name)
        date = m.group(1) if m else ""
        title = (m.group(3) or "").strip() if m else ""
        sort_key = (date, int(m.group(2)) if m and m.group(2).isdigit() else 0)
        parsed.append((sort_key, _SessionEntry(
            ordinal="", date=date, title=title, folder=d.name, path=d,
            teaser=_first_line_teaser(d / "notes.md"),
        )))
    parsed.sort(key=lambda pair: pair[0])
    return [
        replace(entry, ordinal=str(ordinal))
        for ordinal, (_key, entry) in enumerate(parsed, 1)
    ]


async def _tool_list_sessions() -> str:
    """Chronological campaign session roster with 1-based ordinals.

    Hands the model both the ordinal and the exact folder name for a
    follow-up read_section (see _session_roster_entries).
    """
    entries = _session_roster_entries()
    if not entries:
        return "No session logs found under the knowledge base."

    lines = [f"{len(entries)} campaign session(s), oldest → newest:"]
    for e in entries:
        head = f"Session {e.ordinal}"
        if e.date:
            head += f" · {e.date}"
        if e.title:
            head += f" · {e.title}"
        lines.append(f"{head} — folder `{e.folder}`")
        if e.teaser:
            lines.append(f"    {e.teaser}")
    lines.append("")
    lines.append('To read a session\'s full notes: '
                 'read_section(file="<folder name>").')
    return _cap("\n".join(lines))


_TEASER_SKIP = re.compile(
    r"^(session:|started:|ended:|what happened:)|^\(no notes?\)$",
    re.IGNORECASE)


def _first_line_teaser(notes: pathlib.Path, limit: int = 160) -> str:
    """First MEANINGFUL line of a session notes file (teaser for the roster).

    Skips structural lines by their ORIGINAL form — markdown headers
    ('# Session: …', '## Notes', '## Overview'), the '- Started/Ended:'
    metadata, '(no notes)' placeholders and the 'What happened:' lead-in —
    so the teaser is actual content (or the overview's first bullet).
    """
    try:
        text = notes.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("---"):
            continue
        if stripped.startswith("#"):
            continue  # structural header, not content
        line = re.sub(r"^\s*(\*+|-+)?\s*", "", stripped).strip()
        if not line or _TEASER_SKIP.search(line):
            continue
        return re.sub(r"\s+", " ", line)[:limit]
    return "(no notes)"


async def _tool_list_sections(file: str | None = None, dir: str | None = None) -> str:
    """Heading-path outline of one file, or files+sections under a dir."""
    if not file and not dir:
        raise ToolCallParseError("give 'file' or 'dir'")

    from kb.retrievers import _ensure_index_store
    store = await _ensure_index_store(_kb_path())
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        return "The knowledge base index is unavailable right now."

    by_file: dict[str, list[str]] = {}
    for name, _content in idx.iter_chunks():
        file_part = name.split(" [")[0] if " [" in name else name
        if file and not _file_matches(file_part, file):
            continue
        if dir and dir not in file_part:
            continue
        sec = name[len(file_part):].strip(" []") if " [" in name else "(full document)"
        bucket = by_file.setdefault(file_part, [])
        if sec not in bucket:
            bucket.append(sec)

    if not by_file:
        return f"Nothing found for {file or dir!r}"
    lines: list[str] = []
    for f, secs in sorted(by_file.items()):
        lines.append(f"{f} ({len(secs)} section(s))")
        for s in secs[:60]:
            lines.append(f"  - {s}")
        if len(secs) > 60:
            lines.append(f"  … and {len(secs) - 60} more")
    return _cap("\n".join(lines))
