"""In-process KB tools for agent mode (#39, research doc §3.2).

Three tools over the EXISTING vector index + reranker — no new services,
no new VRAM:

  search_kb(query, scope)     → top-k hits as {file, section, snippet, score}
  read_section(file, section) → the FULL chunk text (small-to-big)
  list_sections(file | dir)   → heading-path outline (enumeration support)

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

import json
import pathlib
import logging
import re
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── Tool schemas (validation + the prompt-facing description) ─────────────

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_kb": {
        "description": "Search the D&D rules + campaign knowledge base. "
                       "Returns up to %d hits with file, section path, a short "
                       "snippet and a relevance score (0..1). Use scope "
                       "'rules' for 5e rulebook content, 'lore' for "
                       "campaign/session notes, or 'all'.",
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
                       "optionally a section path fragment to disambiguate.",
        "args": {
            "file": ("str", True, "File (path) from a search_kb hit."),
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
    candidates: list[str] = []
    m = _FENCED_JSON_RE.search(text)
    if m:
        candidates.append(m.group(1))
    # Bare object(s): try every '{' start, keep the first that parses AND
    # validates (models sometimes wrap the call in prose).
    for i, ch in enumerate(text):
        if ch == "{":
            candidates.append(text[i:])

    last_err = "no JSON object found"
    for cand in candidates:
        try:
            obj, _end = json.JSONDecoder().raw_decode(cand.lstrip())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            last_err = "JSON is not an object"
            continue
        name = obj.get("tool")
        if not isinstance(name, str) or name not in TOOL_SCHEMAS:
            known = ", ".join(TOOL_SCHEMAS)
            last_err = (f"unknown or missing tool {name!r} — expected one "
                        f"of: {known}")
            continue
        args_raw = obj.get("args", {})
        if not isinstance(args_raw, dict):
            last_err = "'args' must be a JSON object"
            continue
        name, args = _validate_args(name, args_raw)
        return name, args
    raise ToolCallParseError(last_err)


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
        "where you need more information, reply with EXACTLY ONE JSON object "
        "(no prose around it):",
        '  {"tool": "<name>", "args": {<arguments>}}',
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
        "Rules: search BEFORE answering a KB question; at most a few tool "
        "turns total, then answer in plain text (no JSON) with the findings, "
        "citing file/section per key claim. If nothing you find is relevant, "
        "say plainly that it is not in the rules/campaign notes instead of "
        "guessing.",
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
    from config.settings import RAG_DENSE_TOP_K
    ranked, _q_emb = await idx.query_with_embeddings(query, top_n=RAG_DENSE_TOP_K)
    if keep is not None:
        ranked = [t for t in ranked if keep(t[0])]

    # Rerank the head of the list when the cross-encoder is up; on any
    # failure keep the dense order (degrades exactly like the fast path).
    top_k = RAG_AGENT_SEARCH_TOP_K
    if _rerank_ok() and ranked:
        cands = ranked[: max(top_k * 3, 24)]
        try:
            scored = await rerank(query, [c for _, c, _ in cands])
            by_content = {c: s for c, s in scored}
            ranked_sorted = sorted(
                cands, key=lambda t: by_content.get(t[1], 0.0), reverse=True)
        except Exception as exc:  # noqa: BLE001 — pure enhancement
            log.warning("search_kb rerank failed (%s); dense order", exc)
            ranked_sorted = cands
    else:
        ranked_sorted = ranked

    hits = []
    for name, content, _score in ranked_sorted[:top_k]:
        file_part = name.split(" [")[0] if " [" in name else name
        section = name[len(file_part):].strip(" []") if " [" in name else ""
        snippet = re.sub(r"\s+", " ", content)[:_SNIPPET_CHARS]
        hits.append({"file": file_part, "section": section, "snippet": snippet})

    if not hits:
        return f"No hits for {query!r}" + (f" in scope '{scope}'" if scope else "")
    # Scores only exist on the rerank path; omit the field otherwise so the
    # model doesn't read dense cosine as a calibrated relevance number.
    out = json.dumps(hits, ensure_ascii=False)
    return _cap(f"{len(hits)} hit(s):\n{out}")


def _kb_path() -> pathlib.Path:
    from config.settings import KB_PATH
    return KB_PATH


async def _tool_read_section(file: str, section: str | None = None) -> str:
    """Full chunk text for (file[, section fragment])."""
    from kb.retrievers import _ensure_index_store

    store = await _ensure_index_store(_kb_path())
    idx = store.get_index() if store is not None else None
    if idx is None or idx.is_empty():
        return "The knowledge base index is unavailable right now."

    matches: list[tuple[str, str]] = []
    for name, content in idx.iter_chunks():
        file_part = name.split(" [")[0] if " [" in name else name
        if file_part != file and not file_part.endswith(f"/{file}"):
            continue
        if section:
            sec = name[len(file_part):].strip(" []") if " [" in name else ""
            if section.lower() not in sec.lower():
                continue
        matches.append((name, content))
        if len(matches) >= 3:
            break

    if not matches:
        return f"No chunk found for file={file!r}" + (f" section~{section!r}"
                                                      if section else "")
    parts = []
    for name, content in matches:
        parts.append(f"### {name}\n{content}")
    return _cap("\n\n".join(parts))


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
        if file and not (file_part == file or file_part.endswith(f"/{file}")):
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
