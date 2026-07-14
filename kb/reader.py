"""Read chunks from KB files and filter by relevance.

Replaces the in-line filesystem RAG logic that was previously in main.py's
ask_ai_with_model() with a clean, testable module interface."""
from __future__ import annotations

import os
import pathlib


def read_kb_files(
    kb_path: str | pathlib.Path,
    max_lines_per_file: int = 50,
    max_bytes_per_file: int = 1024 * 1024,  # 1 MB limit for reading files into context
) -> list[tuple[str, str]]:
    """Read raw KB files from shared volume and return [(display_name, content)]."""
    kb_root = pathlib.Path(kb_path)
    if not kb_root.exists():
        return []

    results: list[tuple[str, str]] = []
    for p in sorted(kb_root.rglob("*")):
        if not p.is_file() or "?" in p.name:
            continue

        ext = _extract_ext(p.name)
        if ext not in {".txt", ".md"}:
            continue

        try:
            content_text = p.read_bytes().decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            content_text = p.read_bytes().decode("latin-1", errors="replace")

        if len(content_text) == 0 or len(content_text) > max_bytes_per_file:
            continue

        # Cap lines per file to prevent oversized context injection
        lines = content_text.splitlines()[:max_lines_per_file]
        truncated = "\n".join(lines)
        if len(content_text.splitlines()) > max_lines_per_file:
            truncated += "\n... [truncated]"

        # Build display name from stem (strip UUID prefix)
        base_name = os.path.basename(p.name)
        stem = p.stem
        idx_pos = stem.rfind("_")
        display_name = stem[idx_pos + 1:] if idx_pos > 0 else base_name

        results.append((display_name, truncated))

    return results


def _extract_ext(name: str) -> str:
    base = name.split("?")[0]
    i = base.rfind(".")
    return base[i:].lower() if i > 0 else ""

