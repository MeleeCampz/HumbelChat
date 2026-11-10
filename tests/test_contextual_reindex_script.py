"""CLI behaviour of scripts/contextual_reindex.py (#77 re-embed flags).

Subprocess-level only: every case here is a path that touches no backend
(dry-run, or a run limited to zero chunks), so the suite stays hermetic while
still covering the argument wiring, the live-cache warning, and the dry-run
previews of the new flags.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "contextual_reindex.py"


def _seed_cache(tmp_path, chunks: list[tuple[str, str]], contextualized: int = 0) -> dict:
    """Minimal v3-shaped vector cache + sidecar in a temp persist dir."""
    import hashlib


    persist = tmp_path / ".vector_index_cache"
    persist.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(persist / "vector_index.db"))
    conn.execute(
        "CREATE TABLE document_index (id INTEGER PRIMARY KEY, source_file TEXT, doc_name TEXT,"
        " content TEXT, content_hash TEXT, file_hash TEXT, embedding BLOB, updated_at REAL)"
    )
    lines: dict[str, str] = {}
    for i, (src, content) in enumerate(chunks):
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO document_index (source_file, doc_name, content, content_hash,"
            " file_hash, embedding, updated_at) VALUES (?,?,?,?,?,?,?)",
            (src, f"{src} [Full Document]", content, h, "fh", b"\x00\x00\x00\x00", 0.0),
        )
        if i < contextualized:
            lines[h] = f"context line {i}"
    conn.commit()
    conn.close()
    (persist / "context_lines.json").write_text(json.dumps(lines), encoding="utf-8")
    return {"persist": persist}


def _run(persist_dir, kb_dir, *flags) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--kb-path", str(kb_dir), "--persist-dir", str(persist_dir), *flags],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )


@pytest.fixture
def cache(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "feats.md").write_text("| Feat | Book |\n|-|-|\n| Polearm | PHB |", encoding="utf-8")
    paths = _seed_cache(
        tmp_path,
        [("feats.md", "| Feat | Book |\n|-|-|\n| Polearm | PHB |"), ("other.md", "lorem ipsum")],
        contextualized=1,
    )
    return paths["persist"], kb


class TestDryRun:

    def test_dry_run_previews_the_reembed_scope(self, cache):
        persist, kb = cache
        r = _run(persist, kb, "--dry-run", "--reembed-contextualized", "--prune-orphans")
        assert r.returncode == 0
        assert "dry-run:" in r.stdout
        assert "would re-embed 1 contextualized chunk(s)" in r.stdout
        assert "would prune 0 orphan" in r.stdout
        # Dry run writes nothing at all.
        assert not (persist / "context_lines.json.tmp").exists()


class TestReembedMissing:

    def test_nothing_generated_redirects_to_the_repair_flag(self, cache):
        persist, kb = cache
        r = _run(persist, kb, "--reembed-missing", "--limit", "0")
        assert r.returncode == 0
        assert "nothing was generated this run" in r.stdout
        assert "--reembed-contextualized" in r.stdout
        # Any cache-writing flag warns about the live bot first.
        assert "Stop the bot first" in r.stdout

    def test_reembed_only_skips_generation_entirely(self, cache):
        persist, kb = cache
        r = _run(persist, kb, "--reembed-only", "--dry-run")
        assert r.returncode == 0
        assert "would re-embed 1 contextualized chunk(s)" in r.stdout
        # No "chunks total=…" line: _generate was never invoked.
        assert "chunks total=" not in r.stdout


class TestOrphanPruning:

    def test_orphans_are_dropped_from_the_sidecar(self, cache):
        persist, kb = cache
        sidecar_path = persist / "context_lines.json"
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["deadbeef"] = "line for a chunk that no longer exists"
        sidecar_path.write_text(json.dumps(data), encoding="utf-8")

        r = _run(persist, kb, "--prune-orphans", "--reembed-missing", "--limit", "0")
        assert r.returncode == 0
        assert "removed 1 stale sidecar entry" in r.stdout
        left = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert "deadbeef" not in left
        assert len(left) == 1
