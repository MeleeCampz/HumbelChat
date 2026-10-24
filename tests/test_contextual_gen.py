"""Tests for the Contextual Retrieval generation core (#52).

Covers the shared prompt/parser and generate_missing_context_lines against a
fake store: resumability (existing hashes skipped), per-batch sidecar
persist + live reload, reply-shape retry/give-up, and should_stop aborting at
batch boundaries.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kb.contextual_gen import generate_missing_context_lines, parse_json_array


# ── parser ───────────────────────────────────────────────────────────────────

class TestParseJsonArray:

    def test_plain_array(self):
        assert parse_json_array('["a", "b"]', 2) == ["a", "b"]

    def test_fenced_array_with_prose(self):
        text = 'Sure! Here you go:\n```json\n["x", "y", "z"]\n```\nHope that helps.'
        assert parse_json_array(text, 3) == ["x", "y", "z"]

    def test_bare_string_for_single_chunk(self):
        assert parse_json_array('"Just one line"', 1) == ["Just one line"]

    def test_garbage_returns_none(self):
        assert parse_json_array("I refuse to answer.", 2) is None

    def test_non_string_elements_rejected(self):
        assert parse_json_array('["a", 3]', 2) is None


# ── generation core ──────────────────────────────────────────────────────────

def _fake_store(tmp_path, n_chunks=3, with_lines=0):
    """A duck-typed KBIndexStore: _index._docs + sidecar on disk."""
    from kb.vector_db import _DocEntry, _content_hash

    docs = []
    for i in range(n_chunks):
        # Files are CONSECUTIVE (like DB rows ordered by id within a file) so
        # the generator's consecutive-grouping batches per file: n=3 → [F0,F0],[F1].
        f = i // 2
        d = _DocEntry(display_name=f"file_{f}.md [Full Document]",
                      content=f"Chunk {i} content about topic {i}.",
                      source_file=f"F{f}/file_{f}.md")
        if i < with_lines:
            d.context_line = f"Existing line {i}"
        docs.append(d)
    inner = SimpleNamespace(_docs=docs)
    lines = {_content_hash(d.content): d.context_line for d in docs if d.context_line}
    store = SimpleNamespace(
        _index=inner,
        _context_lines=dict(lines),
        persist_dir=tmp_path,
        save_context_lines=lambda ls: (
            (tmp_path / "context_lines.json").write_text(
                json.dumps(ls, ensure_ascii=False), encoding="utf-8")),
        reload_context_lines=lambda: None,
    )
    return store


def _fake_ai(lines_per_call):
    """Patch the AI client used by generate_missing_context_lines."""
    calls = []

    async def fake_complete(client, *, model, messages, **kw):
        calls.append(messages[0]["content"])
        return lines_per_call.pop(0) if lines_per_call else "[]"

    return patch("bot_core.ai_client.complete_text", new=AsyncMock(side_effect=fake_complete)), \
           patch("bot_core.ai_client._make_client", return_value=object()), \
           patch("bot_core.ai_client._validate_model", new=AsyncMock(return_value="m")), calls


class TestGenerateMissing:

    @pytest.mark.asyncio
    async def test_fills_missing_and_persists_per_batch(self, tmp_path):
        store = _fake_store(tmp_path, n_chunks=3)  # all missing; 2 files → 2 batches
        p1, p2, p3, calls = _fake_ai([
            '["line A0", "line A1"]',   # file_0 chunks (batch of 2)
            '"line B"',                 # file_1 single chunk → bare string ok
        ])
        with p1, p2, p3:
            n = await generate_missing_context_lines(store, batch_pause=0)
        assert n == 3
        sidecar = json.loads((tmp_path / "context_lines.json").read_text(encoding="utf-8"))
        assert len(sidecar) == 3
        # live reload applied the lines to the chunks themselves
        assert all(d.context_line for d in store._index._docs)

    @pytest.mark.asyncio
    async def test_existing_hashes_skipped(self, tmp_path):
        store = _fake_store(tmp_path, n_chunks=2, with_lines=1)  # both F0; one missing
        p1, p2, p3, calls = _fake_ai(['["l1"]'])
        with p1, p2, p3:
            n = await generate_missing_context_lines(store, batch_pause=0)
        assert n == 1
        # the pre-existing line is preserved in the persisted sidecar
        sidecar = json.loads((tmp_path / "context_lines.json").read_text(encoding="utf-8"))
        assert "Existing line 0" in sidecar.values()

    @pytest.mark.asyncio
    async def test_bad_shape_retries_then_gives_up(self, tmp_path):
        store = _fake_store(tmp_path, n_chunks=2)  # one file → one batch of 2
        p1, p2, p3, calls = _fake_ai(["garbage", "still garbage", '["a", "b"]'])
        with p1, p2, p3:
            n = await generate_missing_context_lines(store, batch_pause=0)
        assert n == 2          # third attempt succeeded
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_three_failures(self, tmp_path):
        store = _fake_store(tmp_path, n_chunks=2)
        p1, p2, p3, calls = _fake_ai(["bad"] * 3)
        with p1, p2, p3:
            n = await generate_missing_context_lines(store, batch_pause=0)
        assert n == 0
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_should_stops_at_batch_boundary(self, tmp_path):
        store = _fake_store(tmp_path, n_chunks=4)  # 2 files → 2+ batches
        p1, p2, p3, calls = _fake_ai(['["a", "b"]', '"never"'])
        stops = {"n": 0}

        def should_stop():
            # Checked before each batch AND before each attempt: let the first
            # batch's first attempt through (2 checks), then stop.
            stops["n"] += 1
            return stops["n"] > 2

        with p1, p2, p3:
            n = await generate_missing_context_lines(store, batch_pause=0,
                                                    should_stop=should_stop)
        assert n == 2
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_no_missing_returns_zero_without_ai(self, tmp_path):
        store = _fake_store(tmp_path, n_chunks=2, with_lines=2)
        p1, p2, p3, calls = _fake_ai([])
        with p1, p2, p3:
            n = await generate_missing_context_lines(store, batch_pause=0)
        assert n == 0 and not calls


class TestStoreSidecarMethods:

    def test_save_then_reload_roundtrip(self, tmp_path, monkeypatch):
        from kb.index import KBIndexStore
        store = KBIndexStore.__new__(KBIndexStore)
        store.persist_dir = tmp_path
        store._context_lines = {}
        store.save_context_lines({"abc": "line one"})
        raw = (tmp_path / "context_lines.json").read_text(encoding="utf-8")
        assert json.loads(raw) == {"abc": "line one"}
        assert store.reload_context_lines() == {"abc": "line one"}

    def test_save_is_atomic_no_partial_readers(self, tmp_path):
        from kb.index import KBIndexStore
        store = KBIndexStore.__new__(KBIndexStore)
        store.persist_dir = tmp_path
        store._context_lines = {}
        (tmp_path / "context_lines.json").write_text('{"old": "v"}', encoding="utf-8")
        store.save_context_lines({"new": "v"})
        # no leftover tmp file; the live file is the new one
        assert not (tmp_path / "context_lines.json.tmp").exists()
        assert json.loads((tmp_path / "context_lines.json").read_text(encoding="utf-8")) == {"new": "v"}

    def test_coverage_counts(self):
        from kb.index import KBIndexStore
        from kb.vector_db import _DocEntry
        store = KBIndexStore.__new__(KBIndexStore)
        store._index = None
        assert store.contextual_coverage() == (0, 0)
        d1 = _DocEntry(display_name="a.md", content="x", context_line="ctx")
        d2 = _DocEntry(display_name="b.md", content="y")
        store._index = SimpleNamespace(_docs=[d1, d2])
        assert store.contextual_coverage() == (1, 2)
