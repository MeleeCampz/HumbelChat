"""Tier 2 (#39 item 7): Contextual Retrieval — one-line context per chunk.

The sidecar (``context_lines.json`` in the persist dir) maps a chunk's
ORIGINAL content hash to a one-line context. When present, that line is
prepended to the text BOTH at embed time and at passage-assembly time
(query results, BM25 corpus, iter_chunks). Absent sidecar ⇒ every path is
byte-identical to the legacy behaviour (passage_text() == content).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

import kb.index as I
from kb.vector_db import KBVectorIndex, _DocEntry


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeEmbedder:
    """Captures the exact texts it is asked to embed (no network)."""

    def __init__(self, *a, **k):
        self.embedded: list[str] = []

    async def encode(self, texts):
        self.embedded.extend(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_store(kb: pathlib.Path):
    store = I.KBIndexStore(kb)
    store._embedder = FakeEmbedder()
    return store


@pytest.fixture
def kb(tmp_path) -> pathlib.Path:
    root = tmp_path / "kbctx"
    root.mkdir()
    (root / "one.md").write_text("# Alpha\n\nalpha section body text here")
    (root / "two.md").write_text("whole two file body")
    return root


class TestPassageText:

    def test_no_context_is_legacy_content(self):
        d = _DocEntry(display_name="x.md [S]", content="body text")
        assert d.passage_text() == "body text"

    def test_context_is_prepended_with_blank_line(self):
        d = _DocEntry(display_name="x.md [S]", content="body text",
                      context_line="Context line about this chunk.")
        assert d.passage_text() == "Context line about this chunk.\n\nbody text"


class TestFromEntries:

    def test_matching_hash_carries_context(self):
        idx = KBVectorIndex.from_entries(
            [("a.md [S]", "alpha body", "a.md")],
            [[0.1, 0.2, 0.3, 0.4]],
            context_lines={_h("alpha body"): "Ctx for alpha."},
        )
        assert idx._docs[0].context_line == "Ctx for alpha."
        assert idx._docs[0].passage_text().startswith("Ctx for alpha.")

    def test_unmatched_hash_stays_plain(self):
        idx = KBVectorIndex.from_entries(
            [("a.md [S]", "other body", "a.md")],
            [[0.1, 0.2, 0.3, 0.4]],
            context_lines={_h("alpha body"): "Ctx for alpha."},
        )
        assert idx._docs[0].context_line == ""
        assert idx._docs[0].passage_text() == "other body"

    def test_no_sidecar_map_is_fully_legacy(self):
        idx = KBVectorIndex.from_entries(
            [("a.md [S]", "alpha body", "a.md")],
            [[0.1, 0.2, 0.3, 0.4]],
        )
        assert idx._docs[0].passage_text() == "alpha body"


class TestSidecarIO:

    def test_missing_file_gives_empty_map(self, kb):
        store = _make_store(kb)
        assert store._load_context_lines() == {}

    def test_roundtrip_and_validation(self, kb):
        store = _make_store(kb)
        persist = store.persist_dir
        persist.mkdir(parents=True, exist_ok=True)
        good = _h("alpha section body text here")
        (persist / "context_lines.json").write_text(
            json.dumps({good: "A line.", "not-a-str": 5, "": None}), encoding="utf-8"
        )
        lines = store._load_context_lines()
        assert lines == {good: "A line."}  # non-str values dropped

    def test_reload_picks_up_new_sidecar(self, kb):
        store = _make_store(kb)
        assert store.reload_context_lines() == {}
        persist = store.persist_dir
        persist.mkdir(parents=True, exist_ok=True)
        (persist / "context_lines.json").write_text(
            json.dumps({_h("alpha section body text here"): "Late line."}), encoding="utf-8"
        )
        assert store.reload_context_lines() == {_h("alpha section body text here"): "Late line."}


class TestEmbedAndPassages:

    @pytest.mark.asyncio
    async def test_embed_uses_contextual_passage(self, kb):
        """With a sidecar, the EMBEDDED text is context + original content."""
        store = _make_store(kb)
        persist = store.persist_dir
        persist.mkdir(parents=True, exist_ok=True)
        # Sidecar must exist BEFORE the build so __init__ picks it up. Small
        # files take the whole-file fallback, so the chunk IS the full text.
        (persist / "context_lines.json").write_text(
            json.dumps({_h("# Alpha\n\nalpha section body text here"): "Alpha ctx line."}),
            encoding="utf-8",
        )
        store._context_lines = store.reload_context_lines()
        idx = await store.load(force_rebuild=True)
        embedded = store._embedder.embedded
        assert any(t.startswith("Alpha ctx line.\n\n") for t in embedded)
        # The in-memory doc carries the line, and iter_chunks serves passages.
        alpha = [d for d in idx._docs if "alpha" in d.content][0]
        assert alpha.context_line == "Alpha ctx line."
        passages = dict(idx.iter_chunks())
        assert any(v.startswith("Alpha ctx line.") for v in passages.values())

    @pytest.mark.asyncio
    async def test_no_sidecar_embeds_raw_content(self, kb):
        """Without a sidecar the embedded text is exactly the chunk content."""
        store = _make_store(kb)
        idx = await store.load(force_rebuild=True)
        assert not any("ctx" in t.lower() for t in store._embedder.embedded)
        # passage_text == content for every doc (legacy behaviour)
        assert all(d.passage_text() == d.content for d in idx._docs)

    @pytest.mark.asyncio
    async def test_sidecar_does_not_invalidate_cache(self, kb):
        """Context is keyed by ORIGINAL content hash — a reload with the sidecar
        present must be a cache hit (no re-embedding), since chunk validity is
        decided by the original text's hash."""
        store = _make_store(kb)
        await store.load(force_rebuild=True)
        calls_after_build = len(store._embedder.embedded)

        persist = store.persist_dir
        (persist / "context_lines.json").write_text(
            json.dumps({_h("# Alpha\n\nalpha section body text here"): "Alpha ctx line."}),
            encoding="utf-8",
        )
        # Fresh store, same cache dir: sidecar now present at __init__.
        store2 = I.KBIndexStore(kb)
        emb = FakeEmbedder()
        store2._embedder = emb
        await store2.load()  # incremental — must reuse the cached embeddings
        assert len(emb.embedded) == 0, "cache hit expected: zero embed calls"
        assert calls_after_build > 0
