"""#77: targeted re-embed of chunks whose vector predates their context line.

``KBIndexStore.reembed_contextualized()`` is the missing half of Contextual
Retrieval: the fill (and the offline generator) write a chunk's one-line
context, but the chunk's stored vector was built from the RAW text, so the
dense retrieval leg never sees the framing. These tests cover selection
(only what you asked for), the contextual passage actually being what gets
embedded, cache invalidation (numpy matrix + BM25 corpus are keyed by DOC-LIST
identity, so the swap must replace the list), the incremental SQLite upsert,
and the never-corrupt failure paths.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3

import pytest

import kb.index as I
from kb.index import _unpack_embedding


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vec(text: str) -> list[float]:
    """Fake vector that IDENTIFIES the exact text it was built from.

    First component = a byte of the text's digest, so a test can tell a
    raw-content vector from a contextual-passage vector. Dimension is fixed
    (2) to exercise the dimension guard without numpy-sized vectors.
    """
    return [float(hashlib.sha256(text.encode("utf-8")).digest()[0]), 1.0]


class HashEmbedder:
    """Records every encode call; vectors identify their source text."""

    def __init__(self, *a, **k):
        self.calls: list[list[str]] = []

    async def encode(self, texts, on_batch=None):
        self.calls.append(list(texts))
        return [_vec(t) for t in texts]


class RaisingEmbedder:
    """Always fails — the store must leave index and cache untouched."""

    def __init__(self, *a, **k):
        self.calls = 0

    async def encode(self, texts, on_batch=None):
        self.calls += 1
        raise RuntimeError("embeddings backend down")


class BatchThenSingleEmbedder:
    """Batch encode fails; per-chunk encodes succeed (oversized-chunk case)."""

    def __init__(self, *a, **k):
        self.calls: list[list[str]] = []

    async def encode(self, texts, on_batch=None):
        self.calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("Context size has been exceeded")
        return [_vec(texts[0])]


@pytest.fixture
def kb(tmp_path):
    root = tmp_path / "kbre"
    root.mkdir()
    # Explicit LF bytes (write_text would emit CRLF on Windows and change the
    # content hash the sidecar is keyed by).
    (root / "one.md").write_bytes(b"# Alpha\n\nalpha body text here")
    (root / "two.md").write_bytes(b"# Beta\n\nbeta body text here")
    return root


def _store(kb, embedder=None):
    store = I.KBIndexStore(kb)
    store._embedder = embedder if embedder is not None else HashEmbedder()
    return store


async def _built_store_with_lines(kb, lines: dict[str, str]):
    """Index built WITHOUT context (vectors from raw text), then the lines appear.

    Exactly the state #77 fixes: ``context_line`` set on the live docs and the
    sidecar on disk, while ``embedding`` still describes the raw chunk.
    """
    store = _store(kb)
    idx = await store.load()
    for d in idx._docs:
        if d.content in lines:
            d.context_line = lines[d.content]
    store._context_lines = {_h(c): l for c, l in lines.items()}
    store.save_context_lines(store._context_lines)
    return store


def _rows(db_path) -> dict[str, tuple[str, str | None, list[float]]]:
    """source_file → (content_hash, file_hash, embedding) for every cache row."""
    conn = sqlite3.connect(str(db_path))
    try:
        out = {
            src: (ch, fh, _unpack_embedding(emb))
            for src, ch, fh, emb in conn.execute(
                "SELECT source_file, content_hash, file_hash, embedding FROM document_index"
            ).fetchall()
        }
    finally:
        conn.close()
    return out


def _docs_by_source(store):
    return {d.source(): d for d in store._index._docs}


ALPHA_LINE = "feats-by-book.md table row: Polearm Master (PHB feat)."
BETA_LINE = "two.md section 'Beta': what beta does."


class TestTargetedReembed:

    @pytest.mark.asyncio
    async def test_only_requested_hashes_are_reembedded(self, kb):
        store = await _built_store_with_lines(
            kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE,
                 "# Beta\n\nbeta body text here": BETA_LINE})
        before = _docs_by_source(store)
        vecs_before = {k: list(d.embedding) for k, d in before.items()}
        alpha = before["one.md"]

        n = await store.reembed_contextualized({_h(alpha.content)})

        assert n == 1
        # What was embedded is the CONTEXTUAL passage, not the raw chunk.
        assert store._embedder.calls[-1] == [f"{ALPHA_LINE}\n\n{alpha.content}"]
        after = _docs_by_source(store)
        assert after["one.md"].embedding == _vec(f"{ALPHA_LINE}\n\n{alpha.content}")
        # Untouched chunk keeps its exact old vector.
        assert after["two.md"].embedding == vecs_before["two.md"]
        # Metadata preserved on the rebuilt entry (file_hash feeds the P2 #19 fast path).
        kept = after["one.md"]
        assert kept.content == alpha.content
        assert kept.display_name == alpha.display_name
        assert kept.context_line == ALPHA_LINE
        assert kept.file_hash == alpha.file_hash

    @pytest.mark.asyncio
    async def test_none_target_reembeds_every_contextualized_chunk(self, kb):
        store = await _built_store_with_lines(
            kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE,
                 "# Beta\n\nbeta body text here": BETA_LINE})
        n = await store.reembed_contextualized(None)
        assert n == 2
        assert all(d.embedding == _vec(d.passage_text()) for d in store._index._docs)

    @pytest.mark.asyncio
    async def test_uncontextualized_chunks_are_never_touched(self, kb):
        store = _store(kb)
        await store.load()  # no sidecar at all → legacy state
        calls = len(store._embedder.calls)
        vecs = [list(d.embedding) for d in store._index._docs]
        assert await store.reembed_contextualized(None) == 0
        assert await store.reembed_contextualized({_h("whatever")}) == 0
        assert len(store._embedder.calls) == calls          # no embedding calls made
        assert [d.embedding for d in store._index._docs] == vecs

    @pytest.mark.asyncio
    async def test_empty_hash_set_is_a_noop(self, kb):
        store = await _built_store_with_lines(kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE})
        calls = len(store._embedder.calls)
        assert await store.reembed_contextualized(set()) == 0
        assert len(store._embedder.calls) == calls


class TestCacheInvalidation:
    """Both query caches key on the DOC LIST object; the swap must replace it."""

    @pytest.mark.asyncio
    async def test_numpy_matrix_reflects_the_new_vector(self, kb):
        store = await _built_store_with_lines(kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE})
        idx = store._index
        row = next(i for i, d in enumerate(idx._docs) if d.source() == "one.md")
        alpha = idx._docs[row]
        stale_row = _vec(alpha.content)
        assert idx._ensure_matrix()[0][row].tolist() == stale_row

        assert await store.reembed_contextualized({_h(alpha.content)}) == 1
        # Same doc order, refreshed row — the cache was rebuilt, not reused.
        assert idx._ensure_matrix()[0][row].tolist() == _vec(alpha.passage_text())
        assert idx._mat_cache is not None and idx._mat_cache[2] is idx._docs

    @pytest.mark.asyncio
    async def test_bm25_corpus_is_rebuilt_and_sees_the_context_line(self, kb):
        """BM25 caches tokenized text, so a stale corpus would stay blind."""
        from kb.retrievers import _get_bm25
        store = _store(kb)
        idx = await store.load()
        corpus_raw = _get_bm25(idx)                    # built pre-fill: raw chunks only
        assert corpus_raw.scores("polearm master phb")[0] == 0.0

        # The fill lands: line on the doc + sidecar on disk, vector still raw.
        alpha = next(d for d in idx._docs if d.source() == "one.md")
        alpha.context_line = ALPHA_LINE
        store._context_lines = {_h(alpha.content): ALPHA_LINE}
        store.save_context_lines(store._context_lines)

        assert await store.reembed_contextualized({_h(alpha.content)}) == 1
        corpus_new = _get_bm25(store._index)
        assert corpus_new is not corpus_raw
        assert corpus_new.scores("polearm master phb")[0] > 0.0


class TestPersistence:

    @pytest.mark.asyncio
    async def test_only_the_affected_files_rows_change(self, kb):
        store = await _built_store_with_lines(
            kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE,
                 "# Beta\n\nbeta body text here": BETA_LINE})
        alpha = next(d for d in store._index._docs if d.source() == "one.md")
        # Seed the stale state on disk too (the fill persists lines, not vectors).
        await store._save_to_disk()
        db_before = _rows(store.db_path)

        await store.reembed_contextualized({_h(alpha.content)})
        db_after = _rows(store.db_path)

        assert set(db_after) == set(db_before)                      # no rows lost/added
        assert db_after["one.md"][2] == _vec(f"{ALPHA_LINE}\n\n{alpha.content}")
        assert db_after["one.md"][0] == db_before["one.md"][0]      # content_hash stable
        assert db_after["one.md"][1] == db_before["one.md"][1]      # file_hash stable (P2 #19)
        assert db_after["two.md"] == db_before["two.md"]            # untouched byte-for-byte

    @pytest.mark.asyncio
    async def test_reload_from_disk_carries_the_fresh_vectors(self, kb):
        store = await _built_store_with_lines(kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE})
        alpha = next(d for d in store._index._docs if d.source() == "one.md")
        await store.reembed_contextualized({_h(alpha.content)})
        await store.shutdown()

        # A fresh store = a bot restart: cache HIT, zero embed calls, fresh vectors.
        store2 = _store(kb)
        idx2 = await store2.load()
        assert not store2._embedder.calls
        doc = next(d for d in idx2._docs if d.source() == "one.md")
        assert doc.embedding == _vec(f"{ALPHA_LINE}\n\n{doc.content}")
        assert doc.context_line == ALPHA_LINE


class TestFailurePaths:

    @pytest.mark.asyncio
    async def test_backend_failure_leaves_index_and_cache_untouched(self, kb):
        store = await _built_store_with_lines(kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE})
        alpha = next(d for d in store._index._docs if d.source() == "one.md")
        docs_obj = store._index._docs
        vecs = [list(d.embedding) for d in docs_obj]
        await store._save_to_disk()
        db_before = _rows(store.db_path)

        store._embedder = RaisingEmbedder()
        assert await store.reembed_contextualized({_h(alpha.content)}) == 0

        assert store._index._docs is docs_obj                   # not swapped
        assert [d.embedding for d in store._index._docs] == vecs
        assert _rows(store.db_path) == db_before                # cache unchanged

    @pytest.mark.asyncio
    async def test_batch_failure_falls_back_to_per_chunk(self, kb):
        """One oversized chunk must not take its batch siblings down."""
        store = await _built_store_with_lines(
            kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE,
                 "# Beta\n\nbeta body text here": BETA_LINE})
        store._embedder = BatchThenSingleEmbedder()
        n = await store.reembed_contextualized(None)
        assert n == 2
        assert [c[0] for c in store._embedder.calls if len(c) == 1] == [
            d.passage_text() for d in store._index._docs
        ]
        assert all(d.embedding == _vec(d.passage_text()) for d in store._index._docs)

    @pytest.mark.asyncio
    async def test_wrong_dimension_vector_is_rejected(self, kb):
        """A foreign vector space must never be written into this cache."""
        store = await _built_store_with_lines(kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE})
        alpha = next(d for d in store._index._docs if d.source() == "one.md")
        vecs_before = [list(d.embedding) for d in store._index._docs]

        class WideEmbedder:
            async def encode(self, texts, on_batch=None):
                return [[0.5, 0.5, 0.5] for _ in texts]

        store._embedder = WideEmbedder()
        assert await store.reembed_contextualized({_h(alpha.content)}) == 0
        assert [d.embedding for d in store._index._docs] == vecs_before

    @pytest.mark.asyncio
    async def test_partial_batch_success_still_persists_the_successes(self, kb):
        store = await _built_store_with_lines(
            kb, {"# Alpha\n\nalpha body text here": ALPHA_LINE,
                 "# Beta\n\nbeta body text here": BETA_LINE})
        beta = next(d for d in store._index._docs if d.source() == "two.md")

        class HalfBroken:
            """Second passage in the batch fails individually."""

            def __init__(self):
                self.calls = 0

            async def encode(self, texts, on_batch=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("batch rejected")
                return [_vec(texts[0])] if texts[0] != beta.passage_text() else None

        store._embedder = HalfBroken()
        n = await store.reembed_contextualized(None)
        assert n == 1
        docs = _docs_by_source(store)
        assert docs["one.md"].embedding == _vec(docs["one.md"].passage_text())
        assert docs["two.md"].embedding == _vec(beta.content)  # kept its old (raw) vector
        assert _rows(store.db_path)["one.md"][2] == _vec(docs["one.md"].passage_text())


class TestLegacyStampMigration:
    """The sidecar's re-embed stamp (#80) must survive the load that creates it."""

    async def _contents(self, kb) -> dict[str, str]:
        """source -> CHUNK content, exactly as the index sees it (sidecar keys
        hash over this, not over the file bytes)."""
        store = _store(kb)
        await store.load()
        return {d.source(): d.content for d in store._index._docs}

    def _write_sidecar(self, store, by_source: dict[str, str], *, stamped=None) -> pathlib.Path:
        payload = {_h(c): by_source[c] for c in by_source}
        if stamped is not None:
            payload["__v2__"] = 1
            payload["__v__"] = sorted(stamped)
        path = store.persist_dir / "context_lines.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    async def test_pre_stamp_sidecar_is_stamped_once_and_persisted(self, kb, caplog):
        """Lines older than the stamp must not read as 'vector missing' forever."""
        import logging
        contents = await self._contents(kb)
        lines = {contents["one.md"]: ALPHA_LINE, contents["two.md"]: BETA_LINE}
        store = _store(kb)
        await store.load()                       # vectors built from RAW text
        self._write_sidecar(store, lines)

        caplog.set_level(logging.INFO, logger="kb.index")
        store2 = _store(kb)
        await store2.load()
        assert store2.contextual_coverage() == (2, 2)
        assert store2.contextual_vector_lag() == 0, "a stamped load leaves no lag"
        data = json.loads((store2.persist_dir / "context_lines.json").read_text(encoding="utf-8"))
        assert data["__v2__"] == 1
        assert sorted(data["__v__"]) == sorted([_h(contents["one.md"]), _h(contents["two.md"])])
        assert any("predate the re-embed stamp" in r.getMessage() for r in caplog.records)

        # It stays resolved: a second start re-stamps nothing and re-embeds nothing.
        before = (store2.persist_dir / "context_lines.json").read_bytes()
        store3 = _store(kb)
        await store3.load()
        assert store3.contextual_vector_lag() == 0
        assert (store3.persist_dir / "context_lines.json").read_bytes() == before
        assert await store3.reembed_contextualized(None) == 0

    async def test_stamp_format_sidecar_is_never_blanket_stamped(self, kb):
        """Marker present + partial stamp = the #77 state: it must stay lagging."""
        contents = await self._contents(kb)
        lines = {contents["one.md"]: ALPHA_LINE, contents["two.md"]: BETA_LINE}
        store = _store(kb)
        await store.load()
        path = self._write_sidecar(store, lines, stamped={_h(contents["one.md"])})
        before = path.read_bytes()

        store2 = _store(kb)
        await store2.load()
        assert store2.contextual_coverage() == (2, 2)
        assert store2.contextual_vector_lag() == 1
        assert _h(contents["two.md"]) not in store2._context_vec
        assert path.read_bytes() == before, "a load must not invent a stamp it was given"

    async def test_reembed_records_the_stamp_so_a_second_run_is_a_no_op(self, kb):
        contents = await self._contents(kb)
        store = await _built_store_with_lines(kb, {contents["one.md"]: ALPHA_LINE})
        assert store.contextual_vector_lag() == 1
        n = await store.reembed_contextualized(None)
        assert n == 1
        data = json.loads((store.persist_dir / "context_lines.json").read_text(encoding="utf-8"))
        assert data["__v__"] == [_h(contents["one.md"])]
        assert await store.reembed_contextualized(None) == 0
