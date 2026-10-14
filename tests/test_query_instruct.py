"""Optional query-side instruct wrap (EMBED_QUERY_INSTRUCT, backlog #33).

Instruction-aware embedding models (Qwen3-Embedding) rank queries better when
each query is prefixed with a short English task instruction while documents
stay plain. The wrap:

  * is a no-op while the setting is empty (default) — existing vector spaces
    and every non-Qwen deployment are untouched;
  * applies to QUERIES only (query / query_with_embeddings / rank_texts);
    index-build (document) embedding paths never call it.

Measured on the 33-query golden set with Qwen3-Embedding-4B Q4_K_M:
found 30/33 -> 32/33, hit@10 26 -> 31, worst top-20 rank 16 -> 13.
"""
from __future__ import annotations

import pathlib

import pytest

import config.settings as S
from kb.vector_db import _wrap_query_instruct

from .test_kb_vector_index import install_fake_embedder, make_store, with_fake

INSTR = "retrieve passages that answer the question"


@pytest.fixture
def kb_dir(tmp_path) -> "pathlib.Path":
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "time_system.md").write_text(
        "# Time System\n\nThe realm of humblewood runs on a unique time system "
        "with nine moon cycles per year, each lasting thirty-one days."
    )
    return kb


# ──────────────────────────── unit: the wrapper ─────────────────────────

class TestWrapHelper:
    def test_passthrough_when_empty(self, monkeypatch):
        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", "")
        assert _wrap_query_instruct("hello") == "hello"

    def test_qwen_format(self, monkeypatch):
        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", INSTR)
        # exact format from the Qwen3-Embedding README: 'Instruct: {task}\nQuery:{q}'
        assert _wrap_query_instruct("hello") == f"Instruct: {INSTR}\nQuery:hello"

    def test_reads_setting_at_call_time(self, monkeypatch):
        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", "")
        assert _wrap_query_instruct("q") == "q"
        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", INSTR)
        assert _wrap_query_instruct("q") != "q"


# ──────────────────────────── query paths ───────────────────────────────

class TestQueryPaths:
    @pytest.mark.asyncio
    async def test_query_wraps_the_query_only(self, kb_dir, tmp_path, monkeypatch):
        store = make_store(tmp_path, kb_dir)
        build_fake = install_fake_embedder(store)
        idx = await store.load()

        # documents were embedded plain (no wrap during the build)
        assert all(not t.startswith("Instruct:") for t in build_fake.encoded_texts)

        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", INSTR)
        fake = with_fake(idx, store)
        await idx.query("time system")

        assert fake.encoded_texts == [f"Instruct: {INSTR}\nQuery:time system"]

    @pytest.mark.asyncio
    async def test_query_plain_by_default(self, kb_dir, tmp_path):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx = await store.load()

        fake = with_fake(idx, store)
        await idx.query("time system")
        assert fake.encoded_texts == ["time system"]

    @pytest.mark.asyncio
    async def test_query_with_embeddings_wraps(self, kb_dir, tmp_path, monkeypatch):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx = await store.load()

        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", INSTR)
        fake = with_fake(idx, store)
        results, q_emb = await idx.query_with_embeddings("time system", top_n=3)

        assert fake.encoded_texts == [f"Instruct: {INSTR}\nQuery:time system"]
        assert results, "wrapped query still retrieves"

    @pytest.mark.asyncio
    async def test_rank_texts_wraps_each_in_order(self, kb_dir, tmp_path, monkeypatch):
        store = make_store(tmp_path, kb_dir)
        install_fake_embedder(store)
        idx = await store.load()

        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", INSTR)
        fake = with_fake(idx, store)
        await idx.rank_texts(["alpha query", "beta query"], top_n=3)

        assert fake.encoded_texts == [
            f"Instruct: {INSTR}\nQuery:alpha query",
            f"Instruct: {INSTR}\nQuery:beta query",
        ]


# ──────────────────────── documents never wrapped ───────────────────────

class TestDocumentsUnaffected:
    @pytest.mark.asyncio
    async def test_rebuild_keeps_documents_plain(self, kb_dir, tmp_path, monkeypatch):
        # instruct enabled BEFORE the build — document paths must ignore it
        monkeypatch.setattr(S, "EMBED_QUERY_INSTRUCT", INSTR)
        store = make_store(tmp_path, kb_dir)
        fake = install_fake_embedder(store)
        idx = await store.load()

        assert not idx.is_empty()
        assert all(not t.startswith("Instruct:") for t in fake.encoded_texts), (
            "document/chunk embeddings must never carry the query instruct"
        )
