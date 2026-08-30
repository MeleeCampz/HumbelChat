"""Tests for option A — low-confidence query rewriting in the vector path.

Covers:
  * reciprocal_rank_fusion — the ranking merge (pure function)
  * KBVectorIndex.rank_texts — batched multi-query ranking with one embed call
  * _retrieve_vector trigger logic — confident queries skip the rewriter;
    low-confidence queries rewrite, merge via RRF, and degrade gracefully
    when rewriting fails or is disabled.

All tests use a deterministic fake embedder (no network).
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb.retrievers import _retrieve_vector, reciprocal_rank_fusion
from kb.vector_db import KBVectorIndex


# ─────────────────────────── Fake embedding world ───────────────────────────
# 4-dim one-hot-ish vectors: each keyword lives on its own axis.  Cosine
# similarity between two texts = (shared keywords) / sqrt(|a| * |b|).

KEYWORDS = ("humblewood", "menu", "dungeon", "xyzzy")


def fake_vec(text: str) -> list[float]:
    v = [1.0 if kw in text.lower() else 0.0 for kw in KEYWORDS]
    if all(x == 0 for x in v):
        v[3] = 1.0  # unknown text gets a private axis (still unit-ish)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def make_fake_embedder():
    emb = AsyncMock()

    async def encode(texts):
        emb.encode_calls.append(list(texts))
        return [fake_vec(t) for t in texts]

    emb.encode = encode
    emb.encode_calls = []
    return emb


# (display_name, content, source_file) — C1..C4 as documented in the module.
ENTRIES = [
    ("a.md [Full Document]", "humblewood calendar and dates", "a.md"),
    ("b.md [Full Document]", "dungeon maps and traps", "b.md"),
    ("c.md [Full Document]", "humblewood menu food ingredients", "c.md"),
    ("d.md [Full Document]", "xyzzy filler text", "d.md"),
]


def build_index():
    idx = KBVectorIndex.from_entries(ENTRIES, [fake_vec(c) for _, c, _ in ENTRIES])
    idx._embedder = make_fake_embedder()
    return idx


# ─────────────────────── reciprocal_rank_fusion ────────────────────────────

class TestReciprocalRankFusion:
    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order_and_content(self):
        ranked = [("x.md [S]", "xc", 0.9), ("y.md [S]", "yc", 0.8)]
        merged = reciprocal_rank_fusion([ranked])
        assert [n for n, _, _ in merged] == ["x.md [S]", "y.md [S]"]
        assert {n: c for n, c, _ in merged}["x.md [S]"] == "xc"

    def test_overlap_boosts_chunk(self):
        """A chunk ranked #2 in two lists must beat a chunk ranked #1 in one."""
        list_a = [("p.md [S]", "pc", 0.9), ("q.md [S]", "qc", 0.8)]
        list_b = [("r.md [S]", "rc", 0.9), ("q.md [S]", "qc", 0.7)]
        merged = reciprocal_rank_fusion([list_a, list_b])
        assert [n for n, _, _ in merged][:2] == ["q.md [S]", "p.md [S]"]

    def test_scores_are_summed_rrf_terms(self):
        list_a = [("a.md [S]", "ac", 0.9)]
        list_b = [("a.md [S]", "ac", 0.5), ("b.md [S]", "bc", 0.4)]
        merged = {n: s for n, _, s in reciprocal_rank_fusion([list_a, list_b])}
        k = 60
        assert merged["a.md [S]"] == pytest.approx(1 / (k + 1) * 2)
        assert merged["b.md [S]"] == pytest.approx(1 / (k + 2))

    def test_tie_broken_by_best_rank_then_stable(self):
        # x: rank1+rank2; y: rank2+rank1 → identical fused scores.
        # Best rank is 1 for both → stable insertion order decides (x first).
        list_a = [("x.md [S]", "xc", 0.9), ("y.md [S]", "yc", 0.8)]
        list_b = [("y.md [S]", "yc", 0.9), ("x.md [S]", "xc", 0.8)]
        merged = reciprocal_rank_fusion([list_a, list_b])
        assert merged[0][2] == pytest.approx(merged[1][2])  # true tie
        assert [n for n, _, _ in merged] == ["x.md [S]", "y.md [S]"]

    def test_duplicate_names_in_one_list_accumulate(self):
        list_a = [("a.md [S]", "ac", 0.9), ("a.md [S]", "ac", 0.8)]
        merged = reciprocal_rank_fusion([list_a])
        k = 60
        assert merged[0][2] == pytest.approx(1 / (k + 1) + 1 / (k + 2))


# ─────────────────────── KBVectorIndex.rank_texts ──────────────────────────

class TestRankTexts:
    @pytest.mark.asyncio
    async def test_one_embed_call_for_many_queries(self):
        idx = build_index()
        results = await idx.rank_texts(["humblewood", "menu food", "xyzzy"], top_n=2)
        emb = idx._embedder
        assert len(emb.encode_calls) == 1            # single batched call
        assert emb.encode_calls[0] == ["humblewood", "menu food", "xyzzy"]
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_each_list_sorted_desc_and_capped(self):
        idx = build_index()
        results = await idx.rank_texts(["humblewood"], top_n=2)
        names = [n for n, _, _ in results[0]]
        assert names == ["a.md [Full Document]", "c.md [Full Document]"]
        scores = [s for _, _, s in results[0]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_inputs_and_empty_index(self):
        idx = build_index()
        assert await idx.rank_texts([], top_n=3) == []
        empty = KBVectorIndex.from_entries([], [])
        assert await empty.rank_texts(["x"], top_n=3) == [[]]


# ─────────────── _retrieve_vector: trigger + merge behavior ────────────────

class TestLowConfidenceRewrite:
    def _patch_store(self, monkeypatch, idx):
        store = MagicMock()
        store.get_index = MagicMock(return_value=idx)
        monkeypatch.setattr("kb.retrievers._index_store", store)

    def _patch_rewriter(self, expansions: list[str] | Exception):
        async def fake_expand(query):
            if isinstance(expansions, Exception):
                raise expansions
            return [query, *expansions]

        rw = MagicMock()
        rw.expand = AsyncMock(side_effect=fake_expand)
        return patch("kb.query_rewriter.create_query_rewriter", return_value=rw), rw

    @pytest.mark.asyncio
    async def test_confident_query_never_invokes_rewriter(self, monkeypatch):
        idx = build_index()
        self._patch_store(monkeypatch, idx)
        # High threshold: only an exact-keyword match (top = 1.0) counts as confident.
        monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.9)

        p, rw = self._patch_rewriter(["menu food"])
        with p:
            docs = await _retrieve_vector("humblewood calendar", "kb", top_n=4)

        # Confident (top = 1.0 ≥ 0.9): rewriter never invoked, one embed call.
        assert rw.expand.await_count == 0
        assert len(idx._embedder.encode_calls) == 1
        names = [n for n, _ in docs]
        assert "a.md" in names and "c.md" in names

    @pytest.mark.asyncio
    async def test_low_confidence_query_rewrites_and_merges(self, monkeypatch):
        idx = build_index()
        self._patch_store(monkeypatch, idx)
        # "xyzzy dungeon" partially matches two chunks (top ≈ 0.707 < 0.9).
        monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.9)

        p, rw = self._patch_rewriter(["humblewood menu food ingredients", "menu ingredients"])
        with p:
            docs = await _retrieve_vector("xyzzy dungeon", "kb", top_n=4)

        assert rw.expand.await_count == 1
        # Original embed + one batched call for BOTH expansions.
        assert len(idx._embedder.encode_calls) == 2
        # c.md ranks #1 in both expansion lists → RRF lifts it above the
        # original query's top chunk (b.md, which appears in only one list).
        assert docs[0][0] == "c.md"
        assert "menu food ingredients" in docs[0][1]

    @pytest.mark.asyncio
    async def test_rewrite_returning_only_original_is_noop(self, monkeypatch):
        idx = build_index()
        self._patch_store(monkeypatch, idx)
        monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.9)

        p, rw = self._patch_rewriter([])  # expand() → [original] only
        with p:
            docs = await _retrieve_vector("xyzzy dungeon", "kb", top_n=4)

        assert rw.expand.await_count == 1
        assert len(idx._embedder.encode_calls) == 1  # no expansion embed call
        assert docs[0][0] == "b.md"  # original ranking untouched

    @pytest.mark.asyncio
    async def test_rewrite_failure_falls_back_to_original(self, monkeypatch):
        idx = build_index()
        self._patch_store(monkeypatch, idx)
        monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.9)

        p, rw = self._patch_rewriter(RuntimeError("backend down"))
        with p:
            docs = await _retrieve_vector("xyzzy dungeon", "kb", top_n=4)

        assert rw.expand.await_count == 1
        assert len(idx._embedder.encode_calls) == 1
        assert docs[0][0] == "b.md"

    @pytest.mark.asyncio
    async def test_disabled_via_settings(self, monkeypatch):
        idx = build_index()
        self._patch_store(monkeypatch, idx)
        monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.9)
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", False)

        p, rw = self._patch_rewriter(["menu food"])
        with p:
            docs = await _retrieve_vector("xyzzy dungeon", "kb", top_n=4)

        assert rw.expand.await_count == 0
        assert len(idx._embedder.encode_calls) == 1
        assert docs[0][0] == "b.md"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
