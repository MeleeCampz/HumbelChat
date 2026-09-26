"""Tests for the in-process CPU RAG pipeline: embedding backend selection,
local encoding, the cross-encoder reranker, and the BM25 lexical leg.

All tests are hermetic — no network, no model download. The sentence-transformers
models are stubbed out, and backend availability is forced via monkeypatch so the
tests pass whether or not `sentence-transformers` is installed in the venv.
"""
from __future__ import annotations

import math

import pytest


# ── effective_embedding_model (backend-aware cache identity) ────────────────

def test_effective_embedding_model_local(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "EMBED_BACKEND", "local")
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "BAAI/bge-m3")
    assert settings.effective_embedding_model() == "BAAI/bge-m3"


def test_effective_embedding_model_remote(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "EMBED_BACKEND", "remote")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "some-remote-model")
    assert settings.effective_embedding_model() == "some-remote-model"


# ── Embedder backend selection in __init__ ───────────────────────────────────

def test_embedder_local_selected_when_available(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "local")
    monkeypatch.setattr("kb.embedder.importlib.util.find_spec", lambda name: object())
    emb = E.Embedder()
    assert emb._use_local is True
    # Model name reflects the local model (drives index cache identity).
    assert emb.model_name == "BAAI/bge-m3"


def test_embedder_falls_back_to_remote_when_lib_missing(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "local")
    monkeypatch.setattr("kb.embedder.importlib.util.find_spec", lambda name: None)
    emb = E.Embedder()
    assert emb._use_local is False


def test_embedder_remote_by_default(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "remote")
    emb = E.Embedder()
    assert emb._use_local is False


# ── Local (in-process) encoding path ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_local_encode_routes_and_shapes(monkeypatch):
    import numpy as np
    import kb.embedder as E

    captured = {}

    class FakeModel:
        def encode(self, texts, **kw):
            captured["texts"] = list(texts)
            captured["kw"] = kw
            base = [[1.0, 0.0], [0.0, 1.0]]
            return np.array([base[i % len(base)] for i in range(len(texts))], dtype="float32")

    monkeypatch.setattr(E, "_get_local_model", lambda: FakeModel())
    emb = E.Embedder()
    emb._use_local = True
    out = await emb.encode(["x", "y"])

    assert captured["kw"].get("normalize_embeddings") is True
    assert len(out) == 2
    assert all(isinstance(v, list) for v in out)


@pytest.mark.asyncio
async def test_local_encode_deterministic_and_ordered(monkeypatch):
    import numpy as np
    import kb.embedder as E

    class FakeModel:
        def encode(self, texts, **kw):
            # Deterministic per-text vector so we can assert ordering/dedup mapping.
            vecs = []
            for t in texts:
                v = np.zeros(3)
                v[0] = 1.0
                v[len(t) % 3] = 1.0
                v = v / (np.linalg.norm(v) or 1.0)
                vecs.append(v)
            return np.stack(vecs)

    monkeypatch.setattr(E, "_get_local_model", lambda: FakeModel())
    emb = E.Embedder()
    emb._use_local = True
    out = await emb.encode(["a", "bb", "a"])  # duplicate "a" must map back correctly
    assert len(out) == 3
    assert out[0] == out[2]  # same input -> same vector


# ── Cross-encoder reranker ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_disabled_returns_original_order(monkeypatch):
    from kb import reranker
    monkeypatch.setattr("config.settings.RERANK_ENABLED", False)
    out = await reranker.rerank("q", ["a", "b", "c"])
    assert [c for c, _ in out] == ["a", "b", "c"]
    assert all(s == 0.0 for _, s in out)


@pytest.mark.asyncio
async def test_rerank_reorders_by_score(monkeypatch):
    from kb import reranker
    monkeypatch.setattr("config.settings.RERANK_ENABLED", True)
    monkeypatch.setattr("kb.reranker.importlib.util.find_spec", lambda name: object())

    class FakeCE:
        def predict(self, pairs, **kw):
            return [1.0, 3.0, 2.0]  # b highest, then c, then a

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCE())
    out = await reranker.rerank("q", ["a", "b", "c"])
    assert [c for c, _ in out] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_rerank_falls_back_on_error(monkeypatch):
    from kb import reranker
    monkeypatch.setattr("config.settings.RERANK_ENABLED", True)
    monkeypatch.setattr("kb.reranker.importlib.util.find_spec", lambda name: object())

    class BoomCE:
        def predict(self, pairs, **kw):
            raise RuntimeError("model blew up")

    monkeypatch.setattr(reranker, "_get_model", lambda: BoomCE())
    out = await reranker.rerank("q", ["a", "b"])
    assert [c for c, _ in out] == ["a", "b"]  # original order preserved
    assert all(s == 0.0 for _, s in out)


# ── BM25 lexical leg ─────────────────────────────────────────────────────────

def test_bm25_ranks_relevant_chunk_first():
    from kb.lexical import BM25
    docs = [
        "the dragon hoards gold",
        "the calendar has twelve moons",
        "spells require mana to cast",
    ]
    bm = BM25(docs)
    scores = bm.scores("dragon gold")
    assert scores[0] == max(scores)
    assert scores[0] > 0


def test_lexical_ranking_returns_relevant_first(monkeypatch):
    import kb.retrievers as R

    class FakeDoc:
        def __init__(self, name, content):
            self.name = name
            self.content = content

        def retrieval_name(self):
            return self.name

    class FakeIdx:
        _docs = [FakeDoc("a.md", "dragon hoards gold"),
                 FakeDoc("b.md", "calendar moons")]

    monkeypatch.setattr(R, "_bm25_cache", None)  # reset cache between tests
    ranked = R._lexical_ranking(FakeIdx(), "dragon", limit=5)
    assert ranked and ranked[0][0] == "a.md"
    assert ranked[0][2] > 0


def test_get_bm25_caches_by_docs_identity(monkeypatch):
    import kb.retrievers as R

    class FakeDoc:
        def __init__(self, content):
            self.content = content

    docs = [FakeDoc("hello world")]
    idx = type("I", (), {"_docs": docs})()
    monkeypatch.setattr(R, "_bm25_cache", None)
    first = R._get_bm25(idx)
    second = R._get_bm25(idx)
    assert first is second  # same _docs object -> cached
