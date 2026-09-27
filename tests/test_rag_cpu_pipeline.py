"""Tests for the in-process CPU RAG pipeline: embedding backend selection,
local encoding, the cross-encoder reranker, and the BM25 lexical leg.

All tests are hermetic — no network, no model download. The sentence-transformers
models are stubbed out, and backend availability is forced via monkeypatch so the
tests pass whether or not `sentence-transformers` is installed in the venv.
"""
from __future__ import annotations

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


# ── #4: index-build embedder (GPU backend vs local) + remote fallback ───────

def test_embedder_backend_param_forces_remote(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "local")  # would normally go local
    emb = E.Embedder(model_name="BAAI/bge-m3", backend="remote", fallback_local=True)
    assert emb._use_local is False          # forced remote despite EMBED_BACKEND=local
    assert emb.model_name == "BAAI/bge-m3"  # keeps the passed slug (not LOCAL_EMBED_MODEL)
    assert emb._fallback_local is True


def test_embedder_backend_param_forces_local(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "remote")
    monkeypatch.setattr("kb.embedder.importlib.util.find_spec", lambda name: object())
    emb = E.Embedder(backend="local")
    assert emb._use_local is True
    assert emb.model_name == "BAAI/bge-m3"


@pytest.mark.asyncio
async def test_remote_encode_falls_back_to_local_on_error(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "remote")
    monkeypatch.setattr("kb.embedder.importlib.util.find_spec", lambda name: object())

    captured = {}

    class FakeModel:
        def encode(self, texts, **kw):
            captured["n"] = len(texts)
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(E, "_get_local_model", lambda: FakeModel())

    async def boom(batch):
        raise E.EmbeddingError("backend down")

    emb = E.Embedder(model_name="m", backend="remote", fallback_local=True)
    monkeypatch.setattr(emb, "_call_api", boom)
    out = await emb.encode(["a", "b"])
    assert len(out) == 2
    assert captured["n"] == 2  # local fallback encoded both texts


@pytest.mark.asyncio
async def test_remote_encode_raises_without_fallback(monkeypatch):
    import kb.embedder as E
    monkeypatch.setattr("config.settings.EMBED_BACKEND", "remote")

    async def boom(batch):
        raise E.EmbeddingError("backend down")

    emb = E.Embedder(model_name="m", backend="remote", fallback_local=False)
    monkeypatch.setattr(emb, "_call_api", boom)
    with pytest.raises(E.EmbeddingError):
        await emb.encode(["a"])


def test_store_index_embedder_uses_backend(monkeypatch, tmp_path):
    from kb.index import KBIndexStore
    monkeypatch.setattr("config.settings.INDEX_EMBED_BACKEND", "backend")
    monkeypatch.setattr("config.settings.INDEX_EMBED_MODEL", "BAAI/bge-m3")
    store = KBIndexStore(kb_path=tmp_path / "kb", persist_dir=tmp_path / "cache")
    emb = store._make_index_embedder()
    assert emb._use_local is False       # remote/GPU path
    assert emb.model_name == "BAAI/bge-m3"
    assert emb._fallback_local is True


def test_store_index_embedder_local(monkeypatch, tmp_path):
    from kb.index import KBIndexStore
    monkeypatch.setattr("config.settings.INDEX_EMBED_BACKEND", "local")
    monkeypatch.setattr("kb.embedder.importlib.util.find_spec", lambda name: object())
    store = KBIndexStore(kb_path=tmp_path / "kb", persist_dir=tmp_path / "cache")
    emb = store._make_index_embedder()
    assert emb._use_local is True


# ── #7: last-session context ────────────────────────────────────────────────

def test_build_last_session_context_uses_overview(monkeypatch):
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)

    class S:
        name = "Session 1"
        overview = "Talked about dragons."
        notes = []

    out = A._build_last_session_context(S())
    assert "Overview:" in out and "dragons" in out and "Session 1" in out


def test_build_last_session_context_is_overview_only(monkeypatch):
    """#7: the block is a BRIEF orientation — overview only.  Notes / merged
    log are deliberately NOT attached; RAG retrieves session detail on demand
    from the indexed notes.md."""
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)

    class S:
        name = "S"
        overview = "Dragons were slain."
        notes = [[1, "first note"], [2, "second note"]]

    out = A._build_last_session_context(S())
    assert "Overview:" in out and "Dragons were slain." in out
    assert "Notes:" not in out
    assert "first note" not in out and "second note" not in out


def test_build_last_session_context_no_overview_returns_empty(monkeypatch):
    """#7: without a stored overview there is nothing brief to attach
    (notes alone no longer produce a block)."""
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)

    class S:
        name = "S"
        overview = None
        notes = [[1, "first"], [2, "second"]]

    assert A._build_last_session_context(S()) == ""


def test_build_last_session_context_is_byte_stable(monkeypatch):
    """#7: same session data ⇒ byte-identical block on every call (stable
    across turns, unlike query-dependent RAG chunks)."""
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)

    class S:
        name = "S"
        overview = "Stable overview."
        notes = [[1, "a"], [2, "b"], [3, "c"]]
        started_at = 1000000.0
        ended_at = 2000000.0

    assert A._build_last_session_context(S()) == A._build_last_session_context(S())


def test_build_last_session_context_truncates_overview_over_budget(monkeypatch):
    """#7: over the char budget the overview is truncated from the TAIL with a
    deterministic marker; the block stays within budget and byte-stable."""
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)
    monkeypatch.setattr(A, "LAST_SESSION_MAX_CHARS", 120)

    class S:
        name = "S"
        overview = "Head. " + "x" * 200 + " Tail."
        notes = [[1, "note"]]

    out = A._build_last_session_context(S())
    assert len(out) <= 120
    assert out.startswith("[Previous session — S")
    assert "Head." in out            # head of the overview kept
    assert "Tail." not in out        # tail truncated away
    assert "…(truncated)" in out     # deterministic marker
    assert A._build_last_session_context(S()) == out  # still byte-stable


def test_build_last_session_context_dict_shape(monkeypatch):
    """The live session is a dict (get_last_session returns dict | None)."""
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)
    sess = {"name": "S1", "overview": "Recapped the dungeon.", "notes": [], "ended_at": 123}
    out = A._build_last_session_context(sess)
    assert "Overview:" in out and "dungeon" in out and "S1" in out


def test_build_last_session_context_disabled_or_none(monkeypatch):
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", False)

    class S:
        name = "S"
        overview = "x"
        notes = []

    assert A._build_last_session_context(S()) == ""
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)
    assert A._build_last_session_context(None) == ""


def test_build_last_session_context_capped(monkeypatch):
    """Huge overview vs. small budget: the overview is truncated from the
    tail — always ≤ cap, notes never attached."""
    from bot_core import ai_client as A
    monkeypatch.setattr(A, "LAST_SESSION_CONTEXT_ENABLED", True)
    monkeypatch.setattr(A, "LAST_SESSION_MAX_CHARS", 150)

    class S:
        name = "S"
        overview = "A" * 120
        notes = [[1, "keep me"]]

    out = A._build_last_session_context(S())
    assert len(out) <= 150
    assert "truncated" in out     # overview truncated from the tail
    assert "keep me" not in out   # notes are no longer attached at all


def test_compose_system_prompt_appends_stat_block_rules(monkeypatch):
    """Every character's system prompt gets the stat-block format appendix;
    missing persona falls back to the default, rules still appended."""
    from bot_core import ai_client as A

    class C:
        system_prompt = "You are Marvin."

    out = A._compose_system_prompt(C())
    assert out.startswith("You are Marvin.")
    assert "<stat-block-format>" in out and out.endswith("</stat-block-format>")
    assert "```" in out  # stat blocks are narrow fenced (monospace) tables
    assert "one entry per line" in out.lower()  # comparisons stay fence-free
    assert "NO header row is needed" in out or "no header row is needed" in out
    assert out == C().system_prompt + A.STAT_BLOCK_FORMAT_RULES  # byte-exact append

    class Bare:  # no system_prompt attribute at all
        pass

    fallback = A._compose_system_prompt(Bare())
    assert "helpful AI assistant" in fallback and "<stat-block-format>" in fallback


def test_append_user_message_includes_session_block():
    from bot_core import ai_client as A
    out = A._append_user_message("Bob", "hi", rag_context="RAG", session_context="SESS")
    assert "[Previous session context]" in out
    assert "[Relevant knowledge-base context]" in out
    assert out.index("[Previous session context]") < out.index("[Relevant knowledge-base context]")


def test_append_user_message_rag_only_unchanged():
    from bot_core import ai_client as A
    out = A._append_user_message("Bob", "hi", rag_context="RAG")
    assert "[Previous session context]" not in out
    assert "[Relevant knowledge-base context]\nRAG" in out


# ── #8: min-attachment relevance floor ───────────────────────────────────────

def _fake_store_with_ranked(ranked):
    class FakeIdx:
        def is_empty(self):
            return False

        async def query_with_embeddings(self, q, top_n=5):
            return ranked, [0.0] * 8

    class FakeStore:
        def get_index(self):
            return FakeIdx()

    return FakeStore()


@pytest.mark.asyncio
async def test_min_attach_score_drops_low_chunks(monkeypatch):
    import kb.retrievers as R
    monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.5)
    monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)
    monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.0)  # no rewrite
    monkeypatch.setattr("kb.reranker.is_available", lambda: False)

    ranked = [("a.md", "high chunk", 0.9), ("b.md", "mid chunk", 0.4), ("c.md", "low chunk", 0.1)]

    async def fake_ensure(kb_path):
        return _fake_store_with_ranked(ranked)

    monkeypatch.setattr(R, "_ensure_index_store", fake_ensure)
    docs = await R._retrieve_vector("q", "/tmp/kb", top_n=5)
    names = [n for n, _ in docs]
    assert "a.md" in names       # 0.9 >= 0.5 kept
    assert "b.md" not in names   # 0.4 < 0.5 dropped
    assert "c.md" not in names   # 0.1 < 0.5 dropped


@pytest.mark.asyncio
async def test_min_attach_score_off_keeps_all(monkeypatch):
    import kb.retrievers as R
    monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)  # pre-fusion floor off
    monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)     # attach floor off (isolate pre-fusion knob)
    monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)
    monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.0)
    monkeypatch.setattr("kb.reranker.is_available", lambda: False)

    ranked = [("a.md", "hi", 0.9), ("b.md", "lo", 0.1)]

    async def fake_ensure(kb_path):
        return _fake_store_with_ranked(ranked)

    monkeypatch.setattr(R, "_ensure_index_store", fake_ensure)
    docs = await R._retrieve_vector("q", "/tmp/kb", top_n=5)
    names = [n for n, _ in docs]
    assert "a.md" in names and "b.md" in names  # all kept when both floors off


@pytest.mark.asyncio
async def test_attach_floor_drops_weak_keeps_strong(monkeypatch):
    """RAG_ATTACH_FLOOR gates at ATTACH time on the original dense score."""
    import kb.retrievers as R
    monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)  # pre-fusion off
    monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.5)
    monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)
    monkeypatch.setattr("config.settings.RAG_REWRITE_MIN_SCORE", 0.0)
    monkeypatch.setattr("kb.reranker.is_available", lambda: False)

    ranked = [("a.md", "hi", 0.9), ("b.md", "lo", 0.1)]

    async def fake_ensure(kb_path):
        return _fake_store_with_ranked(ranked)

    monkeypatch.setattr(R, "_ensure_index_store", fake_ensure)
    docs = await R._retrieve_vector("q", "/tmp/kb", top_n=5)
    names = [n for n, _ in docs]
    assert "a.md" in names   # dense 0.9 >= 0.5 attached
    assert "b.md" not in names  # dense 0.1 < 0.5 dropped at attach time
