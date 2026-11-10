"""Regression tests for the 2026-11 RAG code-review fixes (R1–R12).

Hermetic: no network, no models. Covers
  R2a  sidecar circuit breakers (kb.embedder / kb.reranker)
  R2b  dense-leg gating on breaker + GPU-switch mode (kb.retrievers)
  R3   per-score-scale attach floors (RAG_ATTACH_FLOOR_RERANK / _DENSE)
  R4   write-side model stamp in the cache metadata + identity check
  R6   warm retrieval caches after index mutations
  R8   agent tool-arg telemetry in the turn record
  R1   list_sections dir= over the retrieval name-space (real index object)
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from unittest.mock import patch

import kb.embedder as E
import kb.reranker as RK
import kb.retrievers as R
from config import settings


# ═══════════════════════════ R2a: embedder circuit breaker ════════════════

class _Resp:
    def __init__(self, vecs):
        self._vecs = vecs

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": v} for v in self._vecs], "model": "fake"}


class _BoomResp:
    def __init__(self, status: int):
        self.status_code = status

    def raise_for_status(self):
        raise httpx.HTTPStatusError(
            f"HTTP {self.status_code}",
            request=httpx.Request("POST", "http://fake-sidecar.test/v1/embeddings"),
            response=httpx.Response(self.status_code),
        )

    def json(self):
        return {}


def _fake_client_cls(handler):
    class _Client:
        posts = 0

        def __init__(self, *a, **k):
            pass

        async def post(self, url, json=None, headers=None):
            _Client.posts += 1
            return handler(json)

        async def aclose(self):
            pass

    _Client.posts = 0
    return _Client


@pytest.fixture(autouse=True)
def _breaker_isolation(monkeypatch):
    """Deterministic URL + breaker isolation for breaker tests."""
    monkeypatch.setattr(settings, "EMBED_URL", "http://fake-sidecar.test/v1")
    monkeypatch.setattr(settings, "RAG_SIDECAR_CIRCUIT_COOLDOWN_S", 30.0)
    E.reset_circuit_breaker()
    yield
    E.reset_circuit_breaker()


@pytest_asyncio.fixture(autouse=True)
async def _reset_shared_client():
    """Isolate the process-wide httpx client between tests (p2 #21 pattern)."""
    await E.close_client()
    yield
    await E.close_client()


class TestEmbedderCircuit:

    @pytest.mark.asyncio
    async def test_connect_failure_trips_breaker(self):
        def boom(body):
            raise httpx.ConnectError("connection refused (simulated)")

        Client = _fake_client_cls(boom)
        emb = E.Embedder(model_name="m", backend="remote")
        with patch.object(E.httpx, "AsyncClient", lambda *a, **k: Client()):
            with pytest.raises(E.EmbeddingError):
                await emb.encode(["text"])
        assert E.remote_circuit_open() is True

    @pytest.mark.asyncio
    async def test_open_breaker_short_circuits_without_http(self):
        """After a trip, further calls fail instantly with ZERO requests."""
        def boom(body):
            raise httpx.ConnectError("connection refused (simulated)")

        Client = _fake_client_cls(boom)
        emb = E.Embedder(model_name="m", backend="remote")
        with patch.object(E.httpx, "AsyncClient", lambda *a, **k: Client()):
            with pytest.raises(E.EmbeddingError):
                await emb.encode(["text"])
        assert Client.posts >= 1

        attempts2: list[int] = []

        def record(body):
            attempts2.append(1)
            return _Resp([[0.1] * 8 for _ in body.get("input", [])])

        Client2 = _fake_client_cls(record)
        with patch.object(E.httpx, "AsyncClient", lambda *a, **k: Client2()):
            with pytest.raises(E.EmbeddingError, match="circuit-breaker"):
                await emb.encode(["more text"])
        assert attempts2 == [], "open breaker must not attempt any HTTP call"

    @pytest.mark.asyncio
    async def test_http_error_keeps_breaker_closed(self):
        """5xx on a REACHABLE server stays on the retry ladder — no trip."""
        Client = _fake_client_cls(lambda body: _BoomResp(500))
        emb = E.Embedder(model_name="m", backend="remote")
        monkey = patch.object(E, "_RETRY_ATTEMPTS", 1)  # keep the ladder fast
        with monkey, patch.object(E.httpx, "AsyncClient", lambda *a, **k: Client()):
            with pytest.raises(E.EmbeddingError):
                await emb.encode(["text"])
        assert Client.posts >= 1
        assert E.remote_circuit_open() is False

    @pytest.mark.asyncio
    async def test_breaker_cooldown_expiry_closes(self, monkeypatch):
        monkeypatch.setattr(settings, "RAG_SIDECAR_CIRCUIT_COOLDOWN_S", 0.05)

        def boom(body):
            raise httpx.ConnectError("connection refused (simulated)")

        Client = _fake_client_cls(boom)
        emb = E.Embedder(model_name="m", backend="remote")
        with patch.object(E.httpx, "AsyncClient", lambda *a, **k: Client()):
            with pytest.raises(E.EmbeddingError):
                await emb.encode(["text"])
        assert E.remote_circuit_open() is True
        await asyncio.sleep(0.07)
        assert E.remote_circuit_open() is False

    @pytest.mark.asyncio
    async def test_reset_reopens_for_immediate_retry(self):
        E._breaker_trip("http://fake-sidecar.test/v1", ConnectionError("x"))
        assert E.remote_circuit_open() is True

        Client = _fake_client_cls(
            lambda body: _Resp([[0.1] * 8 for _ in body.get("input", [])]))
        emb = E.Embedder(model_name="m", backend="remote")
        E.reset_circuit_breaker()
        with patch.object(E.httpx, "AsyncClient", lambda *a, **k: Client()):
            vecs = await emb.encode(["text"])
        assert vecs and E.remote_circuit_open() is False


class TestRerankerCircuit:

    def test_trip_and_reset(self, monkeypatch):
        monkeypatch.setattr(settings, "RERANK_API_BASE", "http://fake-rerank.test")
        monkeypatch.setattr(settings, "RAG_SIDECAR_CIRCUIT_COOLDOWN_S", 30.0)
        RK.reset_reranker_circuit_breaker()
        assert RK.reranker_circuit_open() is False
        RK._breaker_trip("http://fake-rerank.test", ConnectionError("refused"))
        assert RK.reranker_circuit_open() is True
        RK.reset_reranker_circuit_breaker()
        assert RK.reranker_circuit_open() is False


# ════════════════════════ R2b: dense-leg gate in retrieval ════════════════

class _FakeDoc:
    def __init__(self, name: str, content: str):
        self.display_name = name
        self.content = content

    def retrieval_name(self):
        return self.display_name

    def passage_text(self):
        return self.content


def _gate_index(monkeypatch, ranked):
    """Fake store/idx pair; records whether the dense leg was attempted."""
    calls: list[str] = []

    class FakeIdx:
        _docs = [
            _FakeDoc("notes.md [Full Document]",
                     "the witch hut at the crossroads details"),
            _FakeDoc("equipment.md [Armor]", "Plate Armor AC 18 weight"),
        ]

        def is_empty(self):
            return False

        async def query_with_embeddings(self, text, top_n=5):
            calls.append(text)
            return ranked, [0.1]

        def embedding_map(self):
            return {}

    class FakeStore:
        def get_index(self):
            return FakeIdx()

    monkeypatch.setattr(R, "_index_store", FakeStore())
    return calls


class TestDenseLegGate:

    async def _gate(self, monkeypatch, ranked, *, hybrid):
        monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", hybrid)
        monkeypatch.setattr("config.settings.RERANK_ENABLED", False)
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", False)
        monkeypatch.setattr("config.settings.RAG_REWRITE_ALL_QUERIES", False)
        monkeypatch.setattr("config.settings.RAG_IMAGE_MODE_LEXICAL_ONLY", True)
        monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)
        monkeypatch.setattr(R, "_bm25_cache", None)
        return _gate_index(monkeypatch, ranked)

    @pytest.mark.asyncio
    async def test_image_mode_skips_dense_leg(self, monkeypatch):
        calls = await self._gate(monkeypatch,
                                 [("equipment.md [Armor]", "Plate Armor AC 18", 0.9)],
                                 hybrid=True)
        R.set_gpu_mode("image")

        meta: dict = {}
        docs = await R._retrieve_vector("witch hut crossroads", "/nonexistent-kb",
                                        top_n=4, meta=meta)
        assert calls == [], "dense leg must not be attempted in image mode"
        assert meta["retrieval_legs"] == "lexical"
        assert any("witch hut" in c.lower() for _n, c in docs)

    @pytest.mark.asyncio
    async def test_open_embedder_breaker_skips_dense_leg(self, monkeypatch):
        calls = await self._gate(monkeypatch, [], hybrid=True)
        E._breaker_trip(settings.EMBED_URL.rstrip("/"), ConnectionError("down"))

        meta: dict = {}
        docs = await R._retrieve_vector("witch hut", "/nonexistent-kb",
                                        top_n=4, meta=meta)
        assert calls == []
        assert meta["retrieval_legs"] == "lexical"
        assert docs

    @pytest.mark.asyncio
    async def test_rag_mode_runs_dense_leg(self, monkeypatch):
        calls = await self._gate(
            monkeypatch,
            [("equipment.md [Armor]", "Plate Armor AC 18", 0.9)],
            hybrid=False)
        R.set_gpu_mode("rag")
        meta: dict = {}
        docs = await R._retrieve_vector("plate armor", "/nonexistent-kb",
                                        top_n=4, meta=meta)
        assert calls == ["plate armor"]
        assert meta["retrieval_legs"] == "dense"
        assert docs

    def test_set_gpu_mode_normalizes_unknown(self):
        R.set_gpu_mode("bogus")
        assert R.get_gpu_mode() == "rag"
        R.set_gpu_mode("IMAGE")
        assert R.get_gpu_mode() == "image"

    def test_images_notify_helper_drives_retrievers(self):
        """bot_core/images.py wires the switch result into retrieval."""
        from bot_core.images import _notify_retrievers_gpu_mode
        _notify_retrievers_gpu_mode("image")
        assert R.get_gpu_mode() == "image"
        _notify_retrievers_gpu_mode("rag")
        assert R.get_gpu_mode() == "rag"

    @pytest.mark.asyncio
    async def test_tool_search_skips_dense_leg_when_off(self, monkeypatch):
        from kb import tools
        monkeypatch.setattr(R, "_dense_leg_state", lambda: (True, "simulated off"))
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", False)

        class FakeIdx:
            def is_empty(self):
                return False

            async def query_with_embeddings(self, text, top_n=5):
                # If the dense leg were (wrongly) attempted, its hit would
                # show up in the tool output below.
                return [("bad.md [X]", "SHOULD-NOT-APPEAR", 0.9)], [0.1]

        class FakeStore:
            def get_index(self):
                return FakeIdx()

        monkeypatch.setattr(R, "_index_store", FakeStore())
        monkeypatch.setattr(R, "_lexical_ranking",
                            lambda idx, q, lim: [("notes.md [Full Document]",
                                                  "witch hut details", 3.0)])
        monkeypatch.setattr(RK, "is_available", lambda: False)

        out = await tools.execute_tool("search_kb", {"query": "witch hut"})
        assert "hit(s)" in out and "witch hut details" in out
        assert "SHOULD-NOT-APPEAR" not in out


# ════════════════════════ R3: per-scale attach floors ═══════════════════

class TestAttachFloorSplit:

    async def _retrieve(self, monkeypatch, ranked):
        monkeypatch.setattr("config.settings.RAG_HYBRID_ENABLED", False)
        monkeypatch.setattr("config.settings.RERANK_ENABLED", False)
        monkeypatch.setattr("config.settings.RAG_QUERY_REWRITER", False)
        monkeypatch.setattr("config.settings.RAG_REWRITE_ALL_QUERIES", False)
        monkeypatch.setattr("config.settings.RAG_MIN_ATTACH_SCORE", 0.0)
        _gate_index(monkeypatch, ranked)
        return await R._retrieve_vector("plate armor", "/nonexistent-kb", top_n=4)

    @pytest.mark.asyncio
    async def test_dense_knob_gates_dense_scores(self, monkeypatch):
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)     # legacy off
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR_DENSE", 0.5)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR_SOURCE", "dense")
        docs = await self._retrieve(monkeypatch, [
            ("equipment.md [Armor]", "strong chunk armor AC 18", 0.9),
            ("weak.md [Full]", "weak chunk armor lore", 0.4),
        ])
        names = [n for n, _ in docs]
        assert any("equipment" in n for n in names)
        assert not any("weak.md" in n for n in names), "dense knob must gate weak chunk"

    @pytest.mark.asyncio
    async def test_legacy_floor_still_applies(self, monkeypatch):
        """Split knobs unset → legacy RAG_ATTACH_FLOOR keeps working."""
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.5)
        monkeypatch.setattr(settings, "RAG_ATTACH_FLOOR_DENSE", None)
        monkeypatch.setattr(settings, "RAG_ATTACH_FLOOR_RERANK", None)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR_SOURCE", "dense")
        docs = await self._retrieve(monkeypatch, [
            ("equipment.md [Armor]", "strong chunk armor AC 18", 0.9),
            ("weak.md [Full]", "weak chunk armor lore", 0.4),
        ])
        names = [n for n, _ in docs]
        assert any("equipment" in n for n in names)
        assert not any("weak.md" in n for n in names)

    @pytest.mark.asyncio
    async def test_rerank_source_without_scores_uses_dense_floor(self, monkeypatch,
                                                                  caplog):
        """SOURCE=rerank but reranker OFF: gate falls back to DENSE and the
        floor used is the DENSE-scale one — never the rerank-scale one."""
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR", 0.0)
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR_RERANK", 0.95)  # rerank scale
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR_DENSE", 0.35)   # cosine scale
        monkeypatch.setattr("config.settings.RAG_ATTACH_FLOOR_SOURCE", "rerank")
        with caplog.at_level(logging.INFO, logger="kb.retrievers"):
            docs = await self._retrieve(monkeypatch, [
                ("equipment.md [Armor]", "relevant chunk armor AC 18", 0.45),
            ])
        names = [n for n, _ in docs]
        assert any("equipment" in n for n in names), (
            "0.45 cosine passes the 0.35 dense floor; applying the 0.95 rerank "
            "floor to cosine scores was exactly the R3 bug")
        assert any("gating on DENSE cosine" in r.message for r in caplog.records)


# ════════════════════════ R4: write-model stamp + identity ══════════════

class _StampingEmbedder:
    model_name = "writer-model"

    async def encode(self, texts):
        import hashlib
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:8]]
                for t in texts]


@pytest.fixture
def tmp_store(tmp_path):
    from kb.index import KBIndexStore
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("# Alpha\nplate armor AC 18\n", newline="\n")
    store = KBIndexStore(kb_dir, persist_dir=tmp_path / "cache",
                         model_name="qwen3-embedding-4b")
    store._embedder = _StampingEmbedder()
    return store, kb_dir


class TestWriteModelStamp:

    @pytest.mark.asyncio
    async def test_metadata_stamps_actual_writer(self, tmp_store):
        store, _ = tmp_store
        await store.load()
        meta = dict(sqlite3.connect(str(store._db_path)).execute(
            "SELECT key, value FROM metadata").fetchall())
        assert meta["index_embedding_model"] == "writer-model"

    @pytest.mark.asyncio
    async def test_writer_stamp_mismatch_invalidates_cache(self, tmp_store):
        store, _ = tmp_store
        await store.load()
        conn = sqlite3.connect(str(store._db_path))
        conn.execute("INSERT OR REPLACE INTO metadata (key, value) "
                     "VALUES ('index_embedding_model', 'some-other-model')")
        conn.commit()
        conn.close()
        assert store._cache_identity_ok() is False, (
            "vectors embedded by a different writer model must not be reused")


# ════════════════════════ R6: warm caches after mutation ════════════════

class TestWarmCaches:

    @pytest.mark.asyncio
    async def test_update_warms_matrix_and_bm25(self, tmp_store, monkeypatch):
        store, kb_dir = tmp_store
        monkeypatch.setattr(R, "_bm25_cache", None)
        await store.load()
        (kb_dir / "b.md").write_text("# Beta\nlongsword damage die\n", newline="\n")
        assert await store.update_single_document(kb_dir / "b.md") is True

        idx = store.get_index()
        assert idx is not None and idx._mat_cache is not None, (
            "mutation must leave the numpy rank matrix WARM (no first-query rebuild)")
        cache = R._bm25_cache
        assert cache is not None and cache[1] is idx._docs, (
            "mutation must leave the BM25 corpus cached against the NEW docs list")


# ════════════════════════ R8: tool-arg telemetry in turn logs ═══════════

class TestToolCallTelemetry:

    @pytest.mark.asyncio
    async def test_agent_loop_records_tool_args(self, monkeypatch):
        from bot_core import agent_loop

        async def _fake_execute(name, args):
            return f"RESULT[{name} {args}]"

        monkeypatch.setattr(agent_loop, "execute_tool", _fake_execute)

        class _Comp:
            def __init__(self, script):
                self.script = script

            async def __call__(self, messages, model, *, thinking):
                item = self.script.pop(0)
                await asyncio.sleep(0)
                return item

        comp = _Comp([
            '{"tool": "list_sections", "args": {"dir": "HumbleWood"}}',
            "Answer: the backgrounds file lists the Soldier, Sage, and "
            "Entertainer backgrounds of Humblewood.",
        ])
        text, meta = await agent_loop.run_agent_loop(
            complete=comp, model="test-model", system_prompt="P",
            user_message="list the lore files")
        assert "Answer" in text
        assert meta.tool_calls == ["list_sections"]
        detail = meta.tool_calls_detail
        assert detail[0]["tool"] == "list_sections"
        assert detail[0]["args"] == {"dir": "HumbleWood"}
        assert detail[0]["tool_error"] is False
        assert detail[0]["result_chars"] > 0

    def test_turn_logger_records_detail(self):
        from bot_core import turn_logger as tl
        rec: dict = {}
        meta = SimpleNamespace(
            iterations=2,
            tool_calls=["search_kb", "read_section"],
            tool_calls_detail=[
                {"tool": "search_kb", "args": {"query": "alderheart"},
                 "result_chars": 400, "tool_error": False},
                {"tool": "read_section", "args": {"file": "npcs.md"},
                 "result_chars": 50, "tool_error": True},
            ],
            fallback=False, fallback_reason="", gathered_context="…")
        tl.set_agent(rec, trigger="low rerank", meta=meta)
        detail = rec["agent"]["tool_calls_detail"]
        assert [d["tool"] for d in detail] == ["search_kb", "read_section"]
        assert detail[0]["args"] == {"query": "alderheart"}
        assert detail[1]["tool_error"] is True


# ════════════════════ R1: list_sections on a REAL index object ═══════════

class TestListSectionsRealIndex:
    """The live index stores display_name as a bare basename while source
    paths carry the folder structure — dir= must match the retrieval name."""

    @pytest.fixture
    def real_idx(self, monkeypatch):
        from kb.vector_db import KBVectorIndex
        entries = [
            ("notes.md [Full Document]",
             "They arrive at the witch's hut late evening.",
             "humblewood/HumbleWood/SessionLogs/2026-10-07_02/notes.md"),
            ("notes.md [Full Document]",
             "Bandits attack the city gates of Alderheart.",
             "humblewood/HumbleWood/SessionLogs/2026-10-20_02/notes.md"),
            ("equipment.md [Armor]",
             "Plate Armor: AC 18.",
             "humblewood/DnDRules/DnD5_5/equipment.md"),
        ]
        emb = [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]]
        idx = KBVectorIndex.from_entries(entries, emb)

        class FakeStore:
            def get_index(self):
                return idx

        monkeypatch.setattr(R, "_index_store", FakeStore())
        return idx

    @pytest.mark.asyncio
    async def test_dir_matches_nested_folder(self, real_idx):
        from kb import tools
        out = await tools.execute_tool(
            "list_sections", {"dir": "humblewood/HumbleWood/SessionLogs"})
        assert "2026-10-07_02" in out and "2026-10-20_02" in out
        assert "equipment.md" not in out

    @pytest.mark.asyncio
    async def test_dir_fuzzy_single_segment(self, real_idx):
        from kb import tools
        # Case-insensitive single segment — strict pass misses, fuzzy rescues.
        out = await tools.execute_tool("list_sections", {"dir": "sessionlogs"})
        assert "2026-10-07_02" in out

    @pytest.mark.asyncio
    async def test_dir_no_match_gets_hint(self, real_idx):
        from kb import tools
        out = await tools.execute_tool("list_sections", {"dir": "nowhere/deep"})
        assert "nowhere" in out  # hint mentions the bad dir instead of silence

    @pytest.mark.asyncio
    async def test_find_in_kb_reports_retrieval_paths(self, real_idx):
        from kb import tools
        out = await tools.execute_tool("find_in_kb", {"term": "Alderheart"})
        assert "SessionLogs/2026-10-20_02" in out
