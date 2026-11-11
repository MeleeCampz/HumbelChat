"""Tests for the contextual-sidecar maintenance loop in main.py (#52, #80).

The worker is no longer a boot-only fill: it re-checks sidecar coverage and the
re-embed stamp every CONTEXTUAL_FILL_RECHECK_S and repairs what is missing. These
tests pin the properties that make it safe to leave running forever — readiness
gates, heartbeat yielding, lag-only repair rounds, and never taking the bot down.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _store(covered: int, total: int, lag: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        contextual_coverage=lambda: (covered, total),
        contextual_vector_lag=lambda: lag,
        reembed_contextualized=AsyncMock(return_value=lag),
        _index=object(),
    )


def _prepare(monkeypatch, store, *, fill=True, recheck=0.0, rounds=1, probe=True):
    """Patch the loop's knobs + its readiness probes to deterministic values."""
    import main
    from kb import contextual_gen as gen
    from kb import retrievers
    monkeypatch.setattr(main, "CONTEXTUAL_FILL_INITIAL_DELAY_S", 0)
    monkeypatch.setattr(main, "CONTEXTUAL_FILL_RECHECK_S", recheck)
    monkeypatch.setattr(main, "CONTEXTUAL_FILL_MAX_ROUNDS", rounds)
    monkeypatch.setattr(main, "CONTEXTUAL_FILL_RETRY_S", 0.0)
    monkeypatch.setattr(main, "CONTEXTUAL_BACKGROUND_FILL", fill)
    monkeypatch.setattr(main, "CONTEXTUAL_FILL_BATCH_PAUSE_S", 5.0)
    monkeypatch.setattr(retrievers, "_ensure_index_store", AsyncMock(return_value=store))
    # A real round needs the embedding backend up and the chat model hosted;
    # stub both so a test controls the gate instead of the environment.
    async def _probe(*a, **kw):
        return probe
    monkeypatch.setattr(gen, "make_fill_client", lambda *a, **kw: SimpleNamespace())

    async def _resolve(client, **kw):
        return True, None
    monkeypatch.setattr("kb.embedder.probe_remote", _probe)
    monkeypatch.setattr(gen, "resolve_fill_model", _resolve)
    return main, gen


class TestContextualSidecarWorker:

    @pytest.mark.asyncio
    async def test_warns_on_partial_coverage_without_fill(self, monkeypatch, caplog):
        main, _gen = _prepare(monkeypatch, _store(5, 10), fill=False)
        caplog.set_level(logging.WARNING, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert any("5/10" in r.message for r in caplog.records if r.levelno == logging.WARNING)
        assert any("--reembed-contextualized" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_method_style_is_closed_does_not_kill_the_worker(self, monkeypatch, caplog):
        """Production regression 2026-11-10: discord.py 2.7.1's is_closed is a plain
        METHOD — every test fake used the property form, so the worker read a bound
        method (always truthy), treated a live bot as closed, and exited silently
        after its first nap: the fill loop never ran in production, zero log output.
        """
        main, _gen = _prepare(monkeypatch, _store(5, 10), fill=False)
        caplog.set_level(logging.WARNING, logger="bot")
        live = SimpleNamespace(is_closed=lambda: False)  # discord.Client-style method
        await main._contextual_sidecar_worker(live)
        assert any("5/10" in r.message for r in caplog.records if r.levelno == logging.WARNING)

    @pytest.mark.asyncio
    async def test_method_style_is_closed_true_stops_the_worker(self, monkeypatch, caplog):
        """The callable form must still be RESPECTED: a closing bot stops the loop."""
        main, _gen = _prepare(monkeypatch, _store(5, 10), fill=True, rounds=3)
        caplog.set_level(logging.WARNING, logger="bot")
        closing = SimpleNamespace(is_closed=lambda: True)
        await main._contextual_sidecar_worker(closing)
        assert not [r for r in caplog.records if "sidecar" in r.message.lower()]

    @pytest.mark.asyncio
    async def test_silent_when_fully_covered(self, monkeypatch, caplog):
        main, _gen = _prepare(monkeypatch, _store(10, 10), fill=False)
        caplog.set_level(logging.WARNING, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert not [r for r in caplog.records if "sidecar" in r.message.lower()]

    @pytest.mark.asyncio
    async def test_fill_enabled_generates_missing(self, monkeypatch, caplog):
        store = _store(0, 4)
        main, gen = _prepare(monkeypatch, store, fill=True)
        captured: dict = {}

        async def fake_gen(store_, **kw):
            captured["pause"] = kw.get("batch_pause")
            captured["reembed"] = kw.get("reembed")
            captured["should_stop"] = kw.get("should_stop") is not None
            return 4

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        caplog.set_level(logging.INFO, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert any("Contextual fill round" in r.message for r in caplog.records)
        # #77: the batch pause is configurable and the vectors follow the lines.
        assert captured["pause"] == 5.0
        assert captured["reembed"] is True
        assert captured["should_stop"] is True
        # Generation did the re-embed for its own chunks; the loop must not
        # double up a whole-store pass in the same round.
        assert store.reembed_contextualized.await_count == 0

    @pytest.mark.asyncio
    async def test_lag_only_round_repairs_vectors_without_chat(self, monkeypatch, caplog):
        """Lines exist but their vectors don't carry them → repair, no LLM call."""
        store = _store(5749, 5749, lag=246)
        main, gen = _prepare(monkeypatch, store, fill=True)
        called = False

        async def fake_gen(store_, **kw):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        caplog.set_level(logging.INFO, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert not called, "a lag-only round must not spend chat completions"
        assert store.reembed_contextualized.await_args.args == (None,)
        rec = [r.getMessage() for r in caplog.records if "fill round" in r.getMessage()]
        assert rec and "246 chunk(s) vector-repaired" in rec[0]

    @pytest.mark.asyncio
    async def test_complete_index_does_nothing(self, monkeypatch):
        """The steady state: zero work, zero LLM calls, no log noise."""
        store = _store(5944, 5944)
        main, gen = _prepare(monkeypatch, store, fill=True, rounds=1)
        called = False

        async def fake_gen(store_, **kw):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert not called
        assert store.reembed_contextualized.await_count == 0

    @pytest.mark.asyncio
    async def test_round_defers_when_embedder_is_down(self, monkeypatch, caplog):
        """Half-filling is worse than waiting: no lines generated without re-embed."""
        store = _store(5749, 5995, lag=246)
        main, gen = _prepare(monkeypatch, store, fill=True, probe=False)
        called = False

        async def fake_gen(store_, **kw):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        caplog.set_level(logging.WARNING, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert not called
        assert store.reembed_contextualized.await_count == 0
        assert any("embedding backend unreachable" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_round_defers_when_model_not_hosted(self, monkeypatch, caplog):
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True)

        async def _resolve(client, **kw):
            return False, None

        monkeypatch.setattr(gen, "resolve_fill_model", _resolve)
        called = False

        async def fake_gen(store_, **kw):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert not called

    @pytest.mark.asyncio
    async def test_yields_to_the_offline_script_heartbeat(self, monkeypatch, tmp_path):
        """While contextual_reindex.py is writing the cache, the loop stands down."""
        from kb import index as kb_index
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True)
        called = False

        async def fake_gen(store_, **kw):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        monkeypatch.setattr(kb_index, "DEFAULT_PERSIST_DIR", tmp_path)
        beat = main._fill_heartbeat_path(tmp_path)
        beat.parent.mkdir(parents=True, exist_ok=True)
        beat.write_text(f"{time.time()} pid=1 hits=3\n", encoding="utf-8")
        try:
            await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        finally:
            beat.unlink()
        assert not called
        assert store.reembed_contextualized.await_count == 0

    @pytest.mark.asyncio
    async def test_heartbeat_left_by_a_killed_script_does_not_freeze_the_fill(
        self, monkeypatch, tmp_path, caplog
    ):
        """Fresh mtime, ancient claimed beat → the loop must not sit out 20 min."""
        from kb import index as kb_index
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True)

        async def fake_gen(store_, **kw):
            return 12

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        monkeypatch.setattr(kb_index, "DEFAULT_PERSIST_DIR", tmp_path)
        beat = main._fill_heartbeat_path(tmp_path)
        beat.parent.mkdir(parents=True, exist_ok=True)
        beat.write_text(f"{time.time() - 3600} pid=4242 hits=9\n", encoding="utf-8")
        try:
            caplog.set_level(logging.INFO, logger="bot")
            await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        finally:
            beat.unlink()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("stale" in m and "proceeding" in m for m in msgs)
        assert any("fill round" in m for m in msgs)

    @pytest.mark.asyncio
    async def test_unreadable_heartbeat_is_reported_and_ignored(self, monkeypatch, tmp_path, caplog):
        from kb import index as kb_index
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True)

        async def fake_gen(store_, **kw):
            return 12

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        monkeypatch.setattr(kb_index, "DEFAULT_PERSIST_DIR", tmp_path)
        beat = main._fill_heartbeat_path(tmp_path)
        beat.parent.mkdir(parents=True, exist_ok=True)
        beat.write_text("not a timestamp at all", encoding="utf-8")
        try:
            caplog.set_level(logging.INFO, logger="bot")
            await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        finally:
            beat.unlink()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("unreadable" in m for m in msgs)
        assert any("fill round" in m for m in msgs)

    @pytest.mark.asyncio
    async def test_no_progress_round_backs_off_instead_of_hammering(self, monkeypatch, caplog):
        """Lines generated but no vector refreshed = a backend problem, not a queue."""
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True)

        async def fake_gen(store_, **kw):
            return 0  # nothing landed (all batches failed)

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        caplog.set_level(logging.INFO, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert any("made no progress" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_probe_is_skipped_when_there_is_nothing_to_do(self, monkeypatch):
        """Steady state must cost zero network calls, every interval, forever."""
        store = _store(5944, 5944)
        main, gen = _prepare(monkeypatch, store, fill=True, rounds=1)
        calls = []

        async def _probe(*a, **kw):
            calls.append(1)
            return True

        from kb import embedder as kb_embedder
        monkeypatch.setattr(kb_embedder, "probe_remote", _probe)
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert calls == []

    @pytest.mark.asyncio
    async def test_loop_survives_a_failing_round(self, monkeypatch, caplog):
        """A broken round logs and waits — it must never escape into the bot."""
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True, rounds=2)

        async def fake_gen(store_, **kw):
            raise RuntimeError("backend hiccup")

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        caplog.set_level(logging.WARNING, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        msgs = [r.getMessage() for r in caplog.records if "worker failed" in r.getMessage()]
        assert len(msgs) == 2
        # Type name, not %r: a repr() blow-up must not swallow the diagnosis.
        assert "RuntimeError" in msgs[0]

    @pytest.mark.asyncio
    async def test_stops_when_bot_closing(self, monkeypatch):
        store = _store(0, 12)
        main, gen = _prepare(monkeypatch, store, fill=True, rounds=5)

        async def fake_gen(store_, **kw):
            return 0

        monkeypatch.setattr(gen, "generate_missing_context_lines", fake_gen)
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=True))
        assert store.reembed_contextualized.await_count == 0

    @pytest.mark.asyncio
    async def test_no_store_is_a_noop(self, monkeypatch):
        import main
        from kb import retrievers
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_INITIAL_DELAY_S", 0)
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_MAX_ROUNDS", 1)
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_RETRY_S", 0.0)
        monkeypatch.setattr(retrievers, "_ensure_index_store", AsyncMock(return_value=None))
        # must simply return — no exception
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))


class TestFillRoundReadiness:

    @pytest.mark.asyncio
    async def test_reports_the_two_counters(self, monkeypatch):
        import main
        from kb import contextual_gen as gen
        from kb import embedder as kb_embedder

        async def _probe(*a, **kw):
            return True

        monkeypatch.setattr(kb_embedder, "probe_remote", _probe)
        monkeypatch.setattr(gen, "make_fill_client", lambda *a, **kw: SimpleNamespace())

        async def _resolve(client, **kw):
            return True, "some-model"

        monkeypatch.setattr(gen, "resolve_fill_model", _resolve)
        store = _store(5749, 5995, lag=246)
        go, pending, lag = await main._fill_round_readiness(store)
        assert (go, pending, lag) == (True, 246, 246)

    @pytest.mark.asyncio
    async def test_no_work_means_no_round(self, monkeypatch):
        import main
        store = _store(5944, 5944)
        assert await main._fill_round_readiness(store) == (False, 0, 0)

    @pytest.mark.asyncio
    async def test_vector_repair_needs_no_chat_model(self, monkeypatch):
        """A lag-only round runs even when the chat model isn't hosted."""
        import main
        from kb import contextual_gen as gen
        from kb import embedder as kb_embedder

        async def _probe(*a, **kw):
            return True

        monkeypatch.setattr(kb_embedder, "probe_remote", _probe)

        def _boom():
            raise AssertionError("must not build a chat client for a lag-only round")

        monkeypatch.setattr(gen, "make_fill_client", _boom)
        store = _store(5944, 5944, lag=100)
        go, pending, lag = await main._fill_round_readiness(store)
        assert (go, pending, lag) == (True, 0, 100)


class TestDeferralLogThrottle:
    """A steady-state deferral must not become a WARNING every recheck interval."""

    def test_notices_are_full_twice_then_quiet(self, monkeypatch, caplog):
        import logging

        from kb import contextual_gen as gen
        monkeypatch.setattr(gen, "_defer_notices", 0)
        caplog.set_level(logging.INFO, logger="bot.contextual_gen")
        for _ in range(12):
            gen._defer_log("embedding backend unreachable (%d)", 7)
        kinds = [r.levelno for r in caplog.records]
        assert kinds[:2] == [logging.WARNING, logging.WARNING]
        assert logging.WARNING not in kinds[2:5], "the middle stretch must be quiet"
        assert logging.WARNING in kinds[5:], "and it must resurface periodically"
        assert len(kinds) == 12
