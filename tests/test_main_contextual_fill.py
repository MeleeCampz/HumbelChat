"""Tests for the #52 startup contextual-sidecar worker in main.py."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class TestContextualSidecarWorker:

    @pytest.mark.asyncio
    async def test_warns_on_partial_coverage_without_fill(self, monkeypatch, caplog):
        import main
        from kb import retrievers
        store = SimpleNamespace(contextual_coverage=lambda: (5, 10))
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_INITIAL_DELAY_S", 0)
        monkeypatch.setattr(main, "CONTEXTUAL_BACKGROUND_FILL", False)
        monkeypatch.setattr(retrievers, "_ensure_index_store",
                            AsyncMock(return_value=store))
        caplog.set_level(logging.WARNING, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert any("5/10" in r.message for r in caplog.records if r.levelno == logging.WARNING)

    @pytest.mark.asyncio
    async def test_silent_when_fully_covered(self, monkeypatch, caplog):
        import main
        from kb import retrievers
        store = SimpleNamespace(contextual_coverage=lambda: (10, 10))
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_INITIAL_DELAY_S", 0)
        monkeypatch.setattr(main, "CONTEXTUAL_BACKGROUND_FILL", False)
        monkeypatch.setattr(retrievers, "_ensure_index_store",
                            AsyncMock(return_value=store))
        caplog.set_level(logging.WARNING, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert not [r for r in caplog.records if "sidecar" in r.message.lower()]

    @pytest.mark.asyncio
    async def test_fill_enabled_generates_missing(self, monkeypatch, caplog):
        import main
        from kb import retrievers
        store = SimpleNamespace(contextual_coverage=lambda: (0, 4))
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_INITIAL_DELAY_S", 0)
        monkeypatch.setattr(main, "CONTEXTUAL_BACKGROUND_FILL", True)
        monkeypatch.setattr(retrievers, "_ensure_index_store",
                            AsyncMock(return_value=store))

        async def fake_gen(store_, **kw):
            return 4

        monkeypatch.setattr("kb.contextual_gen.generate_missing_context_lines", fake_gen)
        caplog.set_level(logging.INFO, logger="bot")
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
        assert any("Contextual fill finished" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_store_is_a_noop(self, monkeypatch):
        import main
        from kb import retrievers
        monkeypatch.setattr(main, "CONTEXTUAL_FILL_INITIAL_DELAY_S", 0)
        monkeypatch.setattr(retrievers, "_ensure_index_store", AsyncMock(return_value=None))
        # must simply return — no exception
        await main._contextual_sidecar_worker(SimpleNamespace(is_closed=False))
