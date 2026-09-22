"""Regression tests for the P0 correctness fixes.

P0 #1 — ``/reindex_kb`` must make the live RAG path serve the *freshly built*
       index, not the stale in-memory singleton.  Before the fix the handler
       rebuilt a throwaway ``KBIndexStore`` while ``kb.retrievers._index_store``
       kept serving the old index until the bot restarted.

P0 #2 — conversation history must be persisted on *every* turn.  Before the
       fix ``set_history`` (the only disk write) only ran once a channel
       overflowed its context-window cap, so a restart before that point lost
       every recent turn.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests._shared import Interaction


# ─────────────────────────── helpers ───────────────────────────

def _fake_store(count: int, source: str, is_empty: bool = False) -> MagicMock:
    """A KBIndexStore-shaped mock whose index reports *count* chunks.

    ``get_index`` is a *synchronous* MagicMock (matching the real class) so the
    retrievers' ``store.get_index()`` calls don't yield un-awaited coroutines.
    """
    idx = MagicMock()
    idx.count.return_value = count
    idx.is_empty.return_value = is_empty
    idx.source = source

    store = MagicMock()
    store.get_index = MagicMock(return_value=idx)
    store.load = AsyncMock(return_value=idx)
    store.shutdown = AsyncMock()
    store.idx = idx
    store.source = source
    return store


# ─────────────────── P0 #1: index-store swap / reindex ───────────────────

class TestIndexStoreSwap:

    @pytest.mark.asyncio
    async def test_replace_index_store_points_rag_at_new_index(self):
        """replace_index_store must swap the live RAG path onto the new store."""
        from kb import retrievers
        from kb.retrievers import replace_index_store

        stale = _fake_store(1, "STALE")
        fresh = _fake_store(3, "FRESH")
        retrievers._index_store = stale
        retrievers._kb_path_for_store = "/kb"

        replace_index_store(fresh, "/kb")

        assert retrievers._index_store is fresh
        # The live RAG path resolves the singleton — it must now be the fresh
        # store, never the stale one it started on.
        got = await retrievers._ensure_index_store("/kb")
        assert got is fresh

    @pytest.mark.asyncio
    async def test_reset_index_store_clears_singleton(self):
        """reset_index_store clears the singleton for a reload on next use."""
        from kb import retrievers
        from kb.retrievers import reset_index_store

        stale = _fake_store(1, "STALE")
        retrievers._index_store = stale

        await reset_index_store()

        assert retrievers._index_store is None

    @pytest.mark.asyncio
    async def test_reindex_kb_swaps_fresh_store_into_singleton(self, temp_kb_dir):
        """/reindex_kb must install the *rebuilt* store, not a throwaway.

        This is the core P0 #1 regression: after the command, the module
        singleton (what every subsequent /ai RAG query uses) must be the
        freshly built store.
        """
        from kb import retrievers

        stale = _fake_store(1, "STALE")
        fresh = _fake_store(3, "FRESH")
        retrievers._index_store = stale

        ix = Interaction()
        with patch("kb.index.KBIndexStore", return_value=fresh), \
             patch("kb.retrievers.retrieve_kb_documents",
                   new=AsyncMock(return_value=[("x.md", "hi")])), \
             patch("config.settings.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_reindex_kb
            await handle_reindex_kb(ix)

        assert retrievers._index_store is fresh, (
            "handler must swap the freshly built store into the singleton"
        )
        assert any("rebuilt" in s.lower() for s in ix._sent)

        # The replaced (stale) store is shut down out-of-band.
        for _ in range(3):
            await asyncio.sleep(0)
        stale.shutdown.assert_awaited()

    @pytest.mark.asyncio
    async def test_reindex_kb_empty_does_not_clobber_good_singleton(self, temp_kb_dir):
        """A rebuild that yields an empty index must NOT replace a good one.

        Swapping in an empty store would silently blind RAG; keeping the
        previous index degrades gracefully (still serves cached chunks +
        keyword fallback).
        """
        from kb import retrievers

        good = _fake_store(3, "GOOD")
        empty = _fake_store(0, "EMPTY", is_empty=True)
        retrievers._index_store = good

        ix = Interaction()
        with patch("kb.index.KBIndexStore", return_value=empty), \
             patch("kb.retrievers.retrieve_kb_documents",
                   new=AsyncMock(return_value=[("x.md", "hi")])), \
             patch("config.settings.KB_PATH", temp_kb_dir):
            from commands.kb_commands import handle_reindex_kb
            await handle_reindex_kb(ix)

        assert retrievers._index_store is good, (
            "an empty rebuild must not replace a working index"
        )
        assert any("could not be built" in s for s in ix._sent)


# ─────────────────── P0 #2: history persisted each turn ───────────────────

class TestHistoryPersistedEachTurn:

    @pytest.mark.asyncio
    async def test_first_turn_writes_disk_under_cap(self):
        """A single turn (far under the context-window cap) must persist.

        Before the fix ``set_history`` only ran on cap overflow, so the first
        turn of a channel wrote nothing and a restart lost it.
        """
        from bot_core import ai_client
        from bot_core import history as H

        path = H._get_persist_path()
        if path is not None and path.exists():
            path.unlink()

        g, c = 555, 666
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="PONG"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        with patch.object(ai_client, "_make_client", return_value=client), \
             patch("kb.retrievers.retrieve_kb_documents",
                   new=AsyncMock(return_value=[])):
            reply, _extra = await ai_client.ask_ai(
                user_message="PING",
                model_slug="test-model",
                guild_id=g,
                channel_id=c,
                username="Alice",
                user_id=None,
            )

        assert reply == "PONG"
        assert path is not None, "no persistence path configured (HISTORY_PERSIST_FILE)"
        assert path.exists(), "history was not written to disk on the first turn"

        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = payload["history"][str(g)][str(c)]
        assert {"role": "user", "content": "PING"} in stored
        assert {"role": "assistant", "content": "PONG"} in stored

    @pytest.mark.asyncio
    async def test_history_stays_clean_across_persist(self):
        """Persisting every turn must not leak RAG context or decorations."""
        from bot_core import ai_client
        from bot_core import history as H

        g, c = 777, 888
        marker = "UNIQUE_KB_MARKER_DO_NOT_LEAK"
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Reply text"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_resp)
        client.models.list = AsyncMock(return_value=MagicMock(data=[]))

        with patch.object(ai_client, "_make_client", return_value=client), \
             patch("kb.retrievers.retrieve_kb_documents",
                   new=AsyncMock(return_value=[("doc.md", marker)])):
            await ai_client.ask_ai(
                user_message="hello world",
                model_slug="test-model",
                guild_id=g,
                channel_id=c,
                username="Alice",
                user_id=None,
            )

        stored = H.get_history(g, c)
        assert stored, "no history recorded"
        for m in stored:
            assert marker not in m["content"], "RAG context leaked into persisted history"
            assert "Alice" not in m["content"], "username decoration leaked into persisted history"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
