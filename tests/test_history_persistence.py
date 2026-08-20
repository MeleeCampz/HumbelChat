"""Tests for disk persistence of conversation history (bot_core.history)."""
from __future__ import annotations

import json
import pathlib

import pytest

from bot_core import history as H


@pytest.fixture
def persist_path(tmp_path, monkeypatch):
    p = tmp_path / "history.json"
    monkeypatch.setenv("HISTORY_PERSIST_FILE", str(p))
    monkeypatch.setattr(H, "_persist_path", None, raising=False)
    H._chat_history.clear()
    H._active_characters.clear()
    yield p
    monkeypatch.setattr(H, "_persist_path", None, raising=False)


class TestHistoryPersistence:
    def test_set_history_writes_file(self, persist_path):
        H.set_history(111, 222, [{"role": "user", "content": "hi"}])
        assert persist_path.exists()
        payload = json.loads(persist_path.read_text())
        assert payload["history"]["111"]["222"] == [{"role": "user", "content": "hi"}]

    def test_roundtrip_load(self, persist_path, monkeypatch):
        H.set_history(111, 222, [{"role": "user", "content": "hi"}])
        H.set_active_char_key(111, 222, "Trixy")

        # Simulate a fresh process: wipe memory, keep the file
        H._chat_history.clear()
        H._active_characters.clear()
        assert H.get_history(111, 222) == []

        H.load_persisted()
        assert H.get_history(111, 222) == [{"role": "user", "content": "hi"}]
        assert H.get_active_char_key(111, 222) == "Trixy"

    def test_clear_history_removes_from_disk(self, persist_path):
        H.set_history(111, 222, [{"role": "user", "content": "hi"}])
        H.clear_history(111, 222)
        payload = json.loads(persist_path.read_text())
        assert "222" not in payload["history"].get("111", {})

    def test_corrupt_file_does_not_raise(self, persist_path):
        persist_path.write_text("not json{{{")
        H.load_persisted()  # should log a warning, not raise
        assert H.get_history(1, 1) == []

    def test_persistence_disabled_via_empty_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HISTORY_PERSIST_FILE", "")
        monkeypatch.setattr(H, "_persist_path", None, raising=False)
        H._chat_history.clear()
        H.set_history(111, 222, [{"role": "user", "content": "hi"}])
        assert H.get_history(111, 222) == [{"role": "user", "content": "hi"}]
