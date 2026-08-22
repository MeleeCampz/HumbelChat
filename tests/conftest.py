"""Shared pytest fixtures for discord-ai-bot tests.

Provides:
  ix                   -- fixture that creates a mock Interaction (async-compatible)
  temp_kb_dir          -- temporary KB directory with test files
  temp_characters_file -- temporary characters.json file
"""
from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import MagicMock, AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _disable_history_persistence(tmp_path, monkeypatch):
    """Keep tests from writing to (or reading from) the real history file."""
    monkeypatch.setenv("HISTORY_PERSIST_FILE", str(tmp_path / "history_test.json"))
    from bot_core import history as _h
    monkeypatch.setattr(_h, "_persist_path", None, raising=False)
    _h._chat_history.clear()
    _h._active_characters.clear()


@pytest.fixture(autouse=True)
def _disable_streaming(monkeypatch):
    """Tests mock the non-streaming ask_ai; keep streaming off by default."""
    monkeypatch.setenv("AI_STREAMING", "0")


@pytest.fixture(autouse=True)
def _disable_reminders_persistence(tmp_path, monkeypatch):
    """Keep reminder tests from writing to the real reminders file."""
    monkeypatch.setenv("REMINDERS_PERSIST_FILE", str(tmp_path / "reminders_test.json"))
    from bot_core import reminders as _r
    monkeypatch.setattr(_r, "_store_path", None, raising=False)
    _r._reminders.clear()
    _r._tasks.clear()


def _build_ix(**attrs) -> MagicMock:
    """Return a mock Interaction where followup.send / response.send_message are real async."""
    sent = []

    async def on_send(content="", ephemeral=False):
        sent.append(str(content))

    ix = MagicMock()
    # followup.send is used for the final reply
    ix.followup.send.side_effect   = on_send
    # response.send_message is used for error paths in /remind etc.
    ix.response.send_message.side_effect = on_send
    # response.defer is used for defer-before-followup
    ix.response.defer = AsyncMock()

    ix._sent = sent
    for k, v in attrs.items():
        setattr(ix, k, v)
    return ix


@pytest.fixture
def ix():
    """Return a mock Interaction with async-compatible followup.send / response.send_message."""
    return _build_ix()


@pytest.fixture
def temp_kb_dir(tmp_path) -> pathlib.Path:
    kb_root = tmp_path / "kb_test"
    kb_root.mkdir()
    (kb_root / "test_doc.txt").write_text(
        "Test document content for knowledge base.\nLine 2 of the doc."
    )
    subfolder = kb_root / "subfolder"
    subfolder.mkdir()
    (subfolder / "nested.md").write_text("# Nested KB File\nContent in a subfolder.")
    return kb_root


@pytest.fixture
def temp_characters_file(tmp_path) -> pathlib.Path:
    cfg = {
        "default": "assistant",
        "characters": {
            "system":   {"display": "System",       "model": "", "system_prompt": "Be helpful."},
            "assistant":{"display": "Assistant",    "model": "gemma4:latest"},
        },
    }
    p = tmp_path / "test_characters.json"
    p.write_text(json.dumps(cfg))
    return p
