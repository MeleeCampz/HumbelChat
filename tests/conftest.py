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

# Prevent main.py from attaching production log-file handlers when the test
# suite imports it (code review §1.10). Must be set before `import main`.
os.environ["BOT_NO_LOG_FILES"] = "1"


@pytest.fixture(autouse=True)
def _quiet_file_loggers():
    """Extra safety net: strip any file handlers the main import may have added."""
    import logging
    root = logging.getLogger()
    removed = [h for h in list(root.handlers) if isinstance(h, logging.FileHandler)]
    for h in removed:
        root.removeHandler(h)
        h.close()
    yield
    for h in removed:
        h.close()


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


@pytest.fixture(autouse=True)
def _isolate_sessions(tmp_path, monkeypatch):
    """Keep session tests from writing to the real sessions file / KB.

    Points SESSIONS_PERSIST_FILE at a temp file and redirects the notes
    directory (via ``sessions.notes_dir``) into a temp dir so per-session
    markdown files never land in the real knowledge base.  config.settings
    itself is left untouched — path-resolution tests rely on its constants.
    """
    kb = tmp_path / "kb_sessions_test"
    kb.mkdir(exist_ok=True)
    monkeypatch.setenv("SESSIONS_PERSIST_FILE", str(tmp_path / "sessions_test.json"))
    from bot_core import sessions as _sess
    monkeypatch.setattr(_sess, "notes_dir", lambda: kb, raising=False)
    monkeypatch.setattr(_sess, "_store_path", None, raising=False)
    _sess._state["session"] = None
    _sess._state["last_ended"] = None
    _sess._state["last_start_at"] = None
    _sess._state["next_session_reminders"] = []
    yield


@pytest.fixture(autouse=True)
def _clear_model_list_cache():
    """ai_client caches the backend's model list (300 s TTL); keep tests
    hermetic by clearing it between tests."""
    try:
        from bot_core import ai_client
        ai_client._clear_model_list_cache()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _no_auto_index_on_upload(monkeypatch):
    """Keep /upload_kb tests from touching the real embedding backend.

    The upload handler now auto-indexes new documents via
    ``kb.retrievers.update_kb_document``; stub it out so tests stay fast and
    hermetic (no network, no repo-local index cache writes).
    """
    try:
        from kb import retrievers as _r
        monkeypatch.setattr(_r, "update_kb_document", AsyncMock(return_value=True), raising=False)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _disable_backend_health_probe(monkeypatch):
    """Prevent tests from starting the background liveness probe."""
    monkeypatch.setenv("AI_HEALTH_CHECK_INTERVAL", "0")

    try:
        from bot_core import health
        monkeypatch.setattr(health, "AI_HEALTH_CHECK_INTERVAL", 0)
    except Exception:
        pass

    try:
        import main
        monkeypatch.setattr(main, "start_backend_health_probe", lambda bot=None: None)
    except Exception:
        pass



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
