"""Tests for per-turn AI records (bot_core/turn_logger).

One self-contained JSON file per /ai turn in logs/ai_turns/: complete final
prompt, retrieval details, agent-mode telemetry, full response text, timings.
Design contract under test: the logger NEVER breaks a turn — every public
function is total, and AI_TURN_LOG_ENABLED=0 short-circuits to zero overhead.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _stub_turn_logger_env(monkeypatch, tmp_path: Path):
    """Point the turn logger at a temp dir with a small rotation cap."""
    from bot_core import turn_logger

    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_ENABLED", True)
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_DIR", str(tmp_path / "turns"))
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_KEEP", 3)
    return turn_logger


# ─────────────────────── record lifecycle ───────────────────────


def test_full_lifecycle_writes_json_record(monkeypatch, tmp_path):
    tl = _stub_turn_logger_env(monkeypatch, tmp_path)

    rec = tl.new_turn(
        channel_id=42, username="Alice", char_key="char1",
        model="m-27b", user_message="Welche Feats gibt es für Int?",
    )
    assert rec is not None and rec["status"] == "pending"

    tl.set_retrieval(
        rec, attached_files=["FeatList.md"], rag_chars=1234,
        retrieval_meta={"rerank_top": 0.81, "dense_top": 0.55},
    )
    tl.set_prompt(rec, messages=[
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "[ctx]\n**Alice:** Welche Feats…?"},
    ], total_chars=2000)
    tl.set_agent(rec, trigger=None)  # fast path

    import time
    tl.finish_turn(
        rec, response_text="Hier sind die Feats…",
        started_monotonic=time.monotonic() - 1.5,
    )

    files = list((tmp_path / "turns").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["v"] == 1
    assert data["channel_id"] == 42 and data["username"] == "Alice"
    assert data["char_key"] == "char1" and data["model"] == "m-27b"
    assert data["user_message"].startswith("Welche Feats")
    # retrieval: attached files + rerank/dense top scores
    assert data["retrieval"]["attached_files"] == ["FeatList.md"]
    assert data["retrieval"]["rag_chars"] == 1234
    assert data["retrieval"]["rerank_top"] == 0.81
    # prompt: the COMPLETE final message array, verbatim
    msgs = data["prompt"]["messages"]
    assert msgs[0] == {"role": "system", "content": "persona"}
    assert "Feats" in msgs[1]["content"]
    assert data["prompt"]["total_chars"] == 2000
    # fast path marker
    assert data["agent"] == {"mode": "off"}
    # response + timing + status
    assert data["response"]["text"].startswith("Hier sind")
    assert data["timing_ms"]["total"] >= 1490
    assert data["status"] == "ok" and data["error"] == ""


def test_agent_mode_record_carries_tool_transcript(monkeypatch, tmp_path):
    tl = _stub_turn_logger_env(monkeypatch, tmp_path)
    rec = tl.new_turn(
        channel_id=1, username="Bob", char_key=None, model="m", user_message="q",
    )
    meta = SimpleNamespace(
        iterations=2, tool_calls=["search_kb", "read_section"],
        fallback=False, fallback_reason="",
        gathered_context="[search_kb {\"query\": \"feats\"}]\n<rows…>",
    )
    tl.set_agent(rec, trigger="low rerank score 0.41 < floor 0.50", meta=meta)
    tl.finish_turn(rec, response_text="answer")

    data = json.loads(next((tmp_path / "turns").glob("*.json")).read_text())
    assert data["agent"]["mode"] == "auto"
    assert data["agent"]["trigger"].startswith("low rerank score")
    assert data["agent"]["iterations"] == 2
    assert data["agent"]["tool_calls"] == ["search_kb", "read_section"]
    # the loop's transcript: every "[tool args]" line + its result
    assert "[search_kb" in data["agent"]["gathered_context"]


# ─────────────────────── total-failure contract ───────────────────────


def test_disabled_short_circuits_to_none(monkeypatch, tmp_path):
    tl = _stub_turn_logger_env(monkeypatch, tmp_path)
    monkeypatch.setattr(tl, "AI_TURN_LOG_ENABLED", False)
    assert tl.new_turn(
        channel_id=1, username="x", char_key=None, model="m", user_message="q",
    ) is None


def test_all_mutators_accept_none_record_without_raising():
    from bot_core import turn_logger

    # None record (disabled) must be a silent no-op everywhere.
    turn_logger.set_retrieval(None, attached_files=[], rag_chars=0)
    turn_logger.set_prompt(None, messages=[], total_chars=0)
    turn_logger.set_agent(None, trigger=None)
    turn_logger.finish_turn(None, response_text="x")


def test_write_failure_never_propagates(monkeypatch, tmp_path):
    tl = _stub_turn_logger_env(monkeypatch, tmp_path)
    rec = tl.new_turn(
        channel_id=1, username="x", char_key=None, model="m", user_message="q",
    )
    # Simulate an unwritable target (disk full / permission denied).
    with patch.object(Path, "mkdir",
                      side_effect=PermissionError("read-only fs")):
        # Must NOT raise — the turn's delivery is unaffected.
        tl.finish_turn(rec, response_text="still delivered")


def test_rotation_keeps_only_newest(monkeypatch, tmp_path):
    tl = _stub_turn_logger_env(monkeypatch, tmp_path)  # KEEP=3
    for i in range(5):
        rec = tl.new_turn(
            channel_id=i, username="u", char_key=None, model="m",
            user_message=f"question number {i}",
        )
        tl.finish_turn(rec, response_text=f"answer {i}")

    files = sorted((tmp_path / "turns").glob("*.json"))
    assert len(files) == 3
    # the two oldest (questions 0 and 1) were rotated out
    contents = [f.read_text() for f in files]
    assert not any("question number 0" in c for c in contents)
    assert any("question number 4" in c for c in contents)


def test_filename_is_slug_of_user_message(monkeypatch, tmp_path):
    tl = _stub_turn_logger_env(monkeypatch, tmp_path)
    rec = tl.new_turn(
        channel_id=1, username="x", char_key=None, model="m",
        user_message="Was haben die Banditen in der Wüste gemacht?",
    )
    tl.finish_turn(rec, response_text="ok")
    names = [p.name for p in (tmp_path / "turns").glob("*.json")]
    assert len(names) == 1
    # umlaut-stripped, underscored slug of the message
    # NFKD transliterates ü→ue before non-alphanumerics become underscores;
    # the slug is truncated to 40 chars (long German questions run over).
    assert names[0].endswith("was_haben_die_banditen_in_der_wuste_gema.json")


# ─────────────────── integration: ask_ai records a turn ───────────────────


@pytest.mark.asyncio
async def test_ask_ai_writes_turn_record(monkeypatch, tmp_path):
    """End-to-end: a completed /ai turn lands one JSON file with prompt + reply."""
    from bot_core import ai_client

    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="PONG"))]
    client.chat.completions.create = AsyncMock(return_value=resp)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))

    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    from kb import retrievers
    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[("FeatList.md", "feat rows")]))
    # The repo .env sets RAG_AGENT_MODE=auto — pin it off so this test
    # exercises the FAST path deterministically (agent-path recording is
    # covered by the unit tests above).
    import config.settings as settings
    monkeypatch.setattr(settings, "RAG_AGENT_MODE", "off")

    from bot_core import turn_logger
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_ENABLED", True)
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_DIR", str(tmp_path / "turns"))
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_KEEP", 10)

    reply_text, extra = await ai_client.ask_ai(
        user_message="Which feats exist?", model_slug="test-model",
        guild_id=1, channel_id=2, username="Alice", user_id=None,
    )
    assert reply_text == "PONG"

    files = list((tmp_path / "turns").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["response"]["text"] == "PONG"
    # the complete final prompt was captured at build time
    assert data["prompt"]["messages"]
    joined = json.dumps(data["prompt"]["messages"])
    assert "Which feats exist?" in joined
    # retrieval details carried through from _build_ai_request
    assert data["retrieval"]["attached_files"] == ["FeatList.md"]
    # RAG_AGENT_MODE defaults to "off" in the test env → fast path marker.
    assert data["agent"] == {"mode": "off"}


@pytest.mark.asyncio
async def test_ask_ai_fast_path_records_mode_off(monkeypatch, tmp_path):
    """A non-enumeration question with a high rerank top stays on the fast path."""
    from bot_core import ai_client

    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="PONG"))]
    client.chat.completions.create = AsyncMock(return_value=resp)
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))

    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    from kb import retrievers
    # A high rerank top + non-enumeration shape → no agent trigger.
    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[("FeatList.md", "feat rows")],
                                  side_effect=lambda **kw: (kw["meta"].update(
                                      {"rerank_top": 0.9}), [("FeatList.md", "feat rows")])[1]
                                  if isinstance(kw.get("meta"), dict) else []))

    from bot_core import turn_logger
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_ENABLED", True)
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_DIR", str(tmp_path / "turns"))
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_KEEP", 10)

    await ai_client.ask_ai(
        user_message="What is the feat bonus?", model_slug="test-model",
        guild_id=1, channel_id=2, username="Alice", user_id=None,
    )

    data = json.loads(next((tmp_path / "turns").glob("*.json")).read_text())
    assert data["agent"] == {"mode": "off"}
    assert data["retrieval"]["rerank_top"] == 0.9


@pytest.mark.asyncio
async def test_ask_ai_error_still_records_turn(monkeypatch, tmp_path):
    """A classified backend failure must NOT lose the turn record (status=error)."""
    from bot_core import ai_client

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    client.models.list = AsyncMock(return_value=MagicMock(data=[]))

    monkeypatch.setattr(ai_client, "_make_client", lambda: client)
    monkeypatch.setattr(ai_client, "_validate_model",
                        AsyncMock(return_value="test-model"))
    from kb import retrievers
    monkeypatch.setattr(retrievers, "retrieve_kb_documents",
                        AsyncMock(return_value=[]))

    from bot_core import turn_logger
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_ENABLED", True)
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_DIR", str(tmp_path / "turns"))
    monkeypatch.setattr(turn_logger, "AI_TURN_LOG_KEEP", 10)

    with pytest.raises(Exception):
        await ai_client.ask_ai(
            user_message="doomed question", model_slug="test-model",
            guild_id=1, channel_id=2, username="Alice", user_id=None,
        )

    files = list((tmp_path / "turns").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["status"] == "error"
    assert "boom" in data["error"]
