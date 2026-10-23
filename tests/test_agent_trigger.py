"""Tests for the agent-mode trigger (#39 §3.4) in bot_core.ai_client."""
import pytest

from bot_core.ai_client import _looks_like_enumeration, agent_trigger_reason


@pytest.mark.parametrize("mode", ["off", ""])
def test_mode_off_never_triggers(monkeypatch, mode):
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", mode)
    assert agent_trigger_reason("list all backgrounds in Humblewood", 0.1) is None
    assert agent_trigger_reason("plate armor AC?", 0.1) is None


def test_mode_always_triggers(monkeypatch):
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", "always")
    assert agent_trigger_reason("anything", None) == "always"


def test_unknown_mode_is_off(monkeypatch):
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", "sometimes")
    assert agent_trigger_reason("list all backgrounds", 0.1) is None


@pytest.mark.parametrize("text,expected", [
    ("List all the backgrounds in Humblewood", True),
    ("Which special spells exist?", True),
    ("Nenne alle Hintergründe von Humblewood", True),
    ("Welche Zaubersprüche gibt es?", True),
    ("Zähle die Hintergrundoptionen auf", True),
    ("What is the AC of plate armor?", False),
    ("Which one is better, a longsword or a warhammer?", False),
    ("", False),
])
def test_enumeration_heuristic(text, expected):
    assert _looks_like_enumeration(text) is expected


def test_auto_triggers_on_low_rerank(monkeypatch):
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", "auto")
    monkeypatch.setattr("config.settings.RAG_AGENT_SCORE_GATE", 0.55)
    reason = agent_trigger_reason("plate armor AC?", 0.31)
    assert reason is not None and "low-rerank" in reason


def test_auto_no_trigger_on_confident_rerank(monkeypatch):
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", "auto")
    monkeypatch.setattr("config.settings.RAG_AGENT_SCORE_GATE", 0.55)
    assert agent_trigger_reason("plate armor AC?", 0.87) is None


def test_auto_no_trigger_without_rerank_signal(monkeypatch):
    """A missing rerank score (backend down) must NOT activate the agent —
    the fast path degrades gracefully instead."""
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", "auto")
    assert agent_trigger_reason("plate armor AC?", None) is None


def test_auto_enumeration_triggers_even_when_confident(monkeypatch):
    monkeypatch.setattr("config.settings.RAG_AGENT_MODE", "auto")
    reason = agent_trigger_reason("List all the backgrounds in Humblewood", 0.9)
    assert reason == "enumeration-shape"
