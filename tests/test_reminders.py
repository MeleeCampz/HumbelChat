"""Tests for the persistent reminder store (bot_core.reminders)."""
from __future__ import annotations

import json
import time

import pytest

from bot_core import reminders as R


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    p = tmp_path / "reminders.json"
    monkeypatch.setenv("REMINDERS_PERSIST_FILE", str(p))
    monkeypatch.setattr(R, "_store_path", None, raising=False)
    R._reminders.clear()
    R._tasks.clear()
    yield p
    monkeypatch.setattr(R, "_store_path", None, raising=False)
    R._reminders.clear()
    R._tasks.clear()


class TestReminderStore:

    def test_schedule_writes_file(self, store_path):
        rid = R.schedule_reminder(999, "hello", 300)
        assert store_path.exists()
        payload = json.loads(store_path.read_text())
        assert rid in payload
        assert payload[rid]["message"] == "hello"
        assert payload[rid]["delay_sec"] == 300
        assert payload[rid]["fired"] is False

    def test_schedule_returns_unique_ids(self, store_path):
        r1 = R.schedule_reminder(1, "a", 60)
        r2 = R.schedule_reminder(2, "b", 120)
        assert r1 != r2

    def test_cancel_reminder(self, store_path):
        rid = R.schedule_reminder(999, "hello", 300)
        assert R.cancel_reminder(rid) is True
        assert R.cancel_reminder(rid) is False
        payload = json.loads(store_path.read_text())
        assert rid not in payload

    def test_list_reminders(self, store_path):
        R.schedule_reminder(1, "a", 60)
        R.schedule_reminder(2, "b", 120)
        R._reminders[  # simulate a fired reminder
            "fired1"
        ] = {"channel_id": 1, "message": "x", "delay_sec": 60,
             "fires_at": time.time(), "created_at": time.time(), "fired": True}
        active = R.list_reminders()
        assert len(active) == 2
        assert all(not r["fired"] for r in active)

    def test_corrupt_file_does_not_raise(self, store_path):
        store_path.write_text("{{{not json")
        # rearm should not raise
        count = R.rearm_pending_reminders()
        assert count == 0

    def test_rearm_past_due_reminder(self, store_path):
        # Simulate a reminder that was set 200s ago and is due now
        R._reminders["past1"] = {
            "channel_id": 42,
            "message": "overdue",
            "delay_sec": 300,
            "fires_at": time.time() - 100,  # 100s ago
            "created_at": time.time() - 400,
            "fired": False,
        }
        count = R.rearm_pending_reminders()
        assert count == 1

    def test_rearm_future_reminder(self, store_path):
        R._reminders["future1"] = {
            "channel_id": 42,
            "message": "later",
            "delay_sec": 600,
            "fires_at": time.time() + 500,
            "created_at": time.time() - 100,
            "fired": False,
        }
        count = R.rearm_pending_reminders()
        assert count == 1

    def test_fired_reminder_not_rearmed(self, store_path):
        R._reminders["fired1"] = {
            "channel_id": 42,
            "message": "done",
            "delay_sec": 60,
            "fires_at": time.time() - 300,
            "created_at": time.time() - 400,
            "fired": True,
        }
        count = R.rearm_pending_reminders()
        assert count == 0

    def test_persistence_disabled_via_empty_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REMINDERS_PERSIST_FILE", "")
        monkeypatch.setattr(R, "_store_path", None, raising=False)
        R._reminders.clear()
        rid = R.schedule_reminder(1, "no-persist", 60)
        # In-memory state should still work
        assert rid in R._reminders
        assert R._reminders[rid]["message"] == "no-persist"
