"""Tests for the persistent reminder store (bot_core.reminders)."""
from __future__ import annotations

import json
import time
from unittest.mock import patch

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


class TestP1_15DelayCap:
    """P1 #15: /remind must cap absurd delays (e.g. 10^9 s) so a reminder —
    and its background task — doesn't live forever. Both the store and the
    user-facing command share the same 30-day cap.
    """

    def test_clamp_reminder_delay(self):
        assert R.clamp_reminder_delay(60) == 60                    # under cap unchanged
        assert R.clamp_reminder_delay(R.MAX_REMINDER_DELAY_SEC) == R.MAX_REMINDER_DELAY_SEC
        assert R.clamp_reminder_delay(10 ** 9) == R.MAX_REMINDER_DELAY_SEC
        assert R.clamp_reminder_delay(-5) == -5                    # negatives pass through (min enforced by cmd)

    def test_schedule_clamps_absurd_delay(self, store_path):
        """The TODO's verify case: a 40-day request is stored at <= 30 days."""
        forty_days = 40 * 24 * 3600
        rid = R.schedule_reminder(999, "hello", forty_days)
        stored = R._reminders[rid]
        assert stored["delay_sec"] == R.MAX_REMINDER_DELAY_SEC
        assert stored["delay_sec"] <= 30 * 24 * 3600
        # and fires_at reflects the clamped (not requested) delay
        assert stored["fires_at"] <= time.time() + R.MAX_REMINDER_DELAY_SEC + 1

    def test_schedule_preserves_normal_delay(self, store_path):
        rid = R.schedule_reminder(999, "hello", 3600)
        assert R._reminders[rid]["delay_sec"] == 3600

    @pytest.mark.asyncio
    async def test_remind_command_clamps_and_notifies(self, ix):
        """An over-cap /remind (1000 hours, ~41.6 days) is scheduled at the
        30-day cap with a 'max I can schedule' notice. (The command offers
        s/min/h units, so hours is used to exceed the 30-day cap.)"""
        from commands.utility_commands import handle_remind_command

        with patch("bot_core.reminders.schedule_reminder") as sched:
            await handle_remind_command(
                ix, time_value=1000, time_unit="hours", message="way in the future"
            )

        # Scheduled with the clamped delay, not the requested ~41.6 days.
        sched.assert_called_once()
        delay_arg = sched.call_args.args[2]
        assert delay_arg == R.MAX_REMINDER_DELAY_SEC
        assert delay_arg <= 30 * 24 * 3600

        joined = " ".join(ix._sent)
        assert "30 days" in joined
        assert "max I can schedule" in joined

    @pytest.mark.asyncio
    async def test_remind_command_normal_delay_no_cap_notice(self, ix):
        """Regression guard: a normal delay is unaffected by the cap logic."""
        from commands.utility_commands import handle_remind_command

        with patch("bot_core.reminders.schedule_reminder") as sched:
            await handle_remind_command(ix, time_value=2, time_unit="hours", message="standup")

        assert sched.call_args.args[2] == 2 * 3600
        joined = " ".join(ix._sent)
        assert "max I can schedule" not in joined
        assert "2 hour" in joined
