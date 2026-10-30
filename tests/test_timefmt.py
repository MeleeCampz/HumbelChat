"""utils.timefmt — canonical session timestamp format round-trip (code review §2.2)."""
from __future__ import annotations

import time

from utils.timefmt import DISPLAY_FMT, fmt_local, parse_local


class TestFmtLocal:
    def test_format_shape(self):
        out = fmt_local(0)
        assert len(out) == 16  # YYYY-MM-DD HH:MM
        assert out[4] == "-" and out[7] == "-" and out[10] == " "

    def test_accepts_datetime(self):
        from datetime import datetime
        assert fmt_local(datetime(2026, 3, 14, 9, 26)) == "2026-03-14 09:26"

    def test_format_constant_is_the_shape(self):
        assert DISPLAY_FMT == "%Y-%m-%d %H:%M"


class TestRoundTrip:
    def test_parse_is_inverse_of_fmt(self):
        # Any whole-minute epoch survives fmt → parse unchanged (minute resolution).
        now = int(time.time()) // 60 * 60
        assert parse_local(fmt_local(now)) == now

    def test_malformed_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_local("not a timestamp")

    def test_note_round_trip_stable(self):
        # Sessions re-read note timestamps: reformat(parse(t)) == original string.
        ts = 1_762_000_000
        s = fmt_local(ts)
        assert fmt_local(parse_local(s)) == s
