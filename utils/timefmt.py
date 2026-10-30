"""Canonical human-readable timestamp formatting for session notes & chat surfaces.

One format, one round-trip pair (code review §2.2).  Session notes embed
``YYYY-MM-DD HH:MM`` wall-clock strings in markdown; the old code re-implemented
``datetime.fromtimestamp(...).strftime("%Y-%m-%d %H:%M")`` in five modules and
re-parsed it ad-hoc.  The round-trip here is deliberately NAIVE LOCAL time and
symmetric (``fromtimestamp`` ↔ naive-local ``strptime().timestamp()``), so
re-reading a note yields the same epoch it was written from.  The inherent
ambiguity is confined to wall-clock times inside a repeated DST hour — the
accepted trade-off of storing human-readable times in a markdown note.
"""
from __future__ import annotations

from datetime import datetime

#: The one display format used by session notes, /session info and recap briefs.
DISPLAY_FMT = "%Y-%m-%d %H:%M"


def fmt_local(when: float | datetime) -> str:
    """Format epoch seconds (or a datetime) as the canonical local display string."""
    dt = when if isinstance(when, datetime) else datetime.fromtimestamp(when)
    return dt.strftime(DISPLAY_FMT)


def parse_local(text: str) -> float:
    """Inverse of :func:`fmt_local` — ``YYYY-MM-DD HH:MM`` → epoch seconds (local).

    Raises ``ValueError`` for malformed strings (callers skip those lines).
    """
    return datetime.strptime(text, DISPLAY_FMT).timestamp()


__all__ = ["DISPLAY_FMT", "fmt_local", "parse_local"]
