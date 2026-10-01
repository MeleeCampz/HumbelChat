"""Fuzzy item search for /item_search — deterministic, stdlib only.

Scoring tiers (highest wins; ties break alphabetically):

- exact match on the normalized name        → 1.0
- normalized name starts with the query     → 0.9
- query is a substring of the name          → 0.8
- otherwise ``difflib`` similarity ratio    → kept when ≥ 0.6

Normalization (lowercase, strip non-alphanumerics except spaces, collapse
whitespace) makes emote prefixes invisible: ``"bag of holding"`` matches
``🎒 Bag of Holding``.
"""
from __future__ import annotations

import difflib
import re

from bot_core.item_tables import Item

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

#: Minimum similarity ratio for the fuzzy (non-substring) tier.
FUZZY_THRESHOLD = 0.6


def norm(text: str) -> str:
    """Normalize a name/query for matching (see module docstring)."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def score_item(query_norm: str, item_norm: str) -> float:
    """Score one item against an already-normalized query (0.0 = no match)."""
    if not query_norm or not item_norm:
        return 0.0
    if item_norm == query_norm:
        return 1.0
    if item_norm.startswith(query_norm):
        return 0.9
    if query_norm in item_norm:
        return 0.8
    ratio = difflib.SequenceMatcher(None, query_norm, item_norm).ratio()
    return ratio if ratio >= FUZZY_THRESHOLD else 0.0


def search_items(
    query: str,
    items: list[Item],
    limit: int = 10,
) -> list[tuple[Item, float]]:
    """Return up to ``limit`` ``(item, score)`` pairs, best first.

    Deterministic: sorted by ``(-score, normalized name)``. Items scoring
    below the fuzzy threshold are dropped; an empty query yields ``[]``.
    """
    q = norm(query)
    if not q or limit < 1:
        return []
    scored: list[tuple[Item, float]] = []
    for item in items:
        s = score_item(q, norm(item.name))
        if s > 0.0:
            scored.append((item, s))
    scored.sort(key=lambda pair: (-pair[1], norm(pair[0].name)))
    return scored[:limit]
