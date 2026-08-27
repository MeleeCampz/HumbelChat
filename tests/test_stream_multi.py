"""Tests for multi-message streaming with freeze-on-split.

Pins down the delivery contract:
  * every section of a reply streams in live via edits,
  * once a section's message is complete it is FROZEN — never edited again,
  * a new section starts as its own message and streams the same way,
  * no content is lost or reordered across the split.
"""
from __future__ import annotations

import re

import pytest


# ─────────────────────────── test harness ───────────────────────────

class FakeMessage:
    """Records every edit so tests can assert on edit history."""

    _ids = iter(range(10_000, 20_000))

    def __init__(self, content: str) -> None:
        self.id = next(self._ids)
        self.content = content
        self.edits: list[str] = []   # every edit() call, in order

    async def edit(self, content=None, **kw):
        self.content = content
        self.edits.append(content)


class FakeInteraction:
    """Records send order and returns FakeMessages; shares an event log."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.sent: list[FakeMessage] = []

    async def followup_send(self, content="", **kw):
        msg = FakeMessage(str(content))
        self.sent.append(msg)
        self._events.append(f"send#{len(self.sent)}:{msg.content[:20]!r}")
        return msg


def make_ix(events: list[str]) -> FakeInteraction:
    ix = FakeInteraction(events)
    # StreamToDiscord calls interaction.followup.send(...)
    ix.followup = type("F", (), {"send": staticmethod(ix.followup_send)})()
    return ix


async def feed_all(stream, text: str, chunk_size: int = 37) -> None:
    """Feed *text* in small chunks and close the stream."""
    for i in range(0, len(text), chunk_size):
        await stream.feed(text[i:i + chunk_size])
    await stream.close()


def strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ─────────────────────── split helpers (pure) ───────────────────────

class TestGreedySplit:

    def test_short_text_single_part(self):
        from utils.stream_response import _greedy_split
        assert _greedy_split("hello world") == ["hello world"]

    def test_empty_text_no_parts(self):
        from utils.stream_response import _greedy_split
        assert _greedy_split("") == []

    def test_all_parts_within_limit(self):
        from utils.stream_response import _greedy_split, DISCORD_MSG_LIMIT
        paras = [f"Section {i} " + ("word " * 60) for i in range(20)]
        text = "\n\n".join(paras)
        parts = _greedy_split(text)
        assert len(parts) > 1
        for p in parts:
            assert len(p) <= DISCORD_MSG_LIMIT

    def test_no_newline_hard_split(self):
        from utils.stream_response import _greedy_split, DISCORD_MSG_LIMIT
        text = "a" * 5000
        parts = _greedy_split(text)
        assert len(parts) == 3
        assert [len(p) for p in parts] == [2000, 2000, 1000]

    def test_split_prefers_paragraph_boundary(self):
        from utils.stream_response import _split_for_first
        # \n\n sits past the half-window mark → must be chosen over a line break.
        text = "A" * 1500 + "\n\n" + "B" * 1500
        head, rest = _split_for_first(text, 2000)
        assert len(head) <= 2000
        assert head == "A" * 1500
        assert rest.startswith("B")


# ─────────────────── short reply: one message ───────────────────────

class TestShortReply:

    @pytest.mark.asyncio
    async def test_single_message_streams_then_freezes(self):
        from utils.stream_response import StreamToDiscord
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)

        text = "A short answer. " * 20  # ~300 chars, well under the limit
        await feed_all(stream, text)

        assert len(ix.sent) == 1
        msg = ix.sent[0]
        assert msg.content == text
        assert msg.edits, "the message should have grown via at least one edit"
        # The last thing on record for this message is its final content.
        assert msg.edits[-1] == text


# ─────────────── long reply: freeze + new streamed section ──────────

class TestSplitFreeze:

    def _long_reply(self) -> str:
        # ~2500 chars with clear paragraph boundaries → exactly two sections.
        paras = [f"**Section {i}**\n" + ("Some lore text about Humblewood. " * 8)
                 for i in range(9)]
        return "\n\n".join(paras)

    @pytest.mark.asyncio
    async def test_two_messages_first_never_edited_after_split(self):
        from utils.stream_response import StreamToDiscord
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)
        text = self._long_reply()

        await feed_all(stream, text)

        assert len(ix.sent) == 2
        first, second = ix.sent
        # Both messages carry their final section content.
        assert first.content and second.content
        assert strip_ws(first.content + second.content) == strip_ws(text)

        # The split point: first message must NOT contain the last section's
        # text, second must not repeat the first's content.
        assert "Section 8" not in first.content
        assert "Section 0" not in second.content

    @pytest.mark.asyncio
    async def test_first_message_frozen_before_second_starts(self):
        """Core contract: no edit of message 1 after message 2 was sent."""
        from utils.stream_response import StreamToDiscord
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)
        text = self._long_reply()

        # Spy on the first message's edit(): any call after msg 2 is sent is a
        # contract violation.
        state = {"first": None, "second_sent": False}
        violations: list[str] = []
        real_send = ix.followup.send

        async def send_spy(content="", **kw):
            msg = await real_send(content=content, **kw)
            if state["first"] is None:
                orig_edit = msg.edit

                async def edit_spy(content=None, **kw2):
                    if state["second_sent"]:
                        violations.append(str(content))
                    await orig_edit(content=content, **kw2)

                msg.edit = edit_spy
                state["first"] = msg
            else:
                state["second_sent"] = True
            return msg

        ix.followup.send = send_spy
        ix.followup = type("F", (), {"send": staticmethod(send_spy)})()

        await feed_all(stream, text)

        assert len(ix.sent) == 2
        assert violations == [], (
            f"message 1 was edited after message 2 started: {violations[:2]}"
        )
        # Message 1 ends at its frozen (final section) content.
        first = ix.sent[0]
        if first.edits:
            assert first.edits[-1] == first.content

    @pytest.mark.asyncio
    async def test_small_tail_section_starts_immediately(self):
        """A continuation section under MIN_FLUSH_CHARS must still get its own
        live message right away (no stub suppression for sections 2+)."""
        from utils.stream_response import StreamToDiscord, DISCORD_MSG_LIMIT
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)

        # ~1982 chars (fits msg 1) + a short tail → two messages.
        body = "A" * 1500 + "\n\n" + "B" * 480   # 1982 total, single message so far
        text = body + "\n\nTail section, quite short."
        assert len(body) <= DISCORD_MSG_LIMIT < len(text)

        await feed_all(stream, text, chunk_size=250)

        assert len(ix.sent) == 2
        assert ix.sent[1].content == "Tail section, quite short."

    @pytest.mark.asyncio
    async def test_single_huge_chunk_splits_without_stub(self):
        """Whole reply arrives in one chunk → msg1 sent complete+frozen,
        msg2 started; no 30-char stub message is ever posted."""
        from utils.stream_response import StreamToDiscord
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)
        text = self._long_reply()

        await stream.feed(text)   # everything at once
        await stream.close()

        assert len(ix.sent) == 2
        first, second = ix.sent
        assert first.edits == [], "msg1 was sent complete — it never needed an edit"
        assert strip_ws(first.content + second.content) == strip_ws(text)

    @pytest.mark.asyncio
    async def test_three_sections(self):
        """~4500 chars → three messages, each frozen in order."""
        from utils.stream_response import StreamToDiscord
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)

        paras = [f"**S{i}**\n" + ("Lore about Humblewood. " * 8) for i in range(25)]
        text = "\n\n".join(paras)
        await feed_all(stream, text)

        assert len(ix.sent) == 3
        joined = strip_ws("".join(m.content for m in ix.sent))
        assert joined == strip_ws(text)
        # First paragraph only in msg 1; last paragraph only in the final msg.
        assert "S0" in ix.sent[0].content and "S0" not in ix.sent[1].content
        assert "S24" in ix.sent[2].content and "S24" not in ix.sent[1].content

    @pytest.mark.asyncio
    async def test_failed_freeze_edit_retries_next_chunk(self):
        """A failed edit must not lose content or corrupt the split."""
        from utils.stream_response import StreamToDiscord
        events: list[str] = []
        ix = make_ix(events)
        stream = StreamToDiscord(ix)
        text = self._long_reply()

        # Deterministic sabotage: make the first message's edit fail exactly
        # once, then succeed — content must survive and the split stay intact.
        calls = {"fail": 0}
        first = None
        real_send = ix.followup.send

        async def send_once(content="", **kw):
            nonlocal first
            msg = await real_send(content=content, **kw)
            if first is None:
                first = msg
                orig_edit = msg.edit

                async def flaky_edit(content=None, **kw2):
                    calls["fail"] = calls.get("fail", 0) + 1
                    if calls["fail"] == 1:
                        raise Exception("simulated rate limit")
                    await orig_edit(content=content, **kw2)

                msg.edit = flaky_edit
            return msg

        ix.followup.send = send_once
        # rewire the staticmethod shim used by StreamToDiscord
        ix.followup = type("F", (), {"send": staticmethod(send_once)})()

        await feed_all(stream, text)

        assert len(ix.sent) == 2
        joined = strip_ws("".join(m.content for m in ix.sent))
        assert joined == strip_ws(text), "content must survive a failed edit"


# ─────────────────── wrapper behaviour (regression) ─────────────────

class TestWrapper:

    @pytest.mark.asyncio
    async def test_returns_full_text_across_split(self):
        from utils.stream_response import stream_ai_response
        events: list[str] = []
        ix = make_ix(events)

        paras = [f"**S{i}**\n" + ("Lore. " * 40) for i in range(12)]
        text = "\n\n".join(paras)

        async def chunks():
            for i in range(0, len(text), 53):
                yield text[i:i + 53]

        result = await stream_ai_response(ix, chunks())
        assert result == text
        assert strip_ws("".join(m.content for m in ix.sent)) == strip_ws(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
