"""Tests for the XML history-context block (single-turn history rendering).

The bot packs stored chat history into ONE ``user``-role message rendered by
``_render_history_block`` instead of sending loose role messages.  These tests
cover pairing, username attribution, XML escaping, legacy entries without a
username, malformed input robustness, and the storage side (``_persist_turn``
recording the Discord display name).
"""
from __future__ import annotations

import pytest

from bot_core import ai_client as ac
from bot_core import history as hist_mod


# ─────────────────────── _render_history_block ───────────────────────

class TestRenderHistoryBlock:
    def test_pairs_user_assistant_turns(self):
        msgs = [
            {"role": "user", "content": "hello?", "username": "Alice"},
            {"role": "assistant", "content": "hi there"},
        ]
        out = ac._render_history_block(msgs)
        assert out.startswith("<history_context>")
        assert out.rstrip().endswith("</history_context>")
        assert '<user name="Alice">hello?</user>' in out
        assert "<assistant>hi there</assistant>" in out
        assert out.count("<turn>") == 1

    def test_header_instructs_not_to_reanswer(self):
        out = ac._render_history_block(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        assert "Do NOT re-answer" in out

    def test_legacy_entries_without_username_have_no_name_attr(self):
        # Pre-change persisted entries have no "username" key at all.
        msgs = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        out = ac._render_history_block(msgs)
        assert '<user>old question</user>' in out
        assert 'name=' not in out

    def test_empty_username_omits_attribute(self):
        out = ac._render_history_block(
            [{"role": "user", "content": "q", "username": ""},
             {"role": "assistant", "content": "a"}]
        )
        assert "<user>q</user>" in out
        assert 'name=""' not in out

    def test_xml_special_chars_escaped_in_text_and_attrs(self):
        msgs = [
            {"role": "user", "content": 'a < b & c > d "quoted"', "username": 'We"ird <Name>'},
            {"role": "assistant", "content": "<b>bold</b> & more"},
        ]
        out = ac._render_history_block(msgs)
        assert "a &lt; b &amp; c &gt; d &quot;quoted&quot;" in out
        assert 'name="We&quot;ird &lt;Name&gt;"' in out
        assert "<b>bold</b>" not in out  # assistant content escaped too

    def test_consecutive_user_messages_do_not_hang(self):
        # Regression: unpaired user messages must still advance the cursor.
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply"},
        ]
        out = ac._render_history_block(msgs)
        assert out.count("<turn>") == 2  # 'first' unpaired, 'second'+assistant paired
        assert "<user>first</user>" in out
        assert "<user>second</user>\n<assistant>reply</assistant>" in out

    def test_orphan_assistant_message_is_kept(self):
        out = ac._render_history_block([{"role": "assistant", "content": "orphan"}])
        assert "<turn>\n<assistant>orphan</assistant>\n</turn>" in out

    def test_malformed_entries_never_raise(self):
        msgs = [
            None,  # type: ignore[list-item]
            "not-a-dict",  # type: ignore[list-item]
            {"role": "user"},  # missing content
            {"role": "user", "content": 12345},  # non-str content
            {"role": "assistant", "content": None},
        ]
        out = ac._render_history_block(msgs)  # must not raise
        assert out.startswith("<history_context>")
        assert "12345" in out

    def test_multiple_turns_ordered(self):
        msgs = [
            {"role": "user", "content": "q1", "username": "A"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2", "username": "B"},
            {"role": "assistant", "content": "a2"},
        ]
        out = ac._render_history_block(msgs)
        assert out.index("q1") < out.index("a1") < out.index("q2") < out.index("a2")


# ─────────────────────── _persist_turn stores username ───────────────────────

class TestPersistTurnUsername:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        # Disable disk persistence and start from a clean in-memory store.
        monkeypatch.setenv("HISTORY_PERSIST_FILE", "")
        monkeypatch.setattr(hist_mod, "_persist_path", None)  # reset env cache
        hist_mod._chat_history.clear()
        yield
        hist_mod._chat_history.clear()

    def test_username_stored_and_markdown_stripped(self):
        ac._persist_turn(1, 2, "what is up?", "not much", username="**Melee**Chan")
        msgs = hist_mod.get_history(1, 2)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "what is up?", "username": "MeleeChan"}
        assert msgs[1] == {"role": "assistant", "content": "not much"}

    def test_empty_username_stored_as_empty_string(self):
        ac._persist_turn(1, 2, "q", "a")
        msgs = hist_mod.get_history(1, 2)
        assert msgs[0]["username"] == ""

    def test_roundtrip_through_renderer_keeps_name(self):
        ac._persist_turn(1, 2, "show the table", "here it is", username="MeleeChan")
        out = ac._render_history_block(hist_mod.get_history(1, 2))
        assert '<user name="MeleeChan">show the table</user>' in out
