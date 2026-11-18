"""Automatic session-highlight images (#69) — bot_core.session_images.

Hermetic: model calls, the image pipeline and channel delivery are all
mocked; no network, no live services.  The invariants under test:

* gates (kill switch / no image model / nothing recorded) do NO work;
* ALL prompts are prepared on the resident chat model BEFORE the single
  batched paint window (no rapid model swapping);
* one failure never aborts the batch — successes still post + archive;
* an undeliverable channel never wastes the paint time (disk archive only);
* the module NEVER raises, and broken progress display never fails work.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import bot_core.images as images
import bot_core.session_images as si
from bot_core.channel_delivery import ChannelNotDeliverableError


_MOMENTS_JSON = json.dumps([
    {"title": "The Bridge", "scene": "Trixy crosses the burning bridge."},
    {"title": "Marvin Awakens", "scene": "Marvin opens his eyes in the cellar."},
    {"title": "Bubo's Miracle", "scene": "Bubo heals Romi at the shrine."},
])

# A REAL tiny PNG so the vault-convention webp conversion (_to_webp) runs for
# real in these tests; without Pillow the archive degrades to .png — the
# expected extension follows suit.
try:
    import io as _io
    from PIL import Image as _Image
    _buf = _io.BytesIO()
    _Image.new("RGB", (8, 8), (120, 60, 200)).save(_buf, "PNG")
    _PNG = _buf.getvalue()
    _EXT = "webp"
except ImportError:  # pragma: no cover - pillow is a declared dependency
    _PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    _EXT = "png"


# Moment-title slugs (what the vault filename/embed carries — the TITLE,
# never the prompt) for the three stub moments above.
_MOMENT_SLUGS = ("the_bridge", "marvin_awakens", "bubo_s_miracle")


def _moment_name(i: int) -> str:
    return f"Session_01_moment-{i}_{_MOMENT_SLUGS[i - 1]}.{_EXT}"


def _session(tmp_path) -> dict:
    # Real layout: the session folder lives INSIDE the session-log dir, whose
    # parent carries the Images/ sibling.
    d = tmp_path / "SessionLogs" / "SESSION_01_test"
    d.mkdir(parents=True)
    (d / "notes.md").write_text(
        "# Session: test\n\n- Started: 2026-01-01 10:00\n\nBody text.\n",
        encoding="utf-8")
    return {
        "id": "20260101000000",
        "name": "test",
        "dir": str(d),
        "file": str(d / "notes.md"),
        "notes": [[1, "note one"], [2, "note two"]],
        "merged_log": None,
    }


def _patch_pipeline(
        monkeypatch,
        *,
        moments_raw: str = _MOMENTS_JSON,
        fail_index: int | None = 1,
) -> list:
    """Mock selection + rewrite + batch paint; returns call recorders.

    ``fail_index``: which painted moment fails in the fake batch (None = all
    succeed).
    """
    calls: list = []

    async def fake_select(session, guild_id, channel_id):
        calls.append("select")
        return si._parse_moments(moments_raw)

    async def fake_prepare(text, raw=False, **kw):
        calls.append(f"prepare:{text[:20]}")
        return f"REWRITTEN: {text}"

    async def fake_batch(prompts, size=None, on_phase=None):
        calls.append(("batch", tuple(prompts)))
        if on_phase is not None:
            for name in ("loading", "painting 1/3"):
                res = on_phase(name)
                if hasattr(res, "__await__"):
                    await res
        return [images.ImageGenError(f"nope {i + 1}") if (fail_index is not None and i == fail_index) else _PNG
                for i in range(len(prompts))]

    monkeypatch.setattr(si, "_select_moments", fake_select)
    monkeypatch.setattr(images, "prepare_prompt", fake_prepare)
    monkeypatch.setattr(images, "generate_images", fake_batch)
    return calls


def _chan(monkeypatch, *, deliverable: bool = True):
    """Fake channel + resolve_channel; returns (chan, progress_msg)."""
    progress_msg = MagicMock(edit=AsyncMock(), delete=AsyncMock())
    chan = MagicMock()
    chan.send = AsyncMock(side_effect=[progress_msg] + [MagicMock() for _ in range(6)])
    if deliverable:
        monkeypatch.setattr(
            si, "resolve_channel", AsyncMock(return_value=(chan, "cache")))
    else:
        monkeypatch.setattr(
            si, "resolve_channel",
            AsyncMock(side_effect=ChannelNotDeliverableError(123, "gone")))
    return chan, progress_msg


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    """Feature on + image model configured (per-test opt-out available)."""
    monkeypatch.setattr(si, "SESSION_IMAGES", True)
    monkeypatch.setattr(images, "IMAGE_MODEL", "unsloth/Qwen-Image-x")


# ── Gates ────────────────────────────────────────────────────────────────

class TestGates:
    async def test_kill_switch_does_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(si, "SESSION_IMAGES", False)
        calls = _patch_pipeline(monkeypatch)
        chan, _p = _chan(monkeypatch)
        await si.generate_session_highlights(_session(tmp_path), 123)
        assert calls == []
        chan.send.assert_not_awaited()

    async def test_no_image_model_does_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(images, "IMAGE_MODEL", "")
        calls = _patch_pipeline(monkeypatch)
        chan, _p = _chan(monkeypatch)
        await si.generate_session_highlights(_session(tmp_path), 123)
        assert calls == []
        chan.send.assert_not_awaited()

    async def test_no_channel_does_nothing(self, monkeypatch, tmp_path):
        calls = _patch_pipeline(monkeypatch)
        await si.generate_session_highlights(_session(tmp_path), None)
        assert calls == []

    async def test_empty_session_does_nothing(self, monkeypatch, tmp_path):
        s = _session(tmp_path)
        s["notes"] = []
        calls = _patch_pipeline(monkeypatch)
        chan, _p = _chan(monkeypatch)
        await si.generate_session_highlights(s, 123)
        assert calls == []
        chan.send.assert_not_awaited()

    def test_will_generate_true_with_notes(self, tmp_path):
        assert si.will_generate(_session(tmp_path), 123) is True

    def test_will_generate_false_when_killed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(si, "SESSION_IMAGES", False)
        assert si.will_generate(_session(tmp_path), 123) is False


# ── Orchestration ────────────────────────────────────────────────────────

class TestOrchestration:
    async def test_happy_path_prepares_all_before_one_batch(self, monkeypatch, tmp_path):
        calls = _patch_pipeline(monkeypatch, fail_index=None)
        chan, progress_msg = _chan(monkeypatch)
        s = _session(tmp_path)

        await si.generate_session_highlights(s, 123, guild_id=None)

        # Selection once, ALL three prompts prepared BEFORE the single batch.
        assert calls[0] == "select"
        prep_idx = [i for i, c in enumerate(calls) if isinstance(c, str) and c.startswith("prepare")]
        batch_entry = next(c for c in calls if isinstance(c, tuple) and c[0] == "batch")
        batch_idx = calls.index(batch_entry)
        assert len(prep_idx) == 3
        assert max(prep_idx) < batch_idx
        # Batch got the REWRITTEN prompts (rewrite pipeline was used).
        prompts = batch_entry[1]
        assert all(p.startswith("REWRITTEN:") for p in prompts)
        assert len(prompts) == 3
        # Progress message edited + deleted; three captioned images posted.
        progress_msg.edit.assert_awaited()
        progress_msg.delete.assert_awaited_once()
        sent = [c.kwargs for c in chan.send.await_args_list]
        image_sends = [k for k in sent if "file" in k]
        assert len(image_sends) == 3
        # All three moments archived in the vault Images/ folder (sibling of
        # the session-log dir) under the Session_NN convention, and the posted
        # files carry the SAME names.
        imgs = tmp_path / "Images"
        assert sorted(p.name for p in imgs.iterdir()) == [
            _moment_name(1), _moment_name(2), _moment_name(3)]
        assert [k["file"].filename for k in image_sends] == [
            _moment_name(1), _moment_name(2), _moment_name(3)]
        # … and embedded in notes.md under the ## Momente section.
        notes = (tmp_path / "SessionLogs" / "SESSION_01_test" / "notes.md").read_text(encoding="utf-8")
        assert "## Momente" in notes and "Body text." in notes
        assert all(f"![[{_moment_name(i)}]]" in notes for i in (1, 2, 3))

    async def test_no_usable_moments_posts_friendly_note(self, monkeypatch, tmp_path):
        calls = _patch_pipeline(monkeypatch, moments_raw="I cannot pick moments.")
        chan, _p = _chan(monkeypatch)
        await si.generate_session_highlights(_session(tmp_path), 123)
        assert "batch" not in calls
        texts = [c.args[0] for c in chan.send.await_args_list if c.args]
        assert any("key moments" in t for t in texts)

    async def test_partial_failure_posts_successes_and_notes_failures(
            self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)  # fake_batch fails moment 2 only
        chan, _p = _chan(monkeypatch)
        s = _session(tmp_path)

        await si.generate_session_highlights(s, 123)

        sent = [c.kwargs for c in chan.send.await_args_list]
        image_sends = [k for k in sent if "file" in k]
        assert len(image_sends) == 2  # moments 1 + 3 posted
        texts = [c.args[0] for c in chan.send.await_args_list if c.args]
        assert any("Could not paint" in t and "Marvin Awakens" in t for t in texts)
        imgs = tmp_path / "Images"
        # Only the successful moments are archived.
        assert sorted(p.name for p in imgs.iterdir()) == [_moment_name(1), _moment_name(3)]
        notes = (tmp_path / "SessionLogs" / "SESSION_01_test" / "notes.md").read_text(encoding="utf-8")
        assert f"![[{_moment_name(1)}]]" in notes
        assert f"![[{_moment_name(2)}]]" not in notes

    async def test_whole_window_failure_reports_gracefully(self, monkeypatch, tmp_path):
        """A failed GPU MODE SWITCH (generate_images itself raises) must degrade
        to the friendly total-failure note — never kill the background task."""
        _patch_pipeline(monkeypatch)

        async def window_down(prompts, size=None, on_phase=None):
            raise images.ImageGenError("gpu-switch helper unreachable")

        monkeypatch.setattr(images, "generate_images", window_down)
        chan, _p = _chan(monkeypatch)

        await si.generate_session_highlights(_session(tmp_path), 123)

        texts = [c.args[0] for c in chan.send.await_args_list if c.args]
        assert any("None of the highlights" in t for t in texts)

    async def test_total_failure_posts_single_note(self, monkeypatch, tmp_path):
        calls = _patch_pipeline(monkeypatch)

        async def all_fail(prompts, size=None, on_phase=None):
            calls.append("batch")
            return [images.ImageGenError("down")] * len(prompts)

        monkeypatch.setattr(images, "generate_images", all_fail)
        chan, _p = _chan(monkeypatch)

        await si.generate_session_highlights(_session(tmp_path), 123)

        texts = [c.args[0] for c in chan.send.await_args_list if c.args]
        assert any("None of the highlights" in t for t in texts)
        assert not (tmp_path / "Images").exists()

    async def test_undeliverable_channel_still_paints_and_archives(
            self, monkeypatch, tmp_path):
        calls = _patch_pipeline(monkeypatch)
        _chan(monkeypatch, deliverable=False)
        s = _session(tmp_path)

        await si.generate_session_highlights(s, 123)  # must NOT raise

        assert any(isinstance(c, tuple) and c[0] == "batch" for c in calls)  # paint time not wasted
        imgs = tmp_path / "Images"
        assert len(list(imgs.iterdir())) == 2  # the two successful moments

    async def test_broken_progress_never_fails_the_batch(self, monkeypatch, tmp_path):
        _patch_pipeline(monkeypatch)
        chan = MagicMock()
        chan.send = AsyncMock(side_effect=RuntimeError("discord down"))
        monkeypatch.setattr(
            si, "resolve_channel", AsyncMock(return_value=(chan, "cache")))

        await si.generate_session_highlights(_session(tmp_path), 123)  # must NOT raise


# ── Selection parsing ──────────────────────────────────────────────

class TestArchiveConvention:
    """Vault image convention: Images/ sibling folder, Session_NN_ prefix,
    webp conversion, ## Momente embeds in notes.md."""

    def test_archives_webp_and_embeds_once_idempotent(self, tmp_path):
        s = _session(tmp_path)
        first = si._archive(s, 1, _PNG, "The Bridge")
        again = si._archive(s, 1, _PNG, "The Bridge")
        assert first is not None and again is not None
        assert first[1] == again[1] == _moment_name(1)
        assert sorted(p.name for p in (tmp_path / "Images").iterdir()) == [_moment_name(1)]
        notes = (tmp_path / "SessionLogs" / "SESSION_01_test" / "notes.md").read_text(encoding="utf-8")
        assert notes.count("![[") == 1  # the re-archive did NOT duplicate
        assert notes.count("## Momente") == 1
        if _EXT == "webp":  # the archived bytes really are webp now
            assert (tmp_path / "Images" / _moment_name(1)).read_bytes()[:4] == b"RIFF"

    def test_webp_unavailable_degrades_to_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(si, "_to_webp", lambda png: None)
        s = _session(tmp_path)
        out = si._archive(s, 2, _PNG)
        assert out is not None and out[1] == "Session_01_moment-2.png"
        assert out[0] == _PNG

    def test_title_lands_in_the_filename_not_the_prompt(self, tmp_path):
        """Filename carries a slug of the SHORT title; umlauts/apostrophes are
        folded safely; a missing title keeps the bare moment-<i> name."""
        s = _session(tmp_path)
        out = si._archive(s, 1, _PNG, "Waldbrände in Alderheart!")
        assert out is not None and out[1] == f"Session_01_moment-1_waldbrände_in_alderheart.{_EXT}"
        bare = si._archive(s, 2, _PNG, "   ")
        assert bare is not None and bare[1] == f"Session_01_moment-2.{_EXT}"
        notes = (tmp_path / "SessionLogs" / "SESSION_01_test" / "notes.md").read_text(encoding="utf-8")
        assert f"![[Session_01_moment-1_waldbrände_in_alderheart.{_EXT}]]" in notes
        # slug length is capped — a runaway title never makes a mega-filename
        long = si._slugify_title("word " * 30)
        assert len(long) <= si._TITLE_SLUG_MAX

    def test_legacy_folder_name_still_numbered(self, tmp_path):
        s = _session(tmp_path)
        old = tmp_path / "SessionLogs" / "2026-01-01_07_old"
        old.mkdir(parents=True)
        (old / "notes.md").write_text("# Session: old\n", encoding="utf-8")
        s["dir"], s["file"] = str(old), str(old / "notes.md")
        out = si._archive(s, 1, _PNG)
        assert out is not None and out[1] == f"Session_07_moment-1.{_EXT}"
        assert (tmp_path / "Images" / f"Session_07_moment-1.{_EXT}").is_file()

    def test_images_dir_override_setting(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.SESSION_IMAGES_DIR",
                            str(tmp_path / "custom" / "img"))
        s = _session(tmp_path)
        assert si._archive(s, 1, _PNG, "The Bridge") is not None
        assert (tmp_path / "custom" / "img" / _moment_name(1)).is_file()
        assert not (tmp_path / "Images").exists()

    def test_missing_notes_file_is_created_with_the_embed(self, tmp_path):
        s = _session(tmp_path)
        (tmp_path / "SessionLogs" / "SESSION_01_test" / "notes.md").unlink()
        assert si._archive(s, 1, _PNG, "The Bridge") is not None
        notes = (tmp_path / "SessionLogs" / "SESSION_01_test" / "notes.md").read_text(encoding="utf-8")
        assert f"![[{_moment_name(1)}]]" in notes


# ── Selection parsing ────────────────────────────────────────────────────

class TestParseMoments:
    def test_plain_json_array(self):
        out = si._parse_moments(_MOMENTS_JSON)
        assert out is not None and len(out) == 3
        assert out[0] == {"title": "The Bridge",
                          "scene": "Trixy crosses the burning bridge."}

    def test_code_fences_and_prose_stripped(self):
        raw = 'Sure! Here are the moments:\n```json\n' + _MOMENTS_JSON + '\n```\nHope that helps!'
        assert si._parse_moments(raw) is not None

    def test_bare_strings_accepted_as_scenes(self):
        out = si._parse_moments('["Just a scene description"]')
        assert out == [{"title": "Just a scene description",
                        "scene": "Just a scene description"}]

    def test_truncated_to_cap(self):
        data = json.dumps([{"title": f"t{i}", "scene": f"s{i}"} for i in range(5)])
        assert len(si._parse_moments(data, max_moments=3)) == 3

    def test_empty_entries_dropped(self):
        out = si._parse_moments('[{"title": "", "scene": "   "}, {"title": "ok", "scene": "s"}]')
        assert out == [{"title": "ok", "scene": "s"}]

    @pytest.mark.parametrize("raw", [
        "", "no array here", "[not json", '{"title": "single object"}', "null",
    ])
    def test_unusable_returns_none(self, raw):
        assert si._parse_moments(raw) is None


# ── Selection call (model layer) ────────────────────────────────────────

class TestSelectMoments:
    async def test_happy_path_uses_session_sources(self, monkeypatch, tmp_path):
        captured = {}

        async def fake_complete(client, **kwargs):
            captured.update(kwargs)
            return _MOMENTS_JSON

        monkeypatch.setattr(si, "complete_text", fake_complete)
        monkeypatch.setattr(si, "_validate_model", AsyncMock(return_value="model-x"))
        monkeypatch.setattr(si, "_make_client", lambda: MagicMock())
        monkeypatch.setattr(si, "get_character", lambda key: None)

        s = _session(tmp_path)
        s["dir"] = str(tmp_path / "no_docs")  # no .attachments/.transcripts
        out = await si._select_moments(s, guild_id=None, channel_id=123)

        assert out is not None and len(out) == 3
        user_content = captured["messages"][1]["content"]
        assert "Session name: test" in user_content
        assert "note one" in user_content

    @staticmethod
    def _bare_session() -> dict:
        return {"id": "x", "name": "t", "dir": "/nonexistent/session",
                "file": "/nonexistent/notes.md", "notes": [[1, "n"]],
                "merged_log": None}

    async def test_backend_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            si, "complete_text", AsyncMock(side_effect=RuntimeError("backend down")))
        monkeypatch.setattr(si, "_validate_model", AsyncMock(return_value="model-x"))
        monkeypatch.setattr(si, "_make_client", lambda: MagicMock())
        monkeypatch.setattr(si, "get_character", lambda key: None)

        assert await si._select_moments(self._bare_session(), None, 123) is None

    async def test_no_model_returns_none(self, monkeypatch):
        monkeypatch.setattr(si, "_validate_model", AsyncMock(return_value=""))
        monkeypatch.setattr(si, "_make_client", lambda: MagicMock())
        assert await si._select_moments(self._bare_session(), None, 123) is None


# ── Never raises (entry-point contract) ─────────────────────────────────

class TestNeverRaises:
    async def test_unexpected_internal_error_is_swallowed(self, monkeypatch, tmp_path):
        _chan(monkeypatch)

        async def boom(session, guild_id, channel_id):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(si, "_select_moments", boom)
        await si.generate_session_highlights(_session(tmp_path), 123)  # must NOT raise
