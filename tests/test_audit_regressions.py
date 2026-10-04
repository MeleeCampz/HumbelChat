"""Regression tests for the 2026-07 full-repo audit (see BUG_REPORT.md).

Each test locks in one fixed bug so it can never silently return:

  H1  sessions: unnamed-session folders were invisible to the per-day index
      counter → same-day collision overwrote the previous notes.md.
  H2  history: a persisted "None" guild key (DM turns) aborted the ENTIRE
      load, dropping later channels and all active-character selections.
  M1  voice recovery: real frame logs are ``Name_123.wav.log``; the old
      regex never matched → every speaker became user 0 (one shared WAV).
  M2  kb storage: the UUID-prefix stripper ate any leading all-hex word
      ("cafe_notes.md" → "notes.md", "add_this.md" → "this.md").
  M3  errors: every HTTP 400 was reported as "model not found".
  M4  reminders: fired entries + per-id locks were never pruned (unbounded
      growth); rearm now also prunes legacy ``fired: true`` tombstones.
  M5  kb index: force-rebuild deleted the live on-disk cache up front — a
      failed rebuild left nothing; now the old cache survives until the new
      one is published via the atomic temp-file swap.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bot_core import history as H
from bot_core import reminders as R
from bot_core import sessions as S
from kb.storage import _sanitize_filename


# ──────────────────────────── H1: sessions ───────────────────────────────

class TestSessionIndexUnnamedFolders:
    @pytest.fixture(autouse=True)
    def _notes(self, tmp_path, monkeypatch):
        self.notes = tmp_path / "notes"
        self.notes.mkdir()
        monkeypatch.setattr(S, "notes_dir", lambda: self.notes, raising=False)

    def _today_prefix(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def test_unnamed_folders_are_counted(self):
        p = self._today_prefix()
        (self.notes / f"{p}_01").mkdir()
        (self.notes / f"{p}_02").mkdir()
        assert S._next_session_index(datetime.now()) == 3

    def test_named_and_unnamed_mixed(self):
        p = self._today_prefix()
        (self.notes / f"{p}_01_quest").mkdir()
        (self.notes / f"{p}_02").mkdir()
        assert S._next_session_index(datetime.now()) == 3

    def test_two_unnamed_sessions_same_day_keep_separate_notes(self):
        """The exact data-loss scenario: start → end → start again, no name."""
        s1, _ = S.start_session()
        S.end_session()  # end_session rewrites notes.md — write AFTER it
        f1 = Path(s1["file"])
        f1.write_text("FIRST SESSION NOTES", encoding="utf-8")
        s2, _ = S.start_session()
        f2 = Path(s2["file"])
        assert f1 != f2, "second unnamed session must not reuse the folder"
        assert f1.read_text(encoding="utf-8") == "FIRST SESSION NOTES"
        idx1 = int(f1.parent.name.split("_")[1])
        idx2 = int(f2.parent.name.split("_")[1])
        assert idx2 == idx1 + 1


# ──────────────────────────── H2: history ────────────────────────────────

class TestHistoryLoadWithNoneKeys:
    @pytest.fixture
    def persist_path(self, tmp_path, monkeypatch):
        p = tmp_path / "history.json"
        monkeypatch.setenv("HISTORY_PERSIST_FILE", str(p))
        monkeypatch.setattr(H, "_persist_path", None, raising=False)
        H._chat_history.clear()
        H._active_characters.clear()
        yield p
        monkeypatch.setattr(H, "_persist_path", None, raising=False)
        H._chat_history.clear()
        H._active_characters.clear()

    def test_none_guild_key_loads_everything_after_it(self, persist_path):
        """A DM turn persisted under guild key "None" used to abort the whole
        load (int("None") raised into the outer handler)."""
        payload = {
            "history": {
                "None": {"500": [{"role": "user", "content": "dm turn"}]},
                "111": {"222": [{"role": "user", "content": "guild turn"}]},
            },
            "active_characters": {
                "None:500": "Dana",
                "111:222": "Trixy",
            },
        }
        persist_path.write_text(json.dumps(payload), encoding="utf-8")

        H.load_persisted()

        # Everything loaded — nothing after the "None" key may be dropped.
        assert H.get_history(None, 500) == [{"role": "user", "content": "dm turn"}]
        assert H.get_history(111, 222) == [{"role": "user", "content": "guild turn"}]
        # get_active_char_key() intentionally refuses None-guild lookups, so
        # assert on the map itself — the point is the entry LOADED (it used to
        # be dropped along with everything else after the "None" key).
        assert H._active_characters.get((None, 500)) == "Dana"
        assert H.get_active_char_key(111, 222) == "Trixy"

    def test_roundtrip_with_dm_turn(self, persist_path):
        """Save → wipe memory → load must be symmetric for DM (None) keys."""
        H.set_history(None, 500, [{"role": "user", "content": "dm"}])
        H._active_characters[(None, 500)] = "Dana"  # set_active_char_key skips None guilds
        H._save_to_disk()
        H._chat_history.clear()
        H._active_characters.clear()
        H.load_persisted()
        assert H.get_history(None, 500) == [{"role": "user", "content": "dm"}]
        assert H._active_characters.get((None, 500)) == "Dana"


# ──────────────────────── M1: voice recovery ─────────────────────────────

class TestRecoveryLogNameParsing:
    def test_real_runtime_log_name(self):
        from bot_core.voice.recover import _user_id_from_log_name
        # Name produced by VoiceRecorder._log_path_for: <safe>_<uid>.wav.log
        assert _user_id_from_log_name("Alice_1234567890.wav.log") == 1234567890

    def test_legacy_shape_still_parses(self):
        from bot_core.voice.recover import _user_id_from_log_name
        assert _user_id_from_log_name("Bob_777.log") == 777

    def test_multi_speaker_recovery_writes_one_wav_per_user(self, tmp_path):
        """Two speakers with REAL log names must not collapse into user-0."""
        from bot_core.voice.capture import FRAME_SAMPLES, _SpeakerLog
        from bot_core.voice.recover import recover_orphans

        rec_dir = tmp_path / "recording_20260101-000000"
        rec_dir.mkdir()
        started = time.time() - 3600
        (rec_dir / ".recording").write_text(json.dumps(
            {"started_at": started, "guild_id": 1, "channel_id": 2,
             "channel_name": "room", "pid": 1}))
        for name, uid in (("Alice_111.wav.log", 111), ("Bob_222.wav.log", 222)):
            lg = _SpeakerLog(rec_dir / name)
            t = started + 1.0
            for _ in range(5):
                lg.write_frame(t, b"\x07\x08" * (FRAME_SAMPLES // 2))
                t += 0.02
            lg.close()

        recovered = recover_orphans(tmp_path)
        assert len(recovered) == 1
        speakers = recovered[0]["speakers"]
        assert {s["user_id"] for s in speakers} == {111, 222}
        wavs = sorted(s["wav_file"] for s in speakers)
        assert wavs == ["user-111_111.wav", "user-222_222.wav"]  # _wav_filename("", uid)
        for s in speakers:
            assert (rec_dir / s["wav_file"]).exists()


# ─────────────────────── M2: filename sanitizer ──────────────────────────

class TestSanitizeFilenameHexWords:
    def test_hex_like_first_words_are_kept(self):
        # "cafe", "add", "bad", "face", "dead", "beef" are all valid hex —
        # the old stripper removed them.
        assert _sanitize_filename("cafe_notes.md") == "cafe_notes.md"
        assert _sanitize_filename("add_this.md") == "add_this.md"
        assert _sanitize_filename("dead_end.txt") == "dead_end.txt"

    def test_real_uuid_prefix_still_stripped(self):
        u32 = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        assert _sanitize_filename(f"{u32}_notes.md") == "notes.md"

    def test_leading_underscore_kept(self):
        assert _sanitize_filename("_leading_underscore.md") == "_leading_underscore.md"


# ─────────────────── M3: HTTP 400 classification ─────────────────────────

class TestBadRequestClassification:
    def _exc(self, status: int, msg: str) -> Exception:
        class FakeAPIError(Exception):
            response = type("Response", (), {"status_code": status})

        return FakeAPIError(msg)

    def test_context_length_400_is_not_model_not_found(self):
        from bot_core.errors import BadRequestError, classify_ai_error
        exc = self._exc(400, "context_length_exceeded: this model's maximum context length is 8192 tokens")
        classified = classify_ai_error(exc, model="gemma")
        assert isinstance(classified, BadRequestError)
        assert classified.category == "bad_request"
        assert "context" in classified.user_message.lower()

    def test_400_mentioning_model_is_still_model_not_found(self):
        from bot_core.errors import ModelNotFoundError, classify_ai_error
        classified = classify_ai_error(self._exc(400, "invalid model name"), model="bad-model")
        assert isinstance(classified, ModelNotFoundError)

    def test_404_is_model_not_found(self):
        from bot_core.errors import ModelNotFoundError, classify_ai_error
        classified = classify_ai_error(self._exc(404, "no such route"), model="m")
        assert isinstance(classified, ModelNotFoundError)


# ─────────────────── M4: reminder store pruning ──────────────────────────

class TestReminderTombstonePruning:
    @pytest.fixture(autouse=True)
    def _store(self, tmp_path, monkeypatch):
        self.path = tmp_path / "reminders.json"
        monkeypatch.setenv("REMINDERS_PERSIST_FILE", str(self.path))
        monkeypatch.setattr(R, "_store_path", None, raising=False)
        R._reminders.clear()
        R._tasks.clear()
        yield
        R._reminders.clear()
        R._tasks.clear()

    def test_rearm_prunes_fired_tombstones(self):
        R._reminders["old1"] = {
            "channel_id": 1, "message": "x", "delay_sec": 60,
            "fires_at": time.time() - 300, "created_at": time.time() - 400,
            "fired": True,
        }
        R._reminders["live"] = {
            "channel_id": 2, "message": "y", "delay_sec": 60,
            "fires_at": time.time() + 300, "created_at": time.time(),
            "fired": False,
        }
        R._save()  # persist both, as an older build would
        count = R.rearm_pending_reminders()
        assert count == 1
        assert "old1" not in R._reminders, "tombstone must be pruned from memory"
        payload = json.loads(self.path.read_text())
        assert "old1" not in payload, "pruning must persist to disk"
        assert "live" in payload


# ─────────── M5: force-rebuild keeps the old cache until success ─────────

class _FakeEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def encode(self, texts):
        if self.fail:
            raise RuntimeError("embedding backend down (simulated)")
        import hashlib
        out = []
        for t in texts:
            d = hashlib.sha256(t.encode()).digest()[:8]
            out.append([b / 255.0 for b in d])
        return out


class TestForceRebuildCacheSafety:
    @pytest.fixture
    def kb_dir(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "doc.md").write_text(
            "# Doc\n\nSome sufficiently long text to be worth embedding into a vector cache."
        )
        return kb

    async def test_failed_force_rebuild_preserves_old_cache(self, tmp_path, kb_dir):
        from kb.index import KBIndexStore

        persist = tmp_path / "cache"
        store = KBIndexStore(kb_dir, persist_dir=persist, model_name="test-model")
        store._embedder = _FakeEmbedder()
        await store.load()
        db = store._db_path
        assert db.exists()
        rows_before = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM document_index").fetchone()[0]
        assert rows_before > 0

        # Second store, same persist dir, embedding backend DOWN.
        store2 = KBIndexStore(kb_dir, persist_dir=persist, model_name="test-model")
        store2._embedder = _FakeEmbedder(fail=True)
        idx = await store2.load(force_rebuild=True)
        assert idx.is_empty()  # nothing could be embedded

        # ...but the previous cache must survive intact (old code unlinked it).
        assert db.exists(), "failed rebuild must NOT delete the live cache"
        rows_after = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM document_index").fetchone()[0]
        assert rows_after == rows_before

    async def test_successful_force_rebuild_publishes_new_cache(self, tmp_path, kb_dir):
        from kb.index import KBIndexStore

        persist = tmp_path / "cache"
        store = KBIndexStore(kb_dir, persist_dir=persist, model_name="test-model")
        store._embedder = _FakeEmbedder()
        await store.load()
        db = store._db_path
        assert db.exists()

        store2 = KBIndexStore(kb_dir, persist_dir=persist, model_name="test-model")
        store2._embedder = _FakeEmbedder()
        idx = await store2.load(force_rebuild=True)
        assert not idx.is_empty()
        rows = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM document_index").fetchone()[0]
        assert rows > 0
        # Atomic swap leaves no stray temp files behind.
        assert not list(persist.glob("*.tmp"))


# ─────────────── L2/L4/L8: low-priority polish locks ─────────────────────

class TestLowPrioPolish:
    def test_chunker_hash_is_stable_across_processes(self):
        """hash() is PYTHONHASHSEED-randomized; _hash must not be."""
        import subprocess
        import sys
        code = (
            "from kb.chunker import Chunker;"
            "print(Chunker._hash('Some Section Header'))"
        )
        outs = set()
        for seed in ("0", "123"):
            r = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                env={"PYTHONHASHSEED": seed, "PATH": __import__("os").environ["PATH"]},
                cwd=".",
            )
            assert r.returncode == 0, r.stderr
            outs.add(r.stdout.strip())
        assert len(outs) == 1, f"_hash differs across PYTHONHASHSEED: {outs}"

    @pytest.mark.asyncio
    async def test_remind_unit_wording(self, ix):
        from commands.utility_commands import handle_remind_command
        ix.channel = MagicMock()
        ix.channel.id = 99
        await handle_remind_command(ix, time_value=30, time_unit="seconds", message="x")
        assert "**30 seconds**" in ix._sent[0], "plural must stay plural"
        ix._sent.clear()
        await handle_remind_command(ix, time_value=1, time_unit="hours", message="y")
        assert "**1 hour**" in ix._sent[0], "value 1 must be singular"

    @pytest.mark.parametrize(
        ("env", "expected"),
        [({"RERANK_TOP_K": "7"}, 7), ({"RAG_VECTOR_TOP_K": "9"}, 9)],
    )
    def test_rerank_top_k_env_alias(self, env, expected):
        """RERANK_TOP_K wins; legacy RAG_VECTOR_TOP_K still honored.

        Run in a SUBPROCESS: reloading config.settings in-process re-evaluates
        every module-level constant and leaks into sibling tests (the fail-soft
        embedding tests depend on the exact import-time settings state).
        """
        import os
        import subprocess
        import sys
        code = "import config.settings as s; print(s.RERANK_TOP_K)"
        e = {k: v for k, v in os.environ.items()
             if k not in ("RERANK_TOP_K", "RAG_VECTOR_TOP_K")}
        e.update(env)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=e, cwd=".")
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.strip()) == expected
