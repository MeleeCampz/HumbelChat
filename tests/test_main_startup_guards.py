"""Code review §1.5 + §1.10 — startup guards.

§1.5  The single-instance lock must NOT rely on a port check (nothing ever
       bound port 18765).  It should use only the PID file.
§1.10 Importing ``main`` in a test environment must NOT attach file log
       handlers that would pollute ``logs/bot.log`` / ``logs/dev.log``.
"""
from __future__ import annotations

import inspect
import logging
import os
import pathlib
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


# ── §1.5  No dead port check ────────────────────────────────────────────

class TestSingleInstanceLock:

    def test_enforce_single_instance_exists(self):
        import main
        assert callable(main._enforce_single_instance)

    def test_no_port_check_in_enforce_single_instance(self):
        """The source of _enforce_single_instance must not contain a
        socket connect to a hard-coded port (the dead 18765 check)."""
        import main
        src = inspect.getsource(main._enforce_single_instance)
        # A port check would require socket creation + connect.  Neither
        # should appear (the explanatory comment may mention the old port).
        assert "socket." not in src, "no socket-based port check expected"
        assert ".connect(" not in src, "no .connect() port probe expected"

    def test_pidfile_path_is_absolute(self):
        import main
        assert main.PIDFILE.is_absolute(), "PIDFILE must be an absolute path"
        assert main.PIDFILE.parent == REPO

    def test_stale_pidfile_is_cleaned(self, tmp_path, monkeypatch):
        """A PID file with a dead PID should be unlinked and the bot
        should be able to (re-)acquire the lock."""
        import main

        # Point PIDFILE at a temp file
        stale = tmp_path / ".bot.pid"
        stale.write_text(str(os.getpid() + 999999))  # almost certainly not a live PID
        monkeypatch.setattr(main, "PIDFILE", stale)

        # os.kill(pid, 0) on a non-existent PID raises ProcessLookupError
        main._enforce_single_instance()

        # After a clean acquire the PID file should contain our own PID
        assert stale.exists()
        assert int(stale.read_text()) == os.getpid()

    def test_live_pidfile_causes_exit(self, tmp_path, monkeypatch):
        """If the PID file points to a *live* process, the function
        must call sys.exit(0)."""
        import main

        live = tmp_path / ".bot.pid"
        live.write_text(str(os.getpid()))  # our own PID is alive
        monkeypatch.setattr(main, "PIDFILE", live)

        with pytest.raises(SystemExit) as exc:
            main._enforce_single_instance()
        assert exc.value.code == 0


# ── §1.10  No file-handler pollution in tests ───────────────────────────

class TestLogHygiene:

    def test_no_file_handlers_on_root(self):
        """After importing main (which conftest does via BOT_NO_LOG_FILES=1),
        the root logger must NOT have any FileHandler attached."""
        # main is already imported (conftest imports it, and we imported it
        # above).  Verify no FileHandler slipped in.
        import main  # noqa: F401  (force import)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert not file_handlers, (
            f"root logger has {len(file_handlers)} FileHandler(s) — "
            "BOT_NO_LOG_FILES should prevent this"
        )

    def test_bot_no_log_files_env_respected_in_source(self):
        """main.py source must reference BOT_NO_LOG_FILES."""
        src = (REPO / "main.py").read_text()
        assert "BOT_NO_LOG_FILES" in src

    def test_conftest_sets_bot_no_log_files(self):
        """conftest.py must set BOT_NO_LOG_FILES=1 before importing main."""
        src = (REPO / "tests" / "conftest.py").read_text()
        assert "BOT_NO_LOG_FILES" in src
