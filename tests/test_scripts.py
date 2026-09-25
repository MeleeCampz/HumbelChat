"""Regression tests for P1 #5 and P1 #14 (shell scripts).

P1 #5  — ``./botctl.sh logs`` used to call the nonexistent ``tmux follow``
         command and error out.  Now: a TTY gets ``tmux attach`` (interactive
         live output) and a piped/redirected call gets a one-shot
         ``tmux capture-pane`` dump of the last 2000 lines.

P1 #14 — ``start_bot.sh``'s already-running message pointed at
         ``./stop_bot.sh``, which does not exist.  Stop is ``./botctl.sh stop``.

These tests copy the scripts into a temp dir (so ``SCRIPT_DIR``/``.bot.pid``
point at the temp dir, never the repo) and put a stub ``tmux`` on PATH that
records every call it receives.
"""
from __future__ import annotations

import os

try:
    import pty  # Unix only
except ImportError:  # pragma: no cover - Windows
    pty = None
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    pty is None, reason="requires pty (Unix only; botctl/start_bot are shell scripts)"
)

REPO = Path(__file__).resolve().parent.parent
BOTCTL = REPO / "botctl.sh"
START_BOT = REPO / "start_bot.sh"

STUB_TMUX = """#!/usr/bin/env bash
# Stub tmux for tests — records argv, emulates a live session.
echo "${*}" >> "$STUB_TMUX_LOG"
case "$1" in
    has-session)  exit "${STUB_HAS_SESSION:-0}" ;;
    attach)       echo "ATTACHED-TO-SESSION"; exit 0 ;;
    capture-pane) echo "STUB-CAPTURE: fake pane line 1"; echo "STUB-CAPTURE: fake pane line 2"; exit 0 ;;
    follow)       echo "UNEXPECTED: tmux follow was invoked" >&2; exit 64 ;;
    *)            exit 0 ;;
esac
"""


def _setup(tmp_path: Path) -> dict[str, Path]:
    """Copy the scripts into *tmp_path* and install the stub tmux on PATH."""
    shutil.copy(BOTCTL, tmp_path / "botctl.sh")
    shutil.copy(START_BOT, tmp_path / "start_bot.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "tmux"
    stub.write_text(STUB_TMUX)
    stub.chmod(0o755)
    (tmp_path / ".stub_tmux.log").touch()
    return {
        "botctl": tmp_path / "botctl.sh",
        "start": tmp_path / "start_bot.sh",
        "stub_log": tmp_path / ".stub_tmux.log",
        "env": {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "STUB_TMUX_LOG": str(tmp_path / ".stub_tmux.log"),
        },
    }


def _dead_pid() -> int:
    """A PID that is certainly not running (kill -0 fails with ESRCH)."""
    for pid in (4194303, 4194302, 4194301, 4194300):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
    return 4194303


# ─────────────────────────── P1 #5: botctl.sh logs ───────────────────────────

class TestBotctlLogs:

    def test_syntax_ok(self):
        for script in (BOTCTL, START_BOT):
            subprocess.run(["bash", "-n", str(script)], check=True)

    @pytest.mark.parametrize("has_session", [True, False], ids=["session-alive", "no-session"])
    def test_logs_non_tty(self, tmp_path, has_session: bool):
        """Piped (non-TTY) logs must not use `tmux follow`.

        Session alive  → one-shot capture-pane dump, exit 0.
        No session     → clear error on stderr, exit 1.
        """
        s = _setup(tmp_path)
        env = dict(s["env"])
        env["STUB_HAS_SESSION"] = "0" if has_session else "1"

        proc = subprocess.run(
            ["bash", str(s["botctl"]), "logs"],
            env=env, input="", capture_output=True, text=True, timeout=20,
        )
        calls = s["stub_log"].read_text()

        assert "follow" not in calls, "botctl.sh logs still calls the invalid `tmux follow`"
        if has_session:
            assert proc.returncode == 0
            assert "capture-pane" in calls
            assert "STUB-CAPTURE" in proc.stdout
            assert "attach" not in calls, "non-TTY logs must not try to attach"
        else:
            assert proc.returncode == 1
            assert "No bot session" in proc.stderr

    def test_logs_interactive_tty_attaches(self, tmp_path):
        """With a real TTY, logs must `tmux attach` (live interactive tail)."""
        s = _setup(tmp_path)

        master, slave = pty.openpty()
        try:
            proc = subprocess.Popen(
                ["bash", str(s["botctl"]), "logs"],
                env=s["env"],
                stdin=slave, stdout=slave, stderr=slave,
                close_fds=True,
            )
        finally:
            os.close(slave)

        out = b""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        proc.wait(timeout=10)
        os.close(master)

        calls = s["stub_log"].read_text()
        assert proc.returncode == 0
        assert "attach" in calls, "TTY logs must attach interactively"
        assert b"ATTACHED-TO-SESSION" in out
        assert "capture-pane" not in calls, "TTY logs must not do the one-shot capture"


# ───────────────────────── P1 #14: start_bot.sh stop hint ─────────────────────────

class TestStartBotStopHint:

    def test_no_stop_bot_sh_reference_anywhere(self, tmp_path):
        """Neither script may mention the nonexistent stop_bot.sh anymore.

        (The stale-PID path also must not mention it — a stale pid is cleaned
        up and startup continues, so this doubles as the stale-pid smoke test.)
        """
        s = _setup(tmp_path)
        (tmp_path / ".bot.pid").write_text(f"{_dead_pid()}\n")

        assert "stop_bot.sh" not in s["start"].read_text()
        assert "stop_bot.sh" not in s["botctl"].read_text()

        # Stale pid → guard cleans up, script proceeds to start (no stop hint).
        proc = subprocess.run(
            ["bash", str(s["start"])],
            cwd=tmp_path, env=s["env"],
            capture_output=True, text=True, timeout=20,
        )
        combined = proc.stdout + proc.stderr
        assert "stop_bot.sh" not in combined
        assert not (tmp_path / ".bot.pid").exists(), "stale pid file must be cleaned up"

    def test_running_pid_message_same_hint(self, tmp_path):
        """With a LIVE pid the guard must fire with the same correct hint."""
        s = _setup(tmp_path)
        # os.getpid() is alive — the guard path is identical (kill -0 succeeds).
        (tmp_path / ".bot.pid").write_text(f"{os.getpid()}\n")

        proc = subprocess.run(
            ["bash", str(s["start"])],
            cwd=tmp_path, env=s["env"],
            capture_output=True, text=True, timeout=20,
        )

        combined = proc.stdout + proc.stderr
        assert "already running" in combined
        assert "./botctl.sh stop" in combined
        assert "stop_bot.sh" not in combined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
