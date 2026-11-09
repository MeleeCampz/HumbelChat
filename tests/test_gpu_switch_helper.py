"""scripts/gpu_switch_helper.py — the gpu-switch service internals.

Hermetic: the docker daemon and the Studio API are mocked at the module's
helper-function level. What is under test is the SWITCH LOGIC itself:
  * mode is DERIVED from live container state (no stale memory after a
    helper restart),
  * an image-mode switch is REFUSED before tearing RAG down when the image
    backend is unreachable (the 4070 must never end up with both services
    stopped by a failed switch),
  * GET /mode state reporting stays alive even when Studio is down.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_switch_helper as h  # noqa: E402

_REAL_DOCKER = h._docker                  # the autouse fixture monkeypatches h._docker
_REAL_DAEMON_REACHABLE = h._daemon_reachable  # ... and so does it


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch):
    """Neutralize the real docker daemon + Studio endpoints."""
    state = {"reranker": True, "embedder": True, "docker": [],
             "status": {"loaded": False}, "status_error": None}

    def fake_running(name):
        return state["reranker"] if name == h.RERANKER else state["embedder"]

    def fake_docker(name, action):
        state["docker"].append(f"{action}:{name}")
        state["reranker" if name == h.RERANKER else "embedder"] = (action == "start")

    def fake_status():
        if state["status_error"]:
            raise state["status_error"]
        return state["status"]

    def fake_healthy(_name):
        return True

    monkeypatch.setattr(h, "_docker_running", fake_running)
    monkeypatch.setattr(h, "_docker", fake_docker)
    monkeypatch.setattr(h, "_image_status", fake_status)
    monkeypatch.setattr(h, "_healthy", fake_healthy)
    monkeypatch.setattr(h, "_load_image_model", lambda: None)
    monkeypatch.setattr(h, "_unload_image_model", lambda: None)
    monkeypatch.setattr(h, "_backend_reachable", lambda: True)
    monkeypatch.setattr(h, "_daemon_reachable", lambda: True)
    return state


class TestModeDerivation:
    def test_live_rag_containers_mean_rag_mode(self, _fake_env):
        assert h._current_mode() == "rag"

    def test_stopped_containers_mean_image_mode(self, _fake_env):
        # A helper restart while the RAG containers are stopped must NOT
        # report a stale "rag" (that would make the bot no-op its own switch).
        _fake_env["reranker"] = _fake_env["embedder"] = False
        assert h._current_mode() == "image"

    def test_partially_stopped_still_counts_as_rag(self, _fake_env):
        _fake_env["embedder"] = False
        assert h._current_mode() == "rag"

    def test_already_in_mode_is_a_noop(self, _fake_env):
        assert "already in rag mode" in h.switch("rag")
        assert _fake_env["docker"] == []


class TestSwitchSafety:
    def test_rag_switch_starts_both_containers(self, _fake_env):
        _fake_env["reranker"] = _fake_env["embedder"] = False
        detail = h.switch("rag")
        assert "healthy" in detail
        assert _fake_env["docker"] == [f"start:{h.RERANKER}", f"start:{h.EMBEDDER}"]

    def test_image_switch_stops_both_containers(self, _fake_env, monkeypatch):
        detail = h.switch("image")
        assert "resident" in detail
        assert _fake_env["docker"] == [f"stop:{h.RERANKER}", f"stop:{h.EMBEDDER}"]

    def test_image_switch_refused_when_backend_unreachable(self, _fake_env, monkeypatch):
        """THE safety property: RAG containers are still running → a switch
        with an unreachable Studio must refuse and touch nothing."""
        monkeypatch.setattr(h, "_backend_reachable", lambda: False)
        detail = h.switch("image")
        assert "refused" in detail
        assert _fake_env["docker"] == []          # nothing stopped
        assert _fake_env["reranker"] and _fake_env["embedder"]  # rag still live

    def test_already_image_mode_is_noop_even_with_backend_down(self, _fake_env, monkeypatch):
        """Manual gpu_switch.sh overlap: containers already stopped → the mode
        IS image already, so the switch is a pure no-op (the bot then loads the
        model through its own Studio API; no precheck needed)."""
        _fake_env["reranker"] = _fake_env["embedder"] = False
        monkeypatch.setattr(h, "_backend_reachable", lambda: False)
        detail = h.switch("image")
        assert "already in image mode" in detail
        assert _fake_env["docker"] == []

    def test_unknown_mode_rejected(self, _fake_env):
        with pytest.raises(ValueError):
            h.switch("chaos")


class TestStateEndpoint:
    def test_state_reports_mode_and_containers(self, _fake_env):
        st = h.state()
        assert st["mode"] == "rag"
        assert st["docker_ok"] is True
        assert st["rag"] == {"reranker_running": True, "embedder_running": True}
        assert st["image_loaded"] is False
        assert "image_status_error" not in st

    def test_state_survives_a_down_studio(self, _fake_env, monkeypatch):
        """GET /mode must not 500 when the image backend is down: the real
        _image_status swallows the error into {} — state() must flag it."""
        monkeypatch.setattr(h, "_image_status", lambda: {})
        st = h.state()
        assert st["mode"] == "rag"
        assert st["image_loaded"] is False
        assert st["image_status_error"] is True


class TestDaemonTransport:
    """2026-11-05 live incident regression: the helper MUST speak to the
    daemon over the mounted unix socket. The original urllib version hit
    TCP :80, every call got ECONNREFUSED, _docker_running silently lied
    "not running" for live containers, mode misderived as "image" — and
    the rag heal 500'd with '<urlopen error [Errno 111] Connection refused>'."""

    def test_switch_refuses_when_daemon_unreachable(self, _fake_env, monkeypatch):
        monkeypatch.setattr(h, "_daemon_reachable", lambda: False)
        with pytest.raises(RuntimeError, match="daemon unreachable"):
            h.switch("rag")
        assert _fake_env["docker"] == []          # nothing touched

    def test_resolve_daemon_defaults_to_mounted_socket(self, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        assert h._resolve_daemon() == ("unix", "/var/run/docker.sock")

    def test_resolve_daemon_ignores_windows_npipe(self, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", "npipe:////./pipe/docker_engine")
        assert h._resolve_daemon() == ("unix", "/var/run/docker.sock")

    def test_resolve_daemon_honours_unix_and_http(self, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", "unix:///run/custom.sock")
        assert h._resolve_daemon() == ("unix", "/run/custom.sock")
        monkeypatch.setenv("DOCKER_HOST", "http://dhost:2375")
        kind, target = h._resolve_daemon()
        assert kind == "tcp" and (target.hostname, target.port) == ("dhost", 2375)

    def test_docker_req_over_a_real_unix_socket(self, monkeypatch, tmp_path):
        """End-to-end transport proof: real AF_UNIX listener + the helper's
        _docker_req against it (no mocks)."""
        import json as _json
        import socket as _sock
        import threading

        path = str(tmp_path / "docker-test.sock")
        srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)

        def serve():
            conn, _ = srv.accept()
            with conn:
                conn.recv(4096)  # we only need the request bytes
                body = b'{"APIVersion": "1.45"}'
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        t = threading.Thread(target=serve, daemon=True)
        t.start()
        monkeypatch.setattr(h, "DAEMON", ("unix", path))
        assert h._docker_req("/_ping") == {"APIVersion": "1.45"}
        srv.close()

    def test_plain_text_ping_body_is_not_a_failure(self, monkeypatch, tmp_path):
        """Live 2026-11-09 regression: the real daemon answers /_ping with
        text/plain 'OK'. The JSON-only reader raised JSONDecodeError, so
        _daemon_reachable() reported a HEALTHY daemon as unreachable and switch()
        refused everything — docker_ok:false while container lookups worked.
        A 2xx body must count as reachable whatever its content type."""
        import socket as _sock
        import threading

        path = str(tmp_path / "ping-text.sock")
        srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        srv.bind(path)
        srv.listen(3)
        srv.settimeout(5)

        def serve():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                with conn:
                    conn.recv(4096)
                    body = b"OK"
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)

        threading.Thread(target=serve, daemon=True).start()
        monkeypatch.setattr(h, "DAEMON", ("unix", path))
        monkeypatch.setattr(h, "_daemon_reachable", _REAL_DAEMON_REACHABLE)
        assert h._docker_req("/_ping") == {"raw": "OK"}
        assert h._daemon_reachable() is True
        srv.close()

    def test_docker_idempotency_codes(self, monkeypatch):
        def raise_409(path, method="GET", timeout=30.0):
            raise h.DockerAPIError(409, "already started")
        monkeypatch.setattr(h, "_docker_req", raise_409)
        _REAL_DOCKER("x", "start")  # 409 tolerated

        def raise_404(path, method="GET", timeout=30.0):
            raise h.DockerAPIError(404, "no such container")
        monkeypatch.setattr(h, "_docker_req", raise_404)
        _REAL_DOCKER("x", "stop")   # 404 on stop tolerated
        with pytest.raises(RuntimeError, match="container not found"):
            _REAL_DOCKER("x", "start")  # 404 on start = wrong name -> loud fail


class TestStudioUrlDerivation:
    def test_studio_url_override_wins(self, monkeypatch):
        monkeypatch.setenv("STUDIO_URL", "http://studio-box:8888/")
        monkeypatch.setenv("IMAGE_INFER_URL", "http://image-box:8888/v1")
        assert h._resolve_studio_url() == "http://studio-box:8888"

    def test_v1_suffix_is_stripped(self, monkeypatch):
        monkeypatch.delenv("STUDIO_URL", raising=False)
        monkeypatch.setenv("IMAGE_INFER_URL", "http://host.docker.internal:8888/v1")
        assert h._env_url("IMAGE_INFER_URL", "INFER_URL", "http://x") == "http://host.docker.internal:8888"

    def test_falls_back_to_chat_backend_then_default(self, monkeypatch):
        monkeypatch.delenv("STUDIO_URL", raising=False)
        monkeypatch.delenv("IMAGE_INFER_URL", raising=False)
        monkeypatch.setenv("INFER_URL", "http://chat-box:8888/v1")
        assert h._env_url("IMAGE_INFER_URL", "INFER_URL", "http://x") == "http://chat-box:8888"
        monkeypatch.delenv("INFER_URL", raising=False)
        assert h._env_url("IMAGE_INFER_URL", "INFER_URL", "http://fallback:1") == "http://fallback:1"
