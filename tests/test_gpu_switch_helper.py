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
