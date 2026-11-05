#!/usr/bin/env python3
"""gpu_switch_helper.py — GPU mode switch service (GPU-split 2026-11).

Owns the mode of the 4070 Ti SUPER (GPU 0). Two modes, ONE at a time:

  "rag"    reranker-llama + embedder-llama running on GPU 0 (default state)
  "image"  those containers stopped, the Studio diffusion model loaded on
           GPU 0 (~12.5 GiB measured at Q4_K_M)

The 5070 (chat / Strata) is NEVER touched by anything in here.

Runs as the compose service "gpu-switch": it talks to the Docker daemon via
the mounted socket (sibling containers by name) and to Unsloth Studio's
management API on the host. Pure stdlib (urllib + http.server) — nothing to
install.

HTTP API (loopback/docker-network only — no auth, never publish to the LAN):

  GET  /mode              -> {"mode": "rag"|"image", "rag": {...}, "image": {...}}
  POST /mode  {"mode": "rag"|"image"}   -> {"ok": true, "mode": ..., "detail": "..."}

Both switches are IDEMPOTENT: asking for the mode you are already in no-ops.
The bot (bot_core/images.py) calls this around every generation via
GPU_SWITCH_URL. scripts/gpu_switch.sh is the manual CLI twin (it prefers this
helper and falls back to raw docker compose if the helper is down).

Env:
  RERANKER_CONTAINER / EMBEDDER_CONTAINER  names (default humbelchat-reranker-llama / humbelchat-embedder-llama)
  STUDIO_URL          Studio base WITHOUT /v1 (default http://host.docker.internal:8888)
  STUDIO_API_KEY      bearer key for Studio
  IMAGE_MODEL         Studio model path  (e.g. unsloth/Qwen-Image-2.1-Turbo-GGUF)
  IMAGE_GGUF_FILENAME GGUF file inside that repo
  IMAGE_GPU_IDS       space/comma separated GPU ids for the load (default "0")
  LOAD_TIMEOUT_S      image load readiness deadline (default 300)
  HEALTH_TIMEOUT_S    rag container health deadline (default 120)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] gpu-switch: %(message)s")
log = logging.getLogger("gpu-switch")

RERANKER = os.getenv("RERANKER_CONTAINER", "humbelchat-reranker-llama")
EMBEDDER = os.getenv("EMBEDDER_CONTAINER", "humbelchat-embedder-llama")
STUDIO_URL = os.getenv("STUDIO_URL", "http://host.docker.internal:8888").rstrip("/")
STUDIO_API_KEY = os.getenv("STUDIO_API_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "")
IMAGE_GGUF_FILENAME = os.getenv("IMAGE_GGUF_FILENAME", "")
IMAGE_GPU_IDS = [int(p) for p in
                 os.getenv("IMAGE_GPU_IDS", "0").replace(",", " ").split()
                 if p.lstrip("-").isdigit()] or [0]
LOAD_TIMEOUT_S = float(os.getenv("LOAD_TIMEOUT_S", "300"))
HEALTH_TIMEOUT_S = float(os.getenv("HEALTH_TIMEOUT_S", "120"))
DAEMON = os.getenv("DOCKER_HOST", "http://localhost").rstrip("/")

_lock = threading.Lock()          # switches are serialized (one at a time)
MODE = "rag"                      # assumed startup state (compose starts both)


# ── tiny HTTP helpers ─────────────────────────────────────────────────────

def _req(url: str, method: str = "GET", body: dict | None = None,
         headers: dict | None = None, timeout: float = 30.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return {}


# ── docker (sibling containers via the mounted socket) ────────────────────

def _docker(name: str, action: str) -> None:
    """start|stop a sibling container. 409 'already started' / 404-on-stop
    are tolerated — switches must be idempotent."""
    try:
        _req(f"{DAEMON}/containers/{name}/{action}", "POST", timeout=60)
        log.info("docker %s %s", action, name)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:  # noqa: BLE001
            pass
        if e.code == 409 or (action == "stop" and e.code == 404):
            log.info("docker %s %s: already in state (HTTP %d)", action, name, e.code)
            return
        raise RuntimeError(f"docker {action} {name} -> HTTP {e.code}: {body}")


def _docker_running(name: str) -> bool:
    try:
        info = _req(f"{DAEMON}/containers/{name}/json", timeout=15)
        return bool(info.get("State", {}).get("Running"))
    except Exception:  # noqa: BLE001 — unreachable counts as "not running"
        return False


def _healthy(name: str, port: int = 8080) -> bool:
    """Poll the llama.cpp server's own /v1/models over the compose network."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            _req(f"http://{name}:{port}/v1/models", timeout=3)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


# ── Studio (image model) ──────────────────────────────────────────────────

def _studio_headers() -> dict:
    return {"Authorization": f"Bearer {STUDIO_API_KEY}"} if STUDIO_API_KEY else {}


def _studio(path: str, method: str = "GET", body: dict | None = None,
            timeout: float = 60.0) -> dict:
    return _req(f"{STUDIO_URL}{path}", method, body, _studio_headers(), timeout)


def _image_status() -> dict:
    return _studio("/api/inference/images/status")


def _load_image_model() -> None:
    if _image_status().get("loaded"):
        return
    if not IMAGE_MODEL:
        raise RuntimeError("IMAGE_MODEL not configured for the gpu-switch helper")
    payload = {"model_path": IMAGE_MODEL, "model_kind": "gguf",
               "gguf_filename": IMAGE_GGUF_FILENAME, "gpu_ids": IMAGE_GPU_IDS}
    log.info("loading image model %s on GPU %s", IMAGE_MODEL, IMAGE_GPU_IDS)
    try:
        _studio("/api/inference/images/load", "POST", payload, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code != 409:  # 409 = a load is already in progress -> just poll
            raise RuntimeError(f"images/load -> HTTP {e.code}: {e.read().decode()[:200]}")
        log.info("image load already in progress — polling for ready")
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(10)
        if _image_status().get("loaded"):
            return
    raise RuntimeError(f"image model not ready within {LOAD_TIMEOUT_S:.0f}s")


def _unload_image_model() -> None:
    try:
        if _image_status().get("loaded"):
            _studio("/api/inference/images/unload", "POST", {}, timeout=120)
    except Exception as e:  # noqa: BLE001 — best-effort; containers can start anyway
        log.warning("image unload failed (continuing): %s", e)


# ── mode switches ─────────────────────────────────────────────────────────

def switch(mode: str) -> str:
    global MODE
    if mode not in ("rag", "image"):
        raise ValueError(f"unknown mode {mode!r}")
    if MODE == mode:
        return f"already in {mode} mode"
    if mode == "image":
        _docker(RERANKER, "stop")
        _docker(EMBEDDER, "stop")
        _load_image_model()
        MODE = "image"
        return "reranker+embedder stopped, image model resident on GPU 0"
    # -> rag
    _unload_image_model()
    _docker(RERANKER, "start")
    _docker(EMBEDDER, "start")
    ok_r = _healthy(RERANKER)
    ok_e = _healthy(EMBEDDER)
    MODE = "rag"
    if not (ok_r and ok_e):
        return "rag mode restored BUT health check failed (check docker logs)"
    return "reranker+embedder healthy on GPU 0"


def state() -> dict:
    return {
        "mode": MODE,
        "rag": {"reranker_running": _docker_running(RERANKER),
                "embedder_running": _docker_running(EMBEDDER)},
        "image_loaded": bool(_image_status().get("loaded")),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — http.server API
        if self.path != "/mode":
            return self._send(404, {"error": "unknown path"})
        try:
            with _lock:
                self._send(200, state())
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)[:200]})

    def do_POST(self):  # noqa: N802
        if self.path != "/mode":
            return self._send(404, {"error": "unknown path"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            mode = json.loads(self.rfile.read(n) or b"{}").get("mode", "")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        try:
            with _lock:
                detail = switch(mode)
            self._send(200, {"ok": True, "mode": MODE, "detail": detail})
        except Exception as e:  # noqa: BLE001
            log.error("switch to %s failed: %s", mode, e, exc_info=True)
            self._send(500, {"ok": False, "mode": MODE, "error": str(e)[:300]})

    def log_message(self, fmt, *args):  # quiet the default access log
        log.debug("http " + fmt, *args)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    log.info("gpu-switch helper listening on :%d (mode=%s, studio=%s, image=%s)",
             port, MODE, STUDIO_URL, IMAGE_MODEL or "-")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
