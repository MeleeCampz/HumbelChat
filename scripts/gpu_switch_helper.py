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

  GET  /mode   -> {"mode": "rag"|"image", "rag": {"reranker_running": bool,
                 "embedder_running": bool}, "image_loaded": bool,
                 "image_status_error": true  (only when Studio is unreachable)}
  POST /mode   {"mode": "rag"|"image"} -> {"ok": true, "mode": ..., "detail": "..."}
               refused switch -> 200 with detail starting "refused:"
               hard failure  -> 500 {"ok": false, "mode": ..., "error": "..."}

Both switches are IDEMPOTENT: asking for the mode you are already in no-ops.
The bot (bot_core/images.py) calls this around every generation via
GPU_SWITCH_URL. scripts/gpu_switch.sh is the manual CLI twin (it prefers this
helper and falls back to raw docker compose if the helper is down).

Env (the compose service mounts the repo .env, so the bot's image variables
are the single source of truth; a helper-only var can still override):
  RERANKER_CONTAINER / EMBEDDER_CONTAINER  names (default humbelchat-reranker-llama / humbelchat-embedder-llama)
  DOCKER_SOCKET        daemon unix socket path (default /var/run/docker.sock;
                       DOCKER_HOST unix://... or http(s)://... also honoured,
                       npipe:/tcp:// ignored with a warning)
  STUDIO_URL           explicit Studio base (wins over the chain below)
  IMAGE_INFER_URL      Studio OpenAI base (…/v1) — management endpoints are
                       derived by stripping the /v1 suffix
  INFER_URL            fallback (chat backend) if IMAGE_INFER_URL is unset
  IMAGE_INFER_API_KEY  bearer key (falls back to INFER_API_KEY)
  IMAGE_MODEL          Studio model path  (e.g. unsloth/Qwen-Image-2.1-Turbo-GGUF)
  IMAGE_GGUF_FILENAME  GGUF file inside that repo
  IMAGE_GPU_IDS        space/comma separated GPU ids for the load (default "0")
  IMAGE_LOAD_TIMEOUT_S image load readiness deadline (default 300)
  HEALTH_TIMEOUT_S     rag container health deadline (default 120)
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] gpu-switch: %(message)s")
log = logging.getLogger("gpu-switch")

def _env_url(name: str, fallback_name: str, default: str) -> str:
    """URL env with fallback chain, mirroring config/settings.py's rule: the
    image backend falls back to the chat backend, then to the host default.
    Only the /v1 OpenAI suffix is stripped — never a trailing path segment."""
    raw = (os.getenv(name) or os.getenv(fallback_name) or default).strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw.rstrip("/")


RERANKER = os.getenv("RERANKER_CONTAINER", "humbelchat-reranker-llama")
EMBEDDER = os.getenv("EMBEDDER_CONTAINER", "humbelchat-embedder-llama")


def _resolve_studio_url() -> str:
    """STUDIO_URL (explicit helper override) wins, else the bot's image-backend
    chain: IMAGE_INFER_URL → INFER_URL → host default."""
    override = os.getenv("STUDIO_URL", "").strip().rstrip("/")
    if override:
        return override
    return _env_url("IMAGE_INFER_URL", "INFER_URL", "http://host.docker.internal:8888")


STUDIO_URL = _resolve_studio_url()
STUDIO_API_KEY = os.getenv("IMAGE_INFER_API_KEY") or os.getenv("INFER_API_KEY", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "")
IMAGE_GGUF_FILENAME = os.getenv("IMAGE_GGUF_FILENAME", "")
IMAGE_GPU_IDS = [int(p) for p in
                 os.getenv("IMAGE_GPU_IDS", "0").replace(",", " ").split()
                 if p.lstrip("-").isdigit()] or [0]
LOAD_TIMEOUT_S = float(os.getenv("IMAGE_LOAD_TIMEOUT_S", "300"))
HEALTH_TIMEOUT_S = float(os.getenv("HEALTH_TIMEOUT_S", "120"))
DOCKER_SOCK_PATH = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")


class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client over an AF_UNIX socket (the mounted /var/run/docker.sock).
    urllib has NO unix transport: a plain urllib call to "http://localhost"
    goes to TCP port 80 and fails with ECONNREFUSED. That was the 2026-11-05
    live incident — every daemon call was refused, _docker_running silently
    reported every container as stopped, and mode was misderived as "image"
    while the RAG containers were actually running."""

    def __init__(self, sock_path: str, timeout: float) -> None:
        super().__init__("dockerd", timeout=timeout)
        self._sock_path = sock_path

    def connect(self) -> None:
        # Explicit AF_UNIX (not socket.create_connection(str)) — that form
        # only accepts path strings on some Python versions (3.14 unpacks the
        # str as (host, port) and raises). Production is 3.12; keep it proof.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._sock_path)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


def _resolve_daemon() -> tuple:
    """Docker daemon endpoint as (kind, target). The compose service mounts
    /var/run/docker.sock, so the DEFAULT is that unix socket. An inherited
    DOCKER_HOST is honoured only when usable in here: unix://... -> that
    socket, http(s)://... -> TCP. Windows npipe: and bare tcp:// are useless
    from inside the container — warn and keep the mounted socket."""
    raw = (os.getenv("DOCKER_HOST") or "").strip().rstrip("/")
    if raw.startswith("unix://"):
        return ("unix", raw[len("unix://"):])
    if raw.startswith("http://") or raw.startswith("https://"):
        return ("tcp", urllib.parse.urlparse(raw))
    if raw:
        log.warning("DOCKER_HOST %r not usable from inside the helper "
                    "(no npipe/raw-tcp support) — using the mounted socket %s",
                    raw, DOCKER_SOCK_PATH)
    return ("unix", DOCKER_SOCK_PATH)


DAEMON = _resolve_daemon()


class DockerAPIError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"docker API HTTP {status}: {body}")
        self.status = status
        self.body = body


def _docker_req(path: str, method: str = "GET", timeout: float = 30.0) -> dict:
    """One Docker daemon API call over the resolved transport."""
    kind, target = DAEMON
    if kind == "unix":
        conn: http.client.HTTPConnection = _UnixHTTPConnection(target, timeout)
    else:
        port = target.port or (443 if target.scheme == "https" else 2375)
        conn = http.client.HTTPConnection(target.hostname, port, timeout=timeout)
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status >= 400:
            raise DockerAPIError(resp.status, raw.decode(errors="replace")[:200])
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            # Docker's /_ping answers text/plain "OK", NOT JSON. A JSON-only
            # reader turned a perfectly healthy daemon into "unreachable" and
            # refused EVERY switch (live 2026-11-09: docker_ok false while the
            # container lookups — which really are JSON — read fine).
            # A 2xx/3xx body is a success regardless of its content type.
            return {"raw": raw.decode(errors="replace")[:200]}
    finally:
        conn.close()


def _daemon_reachable() -> bool:
    """Fail LOUD: with the daemon unreachable, container-state checks would
    silently mean "everything stopped" and mode derivation would lie (the
    exact 2026-11-05 failure). Switches refuse instead of acting on lies."""
    try:
        # Non-JSON body is fine (see _docker_req): "OK" text is a good daemon.
        _docker_req("/_ping", timeout=10)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("docker daemon unreachable via %s: %s", DAEMON[1], e)
        return False

_lock = threading.Lock()          # switches are serialized (one at a time)


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
        _docker_req(f"/containers/{name}/{action}", "POST", timeout=60)
        log.info("docker %s %s", action, name)
    except DockerAPIError as e:
        if e.status == 409:
            log.info("docker %s %s: already in state (HTTP 409)", action, name)
            return
        if action == "stop" and e.status == 404:
            log.info("docker stop %s: not running (HTTP 404)", name)
            return
        if action == "start" and e.status == 404:
            # A missing container is NOT an idempotency state — it means a
            # wrong container name or a compose service that was never
            # created. Fail loudly instead of reporting a bogus rag mode.
            raise RuntimeError(
                "docker start " + name + ": container not found — check "
                "RERANKER_CONTAINER/EMBEDDER_CONTAINER + compose project "
                "prefix and whether the container exists: " + e.body)
        raise


def _docker_running(name: str) -> bool:
    try:
        info = _docker_req(f"/containers/{name}/json", timeout=15)
        return bool(info.get("State", {}).get("Running"))
    except Exception:  # noqa: BLE001 — unreachable counts as "not running";
        # switch() pre-checks _daemon_reachable(), so a lying mode can't act.
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
    """Studio images/status — RESILIENT: {} when the backend is unreachable
    (callers decide; a down Studio must not crash GET /mode)."""
    try:
        return _studio("/api/inference/images/status")
    except Exception as e:  # noqa: BLE001
        log.warning("images/status failed: %s", e)
        return {}


def _load_image_model() -> None:
    if not IMAGE_MODEL:
        raise RuntimeError("IMAGE_MODEL not configured for the gpu-switch helper")
    if _image_status().get("loaded"):
        return
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

def _current_mode() -> str:
    """Mode is DERIVED from live container state, never remembered across
    restarts: the helper restarting while the RAG containers are stopped must
    not report a stale "rag" mode (which would make the bot no-op its own
    switch and paint against a half-set card)."""
    if _docker_running(RERANKER) or _docker_running(EMBEDDER):
        return "rag"
    return "image"


def _backend_reachable() -> bool:
    """Is the Studio image backend answering (or the model already resident)?
    Used to refuse an image-mode switch BEFORE tearing RAG down."""
    try:
        _studio("/api/inference/images/status", timeout=15)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("image backend unreachable: %s", e)
        return False


def switch(mode: str) -> str:
    if mode not in ("rag", "image"):
        raise ValueError(f"unknown mode {mode!r}")
    if not _daemon_reachable():
        raise RuntimeError(
            "docker daemon unreachable — refusing to switch (container state "
            "would be unknowable; check the docker.sock mount on this service)")
    current = _current_mode()
    if current == mode:
        return f"already in {mode} mode"
    if mode == "image":
        # Never enter the destructive part (stopping RAG) unless the image
        # backend can actually take the card — otherwise the 4070 lands in a
        # half-switched state with BOTH services down. Containers already
        # stopped (manual gpu_switch.sh overlap) skip the precheck so a
        # recovery attempt can still proceed.
        rag_live = _docker_running(RERANKER) or _docker_running(EMBEDDER)
        if rag_live and not _backend_reachable():
            return ("refused: image backend unreachable — staying in rag mode "
                    f"(check Studio at {STUDIO_URL})")
        _docker(RERANKER, "stop")
        _docker(EMBEDDER, "stop")
        _load_image_model()      # raises -> HTTP 500 to the bot, stays image-side
        return "reranker+embedder stopped, image model resident on GPU 0"
    # -> rag
    _unload_image_model()
    _docker(RERANKER, "start")
    _docker(EMBEDDER, "start")
    ok_r = _healthy(RERANKER)
    ok_e = _healthy(EMBEDDER)
    if not (ok_r and ok_e):
        return "rag mode restored BUT health check failed (check docker logs)"
    return "reranker+embedder healthy on GPU 0"


def state() -> dict:
    st = _image_status()
    out = {
        "mode": _current_mode(),
        "docker_ok": _daemon_reachable(),
        "rag": {"reranker_running": _docker_running(RERANKER),
                "embedder_running": _docker_running(EMBEDDER)},
        "image_loaded": bool(st.get("loaded")),
    }
    if not st:
        out["image_status_error"] = True  # Studio down / unreachable
    return out


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
            self._send(200, {"ok": True, "mode": _current_mode(), "detail": detail})
        except Exception as e:  # noqa: BLE001
            log.error("switch to %s failed: %s", mode, e, exc_info=True)
            self._send(500, {"ok": False, "mode": _current_mode(),
                             "error": str(e)[:300]})

    def log_message(self, fmt, *args):  # quiet the default access log
        log.debug("http " + fmt, *args)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    log.info("gpu-switch helper listening on :%d (mode=%s, studio=%s, image=%s)",
             port, _current_mode(), STUDIO_URL, IMAGE_MODEL or "-")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
