"""Sandbox network shims for eval/probe scripts.

`.env` points EMBED_URL / RERANK_API_BASE at docker-internal service names —
correct inside the bot container, unresolvable from the sandbox. Compose
publishes the same sidecars on host ports, reachable here as
host.docker.internal:<port>. Importing this module (BEFORE `load_dotenv` and
any settings import) applies the overrides only when running inside the
sandbox; host runs are byte-identical.

Sandbox detection: clone mode has /run/sandbox; direct mode mounts the host
tree at a drive letter (/c/Repos/...).
"""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def in_sandbox() -> bool:
    return (Path("/run/sandbox").exists()
            or re.match(r"^/[a-z]{1,2}/", str(REPO)) is not None)


# docker service name (bot container) → published port on the host
SANDBOX_ENDPOINT_SHIMS = {
    "EMBED_URL": "http://host.docker.internal:11435/v1",
    "RERANK_API_BASE": "http://host.docker.internal:8000",
}


def apply_sandbox_shims() -> None:
    """Set the shim env vars when inside the sandbox (idempotent).

    Must be called before `load_dotenv` takes effect for settings — or again
    after it, since load_dotenv never overrides already-set variables.
    """
    if in_sandbox():
        os.environ.update(SANDBOX_ENDPOINT_SHIMS)


apply_sandbox_shims()
