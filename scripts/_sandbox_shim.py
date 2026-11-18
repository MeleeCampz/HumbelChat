"""Sandbox network shims for eval/probe scripts.

`.env` points EMBED_URL / RERANK_API_BASE at docker-internal service names —
correct inside the bot container, unresolvable from the sandbox. Compose
publishes the same sidecars on host ports, reachable here as
host.docker.internal:<port>. Importing this module (BEFORE `load_dotenv` and
any settings import) applies the overrides only when running inside the
sandbox; host runs are byte-identical.

The shim also loads the repo `.env` BY PATH and can refuse a run whose
embedding model does not match the vector cache on disk — see
`guard_vector_cache`, which exists because the sandbox and the live bot SHARE
`data/`, including `data/knowledge/.vector_index_cache/`.

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
    "EMBED_URL": "http://host.docker.internal:8091/v1",   # embedder-llama
    "RERANK_API_BASE": "http://host.docker.internal:8090/v1",  # reranker-llama
}


def load_repo_env() -> bool:
    """Load the repo `.env` by ABSOLUTE path. Returns True when a file was read.

    Why this exists: `load_dotenv()` with no argument resolves through
    `find_dotenv()`, which searches from the CALLING FILE's directory. A probe
    living in /tmp (or piped through stdin) finds nothing and loads NOTHING —
    silently leaving the process on `config.settings` defaults. That is not a
    cosmetic problem here: the default `EMBEDDING_MODEL` is the retired
    `nomic-embed-text:latest`, and a store built on that name re-stamps and
    rebuilds the shared vector cache. On 2026-11-18 a probe in /tmp did exactly
    that to the live production cache.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    path = REPO / ".env"
    if not path.exists():
        return False
    # override=False: an endpoint shim or a caller's explicit export always wins.
    return bool(load_dotenv(path, override=False))


def apply_sandbox_shims() -> None:
    """Load production env, then set the shim env vars (idempotent).

    Must be called before `load_dotenv` takes effect for settings — or again
    after it, since load_dotenv never overrides already-set variables.

    Loading `.env` here (rather than trusting each script to do it) is the
    point: with production env in place, a sandbox run's effective model, KB
    path and backend match what the bot itself would use, so an accidental
    cache re-stamp cannot come from defaults.
    """
    if in_sandbox():
        load_repo_env()
        os.environ.update(SANDBOX_ENDPOINT_SHIMS)


def guard_vector_cache() -> None:
    """Refuse a KB-touching sandbox run whose model would re-stamp the cache.

    The sandbox mount is the LIVE data directory — the same tree the container
    reads as /app/data. `kb.index` treats a model-stamp mismatch as "new
    embedding model" and responds by re-embedding every chunk and rewriting
    `vector_index.db`. A probe that is misconfigured therefore destroys the
    production cache's identity: the bot then pays a full reindex at its next
    start (2026-11-18: 5806 chunks, stamp left as `nomic-embed-text:latest`).

    This runs the store's own READ-ONLY identity check (`_cache_identity_ok`,
    the same gate `load()` uses) against the store the production path would
    build, and raises instead of letting the rebuild happen. Call it from any
    script that touches `kb.*` or `bot_core.*` before it does so.
    """
    if not in_sandbox():
        return
    from config import settings as _s
    from kb.index import KBIndexStore

    store = KBIndexStore(_s.KB_PATH)          # mirrors the production construction
    db = store._db_path
    if not db.exists():
        print(f"[sandbox shim] NOTE: no vector cache at {db} — a KB probe here "
              f"would BUILD the live cache (write-side model "
              f"{getattr(store._embedder, 'model_name', store.model_name)}).")
        return
    if not store._cache_identity_ok():
        raise RuntimeError(
            f"[sandbox shim] REFUSING to run: the vector cache at {db} is stamped "
            f"for a different embedding model than this process would use "
            f"(this process: query={store.model_name!r}, "
            f"write={getattr(store._embedder, 'model_name', store.model_name)!r}). "
            "Letting kb.index proceed would re-embed every chunk and rewrite the "
            "cache the live bot reads. Fix the process env first (import the shim "
            "before settings, or export EMBEDDING_MODEL/INDEX_EMBED_MODEL "
            "explicitly), then call guard_vector_cache() again."
        )
    print(f"[sandbox shim] vector cache identity OK (model {store.model_name}) — "
          f"read-only against {db}")


apply_sandbox_shims()
