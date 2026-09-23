"""Knowledge base file storage — write, validate, auto-chunk on upload."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import pathlib
import uuid
from datetime import datetime, timezone
from functools import partial

from config.settings import KB_PATH

log = logging.getLogger("bot.kb.storage")

MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB

# File extensions accepted for KB storage (mirrors what the KB reader
# indexes). Anything else is rejected before it touches disk.
ALLOWED_EXTENSIONS: set[str] = {".txt", ".md", ".csv", ".html", ".xml", ".rtf"}

# P2 #18: sidecar cache for /list_kb_docs SHA-256s.  The listing used to read
# every file fully on each call (O(total KB size)); now the hash is cached
# keyed by (relpath, size, mtime) and only re-computed when that key changes.
# The file lives INSIDE the KB root with a dot-prefix so list_kb_files's
# existing "skip hidden entries" rule keeps it out of user listings.
SHA256_CACHE_FILENAME = ".sha256_cache.json"


def _infer_extension(raw_filename: str | None) -> str:
    """Return a safe file extension, defaulting to .txt."""
    if not raw_filename:
        return ".txt"
    ext = pathlib.Path(raw_filename).suffix.lower()
    mime_map = {
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/csv": ".csv",
        "text/html": ".html",
        "text/xml": ".xml",
        "application/rtf": ".rtf",
    }
    mime, _ = mimetypes.guess_type(raw_filename)
    if mime:
        guess_ext = mime_map.get(mime)
        if guess_ext:
            return guess_ext
    # Return known extension from filename suffix (fallback)
    return ext if len(ext) >= 1 else ".txt"


def _compute_sha256(data: bytes) -> str:
    """Return hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


def _cache_path_for(kb_root: pathlib.Path) -> pathlib.Path:
    return kb_root / SHA256_CACHE_FILENAME


def _load_sha256_cache(kb_root: pathlib.Path) -> dict[str, dict]:
    """Load the sidecar sha256 cache (P2 #18). Returns {} on missing/corrupt.

    Shape: ``{ relpath: {"size": int, "mtime": float, "sha256": str} }``.
    """
    try:
        with open(_cache_path_for(kb_root), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_sha256_cache(kb_root: pathlib.Path, cache: dict[str, dict]) -> None:
    """Persist the sha256 cache (P2 #18). Best-effort — a failure to write the
    cache just means the next listing re-hashes; it never breaks the listing."""
    try:
        with open(_cache_path_for(kb_root), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        log.debug("Failed to write sha256 cache: %s", e)


def validate_upload(
    data: bytes,
    filename: str | None = "uploaded",
    kb_path: pathlib.Path | None = None,
    subfolder: str | None = None,
) -> tuple[pathlib.Path, dict]:
    """Write uploaded content to KB storage.

    Returns (dest_path, summary_dict) where summary has:
      { "name", "size", "modified", "sha256" }
    """
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {len(data):,} bytes (max {MAX_FILE_SIZE:,})"
        )

    # P3 #36: reject 0-byte uploads. An empty file has nothing to index and
    # would otherwise be stored as a 0 KB document that RAG can never match —
    # a common accident when an attachment is selected but not fully read.
    if len(data) == 0:
        raise ValueError(
            f"File '{filename or 'upload'}' is empty (0 bytes). "
            "Upload a file that actually contains content."
        )

    ext = _infer_extension(filename)

    # Reject unsupported file types before anything touches disk.
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"File type '{ext or 'unknown'}' is not supported for the KB. "
            f"Allowed: {allowed}"
        )

    display_name = _sanitize_filename(filename or "uploaded")
    stem_name = pathlib.Path(display_name).stem

    kb_root = (kb_path if kb_path else KB_PATH).resolve()

    if subfolder:
        # Guard against path traversal (e.g. "../../etc"): the resolved
        # subfolder must stay inside the KB root.
        candidate = (kb_root / subfolder).resolve()
        if not candidate.is_relative_to(kb_root):
            raise ValueError(f"Invalid subfolder: {subfolder!r}")
        kb_root = candidate
        try:
            kb_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error("Failed to create subfolder %s: %s", subfolder, e)
            raise FileNotFoundError(f"Could not create subfolder: {subfolder}")

    dest = kb_root / f"{stem_name}{ext}"

    # Collision handling: if file exists, append a short unique ID
    if dest.exists():
        unique_id = uuid.uuid4().hex[:8]
        dest = kb_root / f"{stem_name}_{unique_id}{ext}"

    dest.write_bytes(data)

    sha = _compute_sha256(data)
    log.info("KB storage: %s → %s (sha256=%s)", filename or "unknown", dest, sha[:12])

    stat = dest.stat()
    return dest, {
        "name": dest.name,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha[:16],
    }


def _sanitize_filename(name: str) -> str:
    """Reduce a filename to safe characters, stripping UUID prefix if present."""
    # Strip leading hex segment (UUID) and underscore separator
    parts = name.split("_", 1)
    if len(parts) == 2 and all(c in "0123456789abcdef" for c in parts[0]):
        name = parts[1]

    safe = "".join(c for c in name if c.isalnum() or c in "._- ")
    return safe.strip()[:60] or "uploaded_doc"


def list_kb_files(
    kb_path: str | pathlib.Path,
    subfolder: str | None = None,
    recursive: bool = True,
) -> list[dict]:
    """Scan the KB directory and return metadata for each file.

    Args:
        kb_path: Root knowledge base directory.
        subfolder: Optional subdirectory to start listing from.
                   If given, files are listed relative to this subfolder.
        recursive: If True (default), scans all subdirectories.
                   If False and no *subfolder*, only returns root-level files.

    Hidden entries (dot-prefixed files or directories, e.g. the vector
    index cache) are always excluded — they are internal state, not docs.
    """
    kb_root = pathlib.Path(kb_path)
    docs: list[dict] = []
    if not kb_root.exists():
        return docs

    # Determine the scanning root
    scan_root = kb_root / subfolder if subfolder else kb_root

    # Determine glob pattern
    pattern = "**/*" if recursive else "*"

    # P2 #18: load the sidecar sha256 cache so unchanged files aren't re-hashed.
    kb_abs = kb_root.resolve()
    cache = _load_sha256_cache(kb_abs)
    listed_keys: set[str] = set()
    dirty = False

    for entry in sorted(scan_root.glob(pattern), key=lambda p: str(p)):
        if not entry.is_file():
            continue
        # Skip hidden files and anything inside hidden dirs (e.g.
        # .vector_index_cache/vector_index.db, .sha256_cache.json) — they are
        # internal state, not user documents.
        try:
            rel_parts = entry.relative_to(kb_root).parts
        except ValueError:
            rel_parts = entry.parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if "?" in entry.name or entry.name.endswith(".chunks.jsonl"):
            continue
        stat = entry.stat()
        # Cache key = KB-relative path; a hit requires matching size AND mtime
        # (a content change alters mtime, so we only re-hash when it might differ).
        key = "/".join(rel_parts)
        listed_keys.add(key)
        hit = cache.get(key)
        if hit is not None and hit.get("size") == stat.st_size \
                and hit.get("mtime") == stat.st_mtime:
            sha: str = hit["sha256"]
        else:
            try:
                raw = entry.read_bytes()
                sha = _compute_sha256(raw)
                cache[key] = {"size": stat.st_size, "mtime": stat.st_mtime,
                              "sha256": sha}
                dirty = True
            except OSError:
                sha = "unreadable"
        docs.append({
            "name": str(entry.relative_to(scan_root)),
            "filename": entry.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "sha256": sha[:16],
        })

    # Prune entries for files no longer present — only safe on a full
    # recursive root scan (a subfolder scan would wrongly drop siblings).
    if subfolder is None and recursive:
        stale = [k for k in cache if k not in listed_keys]
        for k in stale:
            cache.pop(k, None)
        dirty = dirty or bool(stale)

    if dirty:
        _save_sha256_cache(kb_abs, cache)

    return docs


# ─────────────── Async wrappers (P0 #4: keep the event loop free) ───────────────
#
# /upload_kb (write + SHA-256) and /list_kb_docs (rglob + per-file reads +
# hashing) used to run their blocking IO on the event loop thread, freezing
# the whole bot while a large KB was scanned.  These wrappers run the
# identical blocking work in the default thread pool.

async def _run_in_thread(func, *args, **kwargs):
    """Run *func* in the default executor (stub-friendly for tests)."""
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, partial(func, *args, **kwargs))
    return await asyncio.ensure_future(future)


async def validate_upload_async(
    data: bytes,
    filename: str | None = "uploaded",
    kb_path: pathlib.Path | None = None,
    subfolder: str | None = None,
) -> tuple[pathlib.Path, dict]:
    """Thread-pool wrapper around :func:`validate_upload` (P0 #4).

    Raises the same ``ValueError`` / ``FileNotFoundError`` as the sync version.
    """
    return await _run_in_thread(
        validate_upload, data, filename, kb_path, subfolder
    )


async def list_kb_files_async(
    kb_path: str | pathlib.Path,
    subfolder: str | None = None,
    recursive: bool = True,
) -> list[dict]:
    """Thread-pool wrapper around :func:`list_kb_files` (P0 #4)."""
    return await _run_in_thread(
        list_kb_files, kb_path,
        subfolder=subfolder,
        recursive=recursive,
    )
