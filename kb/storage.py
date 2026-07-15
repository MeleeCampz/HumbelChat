"""Knowledge base file storage — write, validate, auto-chunk on upload."""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import pathlib
import uuid
from datetime    import datetime, timezone

from kb.scorch import ChunkIndex

log = logging.getLogger("bot.kb.storage")

# Allowed content types (mirrors what the bot's KB reader supports)
ALLOWED_MIMES: set[str] = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "text/xml",
    "application/rtf",
}

MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB


def _infer_extension(raw_filename: str | None) -> str:
    """Return a safe file extension, defaulting to .txt."""
    if not raw_filename:
        return ".txt"
    ext = pathlib.Path(raw_filename).suffix.lower()
    mime_map = {
        "text/plain":     ".txt",
        "text/markdown":  ".md",
        "text/csv":       ".csv",
        "text/html":      ".html",
        "text/xml":       ".xml",
        "application/rtf": ".rtf",
    }
    mime, _ = mimetypes.guess_type(raw_filename)
    if mime:
        guess_ext = mime_map.get(mime, ".bin")
        return guess_ext
    # Return known extension from filename suffix
    return ext if len(ext) >= 1 else ".txt"


def _compute_sha256(data: bytes) -> str:
    """Return hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


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

    ext = _infer_extension(filename)
    display_name = _sanitize_filename(filename or "uploaded")
    stem_name = pathlib.Path(display_name).stem

    kb_root = kb_path if kb_path else pathlib.Path("/shared-knowledge/kb/uploads")
    
    if subfolder:
        kb_root = kb_root / subfolder
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
) -> list[dict]:
    """Scan the KB directory and return metadata for each file."""
    kb_root = pathlib.Path(kb_path)
    docs: list[dict] = []
    if not kb_root.exists():
        return docs

    for entry in sorted(kb_root.rglob("*"), key=lambda p: str(p)):
        if entry.is_file() and "?" not in entry.name and not entry.name.endswith(".chunks.jsonl"):
            stat = entry.stat()
            try:
                raw = entry.read_bytes()
                sha = _compute_sha256(raw)
            except OSError:
                sha = "unreadable"
            docs.append({
                "name": str(entry.relative_to(kb_root)),
                "filename": entry.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha[:16],
            })

    return docs

def reindex_all_kb_files(kb_path: pathlib.Path) -> int:
    """Iterate through all files in KB and run ChunkIndex indexing."""
    count = 0
    if not kb_path.exists():
        return 0
    for p in kb_path.rglob("*"):
        if p.is_file() and "?" not in p.name and not p.name.endswith(".chunks.jsonl"):
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                chunks = ChunkIndex.from_text(content)
                # Save chunks to a sidecar file for retrieval
                chunk_file = p.with_suffix(p.suffix + ".chunks.jsonl")
                with chunk_file.open("w", encoding="utf-8") as f:
                    for chunk in chunks:
                        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                log.error("Failed to reindex %s: %s", p, e)
    return count

