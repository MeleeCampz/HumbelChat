import pathlib
import logging

log = logging.getLogger("bot")

def log_top_kb_files(kb_path: pathlib.Path, top_n: int = 5):
    """Logs the largest files in the KB directory."""
    if not kb_path.exists():
        return

    files = []
    for p in kb_path.rglob("*"):
        if p.is_file() and not p.name.endswith(".chunks.jsonl") and not p.name.startswith("."):
            try:
                files.append((p.name, p.stat().st_size))
            except OSError:
                continue

    # Sort by size descending
    files.sort(key=lambda x: x[1], reverse=True)

    if not files:
        log.info("No KB files found to rank.")
        return

    log.info("--- Top %d Knowledge Base Files (by size) ---", top_n)
    for i, (name, size) in enumerate(files[:top_n]):
        log.info("%d. %s (%s MB)", i + 1, name, f"{size / (1024*1024):.2f}")
    log.info("---------------------------------------------")
