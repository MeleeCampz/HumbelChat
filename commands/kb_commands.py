"""Knowledge-base commands — /upload_kb, /list_kb_docs and /reindex_kb."""
from __future__ import annotations

import logging
import pathlib

from kb.storage import validate_upload, list_kb_files, reindex_all_kb_files
from kb.scorch import ChunkIndex

log = logging.getLogger("bot.commands.kb")


async def handle_upload_kb(
    interaction,                           # Discord Interaction
    kb_name: str | None = None,            # override for KB folder name
    url: str | None = None,                 # remote URL → download
    attachment=None,                        # discord.Attachment or None
    subfolder: str | None = None,           # optional subfolder
) -> None:
    """Upload a file directly to the local KB storage directory."""
    from config.settings import settings

    # --- step 1: get bytes ---
    if attachment is not None:
        data = await attachment.read()
        fname = attachment.filename or "attachment"
    elif url:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            data = resp.content
        fname = url.split("?")[0].split("/")[-1] or "remote_file"
    else:
        await interaction.response.send_message(
            "Please provide either a URL or file attachment for /upload_kb.", ephemeral=True
        )
        return

    # --- step 2 & 3: validate + write to KB_PATH ---
    try:
        dest, summary = validate_upload(data, filename=fname, kb_path=settings.KB_PATH, subfolder=subfolder)
    except ValueError as exc:
        await interaction.response.send_message(f"Upload rejected: **{exc}**", ephemeral=True)
        return
    except FileNotFoundError as exc:
        await interaction.response.send_message(f"KB storage not found: **{exc}**", ephemeral=True)
        return

    # --- step 4: auto-index chunks (non-blocking) ---
    try:
        import json
        content = pathlib.Path(dest).read_text(encoding="utf-8", errors="replace")
        chunks = ChunkIndex.from_text(content, chunk_size=settings.CHUNK_TARGET)
        # Save chunks to a sidecar file for retrieval
        chunk_file = dest.with_suffix(dest.suffix + ".chunks.jsonl")
        with chunk_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        log.info("Auto-indexed %d chunks for %s", len(chunks), dest.name)
    except Exception as e:
        log.warning("Failed to auto-index %s — will need manual /reindex_chunks: %s", dest.name, e)

    # --- step 5: reply with summary ---
    chunk_hint = ""
    try:
        n = len(pathlib.Path(dest).read_text(encoding="utf-8", errors="replace"))
        approx_chunks = n // settings.CHUNK_TARGET if settings.CHUNK_TARGET else "?"
        approx_chunks_display = " (approx %d chunks)" % approx_chunks
    except Exception:
        approx_chunks_display = ""

    await interaction.response.send_message(
        f"✅ **upload_kb** stored `{summary['name']}` ({summary['size']:,} bytes)\n"
        f"Location: ``{dest.name}``\n"
        f"Hash SHA256 prefix: ``{summary['sha256']}...\\n``\n"
        f"Auto-chunked.{approx_chunks_display}", ephemeral=True
    )


def get_root_directories(kb_path: pathlib.Path) -> list[str]:
    """Return sorted list of subdirectory names at the root level."""
    if not kb_path.exists():
        return []
    dirs = []
    for entry in kb_path.iterdir():
        if entry.is_dir():
            dirs.append(entry.name)
    return sorted(dirs)


async def handle_list_kb_docs(interaction, subfolder_path: str | None = None):
    """List all documents in KB_PATH directory.

    If *subfolder_path* is given, recurses into that subfolder.
    Otherwise shows only root-level items (files + directories).
    """
    from config.settings import settings

    if subfolder_path:
        # Subfolder view: recurse into that path
        docs = list_kb_files(settings.KB_PATH, subfolder=subfolder_path, recursive=True)
        lines: list[str] = [
            f"**Knowledge Base** documents — `{subfolder_path}`",
            "📂 **Subdirectories:**",
        ]
        # Show nested directories within the subfolder
        scan_root = pathlib.Path(settings.KB_PATH) / subfolder_path
        subdirs = get_root_directories(scan_root)
        for d in sorted(subdirs):
            lines.append(f"  📂 `{d}`")
    else:
        # Root view: show directories + root-level files only
        docs = list_kb_files(settings.KB_PATH, subfolder=None, recursive=False)
        dirs = get_root_directories(settings.KB_PATH)
        lines = [
            "**Knowledge Base** documents",
            "📁 **Root directories:**",
        ]
        for d in sorted(dirs):
            lines.append(f"  📂 `{d}`")

    if not docs:
        lines.append("")
        lines.append("(no files found)")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)
        return

    # --- unified file listing (same format for both views) ---
    section_label = "🔹 **Root files:**" if not subfolder_path else "📂 **Files:**"
    lines.append("")
    lines.append(section_label)
    for doc in docs[:30]:  # cap at 30
        size_kb = doc["size"] / 1024 or "0"
        name = doc.get("name", doc.get("filename", "unknown"))
        sha8 = (doc.get("sha256", "?")[:8])
        date = doc.get("modified", "?")[:10]
        lines.append(f"  • `{name}` — {size_kb:.1f} KB — {date} — sha:`{sha8}...`")

    if len(docs) > 30:
        lines.append(f"\n… and {len(docs) - 30} more documents.")

    await interaction.response.send_message("\n".join(lines), ephemeral=True)


async def handle_reindex_kb(interaction):
    """Trigger reindexing of all files in the KB."""
    from config.settings import settings
    await interaction.response.defer(ephemeral=True)
    try:
        count = reindex_all_kb_files(pathlib.Path(settings.KB_PATH))
        await interaction.followup.send(f"✅ Successfully reindexed **{count}** files in the knowledge base.", ephemeral=True)
    except Exception as e:
        log.error("Reindexing failed: %s", e)
        await interaction.followup.send(f"❌ Failed to reindex KB: **{e}**", ephemeral=True)
