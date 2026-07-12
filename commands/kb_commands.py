"""Knowledge-base commands — /upload_kb and /list_kb_docs.

Replaces the OWUI-dependent versions in main.py with a direct filesystem approach:
  - ``validate_upload()`` writes to KB_PATH (kb.storage)
  - ``ChunkIndex.from_text()`` auto-indexes chunks (kb.scorch)
  - ``list_kb_files()`` scans KB_PATH directly
"""
from __future__ import annotations

import logging
import pathlib

from kb.storage import validate_upload, list_kb_files
from kb.scorch import ChunkIndex

log = logging.getLogger("bot.commands.kb")


async def handle_upload_kb(
    interaction,                           # Discord Interaction
    kb_name: str | None = None,            # override for KB folder name
    url: str | None = None,                 # remote URL → download
    attachment=None,                        # discord.Attachment or None
) -> None:
    """Upload a file directly to the local KB storage directory.

    File priority (highest first):
      1. *attachment* — a Discord attachment already in memory
      2. *url* — downloads the remote file via httpx
      3. error if neither provided

    Flow:
      1. Fetch content bytes.
      2. Validate MIME + size via kb.storage.validate_upload().
      3. Write to KB_PATH (with UUID prefix).
      4. Auto-index chunks via ChunkIndex.from_text().
      5. Return summary as ephemeral reply.
    """
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
        await interaction.followup.send(
            "Please provide either a URL or file attachment for /upload_kb.", ephemeral=True
        )
        return

    # --- step 2 & 3: validate + write to KB_PATH ---
    try:
        dest, summary = validate_upload(data, filename=fname, kb_path=settings.KB_PATH)
    except ValueError as exc:
        await interaction.followup.send(f"Upload rejected: **{exc}**", ephemeral=True)
        return
    except FileNotFoundError as exc:
        await interaction.followup.send(f"KB storage not found: **{exc}**", ephemeral=True)
        return

    # --- step 4: auto-index chunks (non-blocking) ---
    try:
        content = pathlib.Path(dest).read_text(encoding="utf-8")
        ChunkIndex.from_text(content, chunk_size=settings.CHUNK_TARGET)
        log.info("Auto-indexed %d chunks for %s", summary["size"], dest.name)
    except Exception:
        log.warning("Failed to auto-index %s — will need manual /reindex_kb", dest.name)

    # --- step 5: reply with summary ---
    chunk_hint = ""
    try:
        n = len(pathlib.Path(dest).read_text())
        approx_chunks = n // settings.CHUNK_TARGET if settings.CHUNK_TARGET else "?"
        approx_chunks_display = " (approx %d chunks)" % approx_chunks
    except Exception:
        approx_chunks_display = ""

    await interaction.followup.send(
        f"✅ **upload_kb** stored `{summary['name']}` ({summary['size']:,} bytes)\n"
        f"Location: ``{dest.name}``\n"
        f"Hash SHA256 prefix: ``{summary['sha256']}...\n``\n"
        f"Auto-chunked.{approx_chunks_display}", ephemeral=True
    )


async def handle_list_kb_docs(interaction):
    """List all documents in KB_PATH directory.

    No longer depends on OWUI API — reads directly from the filesystem.
    """
    from config.settings import settings

    docs = list_kb_files(settings.KB_PATH)
    if not docs:
        await interaction.followup.send("No knowledge-base files found.", ephemeral=True)
        return

    lines: list[str] = ["**Knowledge Base** documents:\n"]
    for doc in docs[:15]:  # cap at 15 to avoid huge messages
        size_kb = doc["size"] / 1024 or "~0"
        name = doc.get("name", doc.get("filename", "unknown"))[:60]
        sha8 = (doc.get("sha256", "?")[:8])
        date = doc.get("modified", "?")[:10]
        lines.append(f"  • `{name}` — {size_kb:.1f} KB — {date} — sha:`{sha8}...`")

    if len(docs) > 15:
        lines.append(f"\n\n… and {len(docs) - 15} more document{'s' if len(docs) > 16 else ''}.")
        # Add a note to the user that they can still see the full list locally.
        full_count = f"({len(docs)} total, showing top 15)"
        lines[-1] += f"{full_count}"

    await interaction.followup.send("\n".join(lines), ephemeral=True)
