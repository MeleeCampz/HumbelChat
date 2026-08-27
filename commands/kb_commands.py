"""Knowledge-base commands — /upload_kb, /list_kb_docs and /reindex_kb."""
from __future__ import annotations

import asyncio
import logging
import pathlib

import httpx

from kb.storage import validate_upload, list_kb_files

log = logging.getLogger("bot.commands.kb")

# URL-upload guard: cap the bytes we will ever pull into memory.
UPLOAD_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # matches storage.MAX_FILE_SIZE
UPLOAD_DOWNLOAD_TIMEOUT = 60.0  # seconds


async def handle_upload_kb(
    interaction,                           # Discord Interaction
    kb_name: str | None = None,            # override for KB folder name
    url: str | None = None,                 # remote URL → download
    attachment=None,                        # discord.Attachment or None
    subfolder: str | None = None,           # optional subfolder
) -> None:
    """Upload a file directly to the local KB storage directory."""
    # Defer first — URL downloads can exceed Discord's 15 s interaction window.
    await interaction.response.defer(ephemeral=True)

    # --- step 1: get bytes ---
    if attachment is not None:
        data = await attachment.read()
        fname = attachment.filename or "attachment"
    elif url:
        try:
            async with httpx.AsyncClient(timeout=UPLOAD_DOWNLOAD_TIMEOUT) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                data = resp.content
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            await interaction.followup.send(
                f"⚠️ Failed to download `{url[:120]}`: {exc.__class__.__name__}", ephemeral=True
            )
            return
        if len(data) > UPLOAD_MAX_DOWNLOAD_BYTES:
            await interaction.followup.send(
                f"⚠️ Remote file too large (> {UPLOAD_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB).", ephemeral=True
            )
            return
        fname = url.split("?")[0].split("/")[-1] or "remote_file"
    else:
        await interaction.followup.send(
            "Please provide either a URL or file attachment for /upload_kb.", ephemeral=True
        )
        return

    # --- step 2 & 3: validate + write to KB_PATH ---
    try:
        dest, summary = validate_upload(data, filename=fname, kb_path=None, subfolder=subfolder)
    except ValueError as exc:
        await interaction.followup.send(f"Upload rejected: **{exc}**", ephemeral=True)
        return
    except FileNotFoundError as exc:
        await interaction.followup.send(f"KB storage not found: **{exc}**", ephemeral=True)
        return

    # --- step 4: index the new document so RAG can find it immediately ---
    indexed = False
    try:
        from kb.retrievers import update_kb_document
        indexed = await update_kb_document(dest)
    except Exception as exc:
        log.warning("Auto-index after upload failed for %s: %s", dest.name, exc)

    index_note = "Indexed for search." if indexed else "⚠️ Not auto-indexed — run `/reindex_kb` to make it searchable."

    # --- step 5: reply with summary ---
    approx_chunks_display = ""
    try:
        from config.settings import CHUNK_TARGET
        n = len(pathlib.Path(dest).read_text(encoding="utf-8", errors="replace"))
        if CHUNK_TARGET:
            approx_chunks_display = f" (approx {n // CHUNK_TARGET} chunks)"
    except Exception:
        pass

    await interaction.followup.send(
        f"✅ **upload_kb** stored `{summary['name']}` ({summary['size']:,} bytes)\n"
        f"Location: ``{dest.name}``\n"
        f"Hash SHA256 prefix: ``{summary['sha256']}...``\n"
        f"Auto-chunked.{approx_chunks_display}\n"
        f"{index_note}", ephemeral=True
    )


def get_root_directories(kb_path: pathlib.Path) -> list[str]:
    """Return sorted list of subdirectory names at the root level.

    Hidden directories (dot-prefixed, e.g. .vector_index_cache) are
    internal state and never listed as KB folders.
    """
    if not kb_path.exists():
        return []
    dirs = []
    for entry in kb_path.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            dirs.append(entry.name)
    return sorted(dirs)


async def handle_list_kb_docs(interaction, subfolder_path: str | None = None):
    """List all documents in KB_PATH directory.

    If *subfolder_path* is given, recurses into that subfolder.
    Otherwise shows only root-level items (files + directories).
    """
    from config.settings import KB_PATH

    if subfolder_path:
        # Subfolder view: recurse into that path
        docs = list_kb_files(KB_PATH, subfolder=subfolder_path, recursive=True)
        lines: list[str] = [
            f"**Knowledge Base** documents — `{subfolder_path}`",
            "📂 **Subdirectories:**",
        ]
        # Show nested directories within the subfolder
        scan_root = pathlib.Path(KB_PATH) / subfolder_path
        subdirs = get_root_directories(scan_root)
        for d in sorted(subdirs):
            lines.append(f"  📂 `{d}`")
    else:
        # Root view: show directories + root-level files only
        docs = list_kb_files(KB_PATH, subfolder=None, recursive=False)
        dirs = get_root_directories(KB_PATH)
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
        size_kb = doc["size"] / 1024  # float; never a string (0-byte files are valid)
        name = doc.get("name", doc.get("filename", "unknown"))
        sha8 = (doc.get("sha256", "?")[:8])
        date = doc.get("modified", "?")[:10]
        lines.append(f"  • `{name}` — {size_kb:.1f} KB — {date} — sha:`{sha8}...`")

    if len(docs) > 30:
        lines.append(f"\n… and {len(docs) - 30} more documents.")

    await interaction.response.send_message("\n".join(lines), ephemeral=True)


async def handle_reindex_kb(interaction):
    """Trigger reindexing of all files in the KB using the vector index."""
    from config.settings import KB_PATH

    kb_path = pathlib.Path(KB_PATH)
    await interaction.response.defer(ephemeral=True)

    # --- Phase 2: use persistent vector index (KBIndexStore) ---
    from kb.index import KBIndexStore
    from kb.retrievers import retrieve_kb_documents, DEFAULT_METHOD

    strategy = DEFAULT_METHOD
    msg_parts: list[str] = []

    try:
        store = KBIndexStore(kb_path)
        idx = await store.load(force_rebuild=True)

        if idx is None or idx.is_empty():
            # Index didn't build — check why (no KB files? no embedding backend?)
            docs = list_kb_files(kb_path, recursive=True)
            msg_parts.append("❌ Vector index could not be built.")
            msg_parts.append(f"KB has {len(docs)} file(s) but 0 chunks.")
        else:
            doc_count = idx.count()
            msg_parts.append(f"✅ Successfully rebuilt the vector index for **{kb_path}**.")
            msg_parts.append(f"   • **{doc_count:,}** chunk(s) indexed")
            msg_parts.append(f"   • Strategy: `{strategy}`")

            # Quick sanity test — does retrieval actually work?
            try:
                sample_query = "test"
                results = await retrieve_kb_documents(sample_query, kb_path, strategy=strategy, top_n=3)
                if results:
                    msg_parts.append(f"   • Retrieval test: {len(results)} document(s) found")
                else:
                    msg_parts.append("   • ⚠️ Retrieval returned 0 documents for a sample query")
            except Exception as re:
                msg_parts.append(f"   • ⚠️ Retrieval sanity check failed: {re}")

        await store.shutdown()

    except Exception as e:
        log.error("Reindexing failed: %s", e, exc_info=True)
        msg_parts = [f"❌ Failed to reindex KB: **{e}**"]

    await interaction.followup.send("\n".join(msg_parts), ephemeral=True)
