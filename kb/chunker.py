"""Smart document chunking for KB vector search.

Splits KB documents into semantically coherent chunks by:
1. Full document strategy for small files (≤8000 chars) — preserves context
2. Smart header-based splitting with min-size merging for larger docs
3. Adaptive paragraph boundaries as fallback with structural awareness

Usage
-----
    from kb.chunker import Chunker

    chunks = await Chunker.split_file("path/to/doc.md")
    # Returns: list[ChunkInfo] with metadata + content

"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import re
from dataclasses import dataclass

logger = logging.getLogger("kb.chunker")


@dataclass
class ChunkInfo:
    """A single semantic chunk extracted from a KB document."""
    display_name: str
    source_file: str  # original filename
    section_path: str  # hierarchical path like "Chapter 1 -> Section A"
    content: str
    header_hash: str = ""  # hash of header text for deduplication


class Chunker:
    """Split documents into semantic chunks for better embedding quality.

    Uses Full Document strategy for small files and Smart Header Splitting with
    minimum-size merging for larger docs to prevent tiny, semantically broken chunks.
    """

    MIN_CHUNK_SIZE = 80   # chars — below this, merge with neighbor
    MAX_CHUNK_SIZE = 7500  # chars — never exceed embedder context safety margin (2048 tokens)
    HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @classmethod
    async def split_file(
        cls,
        file_path: str | pathlib.Path,
    ) -> list[ChunkInfo]:
        """Split a single file into semantic chunks.

        Parameters
        ----------
        file_path : Path to the file to chunk.

        Returns
        -------
        List of ``ChunkInfo`` objects representing semantically coherent sections.

        The (blocking) read + regex pass runs in a worker thread — P0 #4:
        chunking a large document on the event loop froze the whole bot.
        """
        return await asyncio.to_thread(cls.split_file_sync, file_path)

    @classmethod
    def split_file_sync(
        cls,
        file_path: str | pathlib.Path,
    ) -> list[ChunkInfo]:
        """Synchronous chunking core — call via :meth:`split_file` (or
        ``asyncio.to_thread``) from async code.
        """
        root = pathlib.Path(file_path)
        if not root.exists():
            return []

        try:
            if root.stat().st_size > 1024 * 1024:
                logger.warning("Skipping %s — larger than the 1 MB index cap", root.name)
                return []
        except OSError:
            return []

        content_text = root.read_bytes().decode("utf-8", errors="replace")
        if not content_text:
            return []

        source_name = root.name
        display_name = _normalize_display_name(root, source_name)

        # 1. Full Document Strategy: files ≤ 8000 chars stay intact to preserve semantic context
        if len(content_text) <= 8000:
            logger.debug("File %s (%d chars): full document", source_name, len(content_text))
            return [
                ChunkInfo(
                    display_name=display_name,
                    source_file=source_name,
                    section_path="Full Document",
                    content=content_text.strip(),
                    header_hash="",
                )
            ]

        chunks: list[ChunkInfo] = []

        # 2. Smart Header Splitting with Minimum-Size Merging
        header_chunks = cls._split_by_headers(content_text, display_name, source_name)
        if header_chunks:
            chunks.extend(header_chunks)

        # 3. Fallback to adaptive chunking if no headers found
        if not chunks:
            chunks.extend(cls._split_adaptive(content_text, display_name, source_name))

        logger.debug("File %s (%d chars): produced %d chunk(s)", source_name, len(content_text), len(chunks))
        return chunks

    @classmethod
    def _split_by_headers(
        cls, content: str, display_name: str, source_file: str
    ) -> list[ChunkInfo]:
        """Split content by Markdown headers — one chunk per section.

        Each header-delimited section that is at least MIN_CHUNK_SIZE becomes its own
        chunk, so individually meaningful entries (spells, monsters, items) are each
        retrievable on their own. Only tiny fragments below MIN_CHUNK_SIZE are folded
        into a neighbouring section to avoid unusably small orphans, and any single
        oversized section is hard-split to respect MAX_CHUNK_SIZE.

        Headers are split at every level (# through ######) so the finest meaningful
        unit of structure is preserved.
        """
        # Step 1: Collect all header regions (each includes its header line + content until next header)
        headers = list(cls.HEADER_RE.finditer(content))
        if not headers:
            return []

        raw_chunks: list[tuple[str, int]] = []  # (chunk text, size)

        preamble = content[: headers[0].start()].strip()
        if preamble:
            raw_chunks.append((preamble, len(preamble)))

        for i, header_match in enumerate(headers):
            header_text = header_match.group(2).strip()
            level = len(header_match.group(1))
            prefix = "#" * level

            # Get content from this header to the next (or end of file)
            start = header_match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            section_content = content[start:end].strip()
            if not section_content or not section_content.replace('#', '').replace('*', '').strip():
                continue

            # Full chunk including the header line itself
            full_chunk_text = f"{prefix} {header_text}\n{section_content}"
            raw_chunks.append((full_chunk_text, len(full_chunk_text)))

        if not raw_chunks:
            return []

        # Step 2: Emit each real section (>= MIN_CHUNK_SIZE) as its OWN chunk so every
        # self-contained entry (a spell, a monster, an item) stays individually retrievable.
        # Tiny fragments (< MIN_CHUNK_SIZE) are folded into the next real section — or the
        # previous one if they trail the document — so we never emit unusably small orphans.
        # A single section that is itself oversized is hard-split so no chunk exceeds
        # MAX_CHUNK_SIZE (the embedder context safety margin).
        merged: list[str] = []
        pending_small: str = ""

        for text, size in raw_chunks:
            if size >= cls.MIN_CHUNK_SIZE:
                pieces = cls._hard_split(text)
                if pending_small:
                    pieces[0] = pending_small + "\n---\n" + pieces[0]
                    pending_small = ""
                merged.extend(pieces)
            else:
                pending_small = (pending_small + "\n---\n" + text) if pending_small else text

        # Flush trailing tiny fragments: attach to the last real chunk, or emit alone.
        if pending_small:
            if merged:
                merged[-1] = merged[-1] + "\n---\n" + pending_small
            else:
                merged.append(pending_small)

        if not merged:
            return []

        logger.debug("Header split produced %d chunk(s)", len(merged))

        # Step 3: Convert to ChunkInfo objects
        result: list[ChunkInfo] = []
        for idx, section_text in enumerate(merged):
            # Extract the primary header text for section_path
            first_header = re.match(r"^(#{1,6})\s+(.+)$", section_text.strip(), re.MULTILINE)
            if first_header:
                section_path = first_header.group(2).strip()
            elif not section_text.lstrip().startswith("#"):
                section_path = "Preamble"
            else:
                section_path = f"Section {idx + 1}"

            result.append(
                ChunkInfo(
                    display_name=display_name,
                    source_file=source_file,
                    section_path=section_path,
                    content=section_text.strip(),
                    header_hash=cls._hash(section_text),
                )
            )
        return result

    @classmethod
    def _hard_split(cls, text: str) -> list[str]:
        """Split *text* into pieces each <= MAX_CHUNK_SIZE (used for oversized sections).

        Splits on lines so structural content (lists/tables) stays intact as far as
        possible; the first piece keeps the section's header line.
        """
        if len(text) <= cls.MAX_CHUNK_SIZE:
            return [text]
        pieces: list[str] = []
        cur = ""
        size = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            add = len(line) + (1 if cur else 0)
            if cur and size + add > cls.MAX_CHUNK_SIZE:
                pieces.append(cur)
                cur, size = line, len(line)
            else:
                cur = f"{cur}\n{line}" if cur else line
                size += add
        if cur:
            pieces.append(cur)
        return [p for p in pieces if p.strip()]

    @classmethod
    def _split_by_paragraphs(
        cls, content: str, display_name: str, source_file: str
    ) -> list[ChunkInfo]:
        """Split content by paragraphs or fixed-size chunks."""
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content.strip())

        chunks: list[ChunkInfo] = []
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < cls.MIN_CHUNK_SIZE:
                continue

            # If adding this paragraph exceeds max chunk size, emit current chunk
            if current_chunk and current_size + len(para) > cls.MAX_CHUNK_SIZE:
                chunks.append(
                    ChunkInfo(
                        display_name=display_name,
                        source_file=source_file,
                        section_path="Paragraph group",
                        content="\n\n".join(current_chunk),
                        header_hash="",
                    )
                )
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += len(para)

        # Emit remaining chunk
        if current_chunk:
            chunks.append(
                ChunkInfo(
                    display_name=display_name,
                    source_file=source_file,
                    section_path="Paragraph group",
                    content="\n\n".join(current_chunk),
                    header_hash="",
                )
            )

        return chunks

    @classmethod
    def _split_adaptive(
        cls, content: str, display_name: str, source_file: str
    ) -> list[ChunkInfo]:
        """Adaptive Chunking: choose best strategy based on document structure.

        Uses intrinsic metrics (Block Integrity, Structural Coherence) to decide
        whether header-based or paragraph-based chunking is superior for this specific file.
        """
        line_count = len(content.splitlines())

        # If small enough, keep as single block (preserves semantic coherence)
        if line_count < 50:
            logger.debug("File %s: adaptive split to single block (%d lines)", source_file, line_count)
            return [
                ChunkInfo(
                    display_name=display_name,
                    source_file=source_file,
                    section_path="Full Document",
                    content=content.strip(),
                    header_hash="",
                )
            ]

        # Check for list/table markers that might be broken by paragraph splitting
        list_markers = sum(1 for line in content.splitlines() if line.strip().startswith(('-', '*', '•', '>', '|')))
        paragraph_count = len(re.split(r"\n\s*\n", content.strip()))

        # If there are many structural markers but few paragraphs, paragraph splitting
        # would break Block Integrity. Use recursive size-based chunking instead.
        if list_markers > paragraph_count and line_count > 100:
            logger.debug("File %s: adaptive split by structure (dense content, %d lines)", source_file, line_count)
            return cls._split_recursive_by_size(content, display_name, source_file)

        # Default: use standard paragraph splitting for structure-awareness
        return cls._split_by_paragraphs(content, display_name, source_file)

    @classmethod
    def _split_recursive_by_size(
        cls, content: str, display_name: str, source_file: str
    ) -> list[ChunkInfo]:
        """Recursive size-based splitting for dense/unstructured content.

        Splits by individual lines while respecting MAX_CHUNK_SIZE to preserve
        structural integrity (lists, tables) better than paragraph splitting.
        """
        chunks: list[ChunkInfo] = []
        current_lines: list[str] = []
        current_size = 0

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if current_size + len(stripped) > cls.MAX_CHUNK_SIZE and current_lines:
                chunks.append(
                    ChunkInfo(
                        display_name=display_name,
                        source_file=source_file,
                        section_path="Structured chunk",
                        content="\n".join(current_lines),
                        header_hash="",
                    )
                )
                current_lines = []
                current_size = 0

            current_lines.append(stripped)
            current_size += len(stripped)

        if current_lines:
            chunks.append(
                ChunkInfo(
                    display_name=display_name,
                    source_file=source_file,
                    section_path="Structured chunk",
                    content="\n".join(current_lines),
                    header_hash="",
                )
            )

        return chunks

    @staticmethod
    def _hash(text: str) -> str:
        """Simple hash for header deduplication."""
        return hex(abs(hash(text)))[-8:]


def _normalize_display_name(p: pathlib.Path, base_name: str) -> str:
    """Build human-readable display name from path and filename."""
    stem = p.stem
    clean_stem = re.sub(r"^\d+", "", stem)
    return f"{clean_stem}{p.suffix}" if clean_stem else base_name

