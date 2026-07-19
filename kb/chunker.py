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

import logging
import pathlib
import re
from dataclasses import dataclass, field

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
        max_lines_per_file: int = 50,
    ) -> list[ChunkInfo]:
        """Split a single file into semantic chunks.

        Parameters
        ----------
        file_path : Path to the file to chunk.
        max_lines_per_file : Limit content to this many lines (prevents OOM).

        Returns
        -------
        List of ``ChunkInfo`` objects representing semantically coherent sections.
        """
        root = pathlib.Path(file_path)
        if not root.exists():
            return []

        content_text = root.read_bytes().decode("utf-8", errors="replace")
        if not content_text:
            return []

        source_name = root.name
        display_name = cls._normalize_display_name(root, source_name)

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

        # For larger files, cap lines to prevent OOM during processing
        lines = content_text.splitlines()[:max_lines_per_file]
        truncated = "\n".join(lines)
        if len(content_text.splitlines()) > max_lines_per_file:
            truncated += "\n... [truncated]"

        chunks: list[ChunkInfo] = []

        # 2. Smart Header Splitting with Minimum-Size Merging
        header_chunks = cls._split_by_headers(truncated, display_name, source_name)
        if header_chunks:
            chunks.extend(header_chunks)

        # 3. Fallback to adaptive chunking if no headers found
        if not chunks:
            chunks.extend(cls._split_adaptive(truncated, display_name, source_name))

        logger.debug("File %s (%d chars): produced %d chunk(s)", source_name, len(content_text), len(chunks))
        return chunks

    @classmethod
    def _split_by_headers(
        cls, content: str, display_name: str, source_file: str
    ) -> list[ChunkInfo]:
        """Split content by Markdown headers with minimum-size merging.

        Collects all header-based sections, then merges adjacent small chunks together
        so no chunk falls below MIN_CHUNK_SIZE. This prevents the "tiny fragment" problem
        where lots of ## headers break a document into unusable pieces.

        Headers are split at every level (# through ######) to capture full context,
        but merged when necessary.
        """
        # Step 1: Collect all header regions (each includes its header line + content until next header)
        headers = list(cls.HEADER_RE.finditer(content))
        if not headers:
            return []

        raw_chunks: list[tuple[str, str]] = []  # (header_line_text, section_content_after_header)

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

        # Step 2: Merge adjacent chunks if current chunk is below MIN_CHUNK_SIZE.
        # We accumulate into 'merged' until adding the next chunk would exceed MAX_CHUNK_SIZE.
        merged: list[str] = []
        accumulator: str = ""

        for text, size in raw_chunks:
            if size >= cls.MIN_CHUNK_SIZE and not accumulator:
                # This chunk is large enough on its own — emit immediately
                merged.append(text)
                continue

            if accumulator:
                # Try to merge with the current accumulated content
                combined_size = len(accumulator) + size
                if combined_size <= cls.MAX_CHUNK_SIZE:
                    # Merge: insert a separator between old and new section
                    accumulator += "\n---\n" + text
                    continue
                else:
                    # Would exceed max — emit accumulator, start new one
                    merged.append(accumulator)
                    accumulator = text
                    continue

            # First chunk or nothing accumulated yet
            accumulator = text

        # Flush any remaining accumulator
        if accumulator:
            merged.append(accumulator)

        if not merged:
            return []

        logger.debug("Header split + merge produced %d chunk(s)", len(merged))

        # Step 3: Convert to ChunkInfo objects
        result: list[ChunkInfo] = []
        for idx, section_text in enumerate(merged):
            # Extract the primary header text for section_path
            first_header = re.match(r"^(#{1,6})\s+(.+)$", section_text.strip(), re.MULTILINE)
            section_path = first_header.group(2).strip() if first_header else f"Section {idx + 1}"

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
    def _normalize_display_name(p: pathlib.Path, base_name: str) -> str:
        """Build human-readable display name from path and filename."""
        stem = p.stem
        clean_stem = re.sub(r"^\d+", "", stem)
        return f"{clean_stem}{p.suffix}" if clean_stem else base_name

    @staticmethod
    def _hash(text: str) -> str:
        """Simple hash for header deduplication."""
        return hex(abs(hash(text)))[-8:]


def _extract_ext(name: str) -> str:
    """Extract file extension (lowercased), stripping any query-string suffix."""
    base = name.split("?")[0]
    i = base.rfind(".")
    return base[i:].lower() if i > 0 else ""


# Reuse the normalization helper from vector_db for consistency
def _normalize_display_name(p: pathlib.Path, base_name: str) -> str:
    """Build human-readable display name from path and filename."""
    stem = p.stem
    clean_stem = re.sub(r"^\d+", "", stem)
    return f"{clean_stem}{p.suffix}" if clean_stem else base_name
