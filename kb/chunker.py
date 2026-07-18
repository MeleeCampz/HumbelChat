"""Smart document chunking for KB vector search.

Splits KB documents into semantically coherent chunks by:
1. Markdown headers (## level or deeper) — primary splitting strategy
2. Paragraph boundaries (~350-500 chars) — fallback when no headers exist

Each chunk preserves context about which section/file it came from.

Usage
-----
    from kb.chunker import Chunker

    chunks = await Chunker.split_file("path/to/doc.md")
    # Returns: list[ChunkInfo] with metadata + content

"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field


@dataclass
class ChunkInfo:
    """A single semantic chunk extracted from a KB document."""
    display_name: str
    source_file: str  # original filename
    section_path: str  # hierarchical path like "Chapter 1 -> Section A"
    content: str
    header_hash: str = ""  # hash of header text for deduplication


class Chunker:
    """Split documents into semantic chunks for better embedding quality."""

    MAX_CHUNK_SIZE = 500  # chars per chunk (target)
    MIN_CHUNK_SIZE = 80   # skip chunks smaller than this
    HEADER_RE = re.compile(r"^(#{2,6})\s+(.+)$", re.MULTILINE)

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

        lines = content_text.splitlines()[:max_lines_per_file]
        truncated = "\n".join(lines)
        if len(content_text.splitlines()) > max_lines_per_file:
            truncated += "\n... [truncated]"

        source_name = root.name
        display_name = cls._normalize_display_name(root, source_name)

        chunks: list[ChunkInfo] = []

        # Strategy 1: Split by Markdown headers (## through ######)
        header_chunks = cls._split_by_headers(truncated, display_name, source_name)
        if header_chunks:
            chunks.extend(header_chunks)

        # Strategy 2: If no headers found, split by paragraphs/size
        if not chunks:
            chunks.extend(cls._split_by_paragraphs(truncated, display_name, source_name))

        return chunks

    @classmethod
    def _split_by_headers(
        cls, content: str, display_name: str, source_file: str
    ) -> list[ChunkInfo]:
        """Split content by Markdown headers (## through ######)."""
        # Find all header positions and their text
        headers = list(cls.HEADER_RE.finditer(content))
        if len(headers) < 2:
            return []  # need at least 2 headers to split into sections

        chunks: list[ChunkInfo] = []
        
        for i, header_match in enumerate(headers):
            header_text = header_match.group(2).strip()
            level = len(header_match.group(1))
            
            # Skip title-level (#) or very high levels (6+) unless it's the only header
            if level >= 6:
                continue

            # Get content from this header to the next (or end of file)
            start = header_match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            
            section_content = content[start:end].strip()
            
            # Don't apply size filter to header-based chunks — a short section under
            # a meaningful header is still valuable for retrieval (e.g., "Time System" might be one line)
            if not section_content or not section_content.replace('#', '').replace('*', '').strip():
                continue

            chunks.append(
                ChunkInfo(
                    display_name=display_name,
                    source_file=source_file,
                    section_path=header_text,
                    content=section_content,
                    header_hash=cls._hash(header_text),
                )
            )

        return chunks

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
