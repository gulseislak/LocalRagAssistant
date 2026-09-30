"""Deterministic text chunking module for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.ingestion.loader import DocumentPage

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


@dataclass
class TextChunk:
    """Represents a discrete chunk of text with associated document metadata."""

    text: str
    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Splits text and document pages into overlapping fixed-size chunks deterministically."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got: {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step = chunk_size - chunk_overlap

    def split_text(
        self,
        text: str,
        source: str = "unknown",
        page_number: int = 1,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split a single text string into overlapping chunks.

        Args:
            text: Raw or pre-cleaned text to chunk.
            source: Source document filename or identifier.
            page_number: Originating page number.
            base_metadata: Additional metadata to attach to all generated chunks.

        Returns:
            List of TextChunk objects.
        """
        text = text.strip()
        if not text:
            return []

        chunks: List[TextChunk] = []
        text_len = len(text)
        metadata = base_metadata.copy() if base_metadata else {}

        # If text is shorter than or equal to chunk_size, return single chunk
        if text_len <= self.chunk_size:
            chunk_id = f"{source}:p{page_number}:c0"
            return [
                TextChunk(
                    text=text,
                    chunk_id=chunk_id,
                    source=source,
                    page_number=page_number,
                    chunk_index=0,
                    char_start=0,
                    char_end=text_len,
                    metadata=metadata,
                )
            ]

        start = 0
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk_id = f"{source}:p{page_number}:c{chunk_idx}"
                chunks.append(
                    TextChunk(
                        text=chunk_content,
                        chunk_id=chunk_id,
                        source=source,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                        char_start=start,
                        char_end=end,
                        metadata=metadata,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break

            start += self.step

        return chunks

    def chunk_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """Split a list of DocumentPage objects into chunks while maintaining page context.

        Args:
            pages: List of DocumentPage objects extracted from a document.

        Returns:
            Flat list of TextChunk objects across all provided pages.
        """
        all_chunks: List[TextChunk] = []
        for page in pages:
            page_chunks = self.split_text(
                text=page.text,
                source=page.source,
                page_number=page.page_number,
                base_metadata=page.metadata,
            )
            all_chunks.extend(page_chunks)
        return all_chunks


def chunk_document_pages(
    pages: List[DocumentPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Convenience helper to chunk document pages with specified size and overlap."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages)


def chunk_text(
    text: str,
    source: str = "text",
    page_number: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Convenience helper to chunk a standalone text string."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_text(text=text, source=source, page_number=page_number)
