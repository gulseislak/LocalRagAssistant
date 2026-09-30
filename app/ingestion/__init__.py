"""Document ingestion and text chunking subpackage."""

from app.ingestion.loader import (
    DocumentLoaderError,
    DocumentPage,
    PDFLoader,
    load_pdf,
)
from app.ingestion.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    TextChunk,
    TextChunker,
    chunk_document_pages,
    chunk_text,
)

__all__ = [
    "DocumentLoaderError",
    "DocumentPage",
    "PDFLoader",
    "load_pdf",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "TextChunk",
    "TextChunker",
    "chunk_document_pages",
    "chunk_text",
]
