"""Embeddings subpackage for local vector representation."""

from app.embeddings.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
    embed_text,
    embed_texts,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingService",
    "embed_text",
    "embed_texts",
]
