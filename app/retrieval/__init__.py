"""Retrieval subpackage for vector similarity search."""

from app.retrieval.retriever import (
    DEFAULT_TOP_K,
    RetrievalResult,
    VectorRetriever,
    cosine_similarity,
)

__all__ = [
    "DEFAULT_TOP_K",
    "RetrievalResult",
    "VectorRetriever",
    "cosine_similarity",
]
