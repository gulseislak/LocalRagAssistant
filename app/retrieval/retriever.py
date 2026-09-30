"""Vector similarity retrieval module for local RAG assistant."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.embeddings.embedder import EmbeddingService
from app.storage.database import SQLiteStorage, StoredChunk

DEFAULT_TOP_K = 3


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute the cosine similarity between two float vectors.

    Args:
        vec_a: First float vector.
        vec_b: Second float vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0 (or 0.0 for zero-norm vectors).
    """
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Clamp to [-1.0, 1.0] range to guard against float precision overflow
    sim = dot_product / (norm_a * norm_b)
    return max(-1.0, min(1.0, float(sim)))


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with associated similarity score."""

    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorRetriever:
    """Retrieves most relevant document chunks for a query using cosine similarity."""

    def __init__(
        self,
        storage: Optional[SQLiteStorage] = None,
        embedding_service: Optional[EmbeddingService] = None,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.storage = storage or SQLiteStorage()
        self.embedding_service = embedding_service or EmbeddingService()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k most similar document chunks for a user query.

        Args:
            query: User's search query or question.
            top_k: Number of top chunks to return (defaults to default_top_k).
            min_score: Optional minimum similarity threshold filter.

        Returns:
            List of RetrievalResult objects sorted descending by similarity score.

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {k}")

        stored_chunks: List[StoredChunk] = self.storage.get_all_chunks()
        if not stored_chunks:
            return []

        query_embedding: List[float] = self.embedding_service.embed_text(query)

        scored_results: List[RetrievalResult] = []
        for chunk in stored_chunks:
            sim = cosine_similarity(query_embedding, chunk.embedding)
            if min_score is not None and sim < min_score:
                continue

            scored_results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=round(sim, 5),
                    metadata=chunk.metadata,
                )
            )

        # Sort descending by similarity score
        scored_results.sort(key=lambda r: r.score, reverse=True)

        return scored_results[:k]
