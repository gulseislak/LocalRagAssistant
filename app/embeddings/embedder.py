"""Local text embedding service using Foundry Local SDK."""

from __future__ import annotations

from typing import Any, List, Optional
from foundry_local_sdk import Configuration, FoundryLocalManager

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DEFAULT_APP_NAME = "foundry"


class EmbeddingService:
    """Service for generating dense vector embeddings via Foundry Local SDK."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        app_name: str = DEFAULT_APP_NAME,
    ) -> None:
        self.model_name = model_name
        self.app_name = app_name
        self._ensure_initialized()
        self._embedding_client: Any = None
        self._model: Any = None

    def _ensure_initialized(self) -> None:
        """Initialize FoundryLocalManager singleton if not already initialized."""
        if FoundryLocalManager.instance is None:
            config = Configuration(app_name=self.app_name)
            FoundryLocalManager.initialize(config)

    def _get_model(self) -> Any:
        """Retrieve and prepare the embedding model."""
        if self._model is None:
            manager = FoundryLocalManager.instance
            if manager is None:
                raise RuntimeError("FoundryLocalManager is not initialized.")
            model = manager.catalog.get_model(self.model_name)
            if model is None:
                raise ValueError(
                    f"Model '{self.model_name}' was not found in the Foundry Local catalog."
                )
            if not model.is_cached:
                model.download()
            if not model.is_loaded:
                model.load()
            self._model = model
        return self._model

    def _get_client(self) -> Any:
        """Retrieve and cache the EmbeddingClient instance."""
        if self._embedding_client is None:
            model = self._get_model()
            self._embedding_client = model.get_embedding_client()
        return self._embedding_client

    def embed_text(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single text string.

        Args:
            text: Non-empty string to embed.

        Returns:
            List of floats representing the numerical vector embedding.

        Raises:
            ValueError: If the text is empty, whitespace-only, or not a string.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        client = self._get_client()
        response = client.generate_embedding(text.strip())

        if not response.data or len(response.data) == 0:
            raise RuntimeError("Embedding client returned an empty response.")

        return [float(val) for val in response.data[0].embedding]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of text strings.

        Args:
            texts: List of non-empty strings to embed.

        Returns:
            List of vector embeddings, one per input text.

        Raises:
            ValueError: If texts is empty or contains non-string / empty elements.
        """
        if not isinstance(texts, list):
            raise ValueError("Inputs must be a list of non-empty strings.")

        if len(texts) == 0:
            return []

        cleaned_texts: List[str] = []
        for i, item in enumerate(texts):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Item at index {i} must be a non-empty string.")
            cleaned_texts.append(item.strip())

        client = self._get_client()
        response = client.generate_embeddings(cleaned_texts)

        if not response.data:
            raise RuntimeError("Embedding client returned an empty response.")

        return [[float(val) for val in item.embedding] for item in response.data]


def embed_text(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> List[float]:
    """Convenience helper to generate an embedding for a single text string."""
    service = EmbeddingService(model_name=model_name)
    return service.embed_text(text)


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> List[List[float]]:
    """Convenience helper to generate embeddings for multiple text strings."""
    service = EmbeddingService(model_name=model_name)
    return service.embed_texts(texts)
