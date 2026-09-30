"""Smoke and unit tests for the local embedding service module."""

import unittest
from app.embeddings.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
    embed_text,
    embed_texts,
)


class TestEmbeddingService(unittest.TestCase):
    """Test suite for EmbeddingService and embedding helpers."""

    def test_configuration_and_defaults(self):
        """Verify module configuration and default embedding model alias."""
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "qwen3-embedding-0.6b")
        service = EmbeddingService()
        self.assertEqual(service.model_name, "qwen3-embedding-0.6b")

    def test_input_validation(self):
        """Verify validation of empty strings and invalid input types."""
        service = EmbeddingService()

        with self.assertRaises(ValueError):
            service.embed_text("")

        with self.assertRaises(ValueError):
            service.embed_text("   \n\t  ")

        with self.assertRaises(ValueError):
            service.embed_texts(["valid text", ""])

        with self.assertRaises(ValueError):
            service.embed_texts(["valid text", "   "])

        # Empty list should return empty list
        self.assertEqual(service.embed_texts([]), [])

    def test_local_embedding_smoke_test(self):
        """Perform a real local embedding smoke test using qwen3-embedding-0.6b."""
        test_text = "Retrieval Augmented Generation with local embeddings on Apple Silicon."
        embedding = embed_text(test_text)

        # Confirm return type is a list
        self.assertIsInstance(embedding, list)

        # Confirm stable, non-zero dimension (1024 for Qwen3-Embedding-0.6B)
        self.assertEqual(len(embedding), 1024)

        # Confirm all elements are numeric floats
        self.assertTrue(all(isinstance(x, (float, int)) for x in embedding))

        # Test batch embedding
        batch_inputs = [
            "First chunk text for vector generation.",
            "Second chunk text for vector generation.",
        ]
        batch_embeddings = embed_texts(batch_inputs)
        self.assertEqual(len(batch_embeddings), 2)
        self.assertEqual(len(batch_embeddings[0]), 1024)
        self.assertEqual(len(batch_embeddings[1]), 1024)


if __name__ == "__main__":
    unittest.main()
