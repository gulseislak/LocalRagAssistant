"""Unit tests for the vector retrieval and cosine similarity module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from app.retrieval.retriever import (
    RetrievalResult,
    VectorRetriever,
    cosine_similarity,
)
from app.storage.database import SQLiteStorage, StoredChunk


class TestVectorRetriever(unittest.TestCase):
    """Test suite for VectorRetriever and cosine_similarity function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_retrieval.db"
        self.storage = SQLiteStorage(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cosine_similarity_math(self):
        """Verify cosine similarity mathematical properties."""
        # Identical vectors -> 1.0
        v1 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v1, v1), 1.0, places=5)

        # Orthogonal vectors -> 0.0
        v_x = [1.0, 0.0, 0.0]
        v_y = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v_x, v_y), 0.0, places=5)

        # Opposite vectors -> -1.0
        v_pos = [1.0, 2.0]
        v_neg = [-1.0, -2.0]
        self.assertAlmostEqual(cosine_similarity(v_pos, v_neg), -1.0, places=5)

        # Zero norm safety
        v_zero = [0.0, 0.0, 0.0]
        self.assertEqual(cosine_similarity(v_zero, v1), 0.0)

        # Dimension mismatch
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_query_validation(self):
        """Verify invalid queries or top_k values raise ValueError."""
        retriever = VectorRetriever(storage=self.storage)

        with self.assertRaises(ValueError):
            retriever.retrieve("")

        with self.assertRaises(ValueError):
            retriever.retrieve("   \n  ")

        with self.assertRaises(ValueError):
            retriever.retrieve("valid query", top_k=0)

        with self.assertRaises(ValueError):
            retriever.retrieve("valid query", top_k=-2)

    def test_empty_database_retrieval(self):
        """Verify retrieval on an empty database returns an empty list without error."""
        retriever = VectorRetriever(storage=self.storage)
        results = retriever.retrieve("Any query text")
        self.assertEqual(results, [])

    def test_ranking_and_top_k(self):
        """Verify similarity ranking is strictly descending and honors top_k."""
        # Create mock embedding service
        mock_embedder = MagicMock()
        # Query embedding: along x axis [1.0, 0.0, 0.0]
        mock_embedder.embed_text.return_value = [1.0, 0.0, 0.0]

        # Insert 3 chunks with known directions:
        # Chunk A: highly aligned [0.9, 0.1, 0.0]
        # Chunk B: moderately aligned [0.5, 0.5, 0.0]
        # Chunk C: orthogonal [0.0, 1.0, 0.0]
        chunks = [
            StoredChunk(
                id="chunk_c",
                source="doc.pdf",
                page_number=1,
                chunk_index=2,
                text="Orthogonal content",
                embedding=[0.0, 1.0, 0.0],
            ),
            StoredChunk(
                id="chunk_a",
                source="doc.pdf",
                page_number=1,
                chunk_index=0,
                text="High similarity content",
                embedding=[0.9, 0.1, 0.0],
            ),
            StoredChunk(
                id="chunk_b",
                source="doc.pdf",
                page_number=1,
                chunk_index=1,
                text="Medium similarity content",
                embedding=[0.5, 0.5, 0.0],
            ),
        ]
        self.storage.insert_chunks(chunks)

        retriever = VectorRetriever(storage=self.storage, embedding_service=mock_embedder)

        # Retrieve top 2
        results = retriever.retrieve(query="test query", top_k=2)

        self.assertEqual(len(results), 2)
        # Top result should be chunk_a
        self.assertEqual(results[0].chunk_id, "chunk_a")
        self.assertEqual(results[1].chunk_id, "chunk_b")
        self.assertTrue(results[0].score >= results[1].score)

        # Verify all 3
        all_results = retriever.retrieve(query="test query", top_k=3)
        self.assertEqual(len(all_results), 3)
        self.assertEqual(all_results[0].chunk_id, "chunk_a")
        self.assertEqual(all_results[1].chunk_id, "chunk_b")
        self.assertEqual(all_results[2].chunk_id, "chunk_c")


if __name__ == "__main__":
    unittest.main()
