"""Unit and integration tests for SQLite storage module."""

import tempfile
import unittest
from pathlib import Path
from app.storage.database import SQLiteStorage, StoredChunk


class TestSQLiteStorage(unittest.TestCase):
    """Test suite for SQLiteStorage vector and metadata persistence."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_rag.db"
        self.storage = SQLiteStorage(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_initialization_and_table_creation(self):
        """Verify database file and tables/indexes are created properly."""
        self.assertTrue(self.db_path.exists())
        self.assertEqual(self.storage.count_chunks(), 0)

    def test_embedding_serialization_roundtrip(self):
        """Verify 1024-dimensional float embedding serialization and deserialization."""
        original_vector = [float(i) * 0.001 for i in range(1024)]
        blob = self.storage.serialize_embedding(original_vector)

        # 1024 floats * 4 bytes = 4096 bytes
        self.assertEqual(len(blob), 4096)

        restored_vector = self.storage.deserialize_embedding(blob)
        self.assertEqual(len(restored_vector), 1024)

        for orig, rest in zip(original_vector, restored_vector):
            self.assertAlmostEqual(orig, rest, places=5)

    def test_empty_embedding_serialization_error(self):
        """Verify serializing an empty list raises ValueError."""
        with self.assertRaises(ValueError):
            self.storage.serialize_embedding([])

    def test_single_chunk_insertion_and_retrieval(self):
        """Verify inserting and retrieving a single chunk with metadata and embedding."""
        sample_vector = [0.1] * 1024
        chunk = StoredChunk(
            id="doc.pdf:p1:c0",
            source="doc.pdf",
            page_number=1,
            chunk_index=0,
            text="Artificial intelligence on local devices.",
            embedding=sample_vector,
            metadata={"author": "Researcher", "tags": ["AI", "Local"]},
        )

        self.storage.insert_chunk(chunk)
        self.assertEqual(self.storage.count_chunks(), 1)

        retrieved = self.storage.get_chunk("doc.pdf:p1:c0")
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.id, "doc.pdf:p1:c0")
        self.assertEqual(retrieved.source, "doc.pdf")
        self.assertEqual(retrieved.page_number, 1)
        self.assertEqual(retrieved.chunk_index, 0)
        self.assertEqual(retrieved.text, "Artificial intelligence on local devices.")
        self.assertEqual(len(retrieved.embedding), 1024)
        self.assertEqual(retrieved.metadata.get("author"), "Researcher")
        self.assertEqual(retrieved.metadata.get("tags"), ["AI", "Local"])

    def test_batch_insertion(self):
        """Verify inserting multiple chunks in a single transaction."""
        chunks = [
            StoredChunk(
                id=f"report.pdf:p1:c{i}",
                source="report.pdf",
                page_number=1,
                chunk_index=i,
                text=f"Content for chunk number {i}",
                embedding=[float(i)] * 1024,
            )
            for i in range(5)
        ]

        inserted_count = self.storage.insert_chunks(chunks)
        self.assertEqual(inserted_count, 5)
        self.assertEqual(self.storage.count_chunks(), 5)

        all_chunks = self.storage.get_all_chunks()
        self.assertEqual(len(all_chunks), 5)
        self.assertEqual(all_chunks[0].id, "report.pdf:p1:c0")
        self.assertEqual(all_chunks[4].id, "report.pdf:p1:c4")

    def test_empty_batch_insertion(self):
        """Verify inserting an empty list returns 0 and does not error."""
        self.assertEqual(self.storage.insert_chunks([]), 0)
        self.assertEqual(self.storage.count_chunks(), 0)

    def test_database_persistence_across_reopen(self):
        """Verify data survives closing and reopening the SQLite database connection."""
        chunk = StoredChunk(
            id="persist.pdf:p2:c1",
            source="persist.pdf",
            page_number=2,
            chunk_index=1,
            text="Persistent data across storage instances.",
            embedding=[0.42] * 1024,
            metadata={"version": "1.0"},
        )
        self.storage.insert_chunk(chunk)

        # Create a new storage instance pointing to the same file
        reopened_storage = SQLiteStorage(db_path=self.db_path)
        self.assertEqual(reopened_storage.count_chunks(), 1)

        retrieved = reopened_storage.get_chunk("persist.pdf:p2:c1")
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.text, "Persistent data across storage instances.")
        self.assertEqual(len(retrieved.embedding), 1024)
        self.assertEqual(retrieved.metadata.get("version"), "1.0")

    def test_delete_and_clear_operations(self):
        """Verify deleting individual chunks and clearing the table."""
        chunk1 = StoredChunk(
            id="del1", source="s", page_number=1, chunk_index=0, text="t1", embedding=[0.1] * 1024
        )
        chunk2 = StoredChunk(
            id="del2", source="s", page_number=1, chunk_index=1, text="t2", embedding=[0.2] * 1024
        )
        self.storage.insert_chunks([chunk1, chunk2])
        self.assertEqual(self.storage.count_chunks(), 2)

        self.assertTrue(self.storage.delete_chunk("del1"))
        self.assertFalse(self.storage.delete_chunk("del1"))  # Already deleted
        self.assertEqual(self.storage.count_chunks(), 1)

        self.storage.clear()
        self.assertEqual(self.storage.count_chunks(), 0)


if __name__ == "__main__":
    unittest.main()
