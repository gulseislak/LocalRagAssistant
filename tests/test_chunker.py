"""Unit tests for the deterministic text chunker module."""

import unittest
from app.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunk,
    TextChunker,
    chunk_document_pages,
    chunk_text,
)
from app.ingestion.loader import DocumentPage


class TestTextChunker(unittest.TestCase):
    """Test suite for TextChunker, split_text, and chunk_pages."""

    def test_default_initialization(self):
        """Verify default parameters."""
        chunker = TextChunker()
        self.assertEqual(chunker.chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(chunker.chunk_overlap, DEFAULT_CHUNK_OVERLAP)
        self.assertEqual(chunker.step, DEFAULT_CHUNK_SIZE - DEFAULT_CHUNK_OVERLAP)

    def test_invalid_parameters(self):
        """Verify parameter validation raises ValueError."""
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=0)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=-10)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=-1)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_empty_and_whitespace_text(self):
        """Verify empty or whitespace-only text returns no chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        self.assertEqual(chunker.split_text(""), [])
        self.assertEqual(chunker.split_text("   \n\n\t  "), [])

    def test_short_text(self):
        """Verify text shorter than chunk_size produces exactly one chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short sentence."
        chunks = chunker.split_text(text, source="doc.pdf", page_number=1)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].source, "doc.pdf")
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].chunk_id, "doc.pdf:p1:c0")

    def test_chunking_with_overlap(self):
        """Verify sliding window chunking and proper overlap between chunks."""
        # 30 chars per chunk, 10 chars overlap -> step is 20
        chunker = TextChunker(chunk_size=30, chunk_overlap=10)
        # Create 55 character string
        text = "0123456789" * 5 + "12345"  # len 55
        chunks = chunker.split_text(text, source="test.txt", page_number=2)

        # Expected chunks:
        # Chunk 0: chars 0..30
        # Chunk 1: chars 20..50
        # Chunk 2: chars 40..55
        self.assertEqual(len(chunks), 3)

        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[0].char_end, 30)

        self.assertEqual(chunks[1].char_start, 20)
        self.assertEqual(chunks[1].char_end, 50)

        self.assertEqual(chunks[2].char_start, 40)
        self.assertEqual(chunks[2].char_end, 55)

        # Check overlap: end of chunk 0 matches start of chunk 1
        overlap_0_1 = text[20:30]
        self.assertTrue(chunks[0].text.endswith(overlap_0_1))
        self.assertTrue(chunks[1].text.startswith(overlap_0_1))

    def test_chunk_document_pages(self):
        """Verify chunking a list of DocumentPage objects preserves page metadata."""
        pages = [
            DocumentPage(text="Page one text content.", page_number=1, source="manual.pdf"),
            DocumentPage(text="Page two text content.", page_number=2, source="manual.pdf"),
        ]

        chunks = chunk_document_pages(pages, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].source, "manual.pdf")
        self.assertEqual(chunks[1].page_number, 2)
        self.assertEqual(chunks[1].source, "manual.pdf")


if __name__ == "__main__":
    unittest.main()
