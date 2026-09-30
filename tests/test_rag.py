"""Unit and integration tests for the end-to-end RAG pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from app.rag.pipeline import RAGPipeline, RAGResponse, SYSTEM_PROMPT
from app.retrieval.retriever import RetrievalResult, VectorRetriever
from app.storage.database import SQLiteStorage


class TestRAGPipeline(unittest.TestCase):
    """Test suite for RAGPipeline context building, query flow, and integration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_rag_pipeline.db"
        self.storage = SQLiteStorage(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_context_construction(self):
        """Verify formatted context includes all source metadata and chunk text."""
        pipeline = RAGPipeline(storage=self.storage)
        sample_results = [
            RetrievalResult(
                chunk_id="doc.pdf:p1:c0",
                source="doc.pdf",
                page_number=1,
                chunk_index=0,
                text="Information about neural networks.",
                score=0.9123,
            ),
            RetrievalResult(
                chunk_id="doc.pdf:p2:c1",
                source="doc.pdf",
                page_number=2,
                chunk_index=1,
                text="Information about transformer architectures.",
                score=0.8456,
            ),
        ]

        context = pipeline.build_context(sample_results)
        self.assertIn("doc.pdf (Page 1)", context)
        self.assertIn("Information about neural networks.", context)
        self.assertIn("doc.pdf (Page 2)", context)
        self.assertIn("Information about transformer architectures.", context)
        self.assertIn("0.9123", context)

    def test_query_validation(self):
        """Verify invalid or empty questions raise ValueError."""
        pipeline = RAGPipeline(storage=self.storage)
        with self.assertRaises(ValueError):
            pipeline.query("")

        with self.assertRaises(ValueError):
            pipeline.query("   \n\t ")

    def test_empty_database_response(self):
        """Verify pipeline returns a safe message when no document chunks exist."""
        pipeline = RAGPipeline(storage=self.storage)
        response = pipeline.query("What is the meaning of life?")
        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(len(response.sources), 0)
        self.assertIn("not have enough information", response.answer.lower())

    def test_detect_language(self):
        """Verify detect_language identifies Turkish, English, German, etc."""
        self.assertEqual(RAGPipeline.detect_language("Retrieval-Augmented Generation (RAG) nedir?"), "Turkish")
        self.assertEqual(RAGPipeline.detect_language("RAG nasıl çalışır?"), "Turkish")
        self.assertEqual(RAGPipeline.detect_language("What is Retrieval-Augmented Generation?"), "English")
        self.assertEqual(RAGPipeline.detect_language("Was ist Retrieval-Augmented Generation?"), "German")

    def test_end_to_end_rag_with_sample_document(self):
        """Perform a full end-to-end RAG query against sample_document.pdf in English and Turkish."""
        sample_pdf = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_document.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample_document.pdf not found.")

        pipeline = RAGPipeline(storage=self.storage)
        indexed_count = pipeline.ingest_and_index_pdf(sample_pdf)
        self.assertTrue(indexed_count > 0)

        # 1. Answerable English question -> English answer
        en_question = "What is Retrieval-Augmented Generation (RAG) and how does it enhance language models?"
        en_response = pipeline.query(question=en_question, top_k=2)

        self.assertIsInstance(en_response, RAGResponse)
        self.assertEqual(en_response.question, en_question)
        self.assertTrue(len(en_response.sources) > 0)
        self.assertTrue(len(en_response.answer) > 0)
        self.assertEqual(en_response.sources[0].source, "sample_document.pdf")
        self.assertTrue(en_response.sources[0].score > 0.0)

        # 2. Answerable Turkish question -> Turkish answer
        tr_question = "Retrieval-Augmented Generation (RAG) nedir ve dil modellerini nasıl geliştirir?"
        tr_response = pipeline.query(question=tr_question, top_k=2)

        self.assertIsInstance(tr_response, RAGResponse)
        self.assertEqual(tr_response.question, tr_question)
        self.assertTrue(len(tr_response.sources) > 0)
        self.assertTrue(len(tr_response.answer) > 0)
        self.assertEqual(tr_response.sources[0].source, "sample_document.pdf")

        # 3. Unsupported question (Refusal Guard)
        unsupported_q = "What is the capital city of France?"
        unsupported_resp = pipeline.query(question=unsupported_q, top_k=2)
        self.assertIsInstance(unsupported_resp, RAGResponse)
        self.assertEqual(
            unsupported_resp.answer.strip(),
            "I do not have enough information in the provided documents to answer this question.",
        )


if __name__ == "__main__":
    unittest.main()
