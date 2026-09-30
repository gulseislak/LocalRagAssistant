"""End-to-end integration test for the Web UI server workflow."""

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from app.rag.pipeline import RAGPipeline
from app.storage.database import SQLiteStorage
from app.ui.server import LocalRAGServer

PORT = 8998


class TestE2EUIWorkflow(unittest.TestCase):
    """End-to-end test verifying full upload -> status -> query -> refusal flow over HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "e2e_rag.db"
        cls.storage = SQLiteStorage(db_path=cls.db_path)
        cls.pipeline = RAGPipeline(storage=cls.storage)

        cls.server = LocalRAGServer(("127.0.0.1", PORT), pipeline=cls.pipeline)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temp_dir.cleanup()

    def test_full_ui_rag_lifecycle(self):
        """Verify: Upload PDF -> Check Status -> Query RAG -> Unsupported Query -> Clear Index."""
        sample_pdf = Path(__file__).resolve().parent.parent / "data" / "documents" / "sample_document.pdf"
        self.assertTrue(sample_pdf.exists(), "sample_document.pdf should exist")

        # 1. Upload sample_document.pdf
        boundary = "---------------------------123456789"
        file_bytes = sample_pdf.read_bytes()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{sample_pdf.name}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        upload_req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with urllib.request.urlopen(upload_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertTrue(data["indexed_chunks"] > 0)
            print(f"\n[E2E] Uploaded '{data['filename']}': {data['indexed_chunks']} chunks indexed.")

        # 2. Check /api/status
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["total_chunks"] > 0)
            self.assertEqual(len(data["documents"]), 1)
            print(f"[E2E] Status verified: {data['total_chunks']} total chunks in database.")

        # 3. Query RAG (Answerable question)
        query_data = json.dumps({
            "question": "What is Retrieval-Augmented Generation (RAG)?",
            "top_k": 2,
        }).encode("utf-8")

        q_req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/query",
            data=query_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(q_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(len(data["answer"]) > 0)
            self.assertFalse(data["is_unsupported"])
            self.assertTrue(len(data["sources"]) > 0)
            print(f"[E2E] Answerable query response: {data['answer'][:80]}...")
            print(f"[E2E] Source match: {data['sources'][0]['source']} (Page {data['sources'][0]['page_number']})")

        # 4. Query Unsupported question
        unsupported_data = json.dumps({
            "question": "What is the capital city of France?",
            "top_k": 2,
        }).encode("utf-8")

        u_req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/query",
            data=unsupported_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(u_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["is_unsupported"])
            print(f"[E2E] Unsupported query handled: '{data['answer']}' (is_unsupported={data['is_unsupported']})")

        # 5. Clear Database
        clear_req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/clear", method="POST")
        with urllib.request.urlopen(clear_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["total_chunks"], 0)
            print("[E2E] Clear API verified: 0 chunks remaining.")


if __name__ == "__main__":
    unittest.main()
