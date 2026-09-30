"""Unit tests for the Local RAG Assistant HTTP Web Server and REST API."""

import json
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from app.rag.pipeline import RAGPipeline
from app.storage.database import SQLiteStorage
from app.ui.server import LocalRAGServer

TEST_PORT = 8999
TEST_HOST = "127.0.0.1"


class TestUIServerAPI(unittest.TestCase):
    """Test suite verifying UI endpoints and HTTP server functionality."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "test_ui_rag.db"
        cls.storage = SQLiteStorage(db_path=cls.db_path)
        cls.pipeline = RAGPipeline(storage=cls.storage)

        cls.server = LocalRAGServer((TEST_HOST, TEST_PORT), pipeline=cls.pipeline)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temp_dir.cleanup()

    def _url(self, path: str) -> str:
        return f"http://{TEST_HOST}:{TEST_PORT}{path}"

    def test_serve_static_index(self):
        """Verify serving index.html root page."""
        req = urllib.request.Request(self._url("/"))
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("Local RAG Assistant", content)
            self.assertIn("html", content.lower())

    def test_serve_static_css_and_js(self):
        """Verify serving styles.css and app.js assets."""
        with urllib.request.urlopen(self._url("/styles.css")) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/css", resp.headers.get("Content-Type"))

        with urllib.request.urlopen(self._url("/app.js")) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/javascript", resp.headers.get("Content-Type"))

    def test_api_status_and_clear(self):
        """Verify /api/status endpoint and /api/clear endpoint."""
        # Initial status
        with urllib.request.urlopen(self._url("/api/status")) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertIn("total_chunks", data)
            self.assertEqual(data["total_chunks"], 0)

        # Clear API
        req = urllib.request.Request(self._url("/api/clear"), method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["total_chunks"], 0)

    def test_api_upload_invalid_file(self):
        """Verify uploading non-PDF or empty payload returns 400 error."""
        boundary = "---------------------------974767299852498929531610575"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
            f"Content-Type: text/plain\r\n\r\n"
            f"This is not a pdf file\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            self._url("/api/upload"),
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_api_query_validation(self):
        """Verify querying with empty body or empty question returns 400 error."""
        req = urllib.request.Request(
            self._url("/api/query"),
            data=json.dumps({"question": ""}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
