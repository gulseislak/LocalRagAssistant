"""Lightweight local HTTP web server and REST API for the Local RAG Assistant UI."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.ingestion.loader import DocumentLoaderError
from app.rag.pipeline import RAGPipeline
from app.storage.database import SQLiteStorage

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DOCS_DIR = _project_root / "data" / "documents"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class RAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing REST API and serving modern static UI."""

    server_version = "LocalRAGAssistant/1.0"

    @property
    def pipeline(self) -> RAGPipeline:
        return self.server.pipeline  # type: ignore

    def do_GET(self) -> None:
        """Handle GET requests for static assets and status API."""
        url_path = self.path.split("?")[0]

        if url_path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif url_path == "/styles.css" or url_path == "/static/styles.css":
            self._serve_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        elif url_path == "/app.js" or url_path == "/static/app.js":
            self._serve_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        elif url_path == "/api/status":
            self._handle_get_status()
        else:
            self._send_json({"error": "Resource not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle POST requests for PDF uploads, RAG queries, and clearing database."""
        url_path = self.path.split("?")[0]

        if url_path == "/api/query":
            self._handle_post_query()
        elif url_path == "/api/upload":
            self._handle_post_upload()
        elif url_path == "/api/clear":
            self._handle_post_clear()
        else:
            self._send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_get_status(self) -> None:
        """Return system health, document summaries, chunk count, and model metadata."""
        try:
            storage: SQLiteStorage = self.pipeline.storage
            total_chunks = storage.count_chunks()
            doc_summaries = storage.get_document_summaries()

            response_data = {
                "status": "ok",
                "total_chunks": total_chunks,
                "documents": doc_summaries,
                "chat_model": self.pipeline.llm.model_name,
                "embedding_model": self.pipeline.embedding_service.model_name,
                "db_path": str(storage.db_path.name),
                "platform": "Apple Silicon (Foundry Local | Unified Memory)",
            }
            self._send_json(response_data)
        except Exception as e:
            self._send_json({"error": f"Failed to retrieve status: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_post_query(self) -> None:
        """Execute a grounded RAG query and return answer with source citations."""
        try:
            body = self._read_request_body()
            if not body:
                self._send_json({"error": "Empty request body"}, status=HTTPStatus.BAD_REQUEST)
                return

            data = json.loads(body.decode("utf-8"))
            question = data.get("question", "").strip()
            top_k = int(data.get("top_k", 3))

            if not question:
                self._send_json({"error": "Question field is required and cannot be empty"}, status=HTTPStatus.BAD_REQUEST)
                return

            rag_response = self.pipeline.query(question=question, top_k=top_k)

            sources_data = [
                {
                    "chunk_id": s.chunk_id,
                    "source": s.source,
                    "page_number": s.page_number,
                    "chunk_index": s.chunk_index,
                    "score": s.score,
                    "text": s.text,
                }
                for s in rag_response.sources
            ]

            # Detect refusal / unsupported question
            is_unsupported = (
                "not have enough information" in rag_response.answer.lower()
                or "not provided in" in rag_response.answer.lower()
                or not sources_data
                or (sources_data and sources_data[0]["score"] < 0.35)
            )

            result = {
                "question": rag_response.question,
                "answer": rag_response.answer,
                "is_unsupported": is_unsupported,
                "sources": sources_data,
                "context": rag_response.context,
            }
            self._send_json(result)

        except json.JSONDecodeError:
            self._send_json({"error": "Malformed JSON in request body"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as e:
            self._send_json({"error": f"RAG query execution failed: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_post_upload(self) -> None:
        """Receive and ingest an uploaded PDF file into the SQLite vector database."""
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self._send_json(
                    {"error": "Invalid Content-Type. Expected 'multipart/form-data'"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            body = self._read_request_body()
            filename, file_bytes = self._parse_multipart(body, content_type)

            if not filename or not file_bytes:
                self._send_json({"error": "No file uploaded or file is empty."}, status=HTTPStatus.BAD_REQUEST)
                return

            if not filename.lower().endswith(".pdf"):
                self._send_json({"error": "Only PDF (.pdf) documents are supported."}, status=HTTPStatus.BAD_REQUEST)
                return

            # Ensure upload directory exists
            DEFAULT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
            safe_filename = Path(filename).name
            target_path = DEFAULT_DOCS_DIR / safe_filename

            # Save uploaded PDF to data/documents/
            target_path.write_bytes(file_bytes)

            # Ingest and index into SQLite
            indexed_count = self.pipeline.ingest_and_index_pdf(target_path)
            total_chunks = self.pipeline.storage.count_chunks()

            self._send_json(
                {
                    "success": True,
                    "filename": safe_filename,
                    "indexed_chunks": indexed_count,
                    "total_chunks_in_db": total_chunks,
                    "message": f"Successfully extracted, embedded, and indexed {indexed_count} chunks from '{safe_filename}'.",
                }
            )

        except DocumentLoaderError as dle:
            self._send_json({"error": f"PDF parse error: {dle}"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as e:
            self._send_json({"error": f"Upload and indexing failed: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_post_clear(self) -> None:
        """Clear all stored document chunks from SQLite."""
        try:
            self.pipeline.storage.clear()
            self._send_json(
                {
                    "success": True,
                    "message": "Document index cleared successfully.",
                    "total_chunks": 0,
                }
            )
        except Exception as e:
            self._send_json({"error": f"Failed to clear database: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_request_body(self) -> bytes:
        """Read exact content length from request stream."""
        content_len = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_len) if content_len > 0 else b""

    def _send_json(self, data: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Serialize and send a JSON HTTP response."""
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        """Serve a static file from disk with proper headers."""
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": f"File '{file_path.name}' not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    @staticmethod
    def _parse_multipart(body: bytes, content_type: str) -> Tuple[Optional[str], Optional[bytes]]:
        """Extract uploaded filename and binary file content from multipart payload."""
        match = re.search(r"boundary=([^\s;]+)", content_type)
        if not match:
            return None, None

        boundary = match.group(1).strip('"').encode("utf-8")
        parts = body.split(b"--" + boundary)

        for part in parts:
            if b'filename="' in part:
                headers_data = part.split(b"\r\n\r\n", 1)
                if len(headers_data) < 2:
                    continue
                header_raw, file_data = headers_data
                if file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]

                fn_match = re.search(r'filename="([^"]+)"', header_raw.decode("utf-8", errors="ignore"))
                filename = fn_match.group(1) if fn_match else "uploaded.pdf"
                return filename, file_data

        return None, None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress verbose default HTTP request logging to stderr."""
        # Optional: uncomment to enable server access logs
        # sys.stdout.write(f"[HTTP] {self.address_string()} - {format % args}\n")
        pass


class LocalRAGServer(ThreadingHTTPServer):
    """Custom ThreadingHTTPServer holding reference to the shared RAGPipeline."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        super().__init__(server_address, RAGRequestHandler)
        self.pipeline = pipeline or RAGPipeline()


def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    pipeline: Optional[RAGPipeline] = None,
) -> LocalRAGServer:
    """Instantiate and start the Local RAG Web Server."""
    server = LocalRAGServer((host, port), pipeline=pipeline)
    print(f"================================================================================")
    print(f"          LOCAL RAG ASSISTANT - WEB UI SERVER STARTED")
    print(f"================================================================================")
    print(f" Access URL : http://{host}:{port}")
    print(f" Models     : Qwen3-4B (Chat) & Qwen3-Embedding-0.6B (Dense Vectors)")
    print(f" Storage    : SQLite ({server.pipeline.storage.db_path.name})")
    print(f" Hardware   : Apple Silicon M1 (Foundry Local GPU Inference)")
    print(f"================================================================================\n")
    return server


if __name__ == "__main__":
    server = start_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Local RAG Assistant Web Server...")
        server.server_close()
