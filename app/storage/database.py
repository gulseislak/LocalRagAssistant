"""SQLite storage module for vector embeddings, chunks, and metadata."""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "rag.db"


@dataclass
class StoredChunk:
    """Represents a document chunk with embedding vector stored in SQLite."""

    id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    embedding: List[float]
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SQLiteStorage:
    """SQLite-based vector and metadata storage manager."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_parent_directory()
        self.initialize()

    def _ensure_parent_directory(self) -> None:
        """Ensure the parent directory for the database file exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Open a new connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create database tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks(source);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_page ON document_chunks(source, page_number);
                """
            )
            conn.commit()

    @staticmethod
    def serialize_embedding(embedding: List[float]) -> bytes:
        """Serialize a float vector into a compact binary float32 buffer (BLOB).

        Args:
            embedding: List of float values (e.g. 1024-dim vector).

        Returns:
            Compact binary bytes representation (4 bytes per dimension).
        """
        if not embedding:
            raise ValueError("Cannot serialize an empty embedding vector.")
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def deserialize_embedding(blob: bytes) -> List[float]:
        """Deserialize a binary float32 BLOB back into a list of floats.

        Args:
            blob: Binary packed bytes from SQLite BLOB column.

        Returns:
            List of float values.
        """
        if not blob:
            return []
        dim = len(blob) // 4
        return [float(x) for x in struct.unpack(f"{dim}f", blob)]

    def insert_chunk(self, chunk: StoredChunk) -> None:
        """Insert or replace a single document chunk with embedding in SQLite.

        Args:
            chunk: StoredChunk instance to persist.
        """
        blob = self.serialize_embedding(chunk.embedding)
        meta_json = json.dumps(chunk.metadata) if chunk.metadata else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO document_chunks (
                    id, source, page_number, chunk_index, text, embedding, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    chunk.id,
                    chunk.source,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                    blob,
                    meta_json,
                ),
            )
            conn.commit()

    def insert_chunks(self, chunks: List[StoredChunk]) -> int:
        """Insert multiple document chunks in a single atomic transaction.

        Args:
            chunks: List of StoredChunk objects to insert.

        Returns:
            Number of chunks successfully inserted.
        """
        if not chunks:
            return 0

        rows = []
        for chunk in chunks:
            blob = self.serialize_embedding(chunk.embedding)
            meta_json = json.dumps(chunk.metadata) if chunk.metadata else None
            rows.append(
                (
                    chunk.id,
                    chunk.source,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                    blob,
                    meta_json,
                )
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO document_chunks (
                    id, source, page_number, chunk_index, text, embedding, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def get_chunk(self, chunk_id: str) -> Optional[StoredChunk]:
        """Retrieve a single chunk by its unique ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            StoredChunk if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, source, page_number, chunk_index, text, embedding, metadata, created_at
                FROM document_chunks
                WHERE id = ?;
                """,
                (chunk_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_chunk(row)

    def get_all_chunks(self) -> List[StoredChunk]:
        """Retrieve all stored chunks from the database.

        Returns:
            List of all StoredChunk records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, source, page_number, chunk_index, text, embedding, metadata, created_at
                FROM document_chunks
                ORDER BY source, page_number, chunk_index;
                """
            )
            rows = cursor.fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def count_chunks(self) -> int:
        """Get the total number of chunks currently stored in SQLite.

        Returns:
            Integer total chunk count.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM document_chunks;")
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        Args:
            chunk_id: Identifier of chunk to remove.

        Returns:
            True if a record was deleted, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks WHERE id = ?;", (chunk_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all stored chunk records from the table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks;")
            conn.commit()

    def get_document_summaries(self) -> List[Dict[str, Any]]:
        """Retrieve distinct indexed documents with total chunk and page counts.

        Returns:
            List of dictionaries with source, chunk_count, and max_page.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT source, COUNT(*) AS chunk_count, MAX(page_number) AS max_page
                FROM document_chunks
                GROUP BY source
                ORDER BY source;
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "source": row["source"],
                    "chunks": int(row["chunk_count"]),
                    "pages": int(row["max_page"]),
                }
                for row in rows
            ]

    def _row_to_chunk(self, row: sqlite3.Row) -> StoredChunk:
        """Convert a SQLite Row object into a StoredChunk dataclass."""
        embedding = self.deserialize_embedding(row["embedding"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return StoredChunk(
            id=row["id"],
            source=row["source"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            embedding=embedding,
            created_at=str(row["created_at"]) if row["created_at"] else None,
            metadata=metadata,
        )
