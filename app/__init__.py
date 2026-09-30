"""Local RAG Assistant Application Package."""

from app.llm import LocalLLM, generate_response
from app.ingestion import (
    DocumentLoaderError,
    DocumentPage,
    PDFLoader,
    load_pdf,
    TextChunk,
    TextChunker,
    chunk_document_pages,
    chunk_text,
)
from app.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
    embed_text,
    embed_texts,
)
from app.storage import (
    DEFAULT_DB_PATH,
    SQLiteStorage,
    StoredChunk,
)
from app.retrieval import (
    DEFAULT_TOP_K,
    RetrievalResult,
    VectorRetriever,
    cosine_similarity,
)
from app.rag import (
    RAGPipeline,
    RAGResponse,
    SYSTEM_PROMPT,
)
from app.ui import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    LocalRAGServer,
    RAGRequestHandler,
    start_server,
)

__all__ = [
    "LocalLLM",
    "generate_response",
    "DocumentLoaderError",
    "DocumentPage",
    "PDFLoader",
    "load_pdf",
    "TextChunk",
    "TextChunker",
    "chunk_document_pages",
    "chunk_text",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingService",
    "embed_text",
    "embed_texts",
    "DEFAULT_DB_PATH",
    "SQLiteStorage",
    "StoredChunk",
    "DEFAULT_TOP_K",
    "RetrievalResult",
    "VectorRetriever",
    "cosine_similarity",
    "RAGPipeline",
    "RAGResponse",
    "SYSTEM_PROMPT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LocalRAGServer",
    "RAGRequestHandler",
    "start_server",
]
