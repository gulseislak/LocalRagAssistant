# Local RAG Assistant

A fully local, privacy-focused Retrieval-Augmented Generation (RAG) assistant designed for macOS Apple Silicon, built with Microsoft Foundry Local and SQLite.

## Key Features

- **100% Local & Private:** All embeddings and LLM inference run strictly on Apple Silicon GPU (Metal/MPS) using Foundry Local. No external APIs or cloud dependencies.
- **Pure Python Storage & Retrieval:** SQLite vector storage using binary float32 BLOBs and pure Python Cosine Similarity (zero heavy vector DB frameworks).
- **Grounded Responses & Hallucination Guardrails:** The LLM is strictly instructed to use only retrieved context data and explicitly refuses when information is missing.
- **Source Attribution:** Every generated answer displays exact citation metadata: source filename, page number, and similarity match percentage.
- **Modern Presentation Web UI:** A sleek, glassmorphic dark-themed interface supporting drag-and-drop PDF ingestion, live vector index status, one-click demo questions, and interactive RAG chat.

---

## Pipeline Architecture

```
PDF Document
     ↓
Text Extraction (pypdf)
     ↓
Clean Text & Page Tracking
     ↓
Deterministic Chunking (Sliding Window: 500 chars / 100 overlap)
     ↓
Dense Embeddings (Qwen3-Embedding-0.6B → 1024 floats)
     ↓
SQLite Storage (Vector BLOBs + Metadata)
     ↓
User Question (Web UI / CLI)
     ↓
Query Embedding (Qwen3-Embedding-0.6B)
     ↓
Vector Similarity Retrieval (Pure Python Cosine Similarity)
     ↓
Top-K Relevant Chunks
     ↓
Grounded Context Construction
     ↓
Local LLM Generation (Qwen3-4B via Foundry Local)
     ↓
Grounded Answer + Source Citations + Refusal Guard
```

---

## Project Structure

```
local-rag-assistant/
├── app/
│   ├── __init__.py          # Package exports
│   ├── llm.py               # Local LLM interface (Qwen3-4B)
│   ├── ingestion/           # PDF text extraction & deterministic chunker
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── chunker.py
│   ├── embeddings/          # Dense vector embeddings (Qwen3-Embedding-0.6B)
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── storage/             # SQLite vector BLOB & metadata persistence
│   │   ├── __init__.py
│   │   └── database.py
│   ├── retrieval/           # Pure Python Cosine Similarity & top-k ranking
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── rag/                 # Grounded context construction & query pipeline
│   │   ├── __init__.py
│   │   └── pipeline.py
│   └── ui/                  # Lightweight HTTP Server & Presentation Web UI
│       ├── __init__.py
│       ├── server.py
│       └── static/
│           ├── index.html
│           ├── styles.css
│           └── app.js
├── data/
│   ├── documents/           # Local source documents
│   │   └── sample_document.pdf
│   └── rag.db               # SQLite database file (ignored by git)
├── tests/                   # Complete automated test suite (36 tests)
│   ├── test_llm.py
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_storage.py
│   ├── test_retrieval.py
│   ├── test_rag.py
│   └── test_ui_api.py
├── main.py                  # Multi-mode launcher (Web UI server & CLI demo)
├── requirements.txt         # Minimal pinned dependencies
├── pyrightconfig.json       # Language server configuration
├── .gitignore
└── README.md
```

---

## Quick Start Guide

### 1. Activate Environment

```bash
source .venv/bin/activate
```

### 2. Launch Presentation Web UI

```bash
python main.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### 3. Run Automated CLI Evaluation Demo

```bash
python main.py --cli
```

### 4. Run Complete Test Suite

```bash
python -m unittest discover -s tests -v
```

---

## REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the Presentation Web UI (`index.html`) |
| `/api/status` | `GET` | Returns vector store chunk count, indexed documents, and models |
| `/api/upload` | `POST` | Ingests uploaded PDF, chunks, computes embeddings, indexes in SQLite |
| `/api/query` | `POST` | Performs similarity search, context grounding, and Qwen3-4B generation |
| `/api/clear` | `POST` | Clears all document chunks from SQLite |
