"""Main entry point for Local RAG Assistant - Web UI & CLI Demonstration."""

import argparse
import sys
from pathlib import Path
from app.rag import RAGPipeline
from app.ui import DEFAULT_HOST, DEFAULT_PORT, start_server


def run_rag_demonstration() -> None:
    """Execute the full end-to-end Local RAG Assistant demonstration."""
    print("================================================================================")
    print("                LOCAL RAG ASSISTANT - END-TO-END DEMO")
    print("         (Apple Silicon M1 | Microsoft Foundry Local | SQLite)")
    print("================================================================================\n")

    sample_pdf = Path(__file__).parent / "data" / "documents" / "sample_document.pdf"
    db_path = Path(__file__).parent / "data" / "rag.db"

    if not sample_pdf.exists():
        print(f"Error: Sample PDF not found at {sample_pdf}", file=sys.stderr)
        return

    # Initialize RAG pipeline
    print(f"[Step 1] Initializing RAG Pipeline & SQLite storage ({db_path.name})...")
    pipeline = RAGPipeline()

    # Index document if database is empty
    current_count = pipeline.storage.count_chunks()
    if current_count == 0:
        print(f"[Step 2] Indexing document '{sample_pdf.name}' into SQLite vector store...")
        indexed_count = pipeline.ingest_and_index_pdf(sample_pdf, chunk_size=200, chunk_overlap=40)
        print(f"         Successfully extracted, embedded, and indexed {indexed_count} chunks.")
    else:
        print(f"[Step 2] Using existing database index ({current_count} stored chunks).")

    print("\n--------------------------------------------------------------------------------")
    print("               EXECUTING RAG EVALUATION QUESTIONS")
    print("--------------------------------------------------------------------------------\n")

    test_questions = [
        {
            "category": "1. Direct Document Question",
            "question": "What is Retrieval-Augmented Generation (RAG) and how does it enhance language models?",
        },
        {
            "category": "2. Cross-Section Question",
            "question": "What hardware target and runtime backend is used for Foundry Local inference on Apple Silicon?",
        },
        {
            "category": "3. Unsupported Question (Hallucination & Refusal Test)",
            "question": "What is the capital city of France and who won the 1998 FIFA World Cup?",
        },
    ]

    for item in test_questions:
        cat = item["category"]
        q = item["question"]

        print(f"=== [{cat}] ===")
        print(f"Question: \"{q}\"\n")

        print("--> Step 2.1: Retrieving top relevant chunks from SQLite via cosine similarity...")
        response = pipeline.query(question=q, top_k=2)

        print(f"--> Retrieved {len(response.sources)} Relevant Chunks:")
        for idx, src in enumerate(response.sources, 1):
            text_preview = src.text.replace("\n", " ")
            if len(text_preview) > 110:
                text_preview = text_preview[:110] + "..."
            print(f"    [{idx}] Similarity: {src.score:.4f} | Source: {src.source} (Page {src.page_number})")
            print(f"        Text: \"{text_preview}\"")

        print("\n--> Step 2.2: Grounded Qwen3-4B Generation:")
        print("----------------------------------------------------------------")
        print(response.answer)
        print("----------------------------------------------------------------")

        print("--> Sources Attribution:")
        for idx, src in enumerate(response.sources, 1):
            print(f"    - Source {idx}: {src.source} (Page {src.page_number}) [Score: {src.score:.4f}, ID: {src.chunk_id}]")

        print("\n" + "=" * 80 + "\n")

    print("End-to-end Local RAG Assistant demonstration completed successfully!")


def main() -> None:
    """Parse CLI arguments and launch either Web UI server or CLI evaluation demo."""
    parser = argparse.ArgumentParser(description="Local RAG Assistant - Apple Silicon & Microsoft Foundry Local")
    parser.add_argument("--ui", action="store_true", help="Start the presentation Web UI server (default: http://127.0.0.1:8000)")
    parser.add_argument("--cli", action="store_true", help="Run the automated CLI evaluation demo")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Host address for UI server (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port for UI server (default: {DEFAULT_PORT})")

    args = parser.parse_args()

    if args.cli:
        run_rag_demonstration()
    else:
        # Default or --ui launches the Web UI
        server = start_server(host=args.host, port=args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Local RAG Assistant Web Server...")
            server.server_close()


if __name__ == "__main__":
    main()
