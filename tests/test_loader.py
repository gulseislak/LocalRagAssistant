"""Unit tests for the PDF document loader module."""

import os
import tempfile
import unittest
from pathlib import Path
from app.ingestion.loader import (
    DocumentLoaderError,
    DocumentPage,
    PDFLoader,
    load_pdf,
)


def create_test_pdf_bytes(pages_text: list[str]) -> bytes:
    """Generate a minimal valid multi-page PDF in-memory for testing."""
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids_refs = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages_text)))
    objects.append(f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(pages_text)} >>".encode("latin1"))

    for i, text in enumerate(pages_text):
        page_obj_id = 3 + i * 2
        content_obj_id = 4 + i * 2
        font_obj_id = 3 + len(pages_text) * 2

        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj_id} 0 R /Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>"
        ).encode("latin1")
        objects.append(page_obj)

        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_data = f"BT /F1 12 Tf 50 720 Td ({safe_text}) Tj ET".encode("latin1")
        content_obj = (
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin1")
            + stream_data
            + b"\nendstream"
        )
        objects.append(content_obj)

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects):
        obj_id = i + 1
        offsets.append(len(output))
        output.extend(f"{obj_id} 0 obj\n".encode("latin1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("latin1"))

    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1")
    )
    return bytes(output)


class TestPDFLoader(unittest.TestCase):
    """Test suite for PDFLoader and load_pdf helper."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "test_doc.pdf"
        sample_pages = [
            "First page: Overview of the local RAG assistant system.",
            "Second page: Ingestion, chunking, and vector indexing details.",
        ]
        with open(self.pdf_path, "wb") as f:
            f.write(create_test_pdf_bytes(sample_pages))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_pdf(self):
        """Verify successful extraction of pages, text, and metadata."""
        pages = load_pdf(self.pdf_path)
        self.assertEqual(len(pages), 2)

        # Check Page 1
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].source, "test_doc.pdf")
        self.assertIn("First page", pages[0].text)
        self.assertEqual(pages[0].metadata["total_pages"], 2)

        # Check Page 2
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[1].source, "test_doc.pdf")
        self.assertIn("Second page", pages[1].text)

    def test_file_not_found(self):
        """Verify FileNotFoundError is raised for non-existent files."""
        non_existent = Path(self.temp_dir.name) / "does_not_exist.pdf"
        with self.assertRaises(FileNotFoundError):
            load_pdf(non_existent)

    def test_invalid_extension(self):
        """Verify ValueError is raised for non-PDF file extensions."""
        txt_file = Path(self.temp_dir.name) / "document.txt"
        txt_file.write_text("Plain text content")
        with self.assertRaises(ValueError):
            load_pdf(txt_file)

    def test_corrupt_pdf(self):
        """Verify DocumentLoaderError is raised for corrupt or unreadable PDF files."""
        corrupt_file = Path(self.temp_dir.name) / "corrupt.pdf"
        corrupt_file.write_bytes(b"This is not a valid PDF file stream")
        with self.assertRaises(DocumentLoaderError):
            load_pdf(corrupt_file)


if __name__ == "__main__":
    unittest.main()
