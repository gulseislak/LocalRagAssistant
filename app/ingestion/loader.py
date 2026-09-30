"""Document loader module for extracting text and metadata from PDF files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentLoaderError(Exception):
    """Base exception for document loading and extraction errors."""


@dataclass
class DocumentPage:
    """Represents the extracted text and metadata of a single document page."""

    text: str
    page_number: int  # 1-indexed
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if the page text contains only whitespace."""
        return len(self.text.strip()) == 0


class PDFLoader:
    """Loader for PDF documents that extracts text page-by-page."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path)

    def load(self, skip_empty_pages: bool = True) -> List[DocumentPage]:
        """Extract text from the PDF file page-by-page.

        Args:
            skip_empty_pages: Whether to exclude pages with no readable text.

        Returns:
            List of DocumentPage objects containing extracted text and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF.
            DocumentLoaderError: If the PDF cannot be read or parsed.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a regular file: {self.file_path}")

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, but received: {self.file_path.name}")

        try:
            reader = PdfReader(str(self.file_path))
            pages: List[DocumentPage] = []
            source_name = self.file_path.name

            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    raw_text = page.extract_text() or ""
                except Exception as extract_err:
                    raise DocumentLoaderError(
                        f"Failed extracting text from page {page_num} of {source_name}: {extract_err}"
                    ) from extract_err

                # Clean whitespace and standardize line breaks
                cleaned_text = self._clean_text(raw_text)

                doc_page = DocumentPage(
                    text=cleaned_text,
                    page_number=page_num,
                    source=source_name,
                    metadata={
                        "file_path": str(self.file_path.resolve()),
                        "total_pages": len(reader.pages),
                    },
                )

                if skip_empty_pages and doc_page.is_empty():
                    continue

                pages.append(doc_page)

            return pages

        except (PdfReadError, Exception) as e:
            if isinstance(e, DocumentLoaderError):
                raise
            raise DocumentLoaderError(
                f"Failed to read or parse PDF '{self.file_path.name}': {e}"
            ) from e

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and line endings while preserving paragraph breaks."""
        if not text:
            return ""
        # Standardize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip trailing whitespace on each line
        lines = [line.strip() for line in text.split("\n")]
        # Rejoin with single newlines, collapsing 3+ newlines to 2
        cleaned = "\n".join(lines).strip()
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")
        return cleaned


def load_pdf(file_path: Union[str, Path], skip_empty_pages: bool = True) -> List[DocumentPage]:
    """Convenience helper function to load and extract pages from a PDF file."""
    loader = PDFLoader(file_path=file_path)
    return loader.load(skip_empty_pages=skip_empty_pages)

