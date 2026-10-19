"""Text extraction from PDF, DOCX, and TXT files.

Extraction strategy:
  - PDF: PyMuPDF (fitz) primary, pdfplumber fallback for layout-heavy docs.
         Tesseract OCR fallback if both return near-empty text.
  - DOCX: python-docx paragraph iteration.
  - TXT: Direct UTF-8 read.

Every extractor returns a list of (page_number, text) tuples so that
downstream segmentation can track provenance back to source pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Container for text extraction output.

    Attributes:
        pages: List of (1-indexed page number, page text) tuples.
        method: Name of the extraction backend that succeeded.
        warnings: Any non-fatal issues encountered during extraction.
    """

    pages: List[Tuple[int, str]] = field(default_factory=list)
    method: str = "none"
    warnings: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate all pages into a single string."""
        return "\n\n".join(text for _, text in self.pages if text.strip())

    @property
    def is_empty(self) -> bool:
        """True when extraction yielded no meaningful text."""
        return len(self.full_text.strip()) < 20


class TextExtractor:
    """Extracts raw text from PDF, DOCX, and TXT files.

    Usage::

        extractor = TextExtractor()
        result = extractor.extract(Path("term_sheet.pdf"))
        for page_num, text in result.pages:
            print(f"--- Page {page_num} ---")
            print(text)
    """

    # Minimum characters per page to consider extraction successful.
    MIN_CHARS_PER_PAGE = 20

    def extract(self, file_path: Path) -> ExtractionResult:
        """Dispatch extraction to the appropriate backend based on file suffix.

        Args:
            file_path: Path to the document file.

        Returns:
            ExtractionResult with page-level text and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".docx":
            return self._extract_docx(path)
        elif suffix == ".txt":
            return self._extract_txt(path)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .txt"
            )

    # ------------------------------------------------------------------
    # PDF Extraction
    # ------------------------------------------------------------------

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        """Try PyMuPDF first, then pdfplumber, then OCR fallback."""
        result = self._try_pymupdf(path)
        if not result.is_empty:
            return result

        logger.info("PyMuPDF yielded sparse text, trying pdfplumber: %s", path.name)
        result = self._try_pdfplumber(path)
        if not result.is_empty:
            return result

        logger.info("pdfplumber also sparse, attempting OCR fallback: %s", path.name)
        return self._try_ocr(path)

    def _try_pymupdf(self, path: Path) -> ExtractionResult:
        """Extract text using PyMuPDF (fitz)."""
        result = ExtractionResult(method="pymupdf")
        try:
            import fitz  # PyMuPDF
        except ImportError:
            result.warnings.append("PyMuPDF (fitz) not installed; skipping.")
            return result

        try:
            doc = fitz.open(str(path))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    result.pages.append((page_num, text))
            doc.close()
        except Exception as exc:
            result.warnings.append(f"PyMuPDF error: {exc}")

        return result

    def _try_pdfplumber(self, path: Path) -> ExtractionResult:
        """Extract text using pdfplumber (better for table-heavy layouts)."""
        result = ExtractionResult(method="pdfplumber")
        try:
            import pdfplumber
        except ImportError:
            result.warnings.append("pdfplumber not installed; skipping.")
            return result

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        result.pages.append((page_num, text))
        except Exception as exc:
            result.warnings.append(f"pdfplumber error: {exc}")

        return result

    def _try_ocr(self, path: Path) -> ExtractionResult:
        """OCR fallback using PyMuPDF pixmaps + pytesseract."""
        result = ExtractionResult(method="ocr_tesseract")
        try:
            import fitz  # PyMuPDF for rasterisation
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            result.warnings.append(
                f"OCR dependencies not fully installed ({exc}); cannot OCR."
            )
            return result

        try:
            doc = fitz.open(str(path))
            for page_num, page in enumerate(doc, start=1):
                # Render page at 300 DPI for OCR quality
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img) or ""
                if text.strip():
                    result.pages.append((page_num, text))
            doc.close()
        except Exception as exc:
            result.warnings.append(f"OCR error: {exc}")

        return result

    # ------------------------------------------------------------------
    # DOCX Extraction
    # ------------------------------------------------------------------

    def _extract_docx(self, path: Path) -> ExtractionResult:
        """Extract text from DOCX using python-docx."""
        result = ExtractionResult(method="python_docx")
        try:
            from docx import Document
        except ImportError:
            result.warnings.append("python-docx not installed.")
            return result

        try:
            doc = Document(str(path))
            paragraphs: List[str] = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            # DOCX has no native page numbers — treat as single page
            full = "\n".join(paragraphs)
            if full.strip():
                result.pages.append((1, full))
        except Exception as exc:
            result.warnings.append(f"python-docx error: {exc}")

        return result

    # ------------------------------------------------------------------
    # TXT Extraction
    # ------------------------------------------------------------------

    def _extract_txt(self, path: Path) -> ExtractionResult:
        """Read plaintext file as a single page."""
        result = ExtractionResult(method="plaintext")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                result.pages.append((1, text))
        except Exception as exc:
            result.warnings.append(f"TXT read error: {exc}")

        return result
