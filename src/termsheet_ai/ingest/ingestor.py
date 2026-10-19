"""S1 Ingestor: wires TextExtractor + ClauseSegmenter into the BaseIngestor contract.

Produces DocumentMetadata and a list of ClauseSpans ready for S2 classification.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from termsheet_ai.ingest.extractor import ExtractionResult, TextExtractor
from termsheet_ai.ingest.segmenter import ClauseSegmenter, SegmenterConfig
from termsheet_ai.pipeline.base import BaseIngestor
from termsheet_ai.schemas.contracts import ClauseSpan, DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


def _detect_doc_type(path: Path) -> DocumentType:
    """Map file extension to DocumentType enum, defaulting to TXT."""
    ext = path.suffix.lower().lstrip(".")
    try:
        return DocumentType(ext)
    except ValueError:
        return DocumentType.TXT


def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of the file contents (first 64 KiB for speed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(65_536))
    return h.hexdigest()[:16]


class Ingestor(BaseIngestor):
    """Production S1 Ingestor for PDF, DOCX, and TXT term sheets.

    Orchestrates :class:`TextExtractor` for raw text recovery and
    :class:`ClauseSegmenter` for splitting the text into clause-level spans.

    Usage::

        from termsheet_ai.ingest import Ingestor

        ingestor = Ingestor()
        metadata, clauses = ingestor.ingest("path/to/term_sheet.pdf")
        for c in clauses:
            print(c.clause_id, c.section_header, c.raw_text[:80])
    """

    def __init__(
        self,
        extractor: Optional[TextExtractor] = None,
        segmenter: Optional[ClauseSegmenter] = None,
        segmenter_config: Optional[SegmenterConfig] = None,
    ) -> None:
        self.extractor = extractor or TextExtractor()
        self.segmenter = segmenter or ClauseSegmenter(config=segmenter_config)

    def ingest(
        self, file_path: Union[str, Path]
    ) -> Tuple[DocumentMetadata, List[ClauseSpan]]:
        """Parse a term-sheet document and return metadata + segmented clauses.

        Args:
            file_path: Path to the document (PDF, DOCX, or TXT).

        Returns:
            A tuple of (DocumentMetadata, list[ClauseSpan]).
        """
        path = Path(file_path)
        logger.info("S1 Ingest: processing %s", path.name)

        # --- Text extraction ---
        extraction: ExtractionResult = self.extractor.extract(path)

        for warn in extraction.warnings:
            logger.warning("Extraction warning: %s", warn)

        if extraction.is_empty:
            logger.warning(
                "Extraction yielded no text for %s (method=%s). "
                "Returning empty clause list.",
                path.name,
                extraction.method,
            )

        # --- Clause segmentation ---
        raw_segments = self.segmenter.segment(extraction.pages)

        # --- Build output contracts ---
        doc_id = _file_hash(path)

        clauses: List[ClauseSpan] = []
        for idx, seg in enumerate(raw_segments, start=1):
            clauses.append(
                ClauseSpan(
                    clause_id=f"c_{idx}",
                    raw_text=seg.text,
                    page_number=seg.page_number,
                    section_header=seg.section_header,
                )
            )

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=path.name,
            file_type=_detect_doc_type(path),
            total_clauses=len(clauses),
        )

        logger.info(
            "S1 complete: %d clauses extracted from %s (method=%s)",
            len(clauses),
            path.name,
            extraction.method,
        )

        return metadata, clauses
