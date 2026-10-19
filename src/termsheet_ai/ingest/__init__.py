"""S1 Ingest & Segmentation module.

Handles PDF/DOCX/TXT text extraction, OCR fallback, and
clause segmentation for the TermSheet AI pipeline.
"""

from termsheet_ai.ingest.extractor import TextExtractor
from termsheet_ai.ingest.ingestor import Ingestor
from termsheet_ai.ingest.segmenter import ClauseSegmenter

__all__ = ["TextExtractor", "ClauseSegmenter", "Ingestor"]
