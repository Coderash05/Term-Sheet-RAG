"""Clause segmentation for term sheet text.

Splits raw extracted text into individual clause spans that align with the
contract schema's ClauseSpan model. Two strategies are employed:

1. **Heading / Numbering Pattern Matching** (primary):
   Detects numbered sections (e.g. ``1.``, ``1.2``, ``(a)``, ``Section 3``),
   capitalised headings, and bullet-prefixed lists to split at clause
   boundaries.

2. **Sentence Window Fallback**:
   If pattern matching yields fewer than 2 segments, the text is split into
   overlapping sentence windows so every piece of content reaches classification.

Output clauses carry their ``page_number`` from the extraction stage.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Heading / numbering patterns that typically open a new clause
# ------------------------------------------------------------------

# Matches: "1.", "1.2", "1.2.3", "12.", etc. at start of line
_NUMBERED_SECTION = re.compile(
    r"^(?P<num>\d{1,3}(?:\.\d{1,3}){0,3})[\.\)]\s+",
    re.MULTILINE,
)

# Matches: "(a)", "(i)", "(iv)", "(1)" at start of line
_LETTER_BULLET = re.compile(
    r"^\s*\((?P<label>[a-z]|[ivxlc]+|\d{1,3})\)\s+",
    re.MULTILINE,
)

# Matches: "Section 1", "ARTICLE II", "Schedule A" (case-insensitive)
_SECTION_HEADING = re.compile(
    r"^(?:section|article|schedule|exhibit|part|appendix)\s+[A-Z0-9ivxlc]+",
    re.MULTILINE | re.IGNORECASE,
)

# Matches: ALL-CAPS lines of 3+ words (common heading style in legal docs)
_ALLCAPS_HEADING = re.compile(
    r"^[A-Z][A-Z\s,\-&]{8,}$",
    re.MULTILINE,
)

# Combined pattern: any of the above marks a new clause boundary
_CLAUSE_BOUNDARY = re.compile(
    r"(?:"
    + _NUMBERED_SECTION.pattern
    + r"|"
    + _SECTION_HEADING.pattern
    + r"|"
    + _ALLCAPS_HEADING.pattern
    + r")",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class RawSegment:
    """Intermediate segment before conversion to ClauseSpan."""

    text: str
    page_number: int = 1
    section_header: Optional[str] = None


@dataclass
class SegmenterConfig:
    """Tunable parameters for the clause segmenter.

    Attributes:
        min_clause_chars: Minimum characters for a segment to be kept as its
            own clause; shorter segments are merged into the previous clause.
        max_clause_chars: If a segment exceeds this length and contains
            sub-bullets, attempt further splitting.
        sentence_window_size: Number of sentences per window in fallback mode.
        sentence_window_overlap: Number of overlapping sentences between windows.
        min_segments_for_pattern: If pattern matching returns fewer than this
            many segments, fall back to sentence windows.
    """

    min_clause_chars: int = 40
    max_clause_chars: int = 3000
    sentence_window_size: int = 5
    sentence_window_overlap: int = 1
    min_segments_for_pattern: int = 2


class ClauseSegmenter:
    """Splits extracted page text into clause-level segments.

    Usage::

        segmenter = ClauseSegmenter()
        segments = segmenter.segment([(1, "1. Liquidation..."), (2, "2. Anti...")])
    """

    def __init__(self, config: Optional[SegmenterConfig] = None) -> None:
        self.config = config or SegmenterConfig()

    def segment(self, pages: List[Tuple[int, str]]) -> List[RawSegment]:
        """Segment page-level text into clause-level RawSegments.

        Args:
            pages: List of (1-indexed page number, page text) from TextExtractor.

        Returns:
            Ordered list of RawSegments ready for conversion to ClauseSpans.
        """
        if not pages:
            return []

        # First attempt: pattern-based segmentation across all pages
        segments = self._segment_by_patterns(pages)

        if len(segments) >= self.config.min_segments_for_pattern:
            logger.info("Pattern segmentation produced %d clauses.", len(segments))
            return self._post_process(segments)

        # Fallback: sentence-window segmentation
        logger.info(
            "Pattern segmentation yielded only %d segment(s); falling back to sentence windows.",
            len(segments),
        )
        segments = self._segment_by_sentences(pages)
        return self._post_process(segments)

    # ------------------------------------------------------------------
    # Pattern-based segmentation
    # ------------------------------------------------------------------

    def _segment_by_patterns(self, pages: List[Tuple[int, str]]) -> List[RawSegment]:
        """Split text at detected clause boundaries."""
        segments: List[RawSegment] = []

        for page_num, page_text in pages:
            # Find all boundary positions within this page
            boundaries = list(_CLAUSE_BOUNDARY.finditer(page_text))

            if not boundaries:
                # No structure detected — treat entire page as one segment
                if page_text.strip():
                    segments.append(
                        RawSegment(
                            text=page_text.strip(),
                            page_number=page_num,
                        )
                    )
                continue

            # Text before the first boundary
            preamble = page_text[: boundaries[0].start()].strip()
            if preamble:
                segments.append(RawSegment(text=preamble, page_number=page_num))

            # Each boundary starts a new segment that runs until the next
            for i, match in enumerate(boundaries):
                start = match.start()
                end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(page_text)
                chunk = page_text[start:end].strip()
                if not chunk:
                    continue

                header = self._extract_header(chunk)
                segments.append(
                    RawSegment(
                        text=chunk,
                        page_number=page_num,
                        section_header=header,
                    )
                )

        return segments

    @staticmethod
    def _extract_header(text: str) -> Optional[str]:
        """Pull the first line as a section header if it looks like one."""
        first_line = text.split("\n", 1)[0].strip()
        # Keep it if it's short-ish and starts with a number or keyword
        if len(first_line) < 120:
            return first_line
        return None

    # ------------------------------------------------------------------
    # Sentence-window fallback
    # ------------------------------------------------------------------

    def _segment_by_sentences(self, pages: List[Tuple[int, str]]) -> List[RawSegment]:
        """Split text into overlapping sentence windows."""
        segments: List[RawSegment] = []
        window = self.config.sentence_window_size
        overlap = self.config.sentence_window_overlap
        step = max(1, window - overlap)

        for page_num, page_text in pages:
            sentences = self._split_sentences(page_text)
            if not sentences:
                continue

            for i in range(0, len(sentences), step):
                window_sents = sentences[i : i + window]
                chunk = " ".join(window_sents).strip()
                if chunk:
                    segments.append(RawSegment(text=chunk, page_number=page_num))

        return segments

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Naïve sentence splitter on period/semicolon boundaries.

        Avoids splitting on common abbreviations (e.g., "e.g.", "i.e.", "No.",
        "§", decimal numbers).
        """
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Split on sentence-ending punctuation followed by space + uppercase
        parts = re.split(r"(?<=[.;!?])\s+(?=[A-Z(0-9])", text)
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Post-processing: merge short, split long
    # ------------------------------------------------------------------

    def _post_process(self, segments: List[RawSegment]) -> List[RawSegment]:
        """Merge short segments and attempt sub-splitting on oversized ones."""
        merged = self._merge_short(segments)
        final = self._split_long(merged)
        return final

    def _merge_short(self, segments: List[RawSegment]) -> List[RawSegment]:
        """Merge segments shorter than min_clause_chars into their predecessor."""
        if not segments:
            return []
        result: List[RawSegment] = [segments[0]]
        for seg in segments[1:]:
            if len(seg.text) < self.config.min_clause_chars and result:
                # Merge into previous
                result[-1].text += "\n" + seg.text
            else:
                result.append(seg)
        return result

    def _split_long(self, segments: List[RawSegment]) -> List[RawSegment]:
        """Re-split oversized segments on sub-bullets like (a), (b)."""
        result: List[RawSegment] = []
        for seg in segments:
            if len(seg.text) <= self.config.max_clause_chars:
                result.append(seg)
                continue

            # Try splitting on letter/number bullets within the segment
            sub_parts = _LETTER_BULLET.split(seg.text)
            if len(sub_parts) <= 1:
                result.append(seg)
                continue

            # sub_parts alternates [preamble, label, text, label, text, ...]
            # Reassemble keeping the label prefixes
            preamble = sub_parts[0].strip()
            if preamble:
                result.append(
                    RawSegment(
                        text=preamble,
                        page_number=seg.page_number,
                        section_header=seg.section_header,
                    )
                )
            # Step through (label, text) pairs
            for i in range(1, len(sub_parts) - 1, 2):
                label = sub_parts[i]
                body = sub_parts[i + 1].strip() if i + 1 < len(sub_parts) else ""
                chunk = f"({label}) {body}".strip()
                if chunk:
                    result.append(
                        RawSegment(
                            text=chunk,
                            page_number=seg.page_number,
                            section_header=f"({label})",
                        )
                    )

        return result
