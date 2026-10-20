"""Tests for the S1 Ingest module: TextExtractor, ClauseSegmenter, and Ingestor."""

import textwrap
from pathlib import Path

import pytest

from termsheet_ai.ingest.extractor import TextExtractor
from termsheet_ai.ingest.ingestor import Ingestor, _detect_doc_type, _file_hash
from termsheet_ai.ingest.segmenter import ClauseSegmenter, SegmenterConfig
from termsheet_ai.schemas.contracts import DocumentType

# ========================================================================
# Fixtures
# ========================================================================

SAMPLE_TERM_SHEET = textwrap.dedent("""\
    TERM SHEET FOR SERIES A PREFERRED STOCK FINANCING

    1. Liquidation Preference
    In the event of any liquidation, dissolution or winding up of the Company,
    the holders of Series A Preferred shall be entitled to receive a 2.0x
    liquidation preference, participating fully in any remaining proceeds
    distributed to common stockholders on an as-converted basis.

    2. Anti-Dilution Protection
    The Series A Preferred shares shall have broad-based weighted average
    anti-dilution protection in the event of a down-round financing.

    3. Board of Directors
    The Board shall consist of five (5) members: two (2) designated by the
    holders of Series A Preferred, two (2) designated by the Common holders,
    and one (1) mutually agreed independent director.

    4. Protective Provisions
    For so long as any shares of Series A Preferred are outstanding, the
    Company shall not, without the approval of the holders of at least a
    majority of the Series A Preferred:
    (a) alter or change the rights, preferences or privileges of the Series A Preferred;
    (b) create any new class or series of capital stock having a preference over the Series A;
    (c) increase the authorized number of shares of Series A Preferred;
    (d) declare or pay any dividends on Common Stock.

    5. Founder Vesting
    All founders shall be subject to four (4) year vesting with a one (1) year
    cliff. 25% of shares vest after 12 months, with the remainder vesting
    monthly over the following 36 months.
""")


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Write sample term sheet text to a temp .txt file."""
    f = tmp_path / "sample_term_sheet.txt"
    f.write_text(SAMPLE_TERM_SHEET, encoding="utf-8")
    return f


@pytest.fixture
def minimal_txt_file(tmp_path: Path) -> Path:
    """A very short file that will trigger sentence-window fallback."""
    f = tmp_path / "minimal.txt"
    f.write_text(
        "The company agrees to standard terms. Nothing else specified.",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def empty_txt_file(tmp_path: Path) -> Path:
    """An empty file."""
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    return f


# ========================================================================
# TextExtractor tests
# ========================================================================


class TestTextExtractor:
    def test_extract_txt(self, sample_txt_file: Path):
        result = TextExtractor().extract(sample_txt_file)
        assert result.method == "plaintext"
        assert len(result.pages) == 1
        assert not result.is_empty
        assert "Liquidation Preference" in result.full_text

    def test_extract_empty_txt(self, empty_txt_file: Path):
        result = TextExtractor().extract(empty_txt_file)
        assert result.is_empty

    def test_extract_missing_file(self):
        with pytest.raises(FileNotFoundError):
            TextExtractor().extract(Path("/nonexistent/file.pdf"))

    def test_extract_unsupported_format(self, tmp_path: Path):
        f = tmp_path / "file.xyz"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file type"):
            TextExtractor().extract(f)


# ========================================================================
# ClauseSegmenter tests
# ========================================================================


class TestClauseSegmenter:
    def test_pattern_segmentation(self):
        """Numbered sections should produce individual segments."""
        pages = [(1, SAMPLE_TERM_SHEET)]
        segments = ClauseSegmenter().segment(pages)
        # Should produce at least 5 segments (one per numbered section)
        assert len(segments) >= 5
        # First numbered section should mention liquidation
        liq_segments = [s for s in segments if "Liquidation" in s.text]
        assert len(liq_segments) >= 1

    def test_sentence_window_fallback(self):
        """Unstructured text should trigger sentence-window fallback."""
        raw = (
            "The company and investors agree on general terms. "
            "All shares will vest over time. "
            "Standard provisions apply to the transaction. "
            "No special preferences are granted."
        )
        pages = [(1, raw)]
        segments = ClauseSegmenter().segment(pages)
        assert len(segments) >= 1

    def test_empty_input(self):
        segments = ClauseSegmenter().segment([])
        assert segments == []

    def test_merge_short_segments(self):
        """Very short fragments should be merged into predecessors."""
        config = SegmenterConfig(min_clause_chars=50)
        segmenter = ClauseSegmenter(config=config)
        pages = [
            (
                1,
                "1. Section One\nShort.\n2. Section Two with enough text to stand alone as a clause.",
            )
        ]
        segments = segmenter.segment(pages)
        # "Short." alone is < 50 chars, should merge
        assert all(len(s.text) >= 10 for s in segments)

    def test_section_headers_detected(self):
        """Pattern segments should carry section headers."""
        pages = [(1, SAMPLE_TERM_SHEET)]
        segments = ClauseSegmenter().segment(pages)
        headers = [s.section_header for s in segments if s.section_header]
        assert len(headers) >= 3


# ========================================================================
# Ingestor (end-to-end S1) tests
# ========================================================================


class TestIngestor:
    def test_ingest_txt(self, sample_txt_file: Path):
        ingestor = Ingestor()
        metadata, clauses = ingestor.ingest(sample_txt_file)

        # Metadata checks
        assert metadata.filename == "sample_term_sheet.txt"
        assert metadata.file_type == DocumentType.TXT
        assert metadata.total_clauses == len(clauses)
        assert metadata.total_clauses >= 5
        assert len(metadata.doc_id) == 16  # truncated SHA-256

        # Clause checks
        for clause in clauses:
            assert clause.clause_id.startswith("c_")
            assert len(clause.raw_text) > 0
            assert clause.page_number >= 1

    def test_ingest_empty_file(self, empty_txt_file: Path):
        ingestor = Ingestor()
        metadata, clauses = ingestor.ingest(empty_txt_file)
        assert metadata.total_clauses == 0
        assert clauses == []

    def test_ingest_minimal_file(self, minimal_txt_file: Path):
        ingestor = Ingestor()
        metadata, clauses = ingestor.ingest(minimal_txt_file)
        # Should fall back to sentence windows and still produce output
        assert metadata.total_clauses >= 1

    def test_doc_type_detection(self, tmp_path: Path):
        assert _detect_doc_type(Path("foo.pdf")) == DocumentType.PDF
        assert _detect_doc_type(Path("foo.docx")) == DocumentType.DOCX
        assert _detect_doc_type(Path("foo.txt")) == DocumentType.TXT
        assert _detect_doc_type(Path("foo.csv")) == DocumentType.TXT  # fallback

    def test_file_hash_deterministic(self, sample_txt_file: Path):
        h1 = _file_hash(sample_txt_file)
        h2 = _file_hash(sample_txt_file)
        assert h1 == h2
        assert len(h1) == 16

    def test_clause_ids_are_sequential(self, sample_txt_file: Path):
        _, clauses = Ingestor().ingest(sample_txt_file)
        ids = [c.clause_id for c in clauses]
        expected = [f"c_{i}" for i in range(1, len(clauses) + 1)]
        assert ids == expected


# ========================================================================
# Integration: Ingestor with the real PDF in the repo
# ========================================================================


class TestPdfIngestion:
    """Integration test using the actual Ai_project-2.pdf in the repo."""

    PDF_PATH = Path(__file__).resolve().parent.parent / "Ai_project-2.pdf"

    @pytest.mark.skipif(
        not PDF_PATH.exists(),
        reason="Ai_project-2.pdf not found in repo root",
    )
    def test_ingest_real_pdf(self):
        ingestor = Ingestor()
        metadata, clauses = ingestor.ingest(self.PDF_PATH)
        assert metadata.file_type == DocumentType.PDF
        assert metadata.total_clauses > 0
        assert len(clauses) > 0
        # The PDF should yield meaningful text
        all_text = " ".join(c.raw_text for c in clauses)
        assert len(all_text) > 200
