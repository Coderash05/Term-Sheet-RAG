"""Tests for Pydantic interface contracts and schema validation."""

import pytest
from pydantic import ValidationError

from termsheet_ai.schemas.contracts import (
    AntiDilutionFact,
    AntiDilutionType,
    ClauseFamily,
    ClauseSpan,
    DocumentMetadata,
    DocumentType,
    LiquidationPreferenceFact,
    ParticipationType,
    SeverityLevel,
    SourceSpan,
)


def test_source_span_validation():
    # Valid span
    span = SourceSpan(
        text="1.0x non-participating preferred", start_char=0, end_char=32, page_number=1
    )
    assert span.text == "1.0x non-participating preferred"
    assert span.page_number == 1

    # Empty text must raise validation error
    with pytest.raises(ValidationError):
        SourceSpan(text="")


def test_clause_span_defaults():
    clause = ClauseSpan(
        clause_id="c_1",
        raw_text="Founders shall be subject to a 4-year vesting schedule with a 1-year cliff.",
    )
    assert clause.clause_id == "c_1"
    assert clause.detected_family == ClauseFamily.OTHER
    assert clause.classification_confidence == 0.0
    assert not clause.is_unclassified


def test_document_metadata_types():
    meta = DocumentMetadata(
        doc_id="test_doc_01",
        filename="nvca_series_a.pdf",
        file_type=DocumentType.PDF,
        total_clauses=15,
    )
    assert meta.doc_id == "test_doc_01"
    assert meta.file_type == DocumentType.PDF
    assert meta.total_clauses == 15


def test_liquidation_preference_fact_validation():
    span = SourceSpan(text="2.0x fully participating", page_number=2)
    fact = LiquidationPreferenceFact(
        clause_id="c_2",
        multiple=2.0,
        participation=ParticipationType.FULLY_PARTICIPATING,
        source_span=span,
    )
    assert fact.multiple == 2.0
    assert fact.participation == ParticipationType.FULLY_PARTICIPATING

    # Multiple must be non-negative
    with pytest.raises(ValidationError):
        LiquidationPreferenceFact(
            clause_id="c_2",
            multiple=-1.0,
            participation=ParticipationType.NON_PARTICIPATING,
            source_span=span,
        )


def test_anti_dilution_fact():
    span = SourceSpan(text="full-ratchet anti-dilution adjustments", page_number=3)
    fact = AntiDilutionFact(
        clause_id="c_3",
        anti_dilution_type=AntiDilutionType.FULL_RATCHET,
        source_span=span,
    )
    assert fact.anti_dilution_type == AntiDilutionType.FULL_RATCHET


def test_severity_levels():
    assert SeverityLevel.CRITICAL.value == "critical"
    assert SeverityLevel.HIGH.value == "high"
    assert SeverityLevel.MEDIUM.value == "medium"
    assert SeverityLevel.LOW.value == "low"
    assert SeverityLevel.INFO.value == "info"
