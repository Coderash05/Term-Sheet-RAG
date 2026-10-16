"""Tests for end-to-end TermSheetPipeline execution across stages S1 through S5."""

from termsheet_ai.pipeline.runner import TermSheetPipeline
from termsheet_ai.schemas.contracts import ClauseFamily, SeverityLevel


def test_pipeline_end_to_end_execution(tmp_path):
    # Create sample dummy term sheet text file
    sample_ts = tmp_path / "sample_term_sheet.txt"
    sample_ts.write_text(
        "1. Liquidation Preference: Series A Preferred receives a 2.0x liquidation preference, participating fully.\n"
        "2. Anti-Dilution: The shares feature full-ratchet anti-dilution.\n"
        "3. Board: The Board of Directors consists of 3 seats (2 Investor seats, 1 Founder seat).\n",
        encoding="utf-8",
    )

    pipeline = TermSheetPipeline()
    payload = pipeline.run(sample_ts)

    # Validate S1 (Ingest)
    assert payload.metadata.filename == "sample_term_sheet.txt"
    assert len(payload.clauses) == 3

    # Validate S2 (Classify)
    families = [c.detected_family for c in payload.clauses]
    assert ClauseFamily.LIQUIDATION_PREFERENCE in families
    assert ClauseFamily.ANTI_DILUTION in families
    assert ClauseFamily.BOARD_COMPOSITION in families

    # Validate S3 (Extract)
    assert payload.facts is not None
    assert payload.facts.liquidation_preference is not None
    assert payload.facts.liquidation_preference.multiple == 2.0
    assert payload.facts.anti_dilution.anti_dilution_type.value == "full_ratchet"

    # Validate S4 (Reason)
    assert payload.audit_trace is not None
    assert len(payload.audit_trace.flags) >= 2
    critical_flags = [f for f in payload.audit_trace.flags if f.severity == SeverityLevel.CRITICAL]
    assert len(critical_flags) == 1
    assert "Full-Ratchet" in critical_flags[0].title

    # Validate S5 (Explain)
    assert len(payload.summaries) == 3
    assert all(s.numbers_verified for s in payload.summaries)
    assert len(payload.execution_errors) == 0
