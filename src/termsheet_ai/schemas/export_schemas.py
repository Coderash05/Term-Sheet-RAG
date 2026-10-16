"""Utility script to export Pydantic models as standard JSON Schema files."""

import json
from pathlib import Path

from termsheet_ai.schemas.contracts import (
    AuditTrace,
    ClauseSpan,
    DocumentMetadata,
    ExtractedFacts,
    PipelinePayload,
    RiskFlag,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "contracts" / "json_schemas"


def export_all_schemas(output_dir: Path = SCHEMA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "document_metadata.json": DocumentMetadata,
        "clause_span.json": ClauseSpan,
        "extracted_facts.json": ExtractedFacts,
        "risk_flag.json": RiskFlag,
        "audit_trace.json": AuditTrace,
        "pipeline_payload.json": PipelinePayload,
    }

    for filename, model_cls in models.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(model_cls.model_json_schema(), f, indent=2)
        print(f"Exported: {filepath.name}")


if __name__ == "__main__":
    export_all_schemas()
