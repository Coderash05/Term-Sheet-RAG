"""Abstract base interfaces for the 5-stage TermSheet AI pipeline.

Enforces clear contract boundaries between components:
- S1: Ingest & Segmentation
- S2: Clause Classification
- S3: Attribute & Fact Extraction
- S4: Symbolic Reasoning & Risk Scoring
- S5: Explanation & Audit Trace
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from termsheet_ai.schemas.contracts import (
    AuditTrace,
    ClauseSpan,
    ClauseSummary,
    DocumentMetadata,
    ExtractedFacts,
)


class BaseIngestor(ABC):
    """S1 Interface: Ingest PDF/DOCX/TXT and segment into numbered/heuristic clauses."""

    @abstractmethod
    def ingest(self, file_path: Union[str, Path]) -> tuple[DocumentMetadata, List[ClauseSpan]]:
        """Parse document and return metadata along with segmented clause spans."""
        pass


class BaseClassifier(ABC):
    """S2 Interface: SBERT + ChromaDB k-NN clause classification."""

    @abstractmethod
    def classify(self, clauses: List[ClauseSpan]) -> List[ClauseSpan]:
        """Classify each clause into one of the 12 clause families with confidence scores."""
        pass


class BaseExtractor(ABC):
    """S3 Interface: LLM attribute extraction with source span grounding."""

    @abstractmethod
    def extract(self, doc_id: str, clauses: List[ClauseSpan]) -> ExtractedFacts:
        """Extract structured domain attributes anchored to source text spans."""
        pass


class BaseReasoner(ABC):
    """S4 Interface: Symbolic Horn-clause forward chaining and risk scoring."""

    @abstractmethod
    def reason(self, facts: ExtractedFacts) -> AuditTrace:
        """Evaluate market-norm rules against extracted facts and generate justification proof tree."""
        pass


class BaseExplainer(ABC):
    """S5 Interface: Plain-English RAG summaries with consistency checking."""

    @abstractmethod
    def explain(self, clauses: List[ClauseSpan], audit_trace: AuditTrace) -> List[ClauseSummary]:
        """Generate verified plain-English summaries alongside original clauses."""
        pass
