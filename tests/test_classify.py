"""Unit and integration tests for Stage S2 Classify module.

Tests:
1. Exemplar Bank integrity and text preparation
2. TF-IDF Baseline Classifier (training, inference, thresholding, noise handling)
3. SBERT Classifier (k-NN, LR, Hybrid, ChromaDB indexing, OOD guard, immutability, injection)
4. Classification Metrics and cross-validation benchmarking
5. End-to-end integration with the master TermSheetPipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from termsheet_ai.classify import (
    ClassificationReport,
    Exemplar,
    SbertClassifier,
    TfidfBaselineClassifier,
    compute_metrics,
    evaluate_classifiers,
    format_metrics_table,
    get_exemplar_counts,
    get_training_corpus,
    load_exemplar_bank,
    prepare_clause_text,
)
from termsheet_ai.pipeline.runner import TermSheetPipeline
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_clauses() -> List[ClauseSpan]:
    """Sample test clauses representing distinct legal provisions."""
    return [
        ClauseSpan(
            clause_id="c_1",
            raw_text=(
                "In the event of any liquidation, dissolution or winding up of the Company, "
                "holders of Series A Preferred shall receive 1.0x Original Purchase Price non-participating."
            ),
            section_header="Liquidation Preference",
            page_number=1,
        ),
        ClauseSpan(
            clause_id="c_2",
            raw_text="The Series A Conversion Price will adjust on a broad-based weighted average formula.",
            section_header="Anti-Dilution",
            page_number=1,
        ),
        ClauseSpan(
            clause_id="c_3",
            raw_text="The Board shall consist of five directors: two Series A, two Common, one independent.",
            section_header="Board of Directors",
            page_number=2,
        ),
        ClauseSpan(
            clause_id="c_4",
            raw_text="All founder equity shall vest over 4 years with a 1-year cliff and monthly thereafter.",
            section_header="Vesting Schedule",
            page_number=2,
        ),
        ClauseSpan(
            clause_id="c_5",
            raw_text="This agreement is governed by the laws of the State of Delaware.",
            section_header="Governing Law",
            page_number=3,
        ),
        ClauseSpan(
            clause_id="c_6",
            raw_text="   \n\t  ",
            page_number=3,
        ),
    ]


@pytest.fixture
def mock_embed_fn():
    """Deterministic fast mock embedder generating pseudo-embeddings from text hashing."""

    def _embed(texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            # Deterministic pseudo-random 384-d vector seeded by text hash
            seed = abs(hash(text)) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            vectors.append(vec / (norm if norm > 0 else 1.0))
        return np.array(vectors, dtype=np.float32)

    return _embed


# ============================================================================
# 1. Exemplar Bank Tests
# ============================================================================


class TestExemplarBank:
    def test_total_count_and_family_distribution(self):
        """Bank must contain exactly 104 exemplars, exactly 8 per family across all 13 families."""
        bank = load_exemplar_bank()
        assert len(bank) == 104

        counts = get_exemplar_counts()
        assert len(counts) == 13
        for family in ClauseFamily:
            assert counts[family.value] == 8, (
                f"Family {family.value} count is {counts[family.value]} != 8"
            )

    def test_no_empty_strings_or_missing_sources(self):
        """All exemplars must have non-empty text, valid ClauseFamily, and cited sources."""
        bank = load_exemplar_bank()
        for ex in bank:
            assert isinstance(ex, Exemplar)
            assert len(ex.text.strip()) > 30, f"Exemplar text too short: {ex.text}"
            assert isinstance(ex.family, ClauseFamily)
            assert len(ex.source.strip()) > 0

    def test_prepare_clause_text_with_header(self):
        """Header should be fused cleanly when not already in raw text."""
        raw = "Holders will receive a 1.0x preference."
        header = "Liquidation"
        result = prepare_clause_text(raw, header)
        assert result == "Liquidation: Holders will receive a 1.0x preference."

    def test_prepare_clause_text_avoids_duplicate_header(self):
        """Header should not be duplicated if raw text already starts with it."""
        raw = "Liquidation Preference: Holders will receive a 1.0x preference."
        header = "Liquidation Preference"
        result = prepare_clause_text(raw, header)
        assert result == raw

    def test_prepare_clause_text_empty_and_whitespace(self):
        """Empty or whitespace input must return empty string."""
        assert prepare_clause_text("") == ""
        assert prepare_clause_text("   \n\t") == ""
        assert prepare_clause_text("   ", "Header") == ""

    def test_get_training_corpus(self):
        """get_training_corpus must return 3 parallel lists of length 104."""
        texts, families, sources = get_training_corpus()
        assert len(texts) == 104
        assert len(families) == 104
        assert len(sources) == 104
        assert all(isinstance(f, ClauseFamily) for f in families)


# ============================================================================
# 2. TF-IDF Baseline Classifier Tests
# ============================================================================


class TestTfidfBaselineClassifier:
    def test_tfidf_initialization_and_fit(self):
        """Classifier must fit on default bank and recognize all 13 classes."""
        clf = TfidfBaselineClassifier()
        assert clf._is_fitted
        assert len(clf._classes) == 13

    def test_tfidf_predict_known_clauses(self, sample_clauses):
        """Classifier must accurately classify standard legal clauses."""
        clf = TfidfBaselineClassifier(confidence_threshold=0.15)
        classified = clf.classify(sample_clauses[:5])

        assert classified[0].detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert classified[1].detected_family == ClauseFamily.ANTI_DILUTION
        assert classified[2].detected_family == ClauseFamily.BOARD_COMPOSITION
        assert classified[3].detected_family == ClauseFamily.FOUNDER_VESTING
        assert classified[4].detected_family == ClauseFamily.OTHER

        for c in classified:
            assert c.classification_confidence > 0.0
            assert c.is_unclassified is False

    def test_tfidf_confidence_threshold_fallback(self, sample_clauses):
        """When confidence is below threshold, clause must be marked unclassified."""
        clf = TfidfBaselineClassifier(confidence_threshold=0.99)
        classified = clf.classify(sample_clauses[:2])

        for c in classified:
            assert c.detected_family == ClauseFamily.OTHER
            assert c.is_unclassified is True

    def test_tfidf_empty_and_noise_handling(self, sample_clauses):
        """Whitespace and noise clauses must be marked unclassified with 0.0 confidence."""
        clf = TfidfBaselineClassifier()
        noise_clauses = [
            sample_clauses[5],  # whitespace
            ClauseSpan(clause_id="c_noise", raw_text="... ??? 123", page_number=1),
        ]
        classified = clf.classify(noise_clauses)

        for c in classified:
            assert c.detected_family == ClauseFamily.OTHER
            assert c.classification_confidence == 0.0
            assert c.is_unclassified is True

    def test_tfidf_empty_input_list(self):
        """Empty input list returns empty list."""
        clf = TfidfBaselineClassifier()
        assert clf.classify([]) == []

    def test_tfidf_predict_proba_shape(self):
        """predict_proba returns (N, 13) array summing to 1.0."""
        clf = TfidfBaselineClassifier()
        texts = ["Liquidation preference text.", "Anti-dilution provision."]
        probs = clf.predict_proba(texts)
        assert probs.shape == (2, 13)
        assert np.allclose(np.sum(probs, axis=1), 1.0)


# ============================================================================
# 3. SBERT Classifier Tests
# ============================================================================


class TestSbertClassifier:
    def test_sbert_modes_execution(self, sample_clauses):
        """All three modes (knn, lr, hybrid) must execute and return valid predictions."""
        for mode in ["knn", "lr", "hybrid"]:
            clf = SbertClassifier(mode=mode, confidence_threshold=0.15, ood_threshold=0.0)  # type: ignore[arg-type]
            classified = clf.classify(sample_clauses[:3])
            assert len(classified) == 3
            for c in classified:
                assert isinstance(c.detected_family, ClauseFamily)
                assert 0.0 <= c.classification_confidence <= 1.0

    def test_sbert_predict_known_clauses(self, sample_clauses):
        """SBERT classifier must correctly identify canonical clauses."""
        clf = SbertClassifier(mode="hybrid", confidence_threshold=0.20, ood_threshold=0.10)
        classified = clf.classify(sample_clauses[:5])

        assert classified[0].detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert classified[1].detected_family == ClauseFamily.ANTI_DILUTION
        assert classified[2].detected_family == ClauseFamily.BOARD_COMPOSITION
        assert classified[3].detected_family == ClauseFamily.FOUNDER_VESTING
        assert classified[4].detected_family == ClauseFamily.OTHER

        for c in classified:
            assert c.is_unclassified is False

    def test_chromadb_knn_retrieval(self, sample_clauses):
        """retrieve_nearest_exemplars must return top-k matches with valid metadata and similarities."""
        clf = SbertClassifier()
        exemplars = clf.retrieve_nearest_exemplars(sample_clauses[0], top_k=3)

        assert len(exemplars) == 3
        for ex in exemplars:
            assert "text" in ex
            assert isinstance(ex["family"], ClauseFamily)
            assert "source" in ex
            assert 0.0 <= ex["cosine_similarity"] <= 1.0

        # Top match for liquidation clause must be liquidation preference
        assert exemplars[0]["family"] == ClauseFamily.LIQUIDATION_PREFERENCE
        assert exemplars[0]["cosine_similarity"] > 0.50

    def test_sbert_confidence_threshold_fallback(self, sample_clauses):
        """Setting an unreachable confidence threshold must mark clauses unclassified."""
        clf = SbertClassifier(confidence_threshold=0.99)
        classified = clf.classify(sample_clauses[:2])

        for c in classified:
            assert c.detected_family == ClauseFamily.OTHER
            assert c.is_unclassified is True

    def test_sbert_ood_noise_guard(self):
        """Out-of-distribution / foreign text must trigger OOD guard."""
        clf = SbertClassifier(ood_threshold=0.75)
        ood_clause = ClauseSpan(
            clause_id="c_ood",
            raw_text="The recipe calls for 200 grams of flour, two eggs, and baking powder.",
            page_number=1,
        )
        classified = clf.classify([ood_clause])
        assert classified[0].detected_family == ClauseFamily.OTHER
        assert classified[0].is_unclassified is True

    def test_sbert_empty_and_noise_clause(self, sample_clauses):
        """Whitespace strings must return 0.0 confidence and unclassified status."""
        clf = SbertClassifier()
        classified = clf.classify([sample_clauses[5]])
        assert classified[0].detected_family == ClauseFamily.OTHER
        assert classified[0].classification_confidence == 0.0
        assert classified[0].is_unclassified is True

    def test_sbert_batch_embedding_consistency(self, sample_clauses):
        """Batch encoding must produce predictions identical to single-item encoding."""
        clf = SbertClassifier()
        batch_res = clf.classify(sample_clauses[:3])

        single_0 = clf.classify([sample_clauses[0]])[0]
        single_1 = clf.classify([sample_clauses[1]])[0]

        assert batch_res[0].detected_family == single_0.detected_family
        assert np.isclose(
            batch_res[0].classification_confidence, single_0.classification_confidence, atol=1e-3
        )
        assert batch_res[1].detected_family == single_1.detected_family

    def test_sbert_clause_immutability(self, sample_clauses):
        """Original ClauseSpan instances must remain unchanged after classification."""
        clf = SbertClassifier()
        original_family = sample_clauses[0].detected_family
        original_conf = sample_clauses[0].classification_confidence

        classified = clf.classify(sample_clauses[:1])

        # Input clause is untouched
        assert sample_clauses[0].detected_family == original_family
        assert sample_clauses[0].classification_confidence == original_conf

        # Output clause is updated
        assert classified[0].detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert classified[0].classification_confidence > 0.0

    def test_sbert_custom_embed_fn_injection(self, sample_clauses, mock_embed_fn):
        """Custom embed_fn dependency injection must operate correctly."""
        clf = SbertClassifier(embed_fn=mock_embed_fn, confidence_threshold=0.0, ood_threshold=0.0)
        assert clf._is_fitted
        classified = clf.classify(sample_clauses[:2])
        assert len(classified) == 2
        for c in classified:
            assert isinstance(c.detected_family, ClauseFamily)

    def test_sbert_empty_input_list(self):
        """Empty input list returns empty list."""
        clf = SbertClassifier()
        assert clf.classify([]) == []


# ============================================================================
# 4. Metrics Module Tests
# ============================================================================


class TestMetrics:
    def test_compute_metrics_perfect_predictions(self):
        """Perfect predictions yield 1.0 across accuracy, precision, recall, and F1."""
        y_true = [
            ClauseFamily.LIQUIDATION_PREFERENCE,
            ClauseFamily.ANTI_DILUTION,
            ClauseFamily.OTHER,
        ]
        y_pred = [
            ClauseFamily.LIQUIDATION_PREFERENCE,
            ClauseFamily.ANTI_DILUTION,
            ClauseFamily.OTHER,
        ]

        report = compute_metrics(y_true, y_pred)
        assert report.accuracy == 1.0
        assert report.macro_precision == 1.0
        assert report.macro_recall == 1.0
        assert report.macro_f1 == 1.0

    def test_compute_metrics_confusion_matrix_shape(self):
        """Confusion matrix must have 13x13 dimensions corresponding to the 13 classes."""
        y_true = [ClauseFamily.LIQUIDATION_PREFERENCE]
        y_pred = [ClauseFamily.ANTI_DILUTION]

        report = compute_metrics(y_true, y_pred, labels=list(ClauseFamily))
        assert len(report.confusion_matrix) == 13
        assert len(report.confusion_matrix[0]) == 13
        assert len(report.class_labels) == 13

    def test_classification_report_to_dict(self):
        """Report must serialize to dictionary with expected keys."""
        report = compute_metrics(
            [ClauseFamily.LIQUIDATION_PREFERENCE],
            [ClauseFamily.LIQUIDATION_PREFERENCE],
            labels=list(ClauseFamily),
        )
        d = report.to_dict()
        assert "accuracy" in d
        assert "macro_f1" in d
        assert "family_metrics" in d
        assert "confusion_matrix" in d

    def test_format_metrics_table(self):
        """format_metrics_table must generate valid Markdown table lines."""
        report = compute_metrics(
            [ClauseFamily.LIQUIDATION_PREFERENCE],
            [ClauseFamily.LIQUIDATION_PREFERENCE],
            labels=list(ClauseFamily),
        )
        table = format_metrics_table({"model_test": report})
        assert "| Model | Accuracy |" in table
        assert "| `model_test` |" in table

    def test_evaluate_classifiers_benchmark(self):
        """evaluate_classifiers must produce valid reports for all 4 models."""
        reports = evaluate_classifiers(n_splits=2)
        expected_keys = {"tfidf", "sbert_knn", "sbert_lr", "sbert_hybrid"}
        assert set(reports.keys()) == expected_keys

        for _name, rep in reports.items():
            assert isinstance(rep, ClassificationReport)
            assert 0.0 <= rep.accuracy <= 1.0
            assert 0.0 <= rep.macro_f1 <= 1.0


# ============================================================================
# 5. Pipeline End-to-End Integration Tests
# ============================================================================


class TestPipelineIntegration:
    SAMPLE_DOC_TEXT = (
        "1. Liquidation Preference: Series A Preferred receives 2.0x liquidation preference, participating fully.\n"
        "2. Anti-Dilution: Standard broad-based weighted average anti-dilution applies.\n"
        "3. Board of Directors: The board shall consist of 3 members with 2 designated by Investors.\n"
        "4. Founder Vesting: 4-year vesting schedule with a 1-year cliff.\n"
        "5. Governing Law: This term sheet is governed by Delaware law."
    )

    @pytest.fixture
    def sample_term_sheet_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "test_sheet.txt"
        p.write_text(self.SAMPLE_DOC_TEXT, encoding="utf-8")
        return p

    def test_pipeline_with_tfidf_classifier(self, sample_term_sheet_file: Path):
        """TermSheetPipeline executes end-to-end with TfidfBaselineClassifier."""
        classifier = TfidfBaselineClassifier(confidence_threshold=0.15)
        pipeline = TermSheetPipeline(classifier=classifier)
        payload = pipeline.run(sample_term_sheet_file)

        assert payload.metadata.total_clauses >= 4
        assert len(payload.clauses) >= 4
        assert payload.clauses[0].detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert payload.clauses[0].classification_confidence > 0.0
        assert payload.facts is not None
        assert payload.audit_trace is not None

    def test_pipeline_with_sbert_classifier(self, sample_term_sheet_file: Path):
        """TermSheetPipeline executes end-to-end with SbertClassifier."""
        classifier = SbertClassifier(mode="hybrid", confidence_threshold=0.20, ood_threshold=0.10)
        pipeline = TermSheetPipeline(classifier=classifier)
        payload = pipeline.run(sample_term_sheet_file)

        assert payload.metadata.total_clauses >= 4
        assert len(payload.clauses) >= 4
        assert payload.clauses[0].detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert payload.clauses[0].classification_confidence > 0.0
        assert payload.facts is not None
        assert payload.audit_trace is not None
        assert len(payload.summaries) >= 4
        assert len(payload.execution_errors) == 0
