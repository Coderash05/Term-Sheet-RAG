"""Stage S2 Classification module for venture capital term sheets.

Provides:
- Exemplar bank (104 curated clauses across all 13 families)
- TF-IDF + Logistic Regression Baseline Classifier
- SBERT + ChromaDB k-NN + Logistic Regression Classifier (hybrid, k-NN, and LR heads)
- Comprehensive evaluation metrics and benchmarking harness
"""

from termsheet_ai.classify.exemplars import (
    Exemplar,
    get_exemplar_counts,
    get_training_corpus,
    load_exemplar_bank,
    prepare_clause_text,
)
from termsheet_ai.classify.metrics import (
    ClassificationReport,
    compute_metrics,
    evaluate_classifiers,
    format_metrics_table,
)
from termsheet_ai.classify.sbert import SbertClassifier
from termsheet_ai.classify.tfidf import TfidfBaselineClassifier
from termsheet_ai.pipeline.base import BaseClassifier

__all__ = [
    "BaseClassifier",
    "ClassificationReport",
    "Exemplar",
    "SbertClassifier",
    "TfidfBaselineClassifier",
    "compute_metrics",
    "evaluate_classifiers",
    "format_metrics_table",
    "get_exemplar_counts",
    "get_training_corpus",
    "load_exemplar_bank",
    "prepare_clause_text",
]
