"""TF-IDF Baseline Classifier for venture capital term sheet clauses.

Provides an interpretable scikit-learn baseline using word n-grams (1, 2), sublinear term
frequency scaling, and balanced Logistic Regression. Implements the BaseClassifier interface
with confidence thresholding and unclassified fallback support.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from termsheet_ai.classify.exemplars import (
    Exemplar,
    load_exemplar_bank,
    prepare_clause_text,
)
from termsheet_ai.pipeline.base import BaseClassifier
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan

logger = logging.getLogger(__name__)


def _is_noise_or_empty(text: str) -> bool:
    """Return True if text has fewer than 3 alphabetic characters or is empty."""
    alpha_chars = re.findall(r"[A-Za-z]", text)
    return len(alpha_chars) < 3


class TfidfBaselineClassifier(BaseClassifier):
    """TF-IDF + Logistic Regression baseline classifier for term sheet clause classification.

    Fits on curated clause exemplars and outputs calibrated probabilities for each of the
    13 clause families defined in the project schema.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.20,
        c_param: float = 5.0,
        exemplars: Optional[List[Exemplar]] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        """Initialize and fit the TF-IDF baseline classifier.

        Args:
            confidence_threshold: Minimum prediction probability required to assign a family.
                With 13 classes, uniform probability is ~0.077. Clauses below threshold fall
                back to ClauseFamily.OTHER with is_unclassified=True. Default is 0.20 (>2.5x uniform).
            c_param: Inverse of regularization strength for Logistic Regression.
            exemplars: Optional list of training exemplars; defaults to load_exemplar_bank().
            ngram_range: Lower and upper boundary of range of n-values for n-grams.
            sublinear_tf: Apply sublinear tf scaling, replacing tf with 1 + log(tf).
        """
        self.confidence_threshold = confidence_threshold
        self.c_param = c_param
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
            max_features=5000,
        )
        self.clf = LogisticRegression(
            C=self.c_param,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

        self._classes: List[ClauseFamily] = []
        self._is_fitted: bool = False

        training_exemplars = exemplars if exemplars is not None else load_exemplar_bank()
        if training_exemplars:
            texts = [prepare_clause_text(ex.text, ex.section_header) for ex in training_exemplars]
            labels = [ex.family for ex in training_exemplars]
            self.fit(texts, labels)

    def fit(self, texts: List[str], labels: List[ClauseFamily]) -> TfidfBaselineClassifier:
        """Fit the vectorizer and logistic regression head on input texts and labels.

        Args:
            texts: List of clause strings.
            labels: List of target ClauseFamily values.

        Returns:
            Self instance.
        """
        if not texts or not labels:
            raise ValueError("Training texts and labels cannot be empty.")
        if len(texts) != len(labels):
            raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(labels)} labels.")

        label_strs = [
            label.value if isinstance(label, ClauseFamily) else str(label) for label in labels
        ]
        features = self.vectorizer.fit_transform(texts)
        self.clf.fit(features, label_strs)
        self._classes = [ClauseFamily(cls_name) for cls_name in self.clf.classes_]
        self._is_fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Compute class probabilities for a list of clause texts.

        Args:
            texts: List of prepared clause strings.

        Returns:
            2D numpy array of probabilities of shape (len(texts), len(classes)).
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        if not texts:
            return np.empty((0, len(self._classes)), dtype=float)

        features = self.vectorizer.transform(texts)
        return self.clf.predict_proba(features)

    def predict(self, texts: List[str]) -> List[ClauseFamily]:
        """Predict the most probable clause family for each text.

        Args:
            texts: List of prepared clause strings.

        Returns:
            List of predicted ClauseFamily values.
        """
        probabilities = self.predict_proba(texts)
        if len(probabilities) == 0:
            return []
        best_indices = np.argmax(probabilities, axis=1)
        return [self._classes[idx] for idx in best_indices]

    def classify(self, clauses: List[ClauseSpan]) -> List[ClauseSpan]:
        """Classify each input clause, returning new updated ClauseSpan instances.

        Enforces context boosting (fusing section headers), input validation,
        confidence thresholding, and low-confidence fallback.

        Args:
            clauses: List of ClauseSpan objects produced by S1 Ingest.

        Returns:
            New list of classified ClauseSpan objects.
        """
        if not clauses:
            return []

        classified: List[ClauseSpan] = []
        valid_indices: List[int] = []
        valid_texts: List[str] = []

        # Step 1: Pre-process and identify empty/noise clauses
        for idx, clause in enumerate(clauses):
            prepared = prepare_clause_text(clause.raw_text, clause.section_header)
            if _is_noise_or_empty(prepared):
                # Immediate fallback for empty or noise clauses
                classified.append(
                    clause.model_copy(
                        update={
                            "detected_family": ClauseFamily.OTHER,
                            "classification_confidence": 0.0,
                            "is_unclassified": True,
                        }
                    )
                )
            else:
                classified.append(clause)  # Placeholder to be updated
                valid_indices.append(idx)
                valid_texts.append(prepared)

        # Step 2: Batch predict probabilities for valid clauses
        if valid_texts:
            probs = self.predict_proba(valid_texts)
            for local_idx, global_idx in enumerate(valid_indices):
                clause = clauses[global_idx]
                row_probs = probs[local_idx]
                best_class_idx = int(np.argmax(row_probs))
                confidence = float(row_probs[best_class_idx])
                best_family = self._classes[best_class_idx]

                if confidence < self.confidence_threshold:
                    detected = ClauseFamily.OTHER
                    is_unclass = True
                else:
                    detected = best_family
                    is_unclass = False

                classified[global_idx] = clause.model_copy(
                    update={
                        "detected_family": detected,
                        "classification_confidence": round(confidence, 4),
                        "is_unclassified": is_unclass,
                    }
                )

        return classified
