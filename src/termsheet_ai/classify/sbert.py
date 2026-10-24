"""SBERT + ChromaDB k-NN + Logistic Regression Classifier (Stage S2).

Combines Sentence-BERT embeddings (all-MiniLM-L6-v2, 384-d) with:
1. Cosine similarity k-NN lookup via an in-memory ChromaDB vector index, and
2. A balanced Logistic Regression classification head over the 384-d embeddings.

Provides hybrid ensemble scoring, context boosting with section headers,
OOD/noise filtering, and confidence thresholding adhering strictly to the BaseClassifier contract.
"""

from __future__ import annotations

import functools
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import chromadb
import numpy as np
from sklearn.linear_model import LogisticRegression

from termsheet_ai.classify.exemplars import (
    Exemplar,
    load_exemplar_bank,
    prepare_clause_text,
)
from termsheet_ai.pipeline.base import BaseClassifier
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan

logger = logging.getLogger(__name__)

ClassifierMode = Literal["hybrid", "knn", "lr"]


@functools.lru_cache(maxsize=1)
def get_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load and cache the SentenceTransformer model singleton.

    Cached via lru_cache to ensure model weights are loaded into memory
    at most once per Python process / test session.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading SentenceTransformer model: %s", model_name)
    return SentenceTransformer(model_name)


def _is_noise_or_empty(text: str) -> bool:
    """Return True if text has fewer than 3 alphabetic characters or is empty."""
    alpha_chars = re.findall(r"[A-Za-z]", text)
    return len(alpha_chars) < 3


class SbertClassifier(BaseClassifier):
    """Deep learning clause classifier using Sentence-BERT, ChromaDB k-NN, and Logistic Regression.

    Supports three scoring modes:
    - "knn": Cosine-similarity weighted voting over top-k ChromaDB neighbors.
    - "lr": Calibrated Logistic Regression head trained on 384-d SBERT embeddings.
    - "hybrid": Weighted ensemble combination: alpha * P_lr + (1 - alpha) * P_knn.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.40,
        ood_threshold: float = 0.25,
        mode: ClassifierMode = "hybrid",
        k_neighbors: int = 5,
        alpha: float = 0.5,
        batch_size: int = 32,
        exemplars: Optional[List[Exemplar]] = None,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize and fit the SBERT classifier.

        Args:
            model_name: SentenceTransformers model identifier (default: "all-MiniLM-L6-v2").
            confidence_threshold: Probability threshold for assignment; below this,
                clause falls back to ClauseFamily.OTHER with is_unclassified=True.
            ood_threshold: Minimum cosine similarity to any exemplar required to avoid OOD flagging.
            mode: Scoring mode ("hybrid", "knn", or "lr").
            k_neighbors: Number of nearest neighbors to query in ChromaDB.
            alpha: Weight for LR predictions in hybrid mode (1 - alpha for k-NN).
            batch_size: Batch size for sentence embedding encoding.
            exemplars: Optional list of exemplars; defaults to load_exemplar_bank().
            embed_fn: Optional custom embedding callable for dependency injection in fast tests.
            model: Optional pre-loaded SentenceTransformer instance.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.ood_threshold = ood_threshold
        self.mode = mode
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.batch_size = batch_size

        self._custom_embed_fn = embed_fn
        self._model = model

        # Set up isolated in-memory ChromaDB client & collection
        self.chroma_client = chromadb.EphemeralClient()
        collection_id = f"exemplars_{uuid.uuid4().hex[:12]}"
        self.collection = self.chroma_client.create_collection(
            name=collection_id,
            metadata={"hnsw:space": "cosine"},
        )

        # Logistic Regression classification head
        self.lr_head = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

        self._classes: List[ClauseFamily] = list(ClauseFamily)
        self._class_to_idx: Dict[ClauseFamily, int] = {f: i for i, f in enumerate(self._classes)}
        self._is_fitted: bool = False
        self._has_lr_head: bool = False
        self._exemplars: List[Exemplar] = []

        training_exemplars = exemplars if exemplars is not None else load_exemplar_bank()
        if training_exemplars:
            self.fit(training_exemplars)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalized dense embeddings.

        Args:
            texts: List of input strings.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self._custom_embed_fn is not None:
            embeddings = self._custom_embed_fn(texts)
        else:
            active_model = self._model or get_sbert_model(self.model_name)
            embeddings = active_model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms

    def fit(self, exemplars: List[Exemplar]) -> SbertClassifier:
        """Fit the ChromaDB vector index and Logistic Regression head.

        Args:
            exemplars: List of clause exemplars.

        Returns:
            Self instance.
        """
        if not exemplars:
            raise ValueError("Exemplars list cannot be empty.")

        self._exemplars = list(exemplars)
        prepared_texts = [prepare_clause_text(ex.text, ex.section_header) for ex in exemplars]
        embeddings = self._encode(prepared_texts)

        # 1. Populate ChromaDB collection
        ids = [f"ex_{i}" for i in range(len(exemplars))]
        metadatas = [
            {
                "family": ex.family.value,
                "source": ex.source,
                "section_header": ex.section_header or "",
                "notes": ex.notes or "",
            }
            for ex in exemplars
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=prepared_texts,
            metadatas=metadatas,
        )

        # 2. Fit Logistic Regression classification head if >= 2 classes present
        labels_str = [ex.family.value for ex in exemplars]
        unique_classes = set(labels_str)
        if len(unique_classes) >= 2:
            self.lr_head.fit(embeddings, labels_str)
            self._has_lr_head = True
        else:
            self._has_lr_head = False

        self._is_fitted = True
        return self

    def _predict_lr_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute Logistic Regression probabilities mapped to the full ClauseFamily enum."""
        num_samples = embeddings.shape[0]
        if not self._has_lr_head:
            # Fallback to uniform if trained with fewer than 2 distinct classes
            return np.full((num_samples, len(self._classes)), 1.0 / len(self._classes), dtype=float)

        raw_probs = self.lr_head.predict_proba(embeddings)
        lr_classes = [ClauseFamily(c) for c in self.lr_head.classes_]

        aligned_probs = np.zeros((num_samples, len(self._classes)), dtype=float)

        for col_idx, cls_obj in enumerate(lr_classes):
            if cls_obj in self._class_to_idx:
                global_idx = self._class_to_idx[cls_obj]
                aligned_probs[:, global_idx] = raw_probs[:, col_idx]

        return aligned_probs

    def _predict_knn_proba_and_max_sim(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ChromaDB cosine k-NN probabilities and maximum similarity per sample."""
        num_samples = embeddings.shape[0]
        probs = np.zeros((num_samples, len(self._classes)), dtype=float)
        max_sims = np.zeros(num_samples, dtype=float)

        if num_samples == 0:
            return probs, max_sims

        k = min(self.k_neighbors, len(self._exemplars))
        results = self.collection.query(
            query_embeddings=embeddings.tolist(),
            n_results=k,
            include=["metadatas", "distances"],
        )

        for i in range(num_samples):
            distances = results["distances"][i]
            metadatas = results["metadatas"][i]

            weights_by_class = np.zeros(len(self._classes), dtype=float)
            sample_max_sim = 0.0

            for dist, meta in zip(distances, metadatas, strict=True):
                # ChromaDB cosine distance d in [0, 2]; cosine similarity s = 1.0 - d
                sim = max(0.0, 1.0 - float(dist))
                if sim > sample_max_sim:
                    sample_max_sim = sim

                family = ClauseFamily(meta["family"])
                cls_idx = self._class_to_idx[family]
                weights_by_class[cls_idx] += sim

            total_weight = float(np.sum(weights_by_class))
            if total_weight > 0.0:
                probs[i] = weights_by_class / total_weight
            else:
                probs[i, self._class_to_idx[ClauseFamily.OTHER]] = 1.0

            max_sims[i] = sample_max_sim

        return probs, max_sims

    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute class probability distribution and max cosine similarity for texts.

        Args:
            texts: List of prepared clause texts.

        Returns:
            Tuple of (probabilities, max_similarities).
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        if not texts:
            return np.empty((0, len(self._classes)), dtype=float), np.empty(0, dtype=float)

        embeddings = self._encode(texts)
        p_knn, max_sims = self._predict_knn_proba_and_max_sim(embeddings)
        p_lr = self._predict_lr_proba(embeddings)

        if self.mode == "knn":
            return p_knn, max_sims
        elif self.mode == "lr":
            return p_lr, max_sims
        else:  # hybrid
            p_hybrid = self.alpha * p_lr + (1.0 - self.alpha) * p_knn
            return p_hybrid, max_sims

    def retrieve_nearest_exemplars(
        self, clause: Union[ClauseSpan, str], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k nearest exemplars for a clause from the ChromaDB index.

        Provides direct interpretability and serves as the retrieval engine for Stage S5 RAG.

        Args:
            clause: ClauseSpan object or text string.
            top_k: Number of nearest exemplars to return.

        Returns:
            List of dictionaries containing exemplar details and cosine similarity scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")

        if isinstance(clause, ClauseSpan):
            text = prepare_clause_text(clause.raw_text, clause.section_header)
        else:
            text = prepare_clause_text(str(clause))

        embedding = self._encode([text])
        k = min(top_k, len(self._exemplars))
        results = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, dists, strict=True):
                sim = max(0.0, 1.0 - float(dist))
                retrieved.append(
                    {
                        "text": doc,
                        "family": ClauseFamily(meta["family"]),
                        "source": meta["source"],
                        "section_header": meta["section_header"],
                        "notes": meta["notes"],
                        "cosine_similarity": round(sim, 4),
                    }
                )

        return retrieved

    def classify(self, clauses: List[ClauseSpan]) -> List[ClauseSpan]:
        """Classify each input clause span according to the S2 pipeline contract.

        Applies context boosting, noise filtering, SBERT batch embedding,
        hybrid k-NN/LR probability calculation, and low-confidence / OOD thresholding.

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

        # 1. Pre-filter noise, empty, and very short inputs
        for idx, clause in enumerate(clauses):
            prepared = prepare_clause_text(clause.raw_text, clause.section_header)
            if _is_noise_or_empty(prepared):
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
                classified.append(clause)
                valid_indices.append(idx)
                valid_texts.append(prepared)

        # 2. Batch encode and classify valid clauses
        if valid_texts:
            probs, max_sims = self.predict_proba(valid_texts)

            for local_idx, global_idx in enumerate(valid_indices):
                clause = clauses[global_idx]
                row_probs = probs[local_idx]
                max_sim = max_sims[local_idx]

                best_class_idx = int(np.argmax(row_probs))
                confidence = float(row_probs[best_class_idx])
                best_family = self._classes[best_class_idx]

                # Out-of-distribution or low-confidence fallback
                if max_sim < self.ood_threshold or confidence < self.confidence_threshold:
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
