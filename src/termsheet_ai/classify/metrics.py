"""Classification evaluation metrics for venture capital term sheet classifiers.

Computes precision, recall, macro-F1, weighted-F1, and confusion matrices.
Provides cross-validation benchmarking comparing TF-IDF Baseline against SBERT (k-NN, LR, Hybrid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

from termsheet_ai.classify.exemplars import (
    Exemplar,
    load_exemplar_bank,
    prepare_clause_text,
)
from termsheet_ai.classify.sbert import SbertClassifier, get_sbert_model
from termsheet_ai.classify.tfidf import TfidfBaselineClassifier
from termsheet_ai.schemas.contracts import ClauseFamily


@dataclass
class ClassificationReport:
    """Aggregated classification evaluation metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    family_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    class_labels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "family_metrics": self.family_metrics,
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
        }


def compute_metrics(
    y_true: List[ClauseFamily],
    y_pred: List[ClauseFamily],
    labels: Optional[List[ClauseFamily]] = None,
) -> ClassificationReport:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True clause family labels.
        y_pred: Predicted clause family labels.
        labels: Optional explicit list of ClauseFamily labels to evaluate;
            defaults to all 13 standard families.

    Returns:
        ClassificationReport instance.
    """
    if labels is None:
        unique_present = sorted(
            set(y_true).union(set(y_pred)),
            key=lambda x: x.value if isinstance(x, ClauseFamily) else str(x),
        )
        eval_labels = unique_present if unique_present else list(ClauseFamily)
    else:
        eval_labels = labels

    label_strs = [lbl.value if isinstance(lbl, ClauseFamily) else str(lbl) for lbl in eval_labels]

    y_true_str = [y.value if isinstance(y, ClauseFamily) else str(y) for y in y_true]
    y_pred_str = [y.value if isinstance(y, ClauseFamily) else str(y) for y in y_pred]

    acc = float(accuracy_score(y_true_str, y_pred_str)) if y_true_str else 0.0

    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_str,
        y_pred_str,
        labels=label_strs,
        average="macro",
        zero_division=0,
    )

    # Weighted metrics
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true_str,
        y_pred_str,
        labels=label_strs,
        average="weighted",
        zero_division=0,
    )

    # Per-class metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true_str,
        y_pred_str,
        labels=label_strs,
        average=None,
        zero_division=0,
    )

    family_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(label_strs):
        family_metrics[name] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(support[idx]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true_str, y_pred_str, labels=label_strs)
    cm_list: List[List[int]] = cm.tolist()

    return ClassificationReport(
        accuracy=round(acc, 4),
        macro_precision=round(float(p_macro), 4),
        macro_recall=round(float(r_macro), 4),
        macro_f1=round(float(f1_macro), 4),
        weighted_f1=round(float(f1_weighted), 4),
        family_metrics=family_metrics,
        confusion_matrix=cm_list,
        class_labels=label_strs,
    )


def evaluate_classifiers(
    exemplars: Optional[List[Exemplar]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, ClassificationReport]:
    """Run stratified cross-validation comparing TF-IDF against SBERT variants.

    Args:
        exemplars: Optional exemplar list; defaults to load_exemplar_bank().
        n_splits: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping model names ("tfidf", "sbert_knn", "sbert_lr", "sbert_hybrid")
        to their respective ClassificationReport over all out-of-fold predictions.
    """
    bank = exemplars or load_exemplar_bank()
    texts = [prepare_clause_text(ex.text, ex.section_header) for ex in bank]
    labels = [ex.family for ex in bank]
    label_strs = np.array([f.value for f in labels])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    preds: Dict[str, List[Optional[ClauseFamily]]] = {
        "tfidf": [None] * len(bank),
        "sbert_knn": [None] * len(bank),
        "sbert_lr": [None] * len(bank),
        "sbert_hybrid": [None] * len(bank),
    }

    # Pre-encode all bank texts once with SBERT to accelerate cross-validation
    sbert_model = get_sbert_model()
    raw_embeddings = sbert_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    all_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    embed_map = {texts[i]: all_embeddings[i] for i in range(len(bank))}

    def cached_embed_fn(batch_texts: List[str]) -> np.ndarray:
        return np.array([embed_map[t] for t in batch_texts], dtype=np.float32)

    for train_idx, val_idx in skf.split(texts, label_strs):
        train_exemplars = [bank[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        # 1. TF-IDF Baseline
        tfidf_model = TfidfBaselineClassifier(exemplars=train_exemplars, confidence_threshold=0.0)
        tfidf_preds = tfidf_model.predict(val_texts)
        for i, val_pos in enumerate(val_idx):
            preds["tfidf"][val_pos] = tfidf_preds[i]

        # 2. SBERT Models using fast pre-computed embeddings
        for mode_name in ["knn", "lr", "hybrid"]:
            sbert_model = SbertClassifier(
                exemplars=train_exemplars,
                embed_fn=cached_embed_fn,
                mode=mode_name,  # type: ignore[arg-type]
                confidence_threshold=0.0,
                ood_threshold=0.0,
            )
            probs, _ = sbert_model.predict_proba(val_texts)
            best_idx = np.argmax(probs, axis=1)
            pred_families = [sbert_model._classes[b] for b in best_idx]

            key = f"sbert_{mode_name}"
            for i, val_pos in enumerate(val_idx):
                preds[key][val_pos] = pred_families[i]

    reports: Dict[str, ClassificationReport] = {}
    for model_name, model_preds in preds.items():
        assert all(p is not None for p in model_preds), f"Missing predictions for {model_name}"
        reports[model_name] = compute_metrics(
            y_true=labels,
            y_pred=[p for p in model_preds if p is not None],
        )

    return reports


def format_metrics_table(reports: Dict[str, ClassificationReport]) -> str:
    """Format multiple classification reports into a clean Markdown comparison table.

    Args:
        reports: Dictionary of model name to ClassificationReport.

    Returns:
        Formatted Markdown table string.
    """
    lines = [
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, rep in reports.items():
        lines.append(
            f"| `{name}` | {rep.accuracy:.4f} | {rep.macro_precision:.4f} | "
            f"{rep.macro_recall:.4f} | {rep.macro_f1:.4f} | {rep.weighted_f1:.4f} |"
        )
    return "\n".join(lines)
