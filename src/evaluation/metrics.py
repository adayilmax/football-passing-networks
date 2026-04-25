"""
Shared metrics utilities for GNN model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)


CLASS_NAMES = ["Home Win", "Draw", "Away Win"]


def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    """Return accuracy, per-class F1, macro F1, weighted F1, and confusion matrix."""
    acc      = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wt    = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_per   = f1_score(labels, preds, average=None,       zero_division=0, labels=[0,1,2])
    cm       = confusion_matrix(labels, preds, labels=[0, 1, 2])
    report   = classification_report(
        labels, preds, target_names=CLASS_NAMES, zero_division=0
    )
    return {
        "accuracy":    float(acc),
        "f1_macro":    float(f1_macro),
        "f1_weighted": float(f1_wt),
        "f1_per_class": {CLASS_NAMES[i]: float(f1_per[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": preds,
        "labels": labels,
    }
