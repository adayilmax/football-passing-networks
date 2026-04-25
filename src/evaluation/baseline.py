"""
Baseline models for comparison against GNNs.

Both baselines receive the same information as the GNNs — player-level node
features and edge statistics — but treat it as a flat tabular vector rather
than a graph.  This isolates whether the graph structure itself adds value.

Feature vector per match (constructed by `extract_features`):
  For each team (home, away):
    - mean of node features across all players       [28]
    - std  of node features across all players       [28]
    - global edge stats: mean count, mean success, mean length, num_edges [4]
  Total: 2 * (28 + 28 + 4) = 120 features
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

from src.data.dataset import MatchGraphDataset, MatchPair


# ── Feature extraction ──────────────────────────────────────────────────────

def _graph_to_vec(graph) -> np.ndarray:
    """Flatten a single team graph to a fixed-size numpy vector."""
    x = graph.x.numpy()           # [num_nodes, 28]

    mean_x = x.mean(axis=0)       # [28]
    std_x  = x.std(axis=0)        # [28]  captures within-team variation

    ea = graph.edge_attr.numpy()  # [num_edges, 3]
    if ea.shape[0] > 0:
        edge_stats = np.array([
            ea[:, 0].mean(),   # mean normalised pass count
            ea[:, 1].mean(),   # mean pass success rate
            ea[:, 2].mean(),   # mean normalised pass length
            ea.shape[0] / 200, # num_edges normalised (typical max ~200)
        ])
    else:
        edge_stats = np.zeros(4)

    return np.concatenate([mean_x, std_x, edge_stats])  # [60]


def extract_features(pairs) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract flat feature matrix X and label vector y from a list of MatchPairs.
    Returns X of shape [n, 120] and y of shape [n].
    """
    X, y = [], []
    for pair in pairs:
        home_vec = _graph_to_vec(pair.home_graph)
        away_vec = _graph_to_vec(pair.away_graph)
        X.append(np.concatenate([home_vec, away_vec]))
        y.append(pair.label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Model builders ──────────────────────────────────────────────────────────

def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def build_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )),
    ])


# ── Evaluation helper ────────────────────────────────────────────────────────

def evaluate_sklearn(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds    = model.predict(X_test)
    f1_per   = f1_score(y_test, preds, average=None, zero_division=0, labels=[0, 1, 2])
    class_names = ["Home Win", "Draw", "Away Win"]
    return {
        "accuracy":    float(accuracy_score(y_test, preds)),
        "f1_macro":    float(f1_score(y_test, preds, average="macro",     zero_division=0)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted",  zero_division=0)),
        "f1_per_class": {class_names[i]: float(f1_per[i]) for i in range(3)},
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1, 2]).tolist(),
        "predictions": preds.tolist(),
        "labels": y_test.tolist(),
    }
