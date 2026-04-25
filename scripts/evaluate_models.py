"""
Step 4: Full evaluation of all models + result visualisation.

Produces (all saved to results/plots/ and results/metrics/):
  - confusion matrices for GCN, GAT, LR, MLP
  - accuracy + F1 comparison bar chart
  - training loss/accuracy curves for GCN and GAT
  - classification reports printed to stdout
  - summary JSON saved to results/metrics/all_results.json

Run with:  uv run python evaluate_models.py
"""

import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch

from src.data.dataset import MatchGraphDataset
from src.models.gcn   import GCNModel
from src.models.gat   import GATModel
from src.training.trainer  import evaluate
from src.evaluation.metrics  import compute_metrics, CLASS_NAMES
from src.evaluation.baseline import (
    extract_features, build_logistic_regression, build_mlp, evaluate_sklearn,
)

PLOTS_DIR   = Path("results/plots")
METRICS_DIR = Path("results/metrics")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # blue, orange, green, red


# ============================================================
# 1. Load data and best checkpoints
# ============================================================
print("Loading dataset ...")
dataset = MatchGraphDataset(competition_id=11, season_id=27)
train_ds, val_ds, test_ds = dataset.split(train_frac=0.70, val_frac=0.15, seed=42)

# GNN models
gcn = GCNModel(hidden_dim=64, num_layers=3, dropout=0.3)
gcn.load_state_dict(torch.load(METRICS_DIR / "gcn_best.pt", map_location=DEVICE))

gat = GATModel(hidden_dim=64, num_layers=3, heads=4, dropout=0.3)
gat.load_state_dict(torch.load(METRICS_DIR / "gat_best.pt", map_location=DEVICE))

gcn_preds, gcn_labels = evaluate(gcn, test_ds, DEVICE)
gat_preds, gat_labels = evaluate(gat, test_ds, DEVICE)

gcn_metrics = compute_metrics(gcn_preds, gcn_labels)
gat_metrics = compute_metrics(gat_preds, gat_labels)

# Baseline models
print("Training baseline models ...")
X_train, y_train = extract_features(train_ds)
X_val,   y_val   = extract_features(val_ds)
X_test,  y_test  = extract_features(test_ds)

# Combine train+val for sklearn (early stopping is internal)
X_trval = np.concatenate([X_train, X_val])
y_trval = np.concatenate([y_train, y_val])

lr_model = build_logistic_regression()
lr_model.fit(X_trval, y_trval)
lr_metrics = evaluate_sklearn(lr_model, X_test, y_test)

mlp_model = build_mlp()
mlp_model.fit(X_trval, y_trval)
mlp_metrics = evaluate_sklearn(mlp_model, X_test, y_test)

# ── Print classification reports ────────────────────────────────────────────
for name, m in [("GCN", gcn_metrics), ("GAT", gat_metrics)]:
    print(f"\n{'='*50}")
    print(f"{name} Classification Report")
    print(f"{'='*50}")
    print(m["classification_report"])

# ── Collect summary ──────────────────────────────────────────────────────────
all_results = {
    "GCN":  {k: v for k, v in gcn_metrics.items() if k != "classification_report"},
    "GAT":  {k: v for k, v in gat_metrics.items() if k != "classification_report"},
    "LR":   lr_metrics,
    "MLP":  mlp_metrics,
}

with open(METRICS_DIR / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nMetrics summary:")
print(f"  {'Model':<6}  {'Accuracy':>8}  {'F1 Macro':>8}  {'F1 Wtd':>7}")
print(f"  {'-'*40}")
for name, m in all_results.items():
    print(f"  {name:<6}  {m['accuracy']:>8.3f}  {m['f1_macro']:>8.3f}  {m['f1_weighted']:>7.3f}")


# ============================================================
# 2. Confusion matrices (2x2 grid)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.01)

model_data = [
    ("GCN",  gcn_metrics["confusion_matrix"]),
    ("GAT",  gat_metrics["confusion_matrix"]),
    ("LR",   lr_metrics["confusion_matrix"]),
    ("MLP",  mlp_metrics["confusion_matrix"]),
]

for ax, (name, cm) in zip(axes.flat, model_data):
    cm_arr = np.array(cm)
    # Row-normalise for readability
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(
        cm_norm, annot=cm_arr, fmt="d", ax=ax,
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=False, linewidths=0.5, linecolor="lightgrey",
    )
    acc = all_results[name]["accuracy"]
    f1  = all_results[name]["f1_macro"]
    ax.set_title(f"{name}  (acc={acc:.3f}, F1={f1:.3f})", fontsize=11, pad=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True",      fontsize=9)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrices.png", bbox_inches="tight")
plt.close()
print(f"\nSaved: {PLOTS_DIR}/confusion_matrices.png")


# ============================================================
# 3. Accuracy + F1 comparison bar chart
# ============================================================
models     = ["LR", "MLP", "GCN", "GAT"]
accuracies = [all_results[m]["accuracy"]  for m in models]
f1_macros  = [all_results[m]["f1_macro"]  for m in models]
f1_wtds    = [all_results[m]["f1_weighted"] for m in models]

x = np.arange(len(models))
w = 0.26

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w,     accuracies, w, label="Accuracy",     color=PALETTE[0], alpha=0.9)
ax.bar(x,         f1_macros,  w, label="F1 Macro",     color=PALETTE[1], alpha=0.9)
ax.bar(x + w,     f1_wtds,    w, label="F1 Weighted",  color=PALETTE[2], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(["Logistic\nRegression", "MLP\nBaseline", "GCN", "GAT (ours)"],
                   fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("Model Comparison — Test Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.axhline(1/3, color="grey", linestyle="--", linewidth=0.8, label="Random baseline")

# Annotate accuracy bars
for i, acc in enumerate(accuracies):
    ax.text(x[i] - w, acc + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/model_comparison.png")


# ============================================================
# 4. Training curves (GCN and GAT side by side)
# ============================================================
with open(METRICS_DIR / "gcn_history.json") as f:
    gcn_hist = json.load(f)
with open(METRICS_DIR / "gat_history.json") as f:
    gat_hist = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Training Curves", fontsize=13, fontweight="bold")

for col, (name, hist, color) in enumerate([
    ("GCN", gcn_hist, PALETTE[0]),
    ("GAT", gat_hist, PALETTE[1]),
]):
    ep = range(1, len(hist["train_loss"]) + 1)

    # Loss
    axes[0, col].plot(ep, hist["train_loss"], label="Train", color=color,      lw=1.8)
    axes[0, col].plot(ep, hist["val_loss"],   label="Val",   color=color, lw=1.8,
                      linestyle="--", alpha=0.7)
    axes[0, col].set_title(f"{name} — Loss")
    axes[0, col].set_xlabel("Epoch")
    axes[0, col].set_ylabel("Cross-Entropy Loss")
    axes[0, col].legend(fontsize=8)

    # Accuracy
    axes[1, col].plot(ep, hist["train_acc"], label="Train", color=color,      lw=1.8)
    axes[1, col].plot(ep, hist["val_acc"],   label="Val",   color=color, lw=1.8,
                      linestyle="--", alpha=0.7)
    axes[1, col].set_title(f"{name} — Accuracy")
    axes[1, col].set_xlabel("Epoch")
    axes[1, col].set_ylabel("Accuracy")
    axes[1, col].set_ylim(0, 1)
    axes[1, col].legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "training_curves.png", bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/training_curves.png")


# ============================================================
# 5. Per-class F1 breakdown
# ============================================================
f1_data = {
    name: list(all_results[name]["f1_per_class"].values())
    for name in models
}

fig, ax = plt.subplots(figsize=(8, 5))
x  = np.arange(3)
w2 = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (name, color) in enumerate(zip(models, PALETTE)):
    ax.bar(x + offsets[i] * w2, f1_data[name], w2,
           label=name, color=color, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("Per-Class F1 Score by Model", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "per_class_f1.png", bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/per_class_f1.png")

print(f"\nAll results saved to {METRICS_DIR}/all_results.json")
print("Step 4 complete.")
