"""
Step 6: Static (full-match) vs dynamic (first-half) graph comparison.

For each model type (GCN, GAT):
  - Train on full-match graphs, evaluate on test set
  - Train on first-half-only graphs, evaluate on test set
  - Compare accuracy, F1, graph structure statistics

Research question: does the second half of passing network data add predictive
value, or is the structure established in the first 45 minutes sufficient?

Produces:
  results/metrics/temporal_results.json
  results/plots/temporal_comparison.png

Run with:  uv run python compare_temporal.py
"""

import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.dataset      import MatchGraphDataset
from src.models.gcn        import GCNModel
from src.models.gat        import GATModel
from src.training.trainer  import train, evaluate
from src.evaluation.metrics  import compute_metrics, CLASS_NAMES
from src.evaluation.temporal import graph_structure_stats, node_feature_stats

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_DIR = Path("results/metrics")
PLOTS_DIR   = Path("results/plots")
METRICS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Shared hyper-parameters — identical for all runs so comparisons are fair
HYP = dict(hidden_dim=64, num_layers=3, dropout=0.3)
TRAIN_CFG = dict(epochs=120, batch_size=32, lr=1e-3, device=DEVICE)

SPLIT_KWARGS = dict(train_frac=0.70, val_frac=0.15, seed=42)

# ── 1. Build / load both datasets ───────────────────────────────────────────
print("=" * 60)
print("Loading datasets")
print("=" * 60)

ds_full = MatchGraphDataset(competition_id=11, season_id=27, half_only=False)
ds_half = MatchGraphDataset(competition_id=11, season_id=27, half_only=True)

train_full, val_full, test_full = ds_full.split(**SPLIT_KWARGS)
train_half, val_half, test_half = ds_half.split(**SPLIT_KWARGS)

# Use the SAME test-set match IDs for a fair comparison.
# Both splits use seed=42 on the same match list so indices align.
print(f"Full-match — train {len(train_full)} / val {len(val_full)} / test {len(test_full)}")
print(f"First-half — train {len(train_half)} / val {len(val_half)} / test {len(test_half)}")

# ── 2. Graph structure comparison ───────────────────────────────────────────
print("\nGraph structure stats:")
for name, ds in [("Full", ds_full), ("Half", ds_half)]:
    s  = graph_structure_stats(ds)
    nf = node_feature_stats(ds)
    print(f"  {name}: nodes {s['mean_nodes']:.1f}±{s['std_nodes']:.1f}  "
          f"edges {s['mean_edges']:.1f}±{s['std_edges']:.1f}  "
          f"completion {nf['completion_rate']:.3f}  "
          f"pressure {nf['pressure_rate']:.3f}  "
          f"progressive {nf['progressive_rate']:.3f}")

# ── 3. Evaluate full-match and train half-match variants ─────────────────────
# Full-match: load the already-trained checkpoints from train_models.py
# (retraining from scratch introduces variance that contaminates the comparison).
# Half-match: train new models from scratch on first-half-only data and save
# to separate checkpoint paths so the main checkpoints are never overwritten.
results = {}

# -- 3a. Full-match: load existing checkpoints --------------------------------
print(f"\n{'='*60}")
print("Loading full-match models (from train_models.py checkpoints)")
print(f"{'='*60}")

for ModelClass, arch in [(GCNModel, "gcn"), (GATModel, "gat")]:
    name = f"{arch}_full"
    ckpt = METRICS_DIR / f"{arch}_best.pt"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} not found. Run train_models.py first."
        )

    model = ModelClass(**HYP, heads=4) if arch == "gat" else ModelClass(**HYP)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    preds, labels = evaluate(model, test_full, DEVICE)
    m = compute_metrics(preds, labels)

    # Provide a stub history for downstream plotting (no training run here)
    results[name] = {
        "accuracy":    m["accuracy"],
        "f1_macro":    m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
        "f1_per_class": m["f1_per_class"],
        "history":     {"val_loss": [], "train_loss": []},
    }
    print(f"  {name:<12}  acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}")

# -- 3b. Half-match: train new models, save to temporal-specific checkpoints --
print(f"\n{'='*60}")
print("Training half-match models")
print(f"{'='*60}")

for ModelClass, arch in [(GCNModel, "gcn"), (GATModel, "gat")]:
    name = f"{arch}_half"
    # Use a distinct name so we never touch gcn_best.pt / gat_best.pt
    temporal_model_name = f"{arch}_temporal_half"
    ckpt = METRICS_DIR / f"{temporal_model_name}_best.pt"

    model = ModelClass(**HYP, heads=4) if arch == "gat" else ModelClass(**HYP)

    history = train(
        model      = model,
        train_ds   = train_half,
        val_ds     = val_half,
        model_name = temporal_model_name,
        **TRAIN_CFG,
    )

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    preds, labels = evaluate(model, test_half, DEVICE)
    m = compute_metrics(preds, labels)

    results[name] = {
        "accuracy":    m["accuracy"],
        "f1_macro":    m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
        "f1_per_class": m["f1_per_class"],
        "history":     history,
    }
    print(f"  {name:<12}  acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}")

# Save results
with open(METRICS_DIR / "temporal_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "history"}
               for k, v in results.items()}, f, indent=2)

# ── 4. Visualisation ─────────────────────────────────────────────────────────
print("\nGenerating plots ...")

PALETTE = {"gcn_full": "#4C72B0", "gcn_half": "#9DB8D9",
           "gat_full": "#DD8452", "gat_half": "#F0C09A"}
LABELS  = {"gcn_full": "GCN Full", "gcn_half": "GCN Half",
           "gat_full": "GAT Full", "gat_half": "GAT Half"}

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Full-Match vs First-Half Graph Comparison", fontsize=14, fontweight="bold")

gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)

# ── Panel A: accuracy and F1 bar chart
ax_bar = fig.add_subplot(gs[0, 0])
keys   = ["gcn_full", "gcn_half", "gat_full", "gat_half"]
x      = np.arange(2)   # accuracy, F1 macro
w      = 0.18
for i, k in enumerate(keys):
    vals = [results[k]["accuracy"], results[k]["f1_macro"]]
    ax_bar.bar(x + (i - 1.5) * w, vals, w,
               color=PALETTE[k], label=LABELS[k], alpha=0.9)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(["Accuracy", "F1 Macro"], fontsize=10)
ax_bar.set_ylim(0, 1)
ax_bar.set_title("Test Performance", fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=7, ncol=2)
ax_bar.set_ylabel("Score")

# ── Panel B: per-class F1 for GAT full vs half
ax_f1  = fig.add_subplot(gs[0, 1])
x2     = np.arange(3)
w2     = 0.30
for i, k in enumerate(["gat_full", "gat_half"]):
    f1s = [results[k]["f1_per_class"][c] for c in CLASS_NAMES]
    ax_f1.bar(x2 + (i - 0.5) * w2, f1s, w2,
              color=PALETTE[k], label=LABELS[k], alpha=0.9)
ax_f1.set_xticks(x2)
ax_f1.set_xticklabels(CLASS_NAMES, fontsize=9)
ax_f1.set_ylim(0, 1)
ax_f1.set_title("GAT Per-Class F1", fontsize=11, fontweight="bold")
ax_f1.legend(fontsize=8)
ax_f1.set_ylabel("F1 Score")

# ── Panel C: graph structure (edges full vs half)
ax_struct = fig.add_subplot(gs[0, 2])
struct = {
    "Full": graph_structure_stats(ds_full),
    "Half": graph_structure_stats(ds_half),
}
struct_x = np.arange(2)
ax_struct.bar(struct_x, [struct["Full"]["mean_edges"], struct["Half"]["mean_edges"]],
              color=["#4C72B0", "#9DB8D9"], width=0.4, alpha=0.9,
              yerr=[struct["Full"]["std_edges"], struct["Half"]["std_edges"]],
              capsize=5)
ax_struct.set_xticks(struct_x)
ax_struct.set_xticklabels(["Full match", "First half"], fontsize=10)
ax_struct.set_title("Mean Edges per Team Graph", fontsize=11, fontweight="bold")
ax_struct.set_ylabel("Edge count")

# Annotate
for xi, key in zip(struct_x, ["Full", "Half"]):
    ax_struct.text(xi, struct[key]["mean_edges"] + struct[key]["std_edges"] + 1,
                   f'{struct[key]["mean_edges"]:.0f}', ha="center", fontsize=9)

# ── Panels D & E: training loss curve for half-match models + per-class F1 GCN
# Panel D: half-match training loss (only half models have a history)
ax_d = fig.add_subplot(gs[1, 0])
for arch, colour in [("gcn", "#4C72B0"), ("gat", "#DD8452")]:
    h = results[f"{arch}_half"]["history"]
    ep = range(1, len(h["val_loss"]) + 1)
    ax_d.plot(ep, h["val_loss"], color=colour, lw=1.8,
              label=arch.upper(), linestyle="--" if arch == "gcn" else "-")
ax_d.set_title("Val Loss - First-Half Models", fontsize=11, fontweight="bold")
ax_d.set_xlabel("Epoch"); ax_d.set_ylabel("Cross-Entropy Loss")
ax_d.legend(fontsize=9)

# Panel E: per-class F1 for GCN full vs half
ax_e = fig.add_subplot(gs[1, 1])
x_e  = np.arange(3)
w_e  = 0.30
for i, k in enumerate(["gcn_full", "gcn_half"]):
    f1s = [results[k]["f1_per_class"][c] for c in CLASS_NAMES]
    ax_e.bar(x_e + (i - 0.5) * w_e, f1s, w_e,
             color=PALETTE[k], label=LABELS[k], alpha=0.9)
ax_e.set_xticks(x_e)
ax_e.set_xticklabels(CLASS_NAMES, fontsize=9)
ax_e.set_ylim(0, 1)
ax_e.set_title("GCN Per-Class F1", fontsize=11, fontweight="bold")
ax_e.legend(fontsize=8)
ax_e.set_ylabel("F1 Score")

# ── Panel F: delta accuracy (full - half) per class for both models
ax_delta = fig.add_subplot(gs[1, 2])
x3 = np.arange(3)
w3 = 0.30
for i, arch in enumerate(["gcn", "gat"]):
    delta = [results[f"{arch}_full"]["f1_per_class"][c]
             - results[f"{arch}_half"]["f1_per_class"][c]
             for c in CLASS_NAMES]
    colour = "#4C72B0" if arch == "gcn" else "#DD8452"
    bars = ax_delta.bar(x3 + (i - 0.5) * w3, delta, w3,
                        color=colour, label=arch.upper(), alpha=0.9)
ax_delta.axhline(0, color="black", lw=0.8, linestyle="--")
ax_delta.set_xticks(x3)
ax_delta.set_xticklabels(CLASS_NAMES, fontsize=9)
ax_delta.set_title("F1 Gain from Full Match\n(Full minus Half)", fontsize=11, fontweight="bold")
ax_delta.set_ylabel("F1 difference")
ax_delta.legend(fontsize=9)

plt.savefig(PLOTS_DIR / "temporal_comparison.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR}/temporal_comparison.png")

# ── 5. Print summary table ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Model':<12}  {'Accuracy':>8}  {'F1 Macro':>8}  {'F1 Wtd':>7}")
print(f"  {'-'*44}")
for k in ["gcn_full", "gcn_half", "gat_full", "gat_half"]:
    r = results[k]
    print(f"  {k:<12}  {r['accuracy']:>8.3f}  {r['f1_macro']:>8.3f}  {r['f1_weighted']:>7.3f}")

for arch in ["gcn", "gat"]:
    acc_gain = results[f"{arch}_full"]["accuracy"] - results[f"{arch}_half"]["accuracy"]
    f1_gain  = results[f"{arch}_full"]["f1_macro"] - results[f"{arch}_half"]["f1_macro"]
    print(f"\n  {arch.upper()} full vs half: acc delta={acc_gain:+.3f}, F1 delta={f1_gain:+.3f}")

print("\nStep 6 complete.")
