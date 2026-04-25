"""
Step 2: Build all match graphs from La Liga 2015/16 and verify the output.
Run with:  uv run python build_dataset.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MatchGraphDataset
from src.data.graph_builder import NUM_NODE_FEATURES, NUM_EDGE_FEATURES

COMP, SEASON = 11, 27   # La Liga 2015/16

# ── 1. Build (or load from cache) ──────────────────────────────────────────
dataset = MatchGraphDataset(competition_id=COMP, season_id=SEASON)

# ── 2. High-level stats ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total match pairs : {len(dataset)}")
print(f"Node feature dim  : {NUM_NODE_FEATURES}  (25 position + completion + pressure + progressive)")
print(f"Edge feature dim  : {NUM_EDGE_FEATURES}  (count, success_rate, mean_length)")

dist = dataset.label_distribution()
total = sum(dist.values())
print(f"\nLabel distribution:")
for k, v in dist.items():
    print(f"  {k:<10}: {v:>3}  ({100*v/total:.1f}%)")

stats = dataset.graph_stats()
print(f"\nGraph statistics (home + away combined):")
for k, v in stats.items():
    print(f"  {k:<12}: {v:.1f}")

# ── 3. Train / val / test split ─────────────────────────────────────────────
train_ds, val_ds, test_ds = dataset.split(train_frac=0.70, val_frac=0.15)
print(f"\nSplits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"  train labels: {train_ds.label_distribution()}")
print(f"  val   labels: {val_ds.label_distribution()}")
print(f"  test  labels: {test_ds.label_distribution()}")

# ── 4. Inspect one sample ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE GRAPH INSPECTION (first match)")
print("=" * 60)

sample = dataset[0]
print(f"Match ID : {sample.match_id}")
print(f"Label    : {sample.label}  (0=home win, 1=draw, 2=away win)")

for name, g in [("Home", sample.home_graph), ("Away", sample.away_graph)]:
    print(f"\n{name} graph:")
    print(f"  x shape        : {g.x.shape}   (nodes × features)")
    print(f"  edge_index     : {g.edge_index.shape}   (2 × edges)")
    print(f"  edge_attr      : {g.edge_attr.shape}   (edges × edge_features)")
    print(f"  Non-zero nodes : {(g.x.sum(dim=1) > 0).sum().item()} / {g.num_nodes}")
    print(f"  x[0] sample    : {g.x[0].tolist()[:6]} ...")
    if g.edge_attr.shape[0] > 0:
        print(f"  edge_attr[0]   : {g.edge_attr[0].tolist()}")

# ── 5. Verify feature ranges ─────────────────────────────────────────────────
import torch
print("\n" + "=" * 60)
print("FEATURE SANITY CHECKS")
print("=" * 60)

all_x = torch.cat([p.home_graph.x for p in dataset] +
                  [p.away_graph.x for p in dataset], dim=0)

pos_sum = all_x[:, :25].sum(dim=1)
print(f"Position one-hot sum  — min: {pos_sum.min():.0f}, max: {pos_sum.max():.0f}  (all should be 0 or 1)")

for i, name in enumerate(["completion_rate", "pressure_rate", "progressive_rate"]):
    col = all_x[:, 25 + i]
    print(f"{name:<20} — min: {col.min():.3f}, max: {col.max():.3f}, mean: {col.mean():.3f}")

all_ea = torch.cat([p.home_graph.edge_attr for p in dataset if p.home_graph.edge_attr.shape[0] > 0] +
                   [p.away_graph.edge_attr for p in dataset if p.away_graph.edge_attr.shape[0] > 0], dim=0)

for i, name in enumerate(["pass_count_norm", "pass_success_rate", "mean_length_norm"]):
    col = all_ea[:, i]
    print(f"{name:<20} — min: {col.min():.3f}, max: {col.max():.3f}, mean: {col.mean():.3f}")

print("\nStep 2 complete. Graphs are cached in data/processed/")
