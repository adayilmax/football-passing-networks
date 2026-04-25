"""
Step 3: Train GCN and GAT models on the La Liga 2015/16 graph dataset.
Run with:  uv run python train_models.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import MatchGraphDataset
from src.models.gcn   import GCNModel
from src.models.gat   import GATModel
from src.training.trainer import train, evaluate

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT    = 0.3
EPOCHS     = 120
BATCH_SIZE = 32
LR         = 1e-3

# ── Data ────────────────────────────────────────────────────────────────────
print("Loading dataset …")
dataset = MatchGraphDataset(competition_id=11, season_id=27)
train_ds, val_ds, test_ds = dataset.split(train_frac=0.70, val_frac=0.15, seed=42)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
print(f"Device: {DEVICE}")

# ── GCN ─────────────────────────────────────────────────────────────────────
gcn = GCNModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)
gcn_history = train(
    model      = gcn,
    train_ds   = train_ds,
    val_ds     = val_ds,
    model_name = "gcn",
    epochs     = EPOCHS,
    batch_size = BATCH_SIZE,
    lr         = LR,
    device     = DEVICE,
)

# ── GAT ─────────────────────────────────────────────────────────────────────
gat = GATModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, heads=4, dropout=DROPOUT)
gat_history = train(
    model      = gat,
    train_ds   = train_ds,
    val_ds     = val_ds,
    model_name = "gat",
    epochs     = EPOCHS,
    batch_size = BATCH_SIZE,
    lr         = LR,
    device     = DEVICE,
)

# ── Quick test-set peek (full eval is in evaluate_models.py) ─────────────────
print("\n" + "=" * 60)
print("QUICK TEST ACCURACY CHECK")
print("=" * 60)

for name, model in [("GCN", gcn), ("GAT", gat)]:
    ckpt = Path("results/metrics") / f"{name.lower()}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    preds, labels = evaluate(model, test_ds, DEVICE)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"  {name} test accuracy: {acc:.3f}  ({sum(p==l for p,l in zip(preds,labels))}/{len(labels)})")

print("\nBest checkpoints saved to results/metrics/")
print("Training histories saved to results/metrics/*_history.json")
print("\nReady for Step 4: evaluation + visualisation")
