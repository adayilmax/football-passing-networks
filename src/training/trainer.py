"""
Training and evaluation loop for GNN match-outcome models.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.collate import collate_match_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "results"
METRICS_DIR  = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ── Class-weighted loss ──────────────────────────────────────────────────────

def compute_class_weights(dataset, num_classes: int = 3) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle the home-win / draw imbalance.
    Returns a tensor of shape [num_classes].
    """
    counts = torch.zeros(num_classes)
    for pair in dataset:
        counts[pair.label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


# ── Single epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """
    Run one epoch.  Returns (mean_loss, accuracy).
    Pass optimizer=None and train=False for eval mode.
    """
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for home, away, labels in loader:
            home   = home.to(device)
            away   = away.to(device)
            labels = labels.to(device)

            logits = model(home, away)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ── Full training run ────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_ds,
    val_ds,
    model_name: str,
    epochs: int            = 100,
    batch_size: int        = 32,
    lr: float              = 1e-3,
    weight_decay: float    = 1e-4,
    patience: int          = 15,
    device: torch.device   = torch.device("cpu"),
) -> dict:
    """
    Train model with early stopping on validation loss.

    Returns a history dict with loss/accuracy curves and best metrics.
    Saves the best checkpoint to results/metrics/<model_name>_best.pt
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_match_pairs, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_match_pairs,
    )

    class_weights = compute_class_weights(train_ds).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7,
    )

    model.to(device)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_loss   = float("inf")
    epochs_no_improv = 0
    best_ckpt_path  = METRICS_DIR / f"{model_name}_best.pt"

    print(f"\n{'-'*60}")
    print(f"Training {model_name}  |  {model.count_parameters():,} parameters")
    print(f"Device: {device}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'-'*60}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"{'-'*60}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   None,      criterion, device, train=False)

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        marker = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improv = 0
            torch.save(model.state_dict(), best_ckpt_path)
            marker = " *"
        else:
            epochs_no_improv += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.3f}  {vl_loss:>8.4f}  {vl_acc:>6.3f}{marker}")

        if epochs_no_improv >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"{'-'*60}")
    print(f"Training complete in {elapsed:.1f}s  |  Best val loss: {best_val_loss:.4f}")

    # Save history
    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])
    with open(METRICS_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[list[int], list[int]]:
    """
    Return (all_preds, all_labels) over the entire dataset.
    Loads the model in eval mode — caller should load best checkpoint first.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_match_pairs,
    )
    model.eval()
    model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for home, away, labels in loader:
            home   = home.to(device)
            away   = away.to(device)
            logits = model(home, away)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels
