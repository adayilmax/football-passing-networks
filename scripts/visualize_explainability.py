"""
Step 7: GNNExplainer-based interpretability for the GCN model.

For each of 6 selected test matches (2 home wins, 2 draws, 2 away wins) we:
  1. Run GNNExplainer on each team graph (home + away)
  2. Plot a formation diagram with node size = node importance and
     edge thickness/colour = edge importance
  3. Compare the top-k explainer edges against the GAT's top-k attention
     edges for the same match — Jaccard similarity quantifies whether the
     two interpretability methods agree on which relationships matter.

Outputs (results/plots/):
  explainer_formation_<match_id>.png   — one figure per match
  explainer_vs_attention.png           — agreement bar chart across matches
  explainer_agreement.json             — raw agreement numbers

Run with:  uv run python visualize_explainability.py
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import MatchGraphDataset
from src.data.loader import get_events, get_lineups, get_matches
from src.evaluation.attention import extract_attention
from src.evaluation.explainer import (
    edge_set_jaccard,
    explain_team_graph,
    plot_explainer_formation,
)
from src.models.gat import GATModel
from src.models.gcn import GCNModel

PLOTS_DIR   = Path("results/plots")
METRICS_DIR = Path("results/metrics")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")

OUTCOME_NAMES = {0: "Home Win", 1: "Draw", 2: "Away Win"}
TOP_K_EDGES   = 25


# ── Load models and dataset ─────────────────────────────────────────────────
print("Loading GCN and GAT models ...")
gcn = GCNModel(hidden_dim=64, num_layers=3, dropout=0.3)
gcn.load_state_dict(torch.load(METRICS_DIR / "gcn_best.pt", map_location=DEVICE))
gcn.eval()

gat = GATModel(hidden_dim=64, num_layers=3, heads=4, dropout=0.3)
gat.load_state_dict(torch.load(METRICS_DIR / "gat_best.pt", map_location=DEVICE))
gat.eval()

print("Loading dataset ...")
dataset = MatchGraphDataset(competition_id=11, season_id=27)
train_ds, val_ds, test_ds = dataset.split(train_frac=0.70, val_frac=0.15, seed=42)

matches_df = get_matches(11, 27)
mid_to_row = {int(r["match_id"]): r for _, r in matches_df.iterrows()}


# ── Select 6 test-set matches: 2 per outcome class ──────────────────────────
print("\nSelecting 6 test matches (2 per outcome) ...")

by_label: dict[int, list] = {0: [], 1: [], 2: []}
for pair in test_ds:
    by_label[pair.label].append(pair)

selected = []
for label in [0, 1, 2]:
    if len(by_label[label]) < 2:
        # Fall back to repeating the only available pair if class is short
        chosen = by_label[label][:2] or by_label[label]
    else:
        chosen = by_label[label][:2]
    for p in chosen:
        selected.append((p, OUTCOME_NAMES[label]))

# Pre-load raw events / lineups for the selected matches only
print("Loading raw events for selected matches ...")
records = {}
for pair, _ in selected:
    mid = pair.match_id
    row = mid_to_row[mid]
    records[mid] = {
        "match_id":   mid,
        "home_team":  row["home_team"],
        "away_team":  row["away_team"],
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
        "events":     get_events(mid),
        "lineups":    get_lineups(mid),
    }
    print(f"  {row['home_team']} {row['home_score']}-{row['away_score']} "
          f"{row['away_team']}  [{OUTCOME_NAMES[pair.label]}]")


# ── Run GNNExplainer + plot formation diagrams ──────────────────────────────
print(f"\nRunning GNNExplainer on {len(selected)} matches ...")

agreement_records: list[dict] = []

for pair, outcome_name in selected:
    mid = pair.match_id
    rec = records[mid]
    print(f"\n  Match {mid} — {rec['home_team']} vs {rec['away_team']} [{outcome_name}]")

    # Explain both teams
    print("    explaining home graph ...")
    expl_home = explain_team_graph(gcn, pair.home_graph, pair.away_graph, side="home")
    print("    explaining away graph ...")
    expl_away = explain_team_graph(gcn, pair.home_graph, pair.away_graph, side="away")

    # Plot formation diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a4f1a")
    hs, as_ = rec["home_score"], rec["away_score"]
    fig.suptitle(
        f"{rec['home_team']} {hs}-{as_} {rec['away_team']}  |  "
        f"{outcome_name}  |  GNNExplainer (GCN)",
        fontsize=12, color="white", fontweight="bold", y=1.01,
    )

    plot_explainer_formation(
        expl_home, rec, axes[0],
        title=f"{rec['home_team']} (Home)",
        top_k_edges=TOP_K_EDGES,
    )
    plot_explainer_formation(
        expl_away, rec, axes[1],
        title=f"{rec['away_team']} (Away)",
        top_k_edges=TOP_K_EDGES,
    )

    plt.tight_layout()
    out = PLOTS_DIR / f"explainer_formation_{mid}.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    saved: {out.name}")

    # ── Agreement: GNNExplainer edges (GCN) vs GAT attention edges ─────────
    # Run the GAT on the same graphs to get its attention rankings
    gat_home_ei, gat_home_attn = extract_attention(gat, pair.home_graph)
    gat_away_ei, gat_away_attn = extract_attention(gat, pair.away_graph)

    j_home = edge_set_jaccard(
        expl_home.edge_index, expl_home.edge_mask,
        gat_home_ei,           gat_home_attn,
        top_k=TOP_K_EDGES,
    )
    j_away = edge_set_jaccard(
        expl_away.edge_index, expl_away.edge_mask,
        gat_away_ei,           gat_away_attn,
        top_k=TOP_K_EDGES,
    )
    print(f"    Jaccard top-{TOP_K_EDGES}  home: {j_home:.3f}   away: {j_away:.3f}")

    agreement_records.append({
        "match_id":   mid,
        "outcome":    outcome_name,
        "home_team":  rec["home_team"],
        "away_team":  rec["away_team"],
        "jaccard_home": float(j_home),
        "jaccard_away": float(j_away),
    })


# ── Random-baseline reference ────────────────────────────────────────────────
# For top-k from each set independently chosen at random, expected Jaccard is
# small. We compute it empirically for the same edge counts to give context.
def random_jaccard_baseline(num_edges: int, k: int, n_trials: int = 200) -> float:
    if num_edges <= 0 or k <= 0:
        return float("nan")
    k = min(k, num_edges)
    js = []
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        a = set(rng.choice(num_edges, size=k, replace=False).tolist())
        b = set(rng.choice(num_edges, size=k, replace=False).tolist())
        js.append(len(a & b) / len(a | b))
    return float(np.mean(js))

# Use the average edge count across the visualised graphs
avg_edges = float(np.mean([
    pair.home_graph.edge_index.shape[1] for pair, _ in selected
] + [
    pair.away_graph.edge_index.shape[1] for pair, _ in selected
]))
random_baseline = random_jaccard_baseline(int(round(avg_edges)), TOP_K_EDGES)
print(f"\nRandom baseline Jaccard (top-{TOP_K_EDGES} on ~{avg_edges:.0f} edges): {random_baseline:.3f}")


# ── Agreement summary plot ──────────────────────────────────────────────────
print("\nBuilding agreement summary plot ...")

labels    = []
home_vals = []
away_vals = []
for r in agreement_records:
    short = f"{r['home_team'][:10]}\nv {r['away_team'][:10]}\n[{r['outcome']}]"
    labels.append(short)
    home_vals.append(r["jaccard_home"])
    away_vals.append(r["jaccard_away"])

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(labels))
w = 0.38
ax.bar(x - w / 2, home_vals, w, label="Home graph", color="#4878cf")
ax.bar(x + w / 2, away_vals, w, label="Away graph", color="#d43f3a")
ax.axhline(random_baseline, color="grey", linestyle="--", lw=1,
           label=f"Random baseline ({random_baseline:.2f})")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Jaccard similarity (top-25 edges)")
ax.set_title("GNNExplainer vs GAT attention — top-25 edge agreement",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = PLOTS_DIR / "explainer_vs_attention.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ── Persist raw numbers ─────────────────────────────────────────────────────
mean_home = float(np.mean(home_vals))
mean_away = float(np.mean(away_vals))
mean_all  = float(np.mean(home_vals + away_vals))

summary = {
    "top_k": TOP_K_EDGES,
    "matches": agreement_records,
    "mean_jaccard_home": mean_home,
    "mean_jaccard_away": mean_away,
    "mean_jaccard_overall": mean_all,
    "random_baseline": random_baseline,
}
with open(METRICS_DIR / "explainer_agreement.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nMean Jaccard (home): {mean_home:.3f}")
print(f"Mean Jaccard (away): {mean_away:.3f}")
print(f"Mean Jaccard (all):  {mean_all:.3f}")
print(f"Random baseline:     {random_baseline:.3f}")
print(f"\nSaved: {METRICS_DIR / 'explainer_agreement.json'}")

print("\nStep 7 complete.")
