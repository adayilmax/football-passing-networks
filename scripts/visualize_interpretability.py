"""
Step 5: GAT interpretability visualisations + t-SNE of node embeddings.

Produces (all saved to results/plots/):
  attention_formation_<match_id>.png  -- formation diagrams with attention edges
  attention_aggregate_heatmap.png     -- position-pair attention averaged over test set
  tsne_node_embeddings.png            -- 2-D t-SNE of player node embeddings

Run with:  uv run python visualize_interpretability.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.data import Batch

from src.data.dataset     import MatchGraphDataset
from src.data.loader      import get_matches, get_events, get_lineups
from src.data.graph_builder import POSITIONS, build_team_graph
from src.models.gat       import GATModel
from src.evaluation.attention import (
    plot_attention_formation,
    compute_aggregate_attention,
    POS_GROUP, GROUP_COLOUR,
)

PLOTS_DIR   = Path("results/plots")
METRICS_DIR = Path("results/metrics")
DEVICE      = torch.device("cpu")

# ── Load model and dataset ───────────────────────────────────────────────────
print("Loading GAT model and dataset ...")
gat = GATModel(hidden_dim=64, num_layers=3, heads=4, dropout=0.3)
gat.load_state_dict(torch.load(METRICS_DIR / "gat_best.pt", map_location=DEVICE))
gat.eval()

dataset = MatchGraphDataset(competition_id=11, season_id=27)
train_ds, val_ds, test_ds = dataset.split(train_frac=0.70, val_frac=0.15, seed=42)

matches_df = get_matches(11, 27)
mid_to_row  = {int(r["match_id"]): r for _, r in matches_df.iterrows()}

# Index full dataset by match_id for formation diagram lookup
full_by_mid = {pair.match_id: pair for pair in dataset}

BIG_CLUBS = {"Real Madrid", "Barcelona", "Atletico Madrid"}

def _has_big_club(mid):
    row = mid_to_row[mid]
    return row["home_team"] in BIG_CLUBS or row["away_team"] in BIG_CLUBS

# ── Select 3 matches: one per outcome, each featuring a big club.
# Search full dataset (not just test split) so we always find big-club matches.
OUTCOME_NAMES = {0: "Home Win", 1: "Draw", 2: "Away Win"}

by_outcome_big = {0: [], 1: [], 2: []}
for pair in dataset:
    if _has_big_club(pair.match_id):
        by_outcome_big[pair.label].append(pair)

selected = []
for label in [0, 1, 2]:
    candidates = by_outcome_big[label]
    if candidates:
        # Prefer Barcelona > Real Madrid > Atletico Madrid for recognisability
        def _priority(p):
            row = mid_to_row[p.match_id]
            teams = {row["home_team"], row["away_team"]}
            if "Barcelona" in teams:      return 0
            if "Real Madrid" in teams:    return 1
            return 2
        best = min(candidates, key=_priority)
        selected.append((best, OUTCOME_NAMES[label]))
    else:
        # Fallback: any match with this outcome from the full dataset
        fallback = [p for p in dataset if p.label == label]
        if fallback:
            selected.append((fallback[0], OUTCOME_NAMES[label]))

# Pre-load raw events + lineups only for the selected matches
print("Loading raw events for selected matches ...")
viz_records = {}
for pair, _ in selected:
    mid = pair.match_id
    row = mid_to_row[mid]
    viz_records[mid] = {
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

# Also build test_records for the aggregate heatmap (uses test_ds)
print("Loading raw events for test set (aggregate heatmap) ...")
test_records = {}
for pair in test_ds:
    mid = pair.match_id
    if mid in viz_records:
        test_records[mid] = viz_records[mid]
        continue
    row = mid_to_row[mid]
    test_records[mid] = {
        "match_id":   mid,
        "home_team":  row["home_team"],
        "away_team":  row["away_team"],
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
        "events":     get_events(mid),
        "lineups":    get_lineups(mid),
    }

print(f"\nGenerating formation attention diagrams for {len(selected)} matches ...")

for pair, outcome_name in selected:
    mid = pair.match_id
    rec = viz_records[mid]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a4f1a")

    hs, as_ = rec["home_score"], rec["away_score"]
    fig.suptitle(
        f"{rec['home_team']} {hs}–{as_} {rec['away_team']}  |  {outcome_name}",
        fontsize=12, color="white", fontweight="bold", y=1.01,
    )

    plot_attention_formation(gat, rec, "home", axes[0],
                             title=f"{rec['home_team']} (Home)", top_k_edges=25)
    plot_attention_formation(gat, rec, "away", axes[1],
                             title=f"{rec['away_team']} (Away)", top_k_edges=25)

    plt.tight_layout()
    out = PLOTS_DIR / f"attention_formation_{mid}.png"
    plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# 2. Aggregate position-pair attention heatmap
# ============================================================
print("\nComputing aggregate attention heatmap over test set ...")

agg = compute_aggregate_attention(
    gat, list(test_ds), test_records,
    events_map={}, lineups_map={},
)

GROUP_NAMES = ["GK", "DEF", "MID", "ATT"]
fig, ax = plt.subplots(figsize=(6, 5))

mask = (agg == 0)
sns.heatmap(
    agg, annot=True, fmt=".3f", ax=ax,
    cmap="YlOrRd", xticklabels=GROUP_NAMES, yticklabels=GROUP_NAMES,
    mask=mask, linewidths=0.5, linecolor="lightgrey",
    cbar_kws={"label": "Mean attention weight"},
)
ax.set_xlabel("Recipient position group", fontsize=10)
ax.set_ylabel("Passer position group",    fontsize=10)
ax.set_title("GAT Aggregate Attention by Position Pair\n(test set, final layer, mean across heads)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "attention_aggregate_heatmap.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")


# ============================================================
# 3. t-SNE of node embeddings
# ============================================================
print("\nExtracting node embeddings for t-SNE ...")

embeddings, pos_groups, team_won = [], [], []

def get_node_embeddings(model, graph) -> np.ndarray:
    """Return node-level embeddings [num_nodes, hidden_dim] from encoder."""
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list([graph])
        emb   = model.encoder(batch.x, batch.edge_index, batch.edge_attr)
    return emb.cpu().numpy()

for pair in dataset:   # use full dataset for richer t-SNE
    mid = pair.match_id
    if mid not in test_records:
        # Load raw data for non-test matches too
        row = mid_to_row[mid]
        rec = {
            "events":  get_events(mid),
            "lineups": get_lineups(mid),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
        }
    else:
        rec = test_records[mid]

    events  = rec["events"]
    lineups = rec["lineups"]

    for team_key, graph in [("home", pair.home_graph), ("away", pair.away_graph)]:
        team_name = rec["home_team"] if team_key == "home" else rec["away_team"]
        lineup_df = lineups[team_name]

        lineup_ids  = set(lineup_df["player_id"].astype(int).tolist())
        team_passes = events[(events["team"] == team_name) & (events["type"] == "Pass")]
        event_ids   = set(team_passes["player_id"].dropna().astype(int).tolist())
        all_pids    = sorted(lineup_ids | event_ids)
        p2i         = {pid: i for i, pid in enumerate(all_pids)}

        embs = get_node_embeddings(gat, graph)  # [N, 64]

        # Determine if this team won (for marker shape)
        if team_key == "home":
            won = 1 if pair.label == 0 else (0 if pair.label == 2 else -1)
        else:
            won = 1 if pair.label == 2 else (0 if pair.label == 0 else -1)

        for pid in all_pids:
            idx = p2i[pid]
            pos_str = events[(events["player_id"] == pid) & events["position"].notna()]
            pos = pos_str["position"].mode().iloc[0] if not pos_str.empty else "Center Midfield"
            group = POS_GROUP.get(pos, "MID")

            embeddings.append(embs[idx])
            pos_groups.append(group)
            team_won.append(won)

    if len(embeddings) >= 2000:   # cap for speed
        break

embeddings = np.array(embeddings)
print(f"  Running t-SNE on {len(embeddings)} player embeddings ...")

tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, n_jobs=-1)
Z    = tsne.fit_transform(embeddings)

# Plot — colour by position group, marker by win/draw/loss
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("t-SNE of GAT Node Embeddings", fontsize=13, fontweight="bold")

# Panel A: coloured by position group
ax = axes[0]
for grp in ["GK", "DEF", "MID", "ATT"]:
    mask = np.array(pos_groups) == grp
    ax.scatter(Z[mask, 0], Z[mask, 1], c=GROUP_COLOUR[grp],
               label=grp, s=8, alpha=0.55, linewidths=0)
ax.set_title("Coloured by Position Group", fontsize=11)
ax.set_xlabel("t-SNE 1", fontsize=9); ax.set_ylabel("t-SNE 2", fontsize=9)
ax.legend(markerscale=3, fontsize=9, framealpha=0.7)
ax.tick_params(labelsize=7)

# Panel B: coloured by match outcome for that team
ax = axes[1]
outcome_map  = {1: ("Win",  "#2ecc71"), 0: ("Draw", "#f39c12"), -1: ("Loss", "#e74c3c")}
won_arr = np.array(team_won)
for val, (label, colour) in outcome_map.items():
    mask = won_arr == val
    ax.scatter(Z[mask, 0], Z[mask, 1], c=colour,
               label=label, s=8, alpha=0.55, linewidths=0)
ax.set_title("Coloured by Team Match Outcome", fontsize=11)
ax.set_xlabel("t-SNE 1", fontsize=9); ax.set_ylabel("t-SNE 2", fontsize=9)
ax.legend(markerscale=3, fontsize=9, framealpha=0.7)
ax.tick_params(labelsize=7)

plt.tight_layout()
out = PLOTS_DIR / "tsne_node_embeddings.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out.name}")

print("\nStep 5 complete.")
print(f"All plots saved to {PLOTS_DIR}/")
