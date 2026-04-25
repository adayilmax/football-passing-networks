"""
GNNExplainer-based interpretability for the dual-encoder GCN.

GAT attention (in attention.py) is an *intrinsic* explanation: the model
exposes attention coefficients as part of its forward pass.  GNNExplainer is
a *post-hoc* explanation: it learns soft node and edge masks that, when
applied to the input, preserve the model's prediction with as few elements
as possible.  Running both lets us cross-check whether the two methods agree.

The dual-encoder takes TWO graphs (home + away) but PyG's GNNExplainer only
explains a single graph at a time.  ``SingleGraphWrapper`` solves this by
pre-computing the *other* team's pooled embedding once at construction time
and concatenating it inside ``forward``.  Only the explained graph runs
through message passing during the explanation, so GNNExplainer's
edge-mask hook never sees a size mismatch.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explainer, GNNExplainer

from src.data.graph_builder import build_team_graph
from src.evaluation.attention import (
    DEFAULT_XY,
    FORMATION_XY,
    GROUP_COLOUR,
    POS_GROUP,
    _get_starting_xi,
    _node_positions_and_labels,
)
from src.models.base import MatchOutcomeModel


# ────────────────────────────────────────────────────────────────────────────
# Wrapper: makes a dual-encoder model look like a single-graph classifier
# ────────────────────────────────────────────────────────────────────────────

class SingleGraphWrapper(nn.Module):
    """
    Wraps a MatchOutcomeModel so PyG's GNNExplainer can explain it.

    Parameters
    ----------
    model : MatchOutcomeModel
        Trained dual-encoder model (currently GCN).
    other_graph : torch_geometric.data.Data
        The team graph that is NOT being explained.  Its pooled embedding is
        computed once at construction and held fixed.
    side : {"home", "away"}
        Which side of the concatenation the explained graph occupies.
    """

    def __init__(self, model: MatchOutcomeModel, other_graph: Data, side: str) -> None:
        super().__init__()
        if side not in ("home", "away"):
            raise ValueError("side must be 'home' or 'away'")
        self.model = model
        self.side = side

        # Pre-compute the fixed other team's pooled embedding ONCE.
        # No message passing on the other graph runs during forward(), so
        # GNNExplainer's edge-mask hook never sees a wrong-sized mask.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            ob = Batch.from_data_list([other_graph])
            h_nodes = model.encoder(ob.x, ob.edge_index, ob.edge_attr)
            h_other = model._pool_fn(h_nodes, ob.batch)  # [1, hidden]
        model.train(was_training)

        # Buffer follows .to(device) and is excluded from parameters
        self.register_buffer("h_other", h_other.detach())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_this_nodes = self.model.encoder(x, edge_index, edge_attr)
        h_this = self.model._pool_fn(h_this_nodes, batch)  # [1, hidden]

        if self.side == "home":
            z = torch.cat([h_this, self.h_other], dim=1)
        else:
            z = torch.cat([self.h_other, h_this], dim=1)

        return self.model.classifier(z)


# ────────────────────────────────────────────────────────────────────────────
# Explanation extraction
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamExplanation:
    """Result of running GNNExplainer on one team graph."""
    edge_index: np.ndarray   # [2, E] — copy of the input edge_index
    edge_mask:  np.ndarray   # [E]    — importance scores in [0, 1]
    node_mask:  np.ndarray   # [N]    — importance scores in [0, 1]
    target:     int          # predicted class explained
    side:       str          # "home" or "away"


def explain_team_graph(
    model: MatchOutcomeModel,
    home_graph: Data,
    away_graph: Data,
    side: str,
    epochs: int = 200,
    lr: float = 0.01,
) -> TeamExplanation:
    """
    Run GNNExplainer on one side of a match.  The other side is held fixed.
    The explanation target is the model's predicted class for the full match.
    """
    model.eval()

    # First get the model's prediction on the FULL match (both graphs)
    with torch.no_grad():
        h_b = Batch.from_data_list([home_graph])
        a_b = Batch.from_data_list([away_graph])
        logits = model(h_b, a_b)
        pred = int(logits.argmax(dim=1).item())

    # Build the wrapper around the side we want to explain
    other_graph = away_graph if side == "home" else home_graph
    explained_graph = home_graph if side == "home" else away_graph
    wrapper = SingleGraphWrapper(model, other_graph, side=side)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    g_b = Batch.from_data_list([explained_graph])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = explainer(
            x=g_b.x,
            edge_index=g_b.edge_index,
            edge_attr=g_b.edge_attr,
            batch=g_b.batch,
            target=torch.tensor([pred]),
        )

    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    node_mask = explanation.node_mask.detach().cpu().numpy().squeeze(-1)
    edge_index = g_b.edge_index.detach().cpu().numpy()

    return TeamExplanation(
        edge_index=edge_index,
        edge_mask=edge_mask,
        node_mask=node_mask,
        target=pred,
        side=side,
    )


# ────────────────────────────────────────────────────────────────────────────
# Formation-style visualisation (mirrors attention.plot_attention_formation)
# ────────────────────────────────────────────────────────────────────────────

def plot_explainer_formation(
    explanation: TeamExplanation,
    match_record: dict,
    ax: plt.Axes,
    title: str = "",
    top_k_edges: int = 20,
) -> None:
    """
    Draw a formation diagram with GNNExplainer importance overlaid:
      * node SIZE encodes node_mask importance
      * edge THICKNESS and COLOUR encode edge_mask importance
    Only starting-XI nodes are shown; only the top-k highest-importance
    edges between starters are drawn.
    """
    events  = match_record["events"]
    lineups = match_record["lineups"]
    side    = explanation.side
    team_name = match_record["home_team"] if side == "home" else match_record["away_team"]

    # Rebuild the same player-to-index mapping the dataset builder used,
    # so explanation.edge_index aligns with player identities.
    lineup_df   = lineups[team_name]
    lineup_ids  = set(lineup_df["player_id"].astype(int).tolist())
    team_passes = events[(events["team"] == team_name) & (events["type"] == "Pass")]
    event_ids   = set(team_passes["player_id"].dropna().astype(int).tolist())
    all_pids    = sorted(lineup_ids | event_ids)
    p2i         = {pid: i for i, pid in enumerate(all_pids)}

    starting_xi      = _get_starting_xi(lineup_df)
    id_to_name       = dict(zip(lineup_df["player_id"].astype(int), lineup_df["player_name"]))
    starter_node_ids = {p2i[pid] for pid in starting_xi if pid in p2i}
    xy, names, groups = _node_positions_and_labels(p2i, starting_xi, id_to_name)

    edge_idx  = explanation.edge_index
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    # Restrict edges to starting XI both endpoints
    starter_mask = np.array([
        (edge_idx[0, i] in starter_node_ids and edge_idx[1, i] in starter_node_ids)
        for i in range(edge_idx.shape[1])
    ])
    edge_idx_s  = edge_idx[:, starter_mask]
    edge_mask_s = edge_mask[starter_mask]

    # Top-k edges by importance
    if len(edge_mask_s) > top_k_edges:
        top    = np.argsort(edge_mask_s)[-top_k_edges:]
        edge_idx_s  = edge_idx_s[:, top]
        edge_mask_s = edge_mask_s[top]

    # Rank-based normalisation. GNNExplainer's sparsity regulariser tends to
    # push edge masks toward a near-binary distribution (a small cluster of
    # ~saturated edges and a long tail at the noise floor), so straight
    # min-max normalisation collapses to two visual buckets. Using ranks
    # spreads the displayed top-k evenly across [0, 1] while preserving the
    # ordering, which is the part GNNExplainer actually gets right.
    if len(edge_mask_s) > 0:
        order  = np.argsort(edge_mask_s)
        ranks  = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(order))
        e_norm = ranks / max(len(order) - 1, 1)
    else:
        e_norm = edge_mask_s

    # Draw pitch (same style as attention diagrams)
    ax.set_facecolor("#2d7a2d")
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor="white", facecolor="none", zorder=1,
    ))
    ax.axhline(0.50, color="white", lw=1.0, alpha=0.5, zorder=1)
    ax.add_patch(plt.Circle((0.50, 0.50), 0.10, color="white",
                            fill=False, lw=0.8, alpha=0.4, zorder=1))
    for y0, h in [(0.02, 0.17), (0.81, 0.17)]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.20, y0), 0.60, h,
            boxstyle="square,pad=0", linewidth=0.8,
            edgecolor="white", facecolor="none", alpha=0.4, zorder=1,
        ))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    # Edges
    cmap = plt.cm.YlOrRd
    for i in range(edge_idx_s.shape[1]):
        src, dst = int(edge_idx_s[0, i]), int(edge_idx_s[1, i])
        if src not in xy or dst not in xy:
            continue
        xs, ys = xy[src]
        xd, yd = xy[dst]
        a_val  = 0.25 + 0.65 * float(e_norm[i])
        lw     = 0.5  + 3.5  * float(e_norm[i])
        colour = cmap(float(e_norm[i]))
        ax.annotate(
            "", xy=(xd, yd), xytext=(xs, ys),
            arrowprops=dict(
                arrowstyle="->", color=colour,
                lw=lw, alpha=a_val,
                connectionstyle="arc3,rad=0.05",
            ),
        )

    # Nodes — size encodes node importance.
    # Same problem as edges: many non-starter nodes pin at 0 and dominate
    # the min, while starters cluster tightly near the top of the range.
    # Normalise rank-wise across the displayed (starter) nodes only.
    starter_idx_list = [idx for idx in xy if idx < len(node_mask)]
    if len(starter_idx_list) > 0:
        starter_vals = np.array([node_mask[idx] for idx in starter_idx_list])
        order = np.argsort(starter_vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(order))
        ranks = ranks / max(len(order) - 1, 1)
        n_norm_map = {idx: float(r) for idx, r in zip(starter_idx_list, ranks)}
    else:
        n_norm_map = {}

    for idx, (x, y) in xy.items():
        grp    = groups.get(idx, "MID")
        colour = GROUP_COLOUR[grp]
        importance = n_norm_map.get(idx, 0.5)
        size = 120 + 280 * importance      # range ~120 to ~400
        ax.scatter(x, y, s=size, c=colour, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(x, y - 0.055, names.get(idx, ""), ha="center", va="top",
                fontsize=6.5, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    # Colorbar for edge importance
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02,
                        fraction=0.03, aspect=30)
    cbar.set_label("GNNExplainer edge importance (normalised)",
                   fontsize=7, color="white")
    cbar.ax.tick_params(colors="white", labelsize=6)
    cbar.outline.set_edgecolor("white")

    legend_handles = [
        mpatches.Patch(color=GROUP_COLOUR[g], label=g)
        for g in ["GK", "DEF", "MID", "ATT"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
              framealpha=0.4, facecolor="white")

    ax.set_title(title or team_name, fontsize=9, color="white", pad=4)


# ────────────────────────────────────────────────────────────────────────────
# Agreement analysis: GNNExplainer (GCN) vs GAT attention
# ────────────────────────────────────────────────────────────────────────────

def edge_set_jaccard(
    edge_index_a: np.ndarray, scores_a: np.ndarray,
    edge_index_b: np.ndarray, scores_b: np.ndarray,
    top_k: int,
) -> float:
    """
    Take the top-k edges from each ranking, treat them as sets of (src, dst)
    tuples, and return the Jaccard similarity.

    The two edge_index arrays may not contain the same edges (e.g. GAT may
    add self-loops internally) — that's fine, we compare on the (src, dst)
    keys.
    """
    if scores_a.size == 0 or scores_b.size == 0:
        return float("nan")
    k_a = min(top_k, scores_a.size)
    k_b = min(top_k, scores_b.size)
    top_a_idx = np.argsort(scores_a)[-k_a:]
    top_b_idx = np.argsort(scores_b)[-k_b:]
    set_a = {(int(edge_index_a[0, i]), int(edge_index_a[1, i])) for i in top_a_idx}
    set_b = {(int(edge_index_b[0, i]), int(edge_index_b[1, i])) for i in top_b_idx}
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else float("nan")
