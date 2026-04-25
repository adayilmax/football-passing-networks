"""
Utilities for extracting and visualising GAT attention weights.

Two main outputs:
  1. Per-match formation diagram — nodes are players positioned on a
     stylised pitch by their role; edge thickness/colour = mean attention
     weight from the final GAT layer.

  2. Aggregate position-pair attention heatmap — averaged over many matches,
     showing which structural pass relationships the model attends to most.
"""

import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

from src.data.graph_builder import POSITIONS, build_team_graph
from src.models.gat import GATModel

# ── Position groups ──────────────────────────────────────────────────────────
GK   = {"Goalkeeper"}
DEF  = {"Right Back","Right Center Back","Center Back","Left Center Back","Left Back",
        "Right Wing Back","Left Wing Back"}
MID  = {"Right Defensive Midfield","Center Defensive Midfield","Left Defensive Midfield",
        "Right Center Midfield","Center Midfield","Left Center Midfield",
        "Right Midfield","Left Midfield",
        "Right Attacking Midfield","Center Attacking Midfield","Left Attacking Midfield"}
ATT  = {"Right Wing","Left Wing","Right Center Forward","Center Forward",
        "Left Center Forward","Secondary Striker"}

POS_GROUP = {}
for p in POSITIONS:
    if p in GK:   POS_GROUP[p] = "GK"
    elif p in DEF: POS_GROUP[p] = "DEF"
    elif p in MID: POS_GROUP[p] = "MID"
    else:          POS_GROUP[p] = "ATT"

GROUP_COLOUR = {"GK": "#e07b39", "DEF": "#4878cf", "MID": "#6acc65", "ATT": "#d43f3a"}

# Formation coordinates: (x, y) where
#   x = lateral position  — 0.0 = left touchline, 1.0 = right touchline
#   y = depth             — 0.0 = own goal line,  1.0 = opponent goal line
#
# This matches the standard tactical-board view: own half at the bottom,
# attacking half at the top, left flank on the left, right flank on the right.
# "Right Back" therefore appears on the RIGHT side of the image (high x).
FORMATION_XY: dict[str, tuple[float, float]] = {
    "Goalkeeper":               (0.50, 0.05),
    "Right Back":               (0.82, 0.20), "Left Back":                (0.18, 0.20),
    "Right Center Back":        (0.65, 0.17), "Center Back":              (0.50, 0.17),
    "Left Center Back":         (0.35, 0.17),
    "Right Wing Back":          (0.88, 0.30), "Left Wing Back":           (0.12, 0.30),
    "Right Defensive Midfield": (0.72, 0.38), "Center Defensive Midfield":(0.50, 0.38),
    "Left Defensive Midfield":  (0.28, 0.38),
    "Right Center Midfield":    (0.72, 0.53), "Center Midfield":          (0.50, 0.53),
    "Left Center Midfield":     (0.28, 0.53),
    "Right Midfield":           (0.88, 0.47), "Left Midfield":            (0.12, 0.47),
    "Right Attacking Midfield": (0.72, 0.66), "Center Attacking Midfield":(0.50, 0.66),
    "Left Attacking Midfield":  (0.28, 0.66),
    "Right Wing":               (0.92, 0.76), "Left Wing":                (0.08, 0.76),
    "Right Center Forward":     (0.65, 0.82), "Center Forward":           (0.50, 0.82),
    "Left Center Forward":      (0.35, 0.82),
    "Secondary Striker":        (0.50, 0.74),
}
DEFAULT_XY = (0.50, 0.50)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _most_common_pos(events, player_id: int) -> str:
    """Return most frequent position string for a player in this match."""
    rows = events[(events["player_id"] == player_id) & events["position"].notna()]
    if rows.empty:
        return "Center Midfield"
    return rows["position"].mode().iloc[0]


def _get_starting_xi(lineup_df) -> dict[int, str]:
    """
    Read the lineup DataFrame and return {player_id: position_name} for the
    11 players whose first position entry has start_reason == 'Starting XI'.

    The `positions` column in each row is a list of dicts of the form:
        {"position": {"id": ..., "name": "..."}, "start_reason": "Starting XI", ...}
    """
    starters: dict[int, str] = {}
    for _, row in lineup_df.iterrows():
        positions = row.get("positions", [])
        if not isinstance(positions, list):
            continue
        for entry in positions:
            if not isinstance(entry, dict):
                continue
            if entry.get("start_reason") == "Starting XI":
                pos_val  = entry.get("position", "Center Midfield")
                # statsbombpy stores position as a plain string, not a nested dict
                pos_name = pos_val if isinstance(pos_val, str) else pos_val.get("name", "Center Midfield")
                starters[int(row["player_id"])] = pos_name
                break   # one entry per player is enough
    return starters


def _node_positions_and_labels(
    player_to_idx: dict[int, int],
    starting_xi: dict[int, str],       # player_id -> starting position name
    id_to_name: dict[int, str],        # player_id -> full name
) -> tuple[dict[int, tuple[float, float]], dict[int, str], dict[int, str]]:
    """
    Returns
      xy        : node_idx -> (x, y) formation coordinate  (collision-resolved)
      names     : node_idx -> full player name
      groups    : node_idx -> position group string
    """
    from collections import defaultdict

    raw_xy: dict[int, tuple[float, float]] = {}
    names:  dict[int, str]                 = {}
    groups: dict[int, str]                 = {}

    for pid, pos in starting_xi.items():
        if pid not in player_to_idx:
            continue
        idx         = player_to_idx[pid]
        raw_xy[idx] = FORMATION_XY.get(pos, DEFAULT_XY)
        names[idx]  = id_to_name.get(pid, f"Player {pid}")   # full name
        groups[idx] = POS_GROUP.get(pos, "MID")

    # Resolve any residual coordinate collisions (e.g. unusual formations
    # where two players share the same position string).
    coord_to_nodes: dict[tuple, list[int]] = defaultdict(list)
    for idx, coord in raw_xy.items():
        coord_to_nodes[coord].append(idx)

    xy: dict[int, tuple[float, float]] = {}
    for coord, node_list in coord_to_nodes.items():
        n = len(node_list)
        if n == 1:
            xy[node_list[0]] = coord
        else:
            # Spread colliding nodes laterally (x-axis) so they don't overlap
            offsets = np.linspace(-0.05 * (n - 1) / 2,
                                   0.05 * (n - 1) / 2, n)
            for i, idx in enumerate(sorted(node_list)):
                xy[idx] = (coord[0] + offsets[i], coord[1])

    return xy, names, groups


# ── Extract attention for one team graph ────────────────────────────────────

def extract_attention(
    model: GATModel,
    graph,
    layer_idx: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one forward pass and return (edge_index, mean_attention).

    edge_index     : [2, E] numpy int array  (self-loops excluded)
    mean_attention : [E]    mean across heads for the chosen layer
    """
    from torch_geometric.data import Batch
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list([graph])
        model.encoder(batch.x, batch.edge_index, batch.edge_attr)

    weights = model.get_attention_weights()
    if weights is None or len(weights) == 0:
        raise RuntimeError("No attention weights found — run a forward pass first.")

    # weights[layer] is now (edge_index, alpha)
    ret_edge_idx, alpha = weights[layer_idx]
    edge_idx   = ret_edge_idx.cpu().numpy()   # [2, E_with_loops]
    mean_attn  = alpha.cpu().numpy().mean(axis=1)   # [E_with_loops]

    # Remove self-loops (src == dst) so we only visualise real pass edges
    not_self = edge_idx[0] != edge_idx[1]
    return edge_idx[:, not_self], mean_attn[not_self]


# ── Formation diagram ────────────────────────────────────────────────────────

def plot_attention_formation(
    model: GATModel,
    match_record: dict,
    team: str,                  # "home" or "away"
    ax: plt.Axes,
    title: str = "",
    top_k_edges: int = 20,
) -> None:
    """
    Draw a formation-based passing attention diagram on ax.

    Only the 11 starting players are shown.  Only the top_k_edges
    highest-attention directed edges between starters are drawn.
    """
    events  = match_record["events"]
    lineups = match_record["lineups"]

    team_name = match_record["home_team"] if team == "home" else match_record["away_team"]

    # Build the full graph exactly as the dataset builder does (all players),
    # so node indices in the graph match the attention edge_index.
    lineup_df   = lineups[team_name]
    lineup_ids  = set(lineup_df["player_id"].astype(int).tolist())
    team_passes = events[(events["team"] == team_name) & (events["type"] == "Pass")]
    event_ids   = set(team_passes["player_id"].dropna().astype(int).tolist())
    all_pids    = sorted(lineup_ids | event_ids)
    p2i         = {pid: i for i, pid in enumerate(all_pids)}   # full roster index

    graph = build_team_graph(events, team_name, lineups)
    edge_idx, mean_attn = extract_attention(model, graph)

    # Restrict visualisation to starting XI only
    starting_xi = _get_starting_xi(lineup_df)          # {player_id: position_name}
    id_to_name  = dict(zip(lineup_df["player_id"].astype(int), lineup_df["player_name"]))
    starter_node_ids = {p2i[pid] for pid in starting_xi if pid in p2i}

    xy, names, groups = _node_positions_and_labels(p2i, starting_xi, id_to_name)

    # Keep only edges where BOTH endpoints are starting XI nodes
    starter_mask = np.array([
        (edge_idx[0, i] in starter_node_ids and edge_idx[1, i] in starter_node_ids)
        for i in range(edge_idx.shape[1])
    ])
    edge_idx  = edge_idx[:, starter_mask]
    mean_attn = mean_attn[starter_mask]

    # ── Relative attention: α_ij × in-degree(j) ────────────────────────────
    # Raw GAT attention is softmax-normalised per destination node, so nodes
    # with few incoming edges (e.g. the GK with ~4 back-passes) get high
    # per-edge values purely from the small denominator.  Multiplying by the
    # in-degree corrects for this: values > 1 mean the model pays MORE than
    # uniform attention to this edge; values < 1 mean less.
    from collections import Counter
    in_deg = Counter(edge_idx[1].tolist())
    rel_attn = mean_attn * np.array([in_deg[edge_idx[1, i]] for i in range(len(mean_attn))],
                                     dtype=np.float32)

    # Top-k edges by relative attention
    if len(rel_attn) > top_k_edges:
        top_mask  = np.argsort(rel_attn)[-top_k_edges:]
        edge_idx  = edge_idx[:, top_mask]
        rel_attn  = rel_attn[top_mask]

    # Normalise to [0, 1] for visual encoding
    attn_norm = (rel_attn - rel_attn.min()) / (rel_attn.max() - rel_attn.min() + 1e-9)

    # Draw pitch background (standard tactical-board orientation)
    # x = lateral (left→right), y = depth (own goal bottom → opponent goal top)
    ax.set_facecolor("#2d7a2d")
    # Pitch border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor="white", facecolor="none", zorder=1,
    ))
    # Halfway line
    ax.axhline(0.50, color="white", lw=1.0, alpha=0.5, zorder=1)
    # Centre circle (approximate)
    circle = plt.Circle((0.50, 0.50), 0.10, color="white", fill=False,
                        lw=0.8, alpha=0.4, zorder=1)
    ax.add_patch(circle)
    # Penalty areas (rough proportions)
    for y0, h in [(0.02, 0.17), (0.81, 0.17)]:   # own and opponent boxes
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.20, y0), 0.60, h,
            boxstyle="square,pad=0", linewidth=0.8,
            edgecolor="white", facecolor="none", alpha=0.4, zorder=1,
        ))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    # Draw edges
    cmap = plt.cm.YlOrRd
    for i in range(edge_idx.shape[1]):
        src, dst = edge_idx[0, i], edge_idx[1, i]
        if src not in xy or dst not in xy:
            continue
        xs, ys = xy[src]
        xd, yd = xy[dst]
        alpha_val = 0.25 + 0.65 * attn_norm[i]
        lw        = 0.5  + 3.5  * attn_norm[i]
        colour    = cmap(attn_norm[i])
        ax.annotate(
            "", xy=(xd, yd), xytext=(xs, ys),
            arrowprops=dict(
                arrowstyle="->", color=colour,
                lw=lw, alpha=alpha_val,
                connectionstyle="arc3,rad=0.05",
            ),
        )

    # Draw nodes
    for idx, (x, y) in xy.items():
        grp    = groups.get(idx, "MID")
        colour = GROUP_COLOUR[grp]
        ax.scatter(x, y, s=220, c=colour, zorder=5, edgecolors="white", linewidths=1.2)
        # Label sits 0.055 below the node centre in data coords — tight enough
        # to be clearly associated but not overlapping the node circle.
        ax.text(x, y - 0.055, names.get(idx, ""), ha="center", va="top",
                fontsize=6.5, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    # Colorbar for attention intensity
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02,
                        fraction=0.03, aspect=30)
    cbar.set_label("Relative attention  (α × in-degree, normalised)", fontsize=7, color="white")
    cbar.ax.tick_params(colors="white", labelsize=6)
    cbar.outline.set_edgecolor("white")

    # Position group legend
    legend_handles = [
        mpatches.Patch(color=GROUP_COLOUR[g], label=g)
        for g in ["GK", "DEF", "MID", "ATT"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
              framealpha=0.4, facecolor="white")

    label = match_record["home_team"] if team == "home" else match_record["away_team"]
    ax.set_title(title or label, fontsize=9, color="white", pad=4)


# ── Aggregate position-pair heatmap ─────────────────────────────────────────

def compute_aggregate_attention(
    model: GATModel,
    pairs,                          # list[MatchPair]
    match_records: dict,            # match_id -> match_record dict
    events_map: dict,
    lineups_map: dict,
) -> np.ndarray:
    """
    Average attention weight over all matches for each (src_group, dst_group) pair.
    Returns a 4x4 numpy array indexed by [GK, DEF, MID, ATT].
    """
    GROUP_NAMES = ["GK", "DEF", "MID", "ATT"]
    n = len(GROUP_NAMES)
    accum  = np.zeros((n, n))
    counts = np.zeros((n, n))
    g_to_i = {g: i for i, g in enumerate(GROUP_NAMES)}

    import pandas as pd
    for pair in pairs:
        for team_key, graph in [("home", pair.home_graph), ("away", pair.away_graph)]:
            mid = pair.match_id
            if mid not in match_records:
                continue
            rec      = match_records[mid]
            events   = rec["events"]
            lineups  = rec["lineups"]
            team_name = rec["home_team"] if team_key == "home" else rec["away_team"]
            lineup_df = lineups[team_name]

            lineup_ids = set(lineup_df["player_id"].astype(int).tolist())
            team_passes = events[(events["team"] == team_name) & (events["type"] == "Pass")]
            event_ids   = set(team_passes["player_id"].dropna().astype(int).tolist())
            all_pids    = sorted(lineup_ids | event_ids)
            p2i         = {pid: i for i, pid in enumerate(all_pids)}

            try:
                edge_idx, mean_attn = extract_attention(model, graph)
            except Exception:
                continue

            # Map node idx -> position group
            idx_to_group = {}
            for pid in all_pids:
                pos = _most_common_pos(events, pid)
                idx_to_group[p2i[pid]] = POS_GROUP.get(pos, "MID")

            for k in range(edge_idx.shape[1]):
                src, dst = edge_idx[0, k], edge_idx[1, k]
                sg = idx_to_group.get(src, "MID")
                dg = idx_to_group.get(dst, "MID")
                si, di = g_to_i[sg], g_to_i[dg]
                accum[si, di]  += mean_attn[k]
                counts[si, di] += 1

    with np.errstate(invalid="ignore"):
        result = np.where(counts > 0, accum / counts, 0.0)
    return result
