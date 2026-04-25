"""
Graph construction for football passing networks.

Each match produces two PyG Data objects — one per team.
Nodes  = players who appeared in the match.
Edges  = directed pass interactions (passer → recipient).

Node features (NUM_NODE_FEATURES = 28):
  [0:25]  position one-hot (25 StatsBomb positions)
  [25]    pass completion rate
  [26]    pressure rate  (passes made under pressure / total passes)
  [27]    progressive pass rate

Edge features (NUM_EDGE_FEATURES = 3):
  [0]  normalised pass count   (count / max_count_in_graph)
  [1]  pass success rate       (completed / total between pair)
  [2]  mean pass length        (metres / 100)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ── Position vocabulary ────────────────────────────────────────────────────
POSITIONS = [
    "Goalkeeper",
    "Right Back", "Right Center Back", "Center Back",
    "Left Center Back", "Left Back",
    "Right Wing Back", "Left Wing Back",
    "Right Defensive Midfield", "Center Defensive Midfield", "Left Defensive Midfield",
    "Right Center Midfield", "Center Midfield", "Left Center Midfield",
    "Right Midfield", "Left Midfield",
    "Right Attacking Midfield", "Center Attacking Midfield", "Left Attacking Midfield",
    "Right Wing", "Left Wing",
    "Right Center Forward", "Center Forward", "Left Center Forward",
    "Secondary Striker",
]
POSITION_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(POSITIONS)}
NUM_POSITIONS    = len(POSITIONS)   # 25
NUM_NODE_FEATURES = NUM_POSITIONS + 3  # 28
NUM_EDGE_FEATURES = 3

# A pass is "progressive" if it advances the ball ≥10 m along the x-axis
# (StatsBomb pitch: x = 0..120, attacking direction = increasing x)
PROGRESSIVE_X_THRESHOLD = 10.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_progressive(row: pd.Series) -> bool:
    """Return True if this pass event moves the ball ≥10 m forward."""
    loc = row.get("location")
    end = row.get("pass_end_location")
    if not isinstance(loc, list) or not isinstance(end, list):
        return False
    return (end[0] - loc[0]) >= PROGRESSIVE_X_THRESHOLD


def _most_common_position(series: pd.Series) -> str | None:
    """Return the most frequent non-null value, or None."""
    s = series.dropna()
    if s.empty:
        return None
    return s.mode().iloc[0]


# ── Core builder ───────────────────────────────────────────────────────────

def build_team_graph(
    events: pd.DataFrame,
    team_name: str,
    lineups: dict,
    max_period: int | None = None,
) -> Data:
    """
    Build a PyG Data object for one team in one match.

    Parameters
    ----------
    events     : full match events DataFrame
    team_name  : name string matching events['team'] values
    lineups    : dict returned by loader.get_lineups(match_id)
    max_period : if set, only include events from periods <= max_period.
                 Use max_period=1 for first-half-only graphs.

    Returns
    -------
    torch_geometric.data.Data with x, edge_index, edge_attr, num_nodes
    """
    # ── Filter events for this team (and optionally by period) ───────────
    team_events = events[events["team"] == team_name]
    if max_period is not None:
        team_events = team_events[team_events["period"] <= max_period]
    team_passes = team_events[team_events["type"] == "Pass"].copy()

    # ── Player roster ────────────────────────────────────────────────────
    lineup_df = lineups[team_name]
    lineup_ids = set(lineup_df["player_id"].astype(int).tolist())

    # Include any extra players who appear in events but not the lineup sheet
    # (very rare edge case — tactical shifts, red cards + subs)
    event_ids = set(team_passes["player_id"].dropna().astype(int).tolist())
    all_player_ids = sorted(lineup_ids | event_ids)

    # Stable node index mapping: player_id → row index in x
    player_to_idx: dict[int, int] = {pid: i for i, pid in enumerate(all_player_ids)}
    num_nodes = len(player_to_idx)

    # ── Node features ────────────────────────────────────────────────────
    x = np.zeros((num_nodes, NUM_NODE_FEATURES), dtype=np.float32)

    for pid in all_player_ids:
        idx = player_to_idx[pid]

        # All events for this player (for position detection)
        player_events = team_events[team_events["player_id"] == pid]
        player_passes = team_passes[team_passes["player_id"] == pid]

        # 1. Position one-hot
        pos = _most_common_position(player_events["position"])
        if pos is not None:
            pos_idx = POSITION_TO_IDX.get(pos)
            if pos_idx is not None:
                x[idx, pos_idx] = 1.0
            # else: unknown position → all zeros (treated as Unknown)

        total = len(player_passes)
        if total == 0:
            continue  # no passes → scalar features stay 0

        # 2. Pass completion rate  (NaN outcome == complete in StatsBomb)
        completed = int(player_passes["pass_outcome"].isna().sum())
        x[idx, NUM_POSITIONS]     = completed / total

        # 3. Pressure rate
        under_pressure = int(player_passes["under_pressure"].fillna(False).sum())
        x[idx, NUM_POSITIONS + 1] = under_pressure / total

        # 4. Progressive pass rate
        prog = int(player_passes.apply(_is_progressive, axis=1).sum())
        x[idx, NUM_POSITIONS + 2] = prog / total

    # ── Edge construction ─────────────────────────────────────────────────
    # Only passes where recipient is known and in our player set
    pass_with_recip = team_passes.dropna(subset=["pass_recipient_id"]).copy()
    pass_with_recip["player_id"]         = pass_with_recip["player_id"].astype(int)
    pass_with_recip["pass_recipient_id"] = pass_with_recip["pass_recipient_id"].astype(int)

    # Accumulate per (src, dst) pair
    edge_stats: dict[tuple[int, int], dict] = {}

    for _, row in pass_with_recip.iterrows():
        src_pid = int(row["player_id"])
        dst_pid = int(row["pass_recipient_id"])

        if src_pid not in player_to_idx or dst_pid not in player_to_idx:
            continue
        if src_pid == dst_pid:
            continue  # skip self-passes (data artefact)

        key = (player_to_idx[src_pid], player_to_idx[dst_pid])
        if key not in edge_stats:
            edge_stats[key] = {"count": 0, "completed": 0, "lengths": []}

        edge_stats[key]["count"] += 1
        if pd.isna(row.get("pass_outcome")):
            edge_stats[key]["completed"] += 1
        length = row.get("pass_length")
        if pd.notna(length):
            edge_stats[key]["lengths"].append(float(length))

    if not edge_stats:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, NUM_EDGE_FEATURES), dtype=torch.float)
    else:
        max_count = max(v["count"] for v in edge_stats.values())
        edges = sorted(edge_stats.keys())

        src_list, dst_list, feat_list = [], [], []
        for src, dst in edges:
            v = edge_stats[(src, dst)]
            src_list.append(src)
            dst_list.append(dst)
            feat_list.append([
                v["count"] / max_count,
                v["completed"] / v["count"],
                float(np.mean(v["lengths"])) / 100.0 if v["lengths"] else 0.0,
            ])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(feat_list, dtype=torch.float)

    return Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        num_nodes  = num_nodes,
    )


def build_match_graphs(
    match_record: dict,
    max_period: int | None = None,
) -> tuple[Data, Data, int]:
    """
    Build home and away graphs for a single match record (as returned by
    loader.load_competition_matches).

    Parameters
    ----------
    max_period : forwarded to build_team_graph. Use 1 for first-half graphs.

    Returns (home_graph, away_graph, label)
    where label: 0=home win, 1=draw, 2=away win
    """
    from src.data.loader import match_outcome_label

    events  = match_record["events"]
    lineups = match_record["lineups"]
    home    = match_record["home_team"]
    away    = match_record["away_team"]
    label   = match_outcome_label(match_record["home_score"], match_record["away_score"])

    home_graph = build_team_graph(events, home, lineups, max_period=max_period)
    away_graph = build_team_graph(events, away, lineups, max_period=max_period)

    return home_graph, away_graph, label
