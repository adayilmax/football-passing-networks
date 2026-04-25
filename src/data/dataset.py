"""
MatchGraphDataset — builds, caches, and loads PyG graph pairs for all matches.

Each item in the dataset is a MatchPair:
    home_graph : torch_geometric.data.Data
    away_graph : torch_geometric.data.Data
    label      : int  (0=home win, 1=draw, 2=away win)
    match_id   : int
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from src.data.loader import get_matches, load_competition_matches
from src.data.graph_builder import build_match_graphs

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MatchPair:
    home_graph : Data
    away_graph : Data
    label      : int
    match_id   : int


class MatchGraphDataset(Dataset):
    """
    PyTorch Dataset of (home_graph, away_graph, label) triples.

    On first call for a given competition/season the graphs are built from
    raw StatsBomb events and saved to data/processed/.  Subsequent calls
    load the cached file instantly.

    Parameters
    ----------
    competition_id : int
    season_id      : int
    max_matches    : optional int — cap for quick testing
    force_rebuild  : bool — ignore cache and rebuild
    half_only      : bool — if True, build graphs from first-half events only
    """

    def __init__(
        self,
        competition_id: int = 11,
        season_id: int = 27,
        max_matches: int | None = None,
        force_rebuild: bool = False,
        half_only: bool = False,
    ) -> None:
        tag  = f"comp{competition_id}_season{season_id}"
        tag += f"_n{max_matches}" if max_matches else "_all"
        tag += "_half" if half_only else "_full"
        self.cache_path = PROCESSED_DIR / f"match_graphs_{tag}.pkl"
        self.half_only  = half_only

        # Backwards-compatibility: full-match cache was previously saved without
        # the "_full" suffix.  Use the old file if the new path doesn't exist.
        if not half_only and not self.cache_path.exists() and not force_rebuild:
            legacy = PROCESSED_DIR / f"match_graphs_{tag.replace('_full','')}.pkl"
            if legacy.exists():
                self.cache_path = legacy

        if self.cache_path.exists() and not force_rebuild:
            print(f"Loading cached dataset from {self.cache_path.name} ...")
            with open(self.cache_path, "rb") as f:
                self.pairs: list[MatchPair] = pickle.load(f)
        else:
            label = "first-half" if half_only else "full-match"
            print(f"Building {label} graph dataset from raw events ...")
            self.pairs = self._build(competition_id, season_id, max_matches, half_only)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.pairs, f)
            print(f"Saved to {self.cache_path}")

    # ── Dataset protocol ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> MatchPair:
        return self.pairs[idx]

    # ── Builder ─────────────────────────────────────────────────────────────
    def _build(
        self,
        competition_id: int,
        season_id: int,
        max_matches: int | None,
        half_only: bool = False,
    ) -> list[MatchPair]:
        records    = load_competition_matches(competition_id, season_id, max_matches)
        max_period = 1 if half_only else None
        pairs: list[MatchPair] = []
        skipped = 0

        for rec in tqdm(records, desc="Building graphs"):
            try:
                home_g, away_g, label = build_match_graphs(rec, max_period=max_period)
                # Sanity check: both graphs must have at least 2 nodes and 1 edge
                if home_g.num_nodes < 2 or away_g.num_nodes < 2:
                    skipped += 1
                    continue
                if home_g.edge_index.shape[1] == 0 or away_g.edge_index.shape[1] == 0:
                    skipped += 1
                    continue
                pairs.append(MatchPair(
                    home_graph=home_g,
                    away_graph=away_g,
                    label=label,
                    match_id=rec["match_id"],
                ))
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] match {rec['match_id']} failed: {exc}")
                skipped += 1

        print(f"Built {len(pairs)} match pairs  ({skipped} skipped)")
        return pairs

    # ── Split helpers ───────────────────────────────────────────────────────
    def split(
        self,
        train_frac: float = 0.70,
        val_frac: float   = 0.15,
        seed: int         = 42,
    ) -> tuple["MatchGraphDataset", "MatchGraphDataset", "MatchGraphDataset"]:
        """
        Return (train, val, test) sub-datasets via random split.
        test_frac = 1 - train_frac - val_frac.
        """
        import random
        rng = random.Random(seed)
        indices = list(range(len(self.pairs)))
        rng.shuffle(indices)

        n      = len(indices)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_ds = _SubDataset(self, indices[:n_train])
        val_ds   = _SubDataset(self, indices[n_train : n_train + n_val])
        test_ds  = _SubDataset(self, indices[n_train + n_val :])
        return train_ds, val_ds, test_ds

    # ── Stats ────────────────────────────────────────────────────────────────
    def label_distribution(self) -> dict[str, int]:
        labels = [p.label for p in self.pairs]
        return {
            "home_win": labels.count(0),
            "draw":     labels.count(1),
            "away_win": labels.count(2),
        }

    def graph_stats(self) -> dict[str, float]:
        """Mean nodes and edges across all home + away graphs."""
        nodes, edges = [], []
        for p in self.pairs:
            for g in (p.home_graph, p.away_graph):
                nodes.append(g.num_nodes)
                edges.append(g.edge_index.shape[1])
        return {
            "mean_nodes": float(torch.tensor(nodes, dtype=torch.float).mean()),
            "mean_edges": float(torch.tensor(edges, dtype=torch.float).mean()),
            "min_nodes":  min(nodes),
            "max_nodes":  max(nodes),
            "min_edges":  min(edges),
            "max_edges":  max(edges),
        }


class _SubDataset(Dataset):
    """Lightweight index-based view into a MatchGraphDataset."""

    def __init__(self, parent: MatchGraphDataset, indices: list[int]) -> None:
        self._parent  = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> MatchPair:
        return self._parent.pairs[self._indices[idx]]

    def label_distribution(self) -> dict[str, int]:
        labels = [self._parent.pairs[i].label for i in self._indices]
        return {
            "home_win": labels.count(0),
            "draw":     labels.count(1),
            "away_win": labels.count(2),
        }
