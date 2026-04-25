"""
Custom collation for MatchPair batches.

PyG's Batch.from_data_list handles variable-size graphs and builds a
single large batch graph with a `batch` index tensor.  We apply this
separately to home and away graphs so each gets a properly disjoint batch.
"""

import torch
from torch_geometric.data import Batch
from src.data.dataset import MatchPair


def collate_match_pairs(pairs: list[MatchPair]):
    """
    Collate a list of MatchPair into (home_batch, away_batch, labels).

    Returns
    -------
    home_batch : torch_geometric.data.Batch
    away_batch : torch_geometric.data.Batch
    labels     : torch.LongTensor  [batch_size]
    """
    home_batch = Batch.from_data_list([p.home_graph for p in pairs])
    away_batch = Batch.from_data_list([p.away_graph for p in pairs])
    labels     = torch.tensor([p.label for p in pairs], dtype=torch.long)
    return home_batch, away_batch, labels
