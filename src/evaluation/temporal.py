"""
Utilities for the static (full-match) vs dynamic (first-half) comparison.
"""

import numpy as np
import torch


def graph_structure_stats(dataset) -> dict:
    """
    Return mean/std of nodes and edges across all graphs in a dataset.
    Covers both home and away graphs.
    """
    nodes, edges = [], []
    for pair in dataset:
        for g in (pair.home_graph, pair.away_graph):
            nodes.append(g.num_nodes)
            edges.append(g.edge_index.shape[1])
    nodes = np.array(nodes, dtype=float)
    edges = np.array(edges, dtype=float)
    return {
        "mean_nodes": nodes.mean(), "std_nodes": nodes.std(),
        "mean_edges": edges.mean(), "std_edges": edges.std(),
        "min_edges":  edges.min(),  "max_edges":  edges.max(),
    }


def node_feature_stats(dataset) -> dict:
    """
    Return mean of each scalar node feature across all players in the dataset.
    Features: [25] completion_rate, [26] pressure_rate, [27] progressive_rate
    """
    all_x = torch.cat(
        [p.home_graph.x for p in dataset] + [p.away_graph.x for p in dataset], dim=0
    )
    return {
        "completion_rate":  float(all_x[:, 25].mean()),
        "pressure_rate":    float(all_x[:, 26].mean()),
        "progressive_rate": float(all_x[:, 27].mean()),
    }
