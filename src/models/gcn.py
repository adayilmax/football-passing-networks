"""
GCN-based match outcome model.

Uses PyG GCNConv layers.  Edge attributes are projected to a scalar weight
that gates each message, giving the convolution access to pass frequency
and success rate without changing the standard GCN formulation.

Node feature input: 28-dim  (position one-hot + pass stats)
Edge feature input: 3-dim   (count_norm, success_rate, length_norm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm

from src.models.base import MatchOutcomeModel
from src.data.graph_builder import NUM_NODE_FEATURES, NUM_EDGE_FEATURES


class GCNEncoder(nn.Module):
    """
    Stack of GCNConv layers with BatchNorm and ReLU.

    Edge weights are derived from edge_attr via a small linear projection
    (3-dim → 1-dim scalar), then passed as the `edge_weight` argument to
    GCNConv so the adjacency is soft-weighted by pass quality.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()

        # Project edge features to a single scalar weight per edge
        self.edge_weight_proj = nn.Linear(NUM_EDGE_FEATURES, 1, bias=False)

        # Input projection: bring raw node features up to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True))
            self.norms.append(BatchNorm(hidden_dim))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Scalar edge weight in [0, 1] via sigmoid
        if edge_attr.shape[0] > 0:
            edge_weight = torch.sigmoid(self.edge_weight_proj(edge_attr)).squeeze(-1)
        else:
            edge_weight = None

        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # [num_nodes, hidden_dim]


class GCNModel(MatchOutcomeModel):
    """GCN-based match outcome predictor."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float  = 0.3,
        pooling: str    = "mean",
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers,
                         dropout=dropout, pooling=pooling)
        self.encoder = GCNEncoder(
            in_dim     = NUM_NODE_FEATURES,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            dropout    = dropout,
        )

    def encode_graph(self, data: Data) -> torch.Tensor:
        node_emb = self.encoder(data.x, data.edge_index, data.edge_attr)
        return self._pool_fn(node_emb, data.batch)  # [B, hidden_dim]
