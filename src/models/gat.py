"""
GAT-based match outcome model.

Uses PyG GATv2Conv layers, which fix the static-attention limitation of the
original GAT by making attention genuinely dynamic (attending over both
source and target node representations).

Edge features (pass count, success rate, length) are concatenated to the
source node embedding before attention is computed, so the model can attend
based on *who passes to whom and how well* — not just node identity.

Attention weights are stored in self.last_attention_weights after each
forward pass and can be retrieved for interpretability analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, BatchNorm

from src.models.base import MatchOutcomeModel
from src.data.graph_builder import NUM_NODE_FEATURES, NUM_EDGE_FEATURES


class GATEncoder(nn.Module):
    """
    Stack of GATv2Conv layers.

    Edge features are incorporated as edge attributes directly inside
    GATv2Conv (edge_dim parameter), so each attention coefficient α_{ij}
    depends on node_i, node_j, AND the edge feature between them.
    This is key for interpretability: the attention weight on edge (i→j)
    reflects how important the *passing relationship* between players i
    and j is for predicting the outcome.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"

        self.head_dim  = hidden_dim // heads
        self.heads     = heads
        self.dropout   = dropout

        # Input projection to hidden_dim before first GATv2 layer
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels  = hidden_dim,
                out_channels = self.head_dim,
                heads        = heads,
                edge_dim     = NUM_EDGE_FEATURES,
                dropout      = dropout,
                concat       = True,   # output: [nodes, heads * head_dim] = [nodes, hidden_dim]
            ))
            self.norms.append(BatchNorm(hidden_dim))

        # Stores the last set of attention coefficients for inspection
        self.last_attention_weights: list[torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        attention_weights_per_layer = []

        for conv, norm in zip(self.convs, self.norms):
            if edge_attr.shape[0] > 0:
                x, (ret_edge_idx, alpha) = conv(
                    x, edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
            else:
                x, (ret_edge_idx, alpha) = conv(
                    x, edge_index,
                    return_attention_weights=True,
                )
            # Store (edge_index, alpha) together so callers can match them correctly.
            # GATv2Conv may add self-loops, so ret_edge_idx != the input edge_index.
            attention_weights_per_layer.append((ret_edge_idx.detach(), alpha.detach()))
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Keep last layer's attention for interpretability
        self.last_attention_weights = attention_weights_per_layer

        return x  # [num_nodes, hidden_dim]


class GATModel(MatchOutcomeModel):
    """
    GAT-based match outcome predictor.

    Parameters
    ----------
    heads : number of attention heads (hidden_dim must be divisible by heads)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int      = 4,
        dropout: float  = 0.3,
        pooling: str    = "mean",
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers,
                         dropout=dropout, pooling=pooling)
        self.encoder = GATEncoder(
            in_dim     = NUM_NODE_FEATURES,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            heads      = heads,
            dropout    = dropout,
        )

    def encode_graph(self, data: Data) -> torch.Tensor:
        node_emb = self.encoder(data.x, data.edge_index, data.edge_attr)
        return self._pool_fn(node_emb, data.batch)  # [B, hidden_dim]

    def get_attention_weights(self) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """
        Return per-layer (edge_index, alpha) tuples from the most recent forward pass.
        edge_index : [2, E]  (may include self-loops added by GATv2Conv)
        alpha      : [E, num_heads]
        Averaged across heads gives per-edge importance scores.
        """
        return self.encoder.last_attention_weights
