"""
Shared base for GNN match-outcome models.

Architecture
------------
For each match, two team graphs (home, away) pass through the SAME
GNN encoder.  The resulting graph-level embeddings are concatenated and
fed into a small MLP that outputs logits over three classes:
    0 = home win  |  1 = draw  |  2 = away win

Subclasses only need to implement `encode_graph(data) -> Tensor[B, hidden_dim]`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_add_pool

from src.data.graph_builder import NUM_NODE_FEATURES, NUM_EDGE_FEATURES

NUM_CLASSES = 3


class MatchOutcomeModel(nn.Module):
    """
    Abstract base.  Subclasses implement `encode_graph`.

    Parameters
    ----------
    hidden_dim  : width of GNN layers and MLP hidden layer
    num_layers  : number of GNN message-passing layers
    dropout     : dropout probability in the MLP
    pooling     : 'mean' or 'add' graph-level pooling
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float  = 0.3,
        pooling: str    = "mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout

        self._pool_fn = global_mean_pool if pooling == "mean" else global_add_pool

        # MLP: [home_emb | away_emb] → 3 classes
        # Input is 2 * hidden_dim because we concatenate both team embeddings
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    # ── Subclass contract ────────────────────────────────────────────────────
    def encode_graph(self, data: Data) -> torch.Tensor:
        """
        Encode a single team graph into a graph-level embedding.
        Input:  PyG Data with x, edge_index, edge_attr, batch
        Output: Tensor of shape [num_graphs_in_batch, hidden_dim]
        """
        raise NotImplementedError

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, home: Data, away: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        home, away : batched PyG Data objects (built by the DataLoader)

        Returns
        -------
        logits : Tensor[batch_size, 3]
        """
        h_home = self.encode_graph(home)   # [B, hidden_dim]
        h_away = self.encode_graph(away)   # [B, hidden_dim]
        z = torch.cat([h_home, h_away], dim=1)   # [B, 2*hidden_dim]
        return self.classifier(z)          # [B, 3]

    # ── Convenience ──────────────────────────────────────────────────────────
    def predict(self, home: Data, away: Data) -> torch.Tensor:
        """Return class predictions (argmax)."""
        with torch.no_grad():
            logits = self(home, away)
        return logits.argmax(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
