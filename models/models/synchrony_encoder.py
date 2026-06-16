"""
SynchronyEncoder — Graph Neural Network for Inter-Electrode Synchrony.

Models MEA electrodes as nodes in a functional connectivity graph.
Edges are derived from pairwise cross-correlation of spike trains.
A 2-layer GraphSAGE-style aggregation produces a synchrony embedding.

Does NOT require torch_geometric — uses batched matrix operations only.

Input : spike_train [B, n_elec, T]
Output: synchrony_embedding [B, sync_dim]
        synchrony_index [B, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynchronyEncoder(nn.Module):
    """
    Graph-based inter-electrode synchrony encoder.

    Parameters
    ----------
    n_electrodes : int
    n_time_bins : int
    node_feat_dim : int
        Dimensionality of per-electrode feature embedding.
    sync_dim : int
        Output synchrony embedding dimension.
    n_graph_layers : int
        Number of message-passing layers.
    dropout : float
    """

    def __init__(
        self,
        n_electrodes: int = 64,
        n_time_bins: int = 1000,
        node_feat_dim: int = 32,
        sync_dim: int = 64,
        n_graph_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.n_electrodes = n_electrodes
        self.n_time_bins = n_time_bins

        # ── Node feature extractor (per-electrode 1D CNN) ─────────────────
        self.node_encoder = nn.Sequential(
            nn.Conv1d(1, node_feat_dim, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(node_feat_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ── Graph message-passing (manual aggregation) ─────────────────────
        self.msg_layers = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(n_graph_layers):
            self.msg_layers.append(nn.Sequential(
                nn.Linear(in_dim * 2, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # ── Graph-level pooling ────────────────────────────────────────────
        self.pool = nn.Sequential(
            nn.Linear(in_dim, sync_dim),
            nn.GELU(),
        )

        # ── Synchrony index regressor ──────────────────────────────────────
        self.sync_regressor = nn.Sequential(
            nn.Linear(sync_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def _build_adj_matrix(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Build adjacency matrix from pairwise cross-correlation.

        Parameters
        ----------
        spikes : torch.Tensor
            [B, n_elec, T] float32.

        Returns
        -------
        torch.Tensor
            Normalized adjacency [B, n_elec, n_elec], non-negative.
        """
        # Zero-mean per electrode
        s = spikes - spikes.mean(dim=-1, keepdim=True)
        # Norms
        norms = s.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        s_normed = s / norms                            # [B, n_elec, T]
        # Cross-correlation: [B, n_elec, n_elec]
        corr = torch.bmm(s_normed, s_normed.transpose(1, 2)) / self.n_time_bins
        # Keep only positive correlations and add self-loops
        adj = F.relu(corr)
        eye = torch.eye(self.n_electrodes, device=spikes.device).unsqueeze(0)
        adj = adj + eye
        # Row-normalize
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adj = adj / row_sum
        return adj

    def forward(self, spikes: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encode inter-electrode synchrony.

        Parameters
        ----------
        spikes : torch.Tensor
            [B, n_elec, T] float32.

        Returns
        -------
        dict with:
            'sync_embedding' : [B, sync_dim]
            'sync_index'     : [B, 1]  (∈ [0, 1])
        """
        B, n_elec, T = spikes.shape

        # Per-electrode node features via 1D CNN
        s_flat = spikes.reshape(B * n_elec, 1, T)                  # [B*n_elec, 1, T]
        h = self.node_encoder(s_flat).squeeze(-1)                   # [B*n_elec, node_feat_dim]
        h = h.view(B, n_elec, -1)                                   # [B, n_elec, D]

        # Build adjacency
        adj = self._build_adj_matrix(spikes)                        # [B, n_elec, n_elec]

        # Message passing
        for layer in self.msg_layers:
            # Aggregate: weighted neighbor sum
            agg = torch.bmm(adj, h)                                 # [B, n_elec, D]
            msg_in = torch.cat([h, agg], dim=-1)                    # [B, n_elec, 2D]
            msg_in_flat = msg_in.view(B * n_elec, -1)
            h_new = layer(msg_in_flat).view(B, n_elec, -1)
            h = h + h_new                                           # residual

        # Graph-level readout: mean pooling over nodes
        graph_feat = h.mean(dim=1)                                  # [B, D]
        sync_emb = self.pool(graph_feat)                            # [B, sync_dim]
        sync_idx = self.sync_regressor(sync_emb)                    # [B, 1]

        return {"sync_embedding": sync_emb, "sync_index": sync_idx}
