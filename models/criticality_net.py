"""
CriticalityNet — Multi-Scale Temporal CNN for MEA Criticality Classification.

Architecture
------------
Input : [B, n_electrodes, T]  (float32 spike train)
   ↓
Embedding Conv1d(n_electrodes → hidden_dim, kernel=7)
   ↓
4x TCB blocks with exponentially dilated convolutions
   ↓
Channel Attention (squeeze-excitation style)
   ↓
Global Average Pooling → [B, hidden_dim]
   ↓
┌─────────────────────────────────┐
│ Head 1: Criticality (3-class)   │  CrossEntropy
│ Head 2: Branching Ratio (σ)     │  MSE
│ Head 3: Genotype (4-class)      │  CrossEntropy (optional)
└─────────────────────────────────┘

Fits in ~40MB GPU RAM; input batch [32, 64, 1000] → ~8MB.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with dilation, residual skip, and dropout.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dilation : int
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2  # causal-ish padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(p=dropout)

        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        return out + residual


class _ChannelAttention(nn.Module):
    """
    Squeeze-Excitation channel attention over the hidden feature dimension.

    Parameters
    ----------
    hidden_dim : int
    reduction : int
        Bottleneck reduction ratio (default: 8).
    """

    def __init__(self, hidden_dim: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(hidden_dim // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → global avg pool → [B, C] → weights → [B, C, 1]
        gap = x.mean(dim=-1)           # [B, C]
        weights = self.fc(gap)         # [B, C]
        return x * weights.unsqueeze(-1)


class CriticalityNet(nn.Module):
    """
    Multi-scale TCN model for criticality state classification from MEA data.

    Parameters
    ----------
    n_electrodes : int
        Number of input MEA channels.
    n_time_bins : int
        Input time dimension (bins).
    hidden_dim : int
        Number of feature channels in TCN backbone.
    n_tcn_layers : int
        Number of TCN blocks (dilation doubles each block: 1,2,4,8,...).
    tcn_kernel_size : int
        Conv kernel size in each TCN block.
    dropout : float
        Dropout probability.
    n_criticality_classes : int
        Output classes for criticality head (default: 3).
    n_genotype_classes : int
        Output classes for genotype head (default: 4).
    use_genotype_head : bool
        If False, skip genotype classification head.
    use_branching_ratio_head : bool
        If False, skip branching ratio regression head.
    """

    def __init__(
        self,
        n_electrodes: int = 64,
        n_time_bins: int = 1000,
        hidden_dim: int = 128,
        n_tcn_layers: int = 4,
        tcn_kernel_size: int = 3,
        dropout: float = 0.2,
        n_criticality_classes: int = 3,
        n_genotype_classes: int = 4,
        use_genotype_head: bool = True,
        use_branching_ratio_head: bool = True,
    ) -> None:
        super().__init__()

        self.n_electrodes = n_electrodes
        self.n_time_bins = n_time_bins
        self.hidden_dim = hidden_dim
        self.use_genotype_head = use_genotype_head
        self.use_branching_ratio_head = use_branching_ratio_head

        # ── Embedding ──────────────────────────────────────────────────────
        self.embedding = nn.Sequential(
            nn.Conv1d(n_electrodes, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # ── TCN Backbone ───────────────────────────────────────────────────
        self.tcn_blocks = nn.ModuleList()
        for i in range(n_tcn_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                _TCNBlock(hidden_dim, hidden_dim, tcn_kernel_size, dilation, dropout)
            )

        # ── Channel Attention ──────────────────────────────────────────────
        self.attention = _ChannelAttention(hidden_dim)

        # ── Global Average Pooling ─────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)  # [B, H, T] → [B, H, 1]

        # ── Classification / Regression Heads ─────────────────────────────
        self.criticality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_criticality_classes),
        )

        if use_branching_ratio_head:
            self.branching_ratio_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),  # σ ∈ [0, 1] after sigmoid; scale to [0, 2] in forward
            )

        if use_genotype_head:
            self.genotype_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_genotype_classes),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize conv weights with Kaiming, linear with Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        spikes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        spikes : torch.Tensor
            Binary spike train [B, n_electrodes, T].

        Returns
        -------
        dict
            Keys: 'criticality_logits', 'branching_ratio', 'genotype_logits' (optional),
                  'features' (backbone embedding).
        """
        # Embedding: [B, n_elec, T] → [B, H, T]
        x = self.embedding(spikes)

        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)

        # Channel attention
        x = self.attention(x)

        # Global average pool: [B, H, T] → [B, H]
        feats = self.gap(x).squeeze(-1)

        out: dict[str, torch.Tensor] = {"features": feats}

        # Criticality head
        out["criticality_logits"] = self.criticality_head(feats)

        # Branching ratio regression (σ ∈ [0, 2])
        if self.use_branching_ratio_head:
            out["branching_ratio"] = self.branching_ratio_head(feats) * 2.0

        # Genotype classification
        if self.use_genotype_head:
            out["genotype_logits"] = self.genotype_head(feats)

        return out

    @classmethod
    def from_config(cls, model_cfg: dict) -> "CriticalityNet":
        """Instantiate from config dict (subset of full YAML)."""
        return cls(
            n_electrodes=model_cfg.get("n_electrodes", 64),
            n_time_bins=model_cfg.get("n_time_bins", 1000),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            n_tcn_layers=model_cfg.get("n_tcn_layers", 4),
            tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.2),
            n_criticality_classes=model_cfg.get("n_criticality_classes", 3),
            n_genotype_classes=model_cfg.get("n_genotype_classes", 4),
            use_genotype_head=model_cfg.get("use_genotype_head", True),
            use_branching_ratio_head=model_cfg.get("use_branching_ratio_head", True),
        )
