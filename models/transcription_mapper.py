"""
TranscriptionalMapper — Bidirectional Electrophysiology ↔ Transcriptome Map.

Maps a vector of electrophysiology criticality features to a latent
transcriptional state (gene-set activity scores) and vice versa.

Electrophysiology features (input):
  [sigma, tau, H, sync_index, mfr, burst_rate, isi_cv, gamma_power]
  → 8-dimensional feature vector

Transcriptional output:
  Activity scores for N gene sets (synaptic, channel, activity-dependent, etc.)
  Default: 32 gene-set scores.

Training modes
--------------
1. Supervised: paired MEA + scRNA-seq data
2. Genotype-discriminative: predict genotype from transcriptional state
3. Reconstruction: autoencoder loss on electrophysiology features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Gene set names (subset, for annotation)
GENE_SET_NAMES = [
    "Synaptic_SHANK", "Synaptic_PSD95", "Synaptic_NRXN",
    "Nav_channels", "Kv_channels", "Cav_channels",
    "DISC1_pathway", "NRG1_ErbB4", "DTNBP1_dysbindin",
    "DGCR8_microRNA", "TBX1_cardiac", "COMT_dopamine",
    "Arc_activity_dep", "BDNF_TRKB", "FOS_FOSB",
    "GABA_synthesis", "Glutamate_AMPA", "Glutamate_NMDA",
    "Mitochondria_OxPhos", "Ribosome_translation",
    "ER_stress_UPR", "Autophagy_lysosome",
    "Cell_cycle_G1S", "Wnt_signaling", "Notch_signaling",
    "mTOR_signaling", "MAPK_ERK", "PI3K_Akt",
    "Neurogenesis_Sox2", "Neurogenesis_Pax6",
    "Cortical_layering_Ctip2", "Cortical_layering_Cux1",
]

ELEC_FEATURE_NAMES = [
    "branching_ratio_sigma",
    "avalanche_size_exponent_tau",
    "dfa_exponent_H",
    "synchrony_index",
    "mean_firing_rate",
    "burst_rate",
    "isi_cv",
    "gamma_band_power",
]


class TranscriptionalMapper(nn.Module):
    """
    MLP encoder/decoder mapping between electrophysiology features
    and transcriptional gene-set activity scores.

    Parameters
    ----------
    n_elec_features : int
        Number of electrophysiology input features (default: 8).
    n_gene_sets : int
        Number of transcriptional gene-set output dimensions (default: 32).
    latent_dim : int
        Shared latent space dimensionality.
    hidden_dim : int
        Hidden MLP width.
    dropout : float
    n_genotype_classes : int
        For genotype classification head.
    """

    def __init__(
        self,
        n_elec_features: int = 8,
        n_gene_sets: int = 32,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        n_genotype_classes: int = 4,
    ) -> None:
        super().__init__()

        self.n_elec_features = n_elec_features
        self.n_gene_sets = n_gene_sets
        self.latent_dim = latent_dim

        # ── Electrophysiology Encoder ─────────────────────────────────────
        self.elec_encoder = nn.Sequential(
            nn.Linear(n_elec_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ── Transcriptional Decoder ───────────────────────────────────────
        self.transcr_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_gene_sets),
        )

        # ── Genotype Classifier from latent ─────────────────────────────
        self.genotype_clf = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_genotype_classes),
        )

        # ── Transcriptional Encoder (for inverse mapping) ─────────────────
        self.transcr_encoder = nn.Sequential(
            nn.Linear(n_gene_sets, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ── Electrophysiology Decoder (for inverse mapping) ───────────────
        self.elec_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_elec_features),
        )

    def encode_elec(self, elec_features: torch.Tensor) -> torch.Tensor:
        """Encode electrophysiology → latent [B, latent_dim]."""
        return self.elec_encoder(elec_features)

    def decode_transcriptome(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → gene-set scores [B, n_gene_sets]."""
        return self.transcr_decoder(latent)

    def encode_transcriptome(self, gene_scores: torch.Tensor) -> torch.Tensor:
        """Encode gene-set scores → latent [B, latent_dim]."""
        return self.transcr_encoder(gene_scores)

    def decode_elec(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → electrophysiology features [B, n_elec_features]."""
        return self.elec_decoder(latent)

    def forward(
        self,
        elec_features: torch.Tensor,
        gene_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: electrophysiology → transcriptional prediction.

        Parameters
        ----------
        elec_features : torch.Tensor
            [B, n_elec_features] normalized feature vector.
        gene_scores : torch.Tensor, optional
            [B, n_gene_sets] ground-truth gene scores for reconstruction loss.

        Returns
        -------
        dict with:
            'latent'            : [B, latent_dim]
            'pred_gene_scores'  : [B, n_gene_sets]
            'genotype_logits'   : [B, n_genotype_classes]
            'recon_elec'        : [B, n_elec_features]  (from latent)
        """
        latent = self.encode_elec(elec_features)
        pred_gene_scores = self.decode_transcriptome(latent)
        genotype_logits = self.genotype_clf(latent)
        recon_elec = self.decode_elec(latent)

        out = {
            "latent": latent,
            "pred_gene_scores": pred_gene_scores,
            "genotype_logits": genotype_logits,
            "recon_elec": recon_elec,
        }

        if gene_scores is not None:
            # Also encode from transcriptome side and reconstruct
            latent_t = self.encode_transcriptome(gene_scores)
            out["latent_from_transcr"] = latent_t
            out["recon_elec_from_transcr"] = self.decode_elec(latent_t)

        return out
