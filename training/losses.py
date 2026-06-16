"""
Custom Loss Functions for CriticalityNet.

Combined multi-task loss:
  L_total = w_crit * CE(criticality) +
            w_sigma * MSE(branching_ratio) +
            w_geno * CE(genotype) +
            w_cons * L_consistency

Consistency Loss (L_consistency):
  Penalizes predicted σ that is inconsistent with the predicted criticality
  class. E.g., if class=0 (subcritical), σ should be < 0.9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticalityLoss(nn.Module):
    """
    Multi-task loss combining criticality classification, branching ratio
    regression, genotype classification, and consistency regularization.

    Parameters
    ----------
    w_criticality : float
        Weight for criticality state CrossEntropy loss.
    w_sigma : float
        Weight for branching ratio MSE loss.
    w_genotype : float
        Weight for genotype CrossEntropy loss.
    w_consistency : float
        Weight for consistency regularization loss.
    label_smoothing : float
        Label smoothing for CrossEntropy (default: 0.05).
    """

    # Expected sigma ranges per class
    SIGMA_BOUNDS = {
        0: (0.0, 0.90),   # subcritical
        1: (0.90, 1.10),  # critical
        2: (1.10, 2.0),   # supercritical
    }

    def __init__(
        self,
        w_criticality: float = 1.0,
        w_sigma: float = 0.5,
        w_genotype: float = 0.3,
        w_consistency: float = 0.2,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_crit = w_criticality
        self.w_sigma = w_sigma
        self.w_geno = w_genotype
        self.w_cons = w_consistency

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        crit_labels: torch.Tensor,
        sigma_targets: torch.Tensor,
        geno_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss.

        Parameters
        ----------
        outputs : dict
            Model output dict (criticality_logits, branching_ratio, genotype_logits).
        crit_labels : torch.Tensor
            Criticality class labels [B], long.
        sigma_targets : torch.Tensor
            Ground-truth branching ratio [B], float.
        geno_labels : torch.Tensor
            Genotype class labels [B], long.

        Returns
        -------
        total_loss : torch.Tensor (scalar)
        loss_dict : dict[str, float]
            Individual loss components for logging.
        """
        loss_dict: dict[str, float] = {}
        total = torch.tensor(0.0, device=crit_labels.device, requires_grad=True)

        # ── Criticality CE ─────────────────────────────────────────────────
        l_crit = self.ce(outputs["criticality_logits"], crit_labels)
        total = total + self.w_crit * l_crit
        loss_dict["loss_criticality"] = l_crit.item()

        # ── Branching Ratio MSE ────────────────────────────────────────────
        if "branching_ratio" in outputs and sigma_targets is not None:
            pred_sigma = outputs["branching_ratio"].squeeze(-1)
            # Only penalize where sigma_targets > 0 (has ground truth)
            valid_mask = sigma_targets > 0
            if valid_mask.any():
                l_sigma = self.mse(pred_sigma[valid_mask], sigma_targets[valid_mask])
            else:
                l_sigma = torch.tensor(0.0, device=crit_labels.device)
            total = total + self.w_sigma * l_sigma
            loss_dict["loss_sigma"] = l_sigma.item()

        # ── Genotype CE ────────────────────────────────────────────────────
        if "genotype_logits" in outputs and geno_labels is not None:
            l_geno = self.ce(outputs["genotype_logits"], geno_labels)
            total = total + self.w_geno * l_geno
            loss_dict["loss_genotype"] = l_geno.item()

        # ── Consistency Loss ───────────────────────────────────────────────
        if "branching_ratio" in outputs:
            l_cons = self._consistency_loss(
                outputs["branching_ratio"].squeeze(-1),
                outputs["criticality_logits"],
            )
            total = total + self.w_cons * l_cons
            loss_dict["loss_consistency"] = l_cons.item()

        loss_dict["loss_total"] = total.item()
        return total, loss_dict

    def _consistency_loss(
        self,
        pred_sigma: torch.Tensor,
        crit_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize predicted σ that lies outside the expected range for the
        predicted criticality class.

        Parameters
        ----------
        pred_sigma : torch.Tensor [B]
        crit_logits : torch.Tensor [B, 3]

        Returns
        -------
        torch.Tensor
            Scalar consistency penalty.
        """
        pred_class = crit_logits.argmax(dim=-1)  # [B]
        penalty = torch.zeros_like(pred_sigma)

        for cls_id, (low, high) in self.SIGMA_BOUNDS.items():
            mask = pred_class == cls_id
            if not mask.any():
                continue
            sigma_cls = pred_sigma[mask]
            # Soft penalty: distance below lower bound or above upper bound
            below = F.relu(low - sigma_cls)
            above = F.relu(sigma_cls - high)
            penalty[mask] = below + above

        return penalty.mean()
