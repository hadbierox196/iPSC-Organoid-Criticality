"""
Metric computation for criticality classification and branching ratio regression.

Computes: Accuracy, F1-macro, ROC-AUC (OvR), RMSE, Concordance Correlation Coefficient.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(preds: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Compute all evaluation metrics from prediction arrays.

    Parameters
    ----------
    preds : dict
        Output from Evaluator.predict():
          'crit_preds', 'crit_labels', 'sigma_preds',
          'sigma_targets', 'geno_preds', 'geno_labels'

    Returns
    -------
    dict[str, float]
        All metrics as a flat dict suitable for JSON serialization.
    """
    metrics: dict[str, float] = {}

    # ── Criticality Classification ─────────────────────────────────────────
    if len(preds.get("crit_preds", [])) > 0:
        yl = preds["crit_labels"]
        yp = preds["crit_preds"]

        metrics["criticality_accuracy"] = float(accuracy_score(yl, yp))
        metrics["criticality_f1_macro"] = float(
            f1_score(yl, yp, average="macro", zero_division=0)
        )
        metrics["criticality_f1_weighted"] = float(
            f1_score(yl, yp, average="weighted", zero_division=0)
        )

        # ROC-AUC (requires probability outputs — use one-hot encoding here)
        try:
            from sklearn.preprocessing import label_binarize
            classes = sorted(np.unique(yl))
            if len(classes) >= 2:
                yl_bin = label_binarize(yl, classes=classes)
                yp_bin = label_binarize(yp, classes=classes)
                if yl_bin.shape[1] > 1:
                    metrics["criticality_roc_auc_ovr"] = float(
                        roc_auc_score(yl_bin, yp_bin, average="macro",
                                      multi_class="ovr")
                    )
        except Exception as e:
            logger.debug("ROC-AUC skipped: %s", e)

        # Per-class F1
        report = classification_report(yl, yp, output_dict=True, zero_division=0)
        class_names = {0: "subcritical", 1: "critical", 2: "supercritical"}
        for cls_id, cls_name in class_names.items():
            if str(cls_id) in report:
                metrics[f"f1_{cls_name}"] = float(report[str(cls_id)]["f1-score"])

    # ── Branching Ratio Regression ─────────────────────────────────────────
    if len(preds.get("sigma_preds", [])) > 0:
        yt = preds["sigma_targets"]
        yp = preds["sigma_preds"]
        valid = yt > 0
        if valid.any():
            rmse = float(np.sqrt(np.mean((yt[valid] - yp[valid]) ** 2)))
            mae  = float(np.mean(np.abs(yt[valid] - yp[valid])))
            ccc  = _concordance_correlation_coefficient(yt[valid], yp[valid])
            metrics["sigma_rmse"] = rmse
            metrics["sigma_mae"] = mae
            metrics["sigma_ccc"] = ccc

    # ── Genotype Classification ────────────────────────────────────────────
    if len(preds.get("geno_preds", [])) > 0:
        yl = preds["geno_labels"]
        yp = preds["geno_preds"]
        metrics["genotype_accuracy"] = float(accuracy_score(yl, yp))
        metrics["genotype_f1_macro"] = float(
            f1_score(yl, yp, average="macro", zero_division=0)
        )

    return metrics


def _concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Lin's Concordance Correlation Coefficient (CCC).

    CCC = 2 * cov(y, yhat) / (var(y) + var(yhat) + (mean(y) - mean(yhat))^2)

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    float
        CCC ∈ [-1, 1]. 1.0 = perfect concordance.
    """
    mu_t = y_true.mean()
    mu_p = y_pred.mean()
    var_t = y_true.var()
    var_p = y_pred.var()
    cov = np.cov(y_true, y_pred, ddof=0)[0, 1]

    denom = var_t + var_p + (mu_t - mu_p) ** 2
    if denom < 1e-10:
        return 0.0
    return float(2.0 * cov / denom)
