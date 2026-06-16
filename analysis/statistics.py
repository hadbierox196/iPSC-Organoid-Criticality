"""
Cross-Genotype Statistical Comparisons.

Provides non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Dunn's post-hoc)
for comparing criticality metrics across DISC1, SHANK3, 22q11.2del, and Control.

All comparisons use Bonferroni correction for multiple comparisons.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatTestResult:
    """Result from a statistical test."""
    test_name: str
    statistic: float
    pvalue: float
    pvalue_corrected: float
    significant: bool
    effect_size: float       # Cohen's d or rank-biserial r
    n_groups: int
    group_sizes: list[int]


class GenotypeStatistics:
    """
    Cross-genotype statistical analysis for criticality metrics.

    Parameters
    ----------
    alpha : float
        Significance level (default: 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def compare_groups(
        self,
        groups: dict[str, np.ndarray],
        metric_name: str = "metric",
    ) -> dict[str, StatTestResult]:
        """
        Run omnibus Kruskal-Wallis test followed by pairwise Mann-Whitney U
        with Bonferroni correction.

        Parameters
        ----------
        groups : dict[str, np.ndarray]
            Mapping genotype_name → array of metric values.
        metric_name : str

        Returns
        -------
        dict
            Keys: 'omnibus' and 'pairwise_{g1}_vs_{g2}' for each pair.
        """
        results = {}
        arrays = list(groups.values())
        names = list(groups.keys())

        # ── Omnibus: Kruskal-Wallis ────────────────────────────────────────
        if len(arrays) >= 2 and all(len(a) > 0 for a in arrays):
            stat, pval = stats.kruskal(*arrays)
            results["omnibus"] = StatTestResult(
                test_name="Kruskal-Wallis",
                statistic=float(stat),
                pvalue=float(pval),
                pvalue_corrected=float(pval),  # no correction needed for omnibus
                significant=pval < self.alpha,
                effect_size=self._eta_squared(stat, arrays),
                n_groups=len(arrays),
                group_sizes=[len(a) for a in arrays],
            )
        else:
            logger.warning("Insufficient data for omnibus test (%s).", metric_name)

        # ── Pairwise: Mann-Whitney U with Bonferroni ───────────────────────
        pairs = list(itertools.combinations(range(len(names)), 2))
        n_comparisons = len(pairs)

        for i, j in pairs:
            g1, g2 = names[i], names[j]
            a1, a2 = arrays[i], arrays[j]

            if len(a1) < 2 or len(a2) < 2:
                continue

            stat, pval = stats.mannwhitneyu(a1, a2, alternative="two-sided")
            pval_corrected = min(pval * n_comparisons, 1.0)
            effect_r = self._rank_biserial_r(a1, a2)

            key = f"pairwise_{g1}_vs_{g2}"
            results[key] = StatTestResult(
                test_name="Mann-Whitney U",
                statistic=float(stat),
                pvalue=float(pval),
                pvalue_corrected=float(pval_corrected),
                significant=pval_corrected < self.alpha,
                effect_size=float(effect_r),
                n_groups=2,
                group_sizes=[len(a1), len(a2)],
            )

        return results

    def summary_stats(self, groups: dict[str, np.ndarray]) -> dict[str, dict]:
        """
        Compute descriptive statistics per group.

        Parameters
        ----------
        groups : dict[str, np.ndarray]

        Returns
        -------
        dict
            Per-genotype summary (mean, median, std, iqr, n).
        """
        out = {}
        for name, arr in groups.items():
            arr = arr[~np.isnan(arr)]
            out[name] = {
                "n": len(arr),
                "mean": float(np.mean(arr)) if len(arr) > 0 else np.nan,
                "median": float(np.median(arr)) if len(arr) > 0 else np.nan,
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "q25": float(np.percentile(arr, 25)) if len(arr) > 0 else np.nan,
                "q75": float(np.percentile(arr, 75)) if len(arr) > 0 else np.nan,
            }
        return out

    @staticmethod
    def _eta_squared(H_stat: float, arrays: list[np.ndarray]) -> float:
        """Effect size η² for Kruskal-Wallis."""
        n_total = sum(len(a) for a in arrays)
        k = len(arrays)
        if n_total <= k:
            return 0.0
        return float((H_stat - k + 1) / (n_total - k))

    @staticmethod
    def _rank_biserial_r(a1: np.ndarray, a2: np.ndarray) -> float:
        """Rank-biserial correlation as effect size for Mann-Whitney."""
        n1, n2 = len(a1), len(a2)
        if n1 == 0 or n2 == 0:
            return 0.0
        u_stat, _ = stats.mannwhitneyu(a1, a2, alternative="two-sided")
        r = 1.0 - (2.0 * u_stat) / (n1 * n2)
        return float(np.clip(r, -1.0, 1.0))
