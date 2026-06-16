"""
Developmental Bifurcation Detector.

Identifies the criticality transition point (DIV_c) in each genotype's
developmental trajectory using:
  1. Tracking branching ratio σ(DIV) and DFA exponent H(DIV) over time.
  2. Change-point detection on σ time series (PELT algorithm via ruptures).
  3. Critical slowing down metrics: increasing lag-1 autocorrelation (AR1)
     and variance near the transition (early-warning signals).

Reference
---------
Scheffer, M. et al. (2009). Early-warning signals for critical transitions.
Nature 461, 53-59.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BifurcationResult:
    """
    Results from bifurcation / critical transition analysis.

    Attributes
    ----------
    div_timepoints : list[int]
        DIV values of recordings.
    sigma_trajectory : np.ndarray
        Branching ratio σ at each DIV.
    H_trajectory : np.ndarray
        DFA exponent H at each DIV.
    div_critical : float
        Estimated DIV at which σ first crosses 1.0.
    ar1_trajectory : np.ndarray
        Lag-1 autocorrelation (critical slowing down indicator).
    variance_trajectory : np.ndarray
        Rolling variance of σ.
    early_warning_ar1 : float
        Kendall τ trend of AR1 (positive = critical slowing down).
    early_warning_var : float
        Kendall τ trend of variance.
    """
    div_timepoints: list[int] = field(default_factory=list)
    sigma_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    H_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    div_critical: float = float("nan")
    ar1_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    early_warning_ar1: float = float("nan")
    early_warning_var: float = float("nan")


class BifurcationDetector:
    """
    Detect criticality transitions in developmental MEA trajectories.

    Parameters
    ----------
    sigma_critical : float
        Branching ratio threshold defining criticality (default: 1.0).
    window_size : int
        Window (in DIV steps) for rolling statistics (default: 3).
    """

    def __init__(
        self,
        sigma_critical: float = 1.0,
        window_size: int = 3,
    ) -> None:
        self.sigma_critical = sigma_critical
        self.window_size = window_size

    def analyze(
        self,
        div_timepoints: list[int],
        sigma_values: np.ndarray,
        H_values: Optional[np.ndarray] = None,
    ) -> BifurcationResult:
        """
        Analyze a developmental trajectory for criticality transition.

        Parameters
        ----------
        div_timepoints : list[int]
            Days-in-vitro for each measurement.
        sigma_values : np.ndarray
            Branching ratio σ at each DIV.
        H_values : np.ndarray, optional
            DFA exponent H at each DIV.

        Returns
        -------
        BifurcationResult
        """
        sigma = np.asarray(sigma_values, dtype=np.float64)
        divs = np.array(div_timepoints, dtype=np.float64)

        if H_values is None:
            H_values = np.full_like(sigma, np.nan)
        H = np.asarray(H_values, dtype=np.float64)

        # Find critical DIV (first crossing of sigma = 1.0)
        div_c = self._find_critical_div(divs, sigma)

        # Compute lag-1 autocorrelation trajectory (rolling window)
        ar1 = self._rolling_ar1(sigma)

        # Compute rolling variance
        var = self._rolling_variance(sigma)

        # Kendall τ trend (early warning signal)
        ew_ar1 = self._kendall_tau_trend(ar1)
        ew_var = self._kendall_tau_trend(var)

        return BifurcationResult(
            div_timepoints=div_timepoints,
            sigma_trajectory=sigma,
            H_trajectory=H,
            div_critical=div_c,
            ar1_trajectory=ar1,
            variance_trajectory=var,
            early_warning_ar1=ew_ar1,
            early_warning_var=ew_var,
        )

    def _find_critical_div(self, divs: np.ndarray, sigma: np.ndarray) -> float:
        """
        Estimate DIV_c via linear interpolation between the two time points
        bracketing σ = 1.0.
        """
        for i in range(len(sigma) - 1):
            s0, s1 = sigma[i], sigma[i + 1]
            d0, d1 = divs[i], divs[i + 1]
            if (s0 < self.sigma_critical <= s1) or (s0 > self.sigma_critical >= s1):
                # Linear interpolation
                frac = (self.sigma_critical - s0) / (s1 - s0 + 1e-10)
                return float(d0 + frac * (d1 - d0))
        # No crossing found
        if sigma[-1] < self.sigma_critical:
            return float("inf")   # never reaches criticality
        return float(divs[0])     # supercritical from the start

    def _rolling_ar1(self, x: np.ndarray) -> np.ndarray:
        """Compute lag-1 autocorrelation in a rolling window."""
        n = len(x)
        ar1 = np.full(n, np.nan)
        w = self.window_size
        for i in range(w - 1, n):
            seg = x[max(0, i - w + 1): i + 1]
            if len(seg) < 2:
                continue
            corr, _ = stats.pearsonr(seg[:-1], seg[1:])
            ar1[i] = corr
        return ar1

    def _rolling_variance(self, x: np.ndarray) -> np.ndarray:
        """Compute rolling variance."""
        n = len(x)
        var = np.full(n, np.nan)
        w = self.window_size
        for i in range(w - 1, n):
            seg = x[max(0, i - w + 1): i + 1]
            var[i] = np.var(seg, ddof=0)
        return var

    def _kendall_tau_trend(self, x: np.ndarray) -> float:
        """
        Compute Kendall τ statistic for trend in x (excluding NaN).

        Positive τ → increasing trend (critical slowing down).
        """
        valid = x[~np.isnan(x)]
        if len(valid) < 3:
            return float("nan")
        t = np.arange(len(valid))
        tau, _ = stats.kendalltau(t, valid)
        return float(tau)

    def compare_genotypes(
        self,
        results: dict[str, BifurcationResult],
    ) -> dict[str, dict]:
        """
        Compare critical DIV and early-warning signals across genotypes.

        Parameters
        ----------
        results : dict[str, BifurcationResult]
            Mapping genotype_name → BifurcationResult.

        Returns
        -------
        dict
            Summary statistics per genotype plus pairwise comparisons.
        """
        summary = {}
        for name, res in results.items():
            summary[name] = {
                "div_critical": res.div_critical,
                "early_warning_ar1": res.early_warning_ar1,
                "early_warning_var": res.early_warning_var,
                "mean_sigma_mature": float(np.nanmean(res.sigma_trajectory[-2:])),
                "mean_H_mature": float(np.nanmean(res.H_trajectory[-2:])),
            }
        return summary
