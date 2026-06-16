"""
Power-Law Fitting via Maximum Likelihood Estimation.

Implements the Clauset, Shalizi & Newman (2009) MLE approach for discrete
and continuous power-law distributions, including:
  - Exponent estimation (τ for sizes, α for durations)
  - KS test for goodness-of-fit
  - Bootstrap confidence intervals
  - Shape-function collapse analysis (avalanche duration vs. size scaling)

Reference
---------
Clauset, A., Shalizi, C.R. & Newman, M.E.J. (2009). Power-law distributions
in empirical data. SIAM Review 51(4), 661-703.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PowerLawResult:
    """
    Results from power-law fitting.

    Attributes
    ----------
    exponent : float
        Estimated power-law exponent (τ or α).
    xmin : float
        Lower cutoff x_min used in fit.
    xmax : float
        Upper cutoff (max data value).
    ks_stat : float
        Kolmogorov-Smirnov statistic against fitted distribution.
    ks_pvalue : float
        KS test p-value (>0.05 → cannot reject power law).
    n_tail : int
        Number of data points above x_min.
    ci_lower : float
        95% CI lower bound on exponent (bootstrap).
    ci_upper : float
        95% CI upper bound on exponent (bootstrap).
    log_likelihood : float
        Log-likelihood of power-law fit.
    """
    exponent: float
    xmin: float
    xmax: float
    ks_stat: float
    ks_pvalue: float
    n_tail: int
    ci_lower: float
    ci_upper: float
    log_likelihood: float

    def is_power_law(self, alpha: float = 0.05) -> bool:
        """Return True if KS test cannot reject power-law (p > alpha)."""
        return self.ks_pvalue > alpha


class PowerLawFitter:
    """
    Fit a power-law distribution P(x) ∝ x^{-τ} to empirical data using MLE.

    Supports both discrete (integer) and continuous data.

    Parameters
    ----------
    xmin_method : str
        'auto'  → choose x_min that minimizes KS statistic (Clauset 2009).
        'median' → use median of data as x_min.
        float  → fixed x_min.
    n_bootstrap : int
        Number of bootstrap samples for CI estimation (default: 200).
    discrete : bool
        If True, use discrete power-law MLE. If False, use continuous.
    """

    def __init__(
        self,
        xmin_method: str | float = "auto",
        n_bootstrap: int = 200,
        discrete: bool = True,
    ) -> None:
        self.xmin_method = xmin_method
        self.n_bootstrap = n_bootstrap
        self.discrete = discrete

    def fit(self, data: np.ndarray) -> PowerLawResult:
        """
        Fit power-law to data.

        Parameters
        ----------
        data : np.ndarray
            Positive-valued samples (avalanche sizes or durations).

        Returns
        -------
        PowerLawResult
        """
        data = np.asarray(data, dtype=np.float64)
        data = data[data > 0]

        if len(data) < 10:
            logger.warning("Fewer than 10 data points — returning null result.")
            return PowerLawResult(
                exponent=np.nan, xmin=np.nan, xmax=np.nan,
                ks_stat=np.nan, ks_pvalue=np.nan, n_tail=0,
                ci_lower=np.nan, ci_upper=np.nan, log_likelihood=np.nan,
            )

        # Determine x_min
        xmin = self._determine_xmin(data)

        # Fit using data >= x_min
        tail = data[data >= xmin]
        n_tail = len(tail)

        if n_tail < 5:
            xmin = float(np.percentile(data, 25))
            tail = data[data >= xmin]
            n_tail = len(tail)

        # MLE exponent estimation
        exponent = self._mle_exponent(tail, xmin)

        # KS statistic
        ks_stat = self._ks_statistic(tail, exponent, xmin)
        ks_pvalue = self._ks_pvalue(tail, ks_stat, exponent, xmin)

        # Log-likelihood
        ll = self._log_likelihood(tail, exponent, xmin)

        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(tail, xmin)

        return PowerLawResult(
            exponent=float(exponent),
            xmin=float(xmin),
            xmax=float(data.max()),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            n_tail=int(n_tail),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            log_likelihood=float(ll),
        )

    def _determine_xmin(self, data: np.ndarray) -> float:
        """Determine x_min by KS minimization or fixed value."""
        if isinstance(self.xmin_method, (int, float)):
            return float(self.xmin_method)
        if self.xmin_method == "median":
            return float(np.median(data))

        # Auto: scan candidate x_mins, minimize KS statistic
        candidates = np.unique(data)[:-1]
        candidates = candidates[candidates < np.percentile(data, 95)]

        best_xmin = candidates[0]
        best_ks = np.inf

        for xmin in candidates:
            tail = data[data >= xmin]
            if len(tail) < 5:
                continue
            exp = self._mle_exponent(tail, xmin)
            ks = self._ks_statistic(tail, exp, xmin)
            if ks < best_ks:
                best_ks = ks
                best_xmin = xmin

        return float(best_xmin)

    def _mle_exponent(self, tail: np.ndarray, xmin: float) -> float:
        """
        MLE estimator for power-law exponent.

        Continuous: τ = 1 + n [Σ ln(x_i / x_min)]^{-1}
        Discrete:   τ = 1 + n [Σ ln(x_i / (x_min - 0.5))]^{-1}
        """
        n = len(tail)
        if self.discrete:
            ln_sum = np.sum(np.log(tail / (xmin - 0.5)))
        else:
            ln_sum = np.sum(np.log(tail / xmin))

        if ln_sum <= 0:
            return 2.0
        return 1.0 + n / ln_sum

    def _ks_statistic(self, tail: np.ndarray, exponent: float, xmin: float) -> float:
        """Kolmogorov-Smirnov statistic between empirical and fitted CDF."""
        n = len(tail)
        tail_sorted = np.sort(tail)

        # Empirical CDF
        ecdf = np.arange(1, n + 1) / n

        # Theoretical CDF: F(x) = 1 - (x/xmin)^{1-exponent}
        if abs(exponent - 1.0) < 1e-10:
            return 1.0
        tcdf = 1.0 - (tail_sorted / xmin) ** (1.0 - exponent)
        tcdf = np.clip(tcdf, 0.0, 1.0)

        return float(np.max(np.abs(ecdf - tcdf)))

    def _ks_pvalue(
        self,
        tail: np.ndarray,
        ks_stat: float,
        exponent: float,
        xmin: float,
        n_simulations: int = 100,
    ) -> float:
        """Estimate p-value via Monte Carlo simulation (simplified)."""
        rng = np.random.default_rng(0)
        n = len(tail)
        count_greater = 0
        for _ in range(n_simulations):
            # Generate synthetic power-law samples
            u = rng.uniform(0, 1, n)
            synth = xmin * (1.0 - u) ** (1.0 / (1.0 - exponent))
            exp_s = self._mle_exponent(synth, xmin)
            ks_s = self._ks_statistic(synth, exp_s, xmin)
            if ks_s >= ks_stat:
                count_greater += 1
        return count_greater / n_simulations

    def _log_likelihood(self, tail: np.ndarray, exponent: float, xmin: float) -> float:
        """Log-likelihood of power-law fit."""
        if exponent <= 1.0:
            return -np.inf
        ll = len(tail) * np.log(exponent - 1.0) - len(tail) * np.log(xmin)
        ll -= exponent * np.sum(np.log(tail / xmin))
        return float(ll)

    def _bootstrap_ci(
        self,
        tail: np.ndarray,
        xmin: float,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap 95% CI for the exponent estimate."""
        rng = np.random.default_rng(42)
        n = len(tail)
        exponents = []
        for _ in range(self.n_bootstrap):
            sample = rng.choice(tail, size=n, replace=True)
            exp = self._mle_exponent(sample, xmin)
            exponents.append(exp)
        alpha = (1.0 - confidence) / 2.0
        return (
            float(np.percentile(exponents, 100 * alpha)),
            float(np.percentile(exponents, 100 * (1 - alpha))),
        )


def check_scaling_relation(
    size_exponent: float,
    duration_exponent: float,
    size_result: "PowerLawResult",
    duration_result: "PowerLawResult",
) -> dict[str, float]:
    """
    Verify the crackling-noise / shape-function scaling relation.

    At criticality: (α - 1) / (τ - 1) = 1/σ_νz
    Also: mean_size(T) ∝ T^{(α-1)/(τ-1)}

    For Beggs & Plenz 2003 values: τ ≈ 1.5, α ≈ 2.0.
    Expected ratio: (2.0-1) / (1.5-1) = 2.0

    Parameters
    ----------
    size_exponent : float
        τ from avalanche size distribution.
    duration_exponent : float
        α from avalanche duration distribution.

    Returns
    -------
    dict with scaling ratio and deviation from expected value.
    """
    if abs(size_exponent - 1.0) < 1e-6:
        ratio = np.nan
    else:
        ratio = (duration_exponent - 1.0) / (size_exponent - 1.0)

    expected = 2.0  # Beggs & Plenz 2003
    deviation = abs(ratio - expected) if not np.isnan(ratio) else np.nan

    return {
        "size_exponent_tau": size_exponent,
        "duration_exponent_alpha": duration_exponent,
        "scaling_ratio": ratio,
        "expected_ratio": expected,
        "deviation_from_expected": deviation,
        "size_fit_pvalue": size_result.ks_pvalue,
        "duration_fit_pvalue": duration_result.ks_pvalue,
    }
