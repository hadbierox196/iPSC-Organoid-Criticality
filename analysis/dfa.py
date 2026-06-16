"""
Detrended Fluctuation Analysis (DFA) — Hurst Exponent Estimation.

DFA quantifies long-range temporal correlations in non-stationary time series.
For MEA population activity at criticality, the Hurst exponent H ≈ 1.0.

Subcritical systems (σ < 1): H → 0.5 (uncorrelated, Brownian)
Critical systems  (σ = 1): H ≈ 0.9–1.1 (long-range correlated)
Supercritical     (σ > 1): H > 1.1 (persistent, synchronized bursting)

Reference
---------
Peng, C.K. et al. (1994). Mosaic organization of DNA nucleotides.
Phys Rev E 49, 1685.

Linkenkaer-Hansen, K. et al. (2001). Long-range temporal correlations and
scaling behavior in human brain oscillations. J. Neurosci. 21(4), 1370-1377.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_dfa(
    signal: np.ndarray,
    scales: Optional[list[int]] = None,
    fit_range: Optional[tuple[int, int]] = None,
    order: int = 1,
) -> dict[str, float | np.ndarray]:
    """
    Compute Detrended Fluctuation Analysis (DFA) on a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D time series (e.g., population spike count per bin). Length ≥ 64.
    scales : list[int], optional
        Window sizes (n) to use. Defaults to [4, 8, 16, 32, 64, 128, 256, 512].
    fit_range : tuple[int, int], optional
        (min_scale, max_scale) to use for H estimation (exclude boundary effects).
        Defaults to (8, 256).
    order : int
        Detrending polynomial order (1=linear, 2=quadratic). Default: 1.

    Returns
    -------
    dict with keys:
        'H'            : float, Hurst exponent
        'scales'       : np.ndarray, window sizes used
        'fluctuations' : np.ndarray, F(n) per scale
        'r2'           : float, R² of log-log linear fit
        'intercept'    : float, log-log intercept
    """
    signal = np.asarray(signal, dtype=np.float64)
    n_total = len(signal)

    if scales is None:
        scales = [4, 8, 16, 32, 64, 128, 256, 512]
    if fit_range is None:
        fit_range = (8, 256)

    # 1. Integrate (cumulative sum of demeaned signal)
    y = np.cumsum(signal - signal.mean())

    fluctuations = []
    valid_scales = []

    for n in scales:
        if n >= n_total // 4:
            continue  # skip scales too large for reliable estimate

        n_windows = n_total // n
        if n_windows < 4:
            continue

        F_sq_sum = 0.0
        for i in range(n_windows):
            segment = y[i * n: (i + 1) * n]
            t = np.arange(n, dtype=np.float64)
            # Polynomial detrending
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            F_sq_sum += np.mean((segment - trend) ** 2)

        F = np.sqrt(F_sq_sum / n_windows)
        fluctuations.append(F)
        valid_scales.append(n)

    scales_arr = np.array(valid_scales, dtype=np.float64)
    fluct_arr = np.array(fluctuations, dtype=np.float64)

    if len(scales_arr) < 3:
        logger.warning("DFA: too few valid scales (%d). Signal length: %d", len(scales_arr), n_total)
        return {
            "H": float("nan"),
            "scales": scales_arr,
            "fluctuations": fluct_arr,
            "r2": float("nan"),
            "intercept": float("nan"),
        }

    # 2. Fit log-log within fit_range
    mask = (scales_arr >= fit_range[0]) & (scales_arr <= fit_range[1])
    if mask.sum() < 2:
        mask = np.ones(len(scales_arr), dtype=bool)

    log_s = np.log10(scales_arr[mask])
    log_f = np.log10(fluct_arr[mask])

    coeffs_fit = np.polyfit(log_s, log_f, 1)
    H = coeffs_fit[0]
    intercept = coeffs_fit[1]

    # R² of fit
    log_f_pred = np.polyval(coeffs_fit, log_s)
    ss_res = np.sum((log_f - log_f_pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "H": float(np.clip(H, 0.0, 2.0)),
        "scales": scales_arr,
        "fluctuations": fluct_arr,
        "r2": r2,
        "intercept": float(intercept),
    }


def dfa_batch(
    population_activities: list[np.ndarray],
    scales: Optional[list[int]] = None,
    fit_range: Optional[tuple[int, int]] = None,
) -> list[dict]:
    """
    Apply DFA to a list of population activity signals.

    Parameters
    ----------
    population_activities : list of np.ndarray
        Each element is a 1D population activity time series.
    scales : list[int], optional
    fit_range : tuple, optional

    Returns
    -------
    list of dicts (one per signal), each with 'H', 'r2', etc.
    """
    results = []
    for sig in population_activities:
        results.append(compute_dfa(sig, scales=scales, fit_range=fit_range))
    return results


def interpret_hurst(H: float) -> str:
    """
    Qualitative interpretation of the Hurst exponent in the MEA context.

    Parameters
    ----------
    H : float

    Returns
    -------
    str
        Qualitative interpretation string.
    """
    if H < 0.5:
        return "Anti-persistent (subcritical, mean-reverting activity)"
    elif H < 0.75:
        return "Weakly persistent (transitioning, sub-to-critical range)"
    elif H < 1.1:
        return "Long-range correlated (critical, optimal coding range)"
    elif H < 1.5:
        return "Strongly persistent (supercritical, synchronized bursting)"
    else:
        return "Non-stationary / 1/f noise or pathological synchrony"
