"""
Neuronal Avalanche Extractor.

Converts multi-electrode binary spike trains into neuronal avalanche
sequences following Beggs & Plenz (2003) methodology:

  1. Sum population activity across electrodes per time bin.
  2. Define an avalanche as a contiguous sequence of active bins (>= threshold)
     bracketed by silent bins.
  3. Record avalanche size (total spikes) and duration (# bins).

Also estimates the branching ratio σ from the spike train.

Reference
---------
Beggs, J.M. & Plenz, D. (2003) Neuronal avalanches in neocortical circuits.
J. Neurosci. 23(35), 11167-11177.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AvalancheData:
    """
    Container for neuronal avalanche statistics.

    Attributes
    ----------
    sizes : np.ndarray
        Avalanche sizes (total spike count across all electrodes).
    durations : np.ndarray
        Avalanche durations in time bins.
    branching_ratio : float
        Estimated branching ratio σ.
    mean_firing_rate : float
        Mean firing rate in spikes / electrode / bin.
    population_activity : np.ndarray
        Sum of active electrodes per time bin [n_bins].
    """
    sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    durations: np.ndarray = field(default_factory=lambda: np.array([]))
    branching_ratio: float = 0.0
    mean_firing_rate: float = 0.0
    population_activity: np.ndarray = field(default_factory=lambda: np.array([]))


class AvalancheExtractor:
    """
    Extracts neuronal avalanche statistics from binary spike trains.

    Parameters
    ----------
    threshold : int
        Minimum number of active electrodes to define an active time bin
        (default: 0, i.e., at least 1 electrode active).
    min_size : int
        Minimum avalanche size (spikes) to include (default: 2).
    max_duration_bins : int
        Maximum allowed avalanche duration in bins (default: 500).
    """

    def __init__(
        self,
        threshold: int = 0,
        min_size: int = 2,
        max_duration_bins: int = 500,
    ) -> None:
        self.threshold = threshold
        self.min_size = min_size
        self.max_duration_bins = max_duration_bins

    def extract(self, spike_train: np.ndarray) -> AvalancheData:
        """
        Extract avalanche statistics from a binary spike train.

        Parameters
        ----------
        spike_train : np.ndarray
            Binary spike train [n_electrodes, n_bins], dtype uint8 or bool.

        Returns
        -------
        AvalancheData
            Extracted statistics including sizes, durations, and σ.
        """
        spike_train = np.asarray(spike_train, dtype=np.int32)
        n_electrodes, n_bins = spike_train.shape

        # Population activity (total active electrodes per bin)
        pop_act = spike_train.sum(axis=0)

        # Define active bins
        active = pop_act > self.threshold  # [n_bins] bool

        # Find avalanche boundaries (transitions 0→1 and 1→0)
        sizes, durations = self._segment_avalanches(active, pop_act)

        # Filter by minimum size
        valid = sizes >= self.min_size
        sizes = sizes[valid]
        durations = durations[valid]

        # Estimate branching ratio
        sigma = self._estimate_branching_ratio(pop_act)

        # Mean firing rate
        mfr = spike_train.mean()  # spikes/electrode/bin

        return AvalancheData(
            sizes=sizes,
            durations=durations,
            branching_ratio=float(sigma),
            mean_firing_rate=float(mfr),
            population_activity=pop_act.astype(np.float32),
        )

    def _segment_avalanches(
        self,
        active: np.ndarray,
        pop_act: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment population activity into discrete avalanches.

        Parameters
        ----------
        active : np.ndarray
            Boolean array of active bins [n_bins].
        pop_act : np.ndarray
            Population activity per bin [n_bins].

        Returns
        -------
        sizes : np.ndarray
            Total spikes per avalanche.
        durations : np.ndarray
            Duration in bins per avalanche.
        """
        sizes_list = []
        durations_list = []

        n_bins = len(active)
        in_avalanche = False
        start = 0
        current_size = 0
        current_duration = 0

        for t in range(n_bins):
            if active[t]:
                if not in_avalanche:
                    in_avalanche = True
                    start = t
                    current_size = 0
                    current_duration = 0
                current_size += int(pop_act[t])
                current_duration += 1
            else:
                if in_avalanche:
                    in_avalanche = False
                    if current_duration <= self.max_duration_bins:
                        sizes_list.append(current_size)
                        durations_list.append(current_duration)

        # Handle avalanche at end of recording
        if in_avalanche and current_duration <= self.max_duration_bins:
            sizes_list.append(current_size)
            durations_list.append(current_duration)

        return np.array(sizes_list, dtype=np.float64), np.array(durations_list, dtype=np.float64)

    def _estimate_branching_ratio(self, pop_act: np.ndarray) -> float:
        """
        Estimate the branching ratio σ = E[descendants] / E[ancestors].

        Uses the method of Beggs & Plenz (2003): σ is estimated from the
        ratio of mean descendants (t+1) to mean ancestors (t) across all
        consecutive pairs of active time bins.

        Parameters
        ----------
        pop_act : np.ndarray
            Population activity time series [n_bins].

        Returns
        -------
        float
            Estimated branching ratio σ. Returns 0.0 if insufficient data.
        """
        ancestors = pop_act[:-1].astype(np.float64)
        descendants = pop_act[1:].astype(np.float64)

        # Only use pairs where ancestors > 0
        mask = ancestors > 0
        if mask.sum() < 10:
            return 0.0

        # σ = mean(descendants[mask] / ancestors[mask])
        sigma = np.mean(descendants[mask] / ancestors[mask])
        return float(np.clip(sigma, 0.0, 5.0))

    def compute_synchrony_index(self, spike_train: np.ndarray) -> float:
        """
        Compute the network synchrony index (pairwise correlation mean).

        Parameters
        ----------
        spike_train : np.ndarray
            Binary spike train [n_electrodes, n_bins].

        Returns
        -------
        float
            Mean pairwise Pearson correlation across electrode pairs.
        """
        st = spike_train.astype(np.float32)
        n_elec = st.shape[0]

        if n_elec < 2:
            return 0.0

        # Subtract mean, compute covariance matrix
        st_zm = st - st.mean(axis=1, keepdims=True)
        norms = np.sqrt((st_zm ** 2).sum(axis=1))
        norms[norms == 0] = 1.0  # avoid division by zero

        # Pearson correlation matrix
        corr = (st_zm @ st_zm.T) / np.outer(norms, norms)
        np.fill_diagonal(corr, 0)

        # Mean of upper triangle
        idx_upper = np.triu_indices(n_elec, k=1)
        mean_corr = corr[idx_upper].mean()
        return float(np.clip(mean_corr, -1.0, 1.0))
