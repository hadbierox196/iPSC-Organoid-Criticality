"""
MEA Signal Preprocessor.

Applies bandpass filtering (MNE), spike detection, and LFP extraction to
raw voltage traces stored in HDF5 format. Outputs a processed HDF5 with
binary spike trains and downsampled LFP.

Supports chunked processing (no >4GB RAM loads).

Usage
-----
python -m data.preprocessor --config configs/default_config.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import h5py
import mne
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MEAPreprocessor:
    """
    Multi-electrode array (MEA) signal preprocessor using MNE.

    Performs:
    1. Butterworth bandpass filter (300–3000 Hz) for spike detection.
    2. Threshold-crossing spike detection (negative peak, per electrode).
    3. Binary spike train generation at 1ms bins.
    4. Optional LFP extraction (1–100 Hz, downsampled to 1 kHz).

    Parameters
    ----------
    sampling_rate : int
        Raw signal sampling rate in Hz.
    spike_threshold_uv : float
        Negative threshold for spike detection (default: -30 μV).
    hp_cutoff_hz : float
        High-pass cutoff for spike band (default: 300 Hz).
    lp_cutoff_hz : float
        Low-pass cutoff for spike band (default: 3000 Hz).
    lfp_cutoff_hz : float
        Low-pass cutoff for LFP extraction (default: 100 Hz).
    lfp_downsample_factor : int
        Downsample factor for LFP (default: 20, 20kHz → 1kHz).
    bin_size_ms : float
        Spike bin width in milliseconds (default: 1.0 ms).
    """

    def __init__(
        self,
        sampling_rate: int = 20000,
        spike_threshold_uv: float = -30.0,
        hp_cutoff_hz: float = 300.0,
        lp_cutoff_hz: float = 3000.0,
        lfp_cutoff_hz: float = 100.0,
        lfp_downsample_factor: int = 20,
        bin_size_ms: float = 1.0,
    ) -> None:
        self.fs = sampling_rate
        self.threshold = spike_threshold_uv
        self.hp = hp_cutoff_hz
        self.lp = lp_cutoff_hz
        self.lfp_lp = lfp_cutoff_hz
        self.lfp_ds = lfp_downsample_factor
        self.bin_size_ms = bin_size_ms
        self._samples_per_bin = int(sampling_rate * bin_size_ms / 1000.0)
        assert self._samples_per_bin >= 1, "bin_size_ms too small for sampling_rate."

    def process_voltage_trace(
        self,
        voltage: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a multi-electrode voltage recording.

        Parameters
        ----------
        voltage : np.ndarray
            Raw voltage traces [n_electrodes, n_samples] in μV.

        Returns
        -------
        spike_train : np.ndarray
            Binary spike train [n_electrodes, n_bins], dtype uint8.
        lfp : np.ndarray
            Downsampled LFP [n_electrodes, n_lfp_samples], dtype float32.
        """
        n_electrodes, n_samples = voltage.shape
        n_bins = n_samples // self._samples_per_bin

        # ── Spike band filtering via MNE RawArray ────────────────────────
        ch_names = [f"E{i:03d}" for i in range(n_electrodes)]
        ch_types = ["eeg"] * n_electrodes
        info = mne.create_info(ch_names=ch_names, sfreq=self.fs, ch_types=ch_types)

        # MNE expects V, not μV
        raw = mne.io.RawArray(voltage * 1e-6, info, verbose=False)
        raw.filter(
            l_freq=self.hp,
            h_freq=self.lp,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            verbose=False,
        )
        filtered_spike = (raw.get_data() * 1e6).astype(np.float32)  # back to μV

        # ── Spike detection (threshold crossing, negative deflection) ─────
        spike_train = self._detect_spikes_binary(filtered_spike, n_bins)

        # ── LFP extraction ────────────────────────────────────────────────
        raw_lfp = mne.io.RawArray(voltage * 1e-6, info, verbose=False)
        raw_lfp.filter(l_freq=1.0, h_freq=self.lfp_lp, method="iir", verbose=False)
        raw_lfp.resample(sfreq=self.fs // self.lfp_ds, verbose=False)
        lfp = (raw_lfp.get_data() * 1e6).astype(np.float32)

        return spike_train, lfp

    def _detect_spikes_binary(
        self,
        filtered: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """
        Convert filtered voltage to binary spike train by threshold crossing.

        Parameters
        ----------
        filtered : np.ndarray
            Bandpass-filtered voltage [n_electrodes, n_samples], μV.
        n_bins : int
            Number of output bins.

        Returns
        -------
        np.ndarray
            Binary spike train [n_electrodes, n_bins], uint8.
        """
        n_electrodes, n_samples = filtered.shape
        spike_train = np.zeros((n_electrodes, n_bins), dtype=np.uint8)
        spb = self._samples_per_bin

        for e in range(n_electrodes):
            signal = filtered[e]
            # Threshold crossing: signal drops below threshold
            crossings = np.where(
                (signal[:-1] >= self.threshold) & (signal[1:] < self.threshold)
            )[0]
            # Convert sample indices to bins
            bin_indices = crossings // spb
            # Only keep valid bins
            valid = bin_indices < n_bins
            spike_train[e, bin_indices[valid]] = 1

        return spike_train

    def process_h5_dataset(
        self,
        input_path: str,
        output_path: str,
        chunk_size_samples: int = 200000,
    ) -> None:
        """
        Batch-process all raw recordings in an HDF5 file.

        Iterates over samples in the HDF5 store and writes processed spike
        trains and LFP to a new HDF5 output file.

        Parameters
        ----------
        input_path : str
            Path to raw HDF5 file. Expects groups named 'sample_XXXX' with
            dataset 'voltage' [n_electrodes, n_samples] OR 'spikes' (pre-binned).
        output_path : str
            Path for processed output HDF5.
        chunk_size_samples : int
            Number of raw samples to process at a time (memory guard).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(input_path, "r") as inp, h5py.File(output_path, "w") as out_f:
            sample_keys = sorted([k for k in inp.keys() if k.startswith("sample_")])
            logger.info("Processing %d samples ...", len(sample_keys))

            for key in tqdm(sample_keys, desc="Preprocessing", leave=False):
                grp_in = inp[key]

                # If dataset already has binary spikes (synthetic), copy directly
                if "spikes" in grp_in and "voltage" not in grp_in:
                    spikes = grp_in["spikes"][:]
                    grp_out = out_f.create_group(key)
                    grp_out.create_dataset("spikes", data=spikes)
                    # Copy attributes
                    for attr_key, val in grp_in.attrs.items():
                        grp_out.attrs[attr_key] = val
                    continue

                # Process raw voltage in chunks
                voltage_all = grp_in["voltage"][:]  # [n_elec, n_samples]
                spike_train, lfp = self.process_voltage_trace(voltage_all)

                grp_out = out_f.create_group(key)
                grp_out.create_dataset(
                    "spikes",
                    data=spike_train,
                    chunks=(spike_train.shape[0], min(10000, spike_train.shape[1])),
                    compression="gzip",
                )
                grp_out.create_dataset(
                    "lfp",
                    data=lfp,
                    chunks=(lfp.shape[0], min(10000, lfp.shape[1])),
                    compression="gzip",
                )
                for attr_key, val in grp_in.attrs.items():
                    grp_out.attrs[attr_key] = val

        logger.info("Preprocessing complete → %s", output_path)


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    proc = MEAPreprocessor(
        sampling_rate=d["sampling_rate"],
        spike_threshold_uv=d["spike_threshold_uv"],
        hp_cutoff_hz=d["hp_cutoff_hz"],
        lp_cutoff_hz=d["lp_cutoff_hz"],
        lfp_cutoff_hz=d["lfp_cutoff_hz"],
        lfp_downsample_factor=d["lfp_downsample_factor"],
        bin_size_ms=d["bin_size_ms"],
    )
    proc.process_h5_dataset(
        input_path=d["local_path"],
        output_path=d["processed_path"],
    )


if __name__ == "__main__":
    main()
