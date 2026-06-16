"""
Streaming downloader with SHA-256 integrity validation.

Supports chunked HTTP download from Zenodo or any direct-link URL.
Falls back to synthetic data generation if source == 'synthetic'.

Usage
-----
python -m data.downloader --config configs/default_config.yaml
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Public API ──────────────────────────────────────────────────────────────

def stream_download(
    url: str,
    dest_path: str,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 8192,
    timeout: int = 60,
) -> str:
    """
    Download a file in streaming chunks with progress bar and SHA-256 validation.

    Parameters
    ----------
    url : str
        Direct download URL.
    dest_path : str
        Local path where the file will be saved.
    expected_sha256 : str, optional
        If provided, validates the downloaded file's checksum.
    chunk_size : int
        Download chunk size in bytes (default: 8192).
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Absolute path to the downloaded file.

    Raises
    ------
    ValueError
        If SHA-256 checksum does not match expected value.
    requests.exceptions.RequestException
        On network failure.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("File already exists at %s — skipping download.", dest)
        if expected_sha256:
            _validate_checksum(dest, expected_sha256)
        return str(dest.resolve())

    logger.info("Downloading from %s → %s", url, dest)
    t0 = time.time()
    sha256 = hashlib.sha256()
    total_bytes = 0

    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))

            with open(dest, "wb") as fh:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    leave=False,
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            sha256.update(chunk)
                            total_bytes += len(chunk)
                            pbar.update(len(chunk))
    except requests.exceptions.RequestException as exc:
        logger.error("Download failed: %s", exc)
        if dest.exists():
            dest.unlink()
        raise

    elapsed = time.time() - t0
    logger.info(
        "Downloaded %.2f MB in %.1fs (%.1f MB/s).",
        total_bytes / 1e6,
        elapsed,
        total_bytes / 1e6 / elapsed,
    )

    if expected_sha256:
        computed = sha256.hexdigest()
        if computed != expected_sha256.lower():
            dest.unlink()
            raise ValueError(
                f"SHA-256 mismatch!\n  expected: {expected_sha256}\n  got:      {computed}"
            )
        logger.info("SHA-256 checksum validated ✓")

    return str(dest.resolve())


def generate_synthetic_dataset(
    output_path: str,
    n_organoids_per_genotype: int = 4,
    recording_duration_s: float = 120.0,
    n_electrodes: int = 64,
    sampling_rate: int = 20000,
    bin_size_ms: float = 1.0,
    seed: int = 42,
) -> str:
    """
    Generate a synthetic HDF5 dataset mimicking MEA organoid recordings.

    Produces branching-process spike trains for four genotypes, each with
    DIV-dependent branching ratio profiles characteristic of that genotype.

    Parameters
    ----------
    output_path : str
        Path to write the synthetic HDF5 file.
    n_organoids_per_genotype : int
        Number of organoid replicates per genotype.
    recording_duration_s : float
        Duration of each synthetic recording in seconds.
    n_electrodes : int
        Number of MEA electrodes (channels).
    sampling_rate : int
        Sampling rate in Hz (used to define metadata only; spike trains in ms).
    bin_size_ms : float
        Bin size for spike trains in milliseconds.
    seed : int
        Global random seed for reproducibility.

    Returns
    -------
    str
        Path to the generated HDF5 file.
    """
    import h5py

    rng = np.random.default_rng(seed)
    n_bins = int(recording_duration_s * 1000.0 / bin_size_ms)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Genotype sigma profiles: {genotype: {div: sigma}}
    sigma_profiles: dict[str, dict[int, float]] = {
        "Control":  {0: 0.3, 7: 0.6, 14: 0.9, 21: 1.0, 28: 1.0, 35: 1.0, 42: 1.0},
        "DISC1":    {0: 0.3, 7: 0.5, 14: 0.7, 21: 0.82, 28: 0.88, 35: 0.9, 42: 0.93},
        "SHANK3":   {0: 0.5, 7: 0.9, 14: 1.15, 21: 1.10, 28: 1.05, 35: 1.02, 42: 1.0},
        "22q11":    {0: 0.3, 7: 0.55, 14: 0.75, 21: 0.91, 28: 0.95, 35: 0.97, 42: 0.99},
    }
    genotype_ids = {"Control": 0, "DISC1": 1, "SHANK3": 2, "22q11": 3}
    div_timepoints = [0, 7, 14, 21, 28, 35, 42]

    logger.info("Generating synthetic dataset → %s", output)

    with h5py.File(output, "w") as f:
        f.attrs["description"] = "Synthetic iPSC organoid MEA data for testing"
        f.attrs["n_electrodes"] = n_electrodes
        f.attrs["sampling_rate"] = sampling_rate
        f.attrs["bin_size_ms"] = bin_size_ms
        f.attrs["n_bins"] = n_bins

        idx = 0
        for genotype, sigma_by_div in sigma_profiles.items():
            for org_id in range(n_organoids_per_genotype):
                for div in div_timepoints:
                    sigma = sigma_by_div[div]
                    spikes = _branching_process(
                        n_neurons=n_electrodes,
                        n_steps=n_bins,
                        sigma=sigma,
                        spontaneous_rate=0.005,
                        rng=rng,
                    )
                    grp_name = f"sample_{idx:04d}"
                    grp = f.create_group(grp_name)
                    grp.create_dataset(
                        "spikes",
                        data=spikes.astype(np.uint8),
                        chunks=(n_electrodes, min(n_bins, 10000)),
                        compression="gzip",
                        compression_opts=4,
                    )
                    grp.attrs["genotype"] = genotype
                    grp.attrs["genotype_id"] = genotype_ids[genotype]
                    grp.attrs["organoid_id"] = org_id
                    grp.attrs["div"] = div
                    grp.attrs["true_sigma"] = sigma
                    idx += 1

    logger.info("Synthetic dataset written: %d samples.", idx)
    return str(output.resolve())


# ─── Private Helpers ──────────────────────────────────────────────────────────

def _branching_process(
    n_neurons: int,
    n_steps: int,
    sigma: float,
    spontaneous_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a branching process spiking network.

    Parameters
    ----------
    n_neurons : int
        Number of neurons / MEA electrodes.
    n_steps : int
        Number of time bins.
    sigma : float
        Branching ratio (mean number of descendants per spike).
        sigma < 1 → subcritical; sigma = 1 → critical; sigma > 1 → supercritical.
    spontaneous_rate : float
        Probability of spontaneous spike per neuron per bin.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Boolean spike train [n_neurons, n_steps].
    """
    spikes = np.zeros((n_neurons, n_steps), dtype=bool)

    for t in range(1, n_steps):
        # Spontaneous activations
        active = rng.random(n_neurons) < spontaneous_rate
        # Descendant activations from previous time step
        n_prev_active = int(spikes[:, t - 1].sum())
        if n_prev_active > 0:
            desc_prob = min(sigma * n_prev_active / n_neurons, 1.0)
            active |= rng.random(n_neurons) < desc_prob
        spikes[:, t] = active

    return spikes


def _validate_checksum(path: Path, expected: str) -> None:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    computed = sha256.hexdigest()
    if computed != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}!\n  expected: {expected}\n  got: {computed}"
        )
    logger.info("SHA-256 validated ✓ for %s", path.name)


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Download or generate organoid MEA dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    source = cfg["data"]["source"]
    out_path = cfg["data"]["local_path"]

    if source == "synthetic":
        sc = cfg["data"]["synthetic"]
        generate_synthetic_dataset(
            output_path=out_path,
            n_organoids_per_genotype=sc.get("n_organoids_per_genotype", 4),
            recording_duration_s=sc.get("recording_duration_s", 120.0),
            n_electrodes=cfg["data"]["n_electrodes"],
            sampling_rate=cfg["data"]["sampling_rate"],
            bin_size_ms=cfg["data"]["bin_size_ms"],
            seed=cfg["project"]["seed"],
        )
    elif source == "zenodo":
        stream_download(
            url=cfg["data"]["zenodo_url"],
            dest_path=out_path,
            expected_sha256=cfg["data"].get("zenodo_sha256"),
            chunk_size=cfg["data"]["chunk_size"],
        )
    else:
        logger.info("Using local path: %s", out_path)
        if not Path(out_path).exists():
            logger.error("Local file not found: %s", out_path)
            sys.exit(1)


if __name__ == "__main__":
    main()
