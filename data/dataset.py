"""
PyTorch Dataset for iPSC Organoid MEA Data.

Provides memory-mapped, lazy HDF5 loading suitable for Colab Free Tier
(avoids loading entire dataset into RAM). Supports stratified train/val/test
splits by organoid line (not by window) to prevent data leakage.

Usage
-----
from data.dataset import OrganoiMEADataset, build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(cfg)
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class OrganoiMEADataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for MEA organoid spike trains.

    Loads one window at a time from HDF5 without caching the full array,
    preventing OOM errors on Colab.

    Parameters
    ----------
    h5_path : str
        Path to processed HDF5 file (output of preprocessor.py).
    sample_keys : list[str]
        List of HDF5 group keys to include.
    window_size_bins : int
        Number of bins per input window.
    stride_bins : int
        Stride between consecutive windows.
    n_electrodes : int
        Number of MEA electrodes (channels).
    augment : bool
        If True, apply random electrode dropout (training only).
    dropout_rate : float
        Fraction of electrodes to randomly zero-out during augmentation.
    """

    def __init__(
        self,
        h5_path: str,
        sample_keys: list[str],
        window_size_bins: int = 1000,
        stride_bins: int = 500,
        n_electrodes: int = 64,
        augment: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:
        self.h5_path = h5_path
        self.sample_keys = sample_keys
        self.window_size = window_size_bins
        self.stride = stride_bins
        self.n_electrodes = n_electrodes
        self.augment = augment
        self.dropout_rate = dropout_rate

        # Build index: (sample_key, window_start_bin)
        self._index: list[tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Pre-compute all (sample_key, window_start) pairs."""
        with h5py.File(self.h5_path, "r") as f:
            for key in self.sample_keys:
                n_bins = f[key]["spikes"].shape[1]
                starts = range(0, n_bins - self.window_size + 1, self.stride)
                self._index.extend([(key, s) for s in starts])
        logger.debug("Dataset index: %d windows from %d samples.", len(self._index), len(self.sample_keys))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Load one window from HDF5.

        Returns
        -------
        dict with keys:
            'spikes'      : FloatTensor [n_electrodes, window_size]
            'genotype_id' : LongTensor scalar
            'div'         : FloatTensor scalar
            'true_sigma'  : FloatTensor scalar
            'criticality_label' : LongTensor scalar (0=sub, 1=crit, 2=super)
        """
        key, start = self._index[idx]

        with h5py.File(self.h5_path, "r") as f:
            grp = f[key]
            spikes = grp["spikes"][:, start: start + self.window_size].astype(np.float32)
            genotype_id = int(grp.attrs["genotype_id"])
            div = float(grp.attrs["div"])
            true_sigma = float(grp.attrs.get("true_sigma", 0.0))

        # Pad if last window is short
        if spikes.shape[1] < self.window_size:
            pad_w = self.window_size - spikes.shape[1]
            spikes = np.pad(spikes, ((0, 0), (0, pad_w)))

        # Pad / crop electrodes
        if spikes.shape[0] < self.n_electrodes:
            spikes = np.pad(spikes, ((0, self.n_electrodes - spikes.shape[0]), (0, 0)))
        elif spikes.shape[0] > self.n_electrodes:
            spikes = spikes[: self.n_electrodes]

        # Augmentation: random electrode dropout
        if self.augment:
            mask = np.random.rand(self.n_electrodes) > self.dropout_rate
            spikes = spikes * mask[:, np.newaxis]

        # Criticality label from sigma
        crit_label = self._sigma_to_label(true_sigma)

        return {
            "spikes": torch.from_numpy(spikes),
            "genotype_id": torch.tensor(genotype_id, dtype=torch.long),
            "div": torch.tensor(div, dtype=torch.float32),
            "true_sigma": torch.tensor(true_sigma, dtype=torch.float32),
            "criticality_label": torch.tensor(crit_label, dtype=torch.long),
        }

    @staticmethod
    def _sigma_to_label(sigma: float) -> int:
        """
        Convert branching ratio σ to 3-class criticality label.

        Returns
        -------
        int
            0 = subcritical (σ < 0.90)
            1 = critical    (0.90 ≤ σ ≤ 1.10)
            2 = supercritical (σ > 1.10)
        """
        if sigma < 0.90:
            return 0
        elif sigma <= 1.10:
            return 1
        else:
            return 2


def build_dataloaders(
    cfg: dict,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build stratified train/val/test DataLoaders from a config dict.

    Splits by organoid_id to prevent data leakage across windows from the
    same organoid appearing in both train and test.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary (from YAML).
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    rng = np.random.default_rng(seed)
    h5_path = cfg["data"]["processed_path"]
    d = cfg["data"]
    train_cfg = cfg["training"]

    # Collect all sample keys and their (genotype, organoid_id)
    with h5py.File(h5_path, "r") as f:
        all_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
        meta: list[dict] = []
        for key in all_keys:
            meta.append({
                "key": key,
                "genotype": f[key].attrs["genotype"],
                "organoid_id": int(f[key].attrs["organoid_id"]),
            })

    # Stratified split by (genotype, organoid_id) → each organoid goes entirely to one split
    organoid_ids_by_genotype: dict[str, set[int]] = {}
    for m in meta:
        g = m["genotype"]
        organoid_ids_by_genotype.setdefault(g, set()).add(m["organoid_id"])

    train_keys, val_keys, test_keys = [], [], []
    for genotype, org_ids in organoid_ids_by_genotype.items():
        org_list = sorted(org_ids)
        rng.shuffle(org_list)  # type: ignore
        n = len(org_list)
        n_train = max(1, round(n * d["train_frac"]))
        n_val = max(1, round(n * d["val_frac"]))
        train_orgs = set(org_list[:n_train])
        val_orgs = set(org_list[n_train: n_train + n_val])
        test_orgs = set(org_list[n_train + n_val:])

        for m in meta:
            if m["genotype"] != genotype:
                continue
            if m["organoid_id"] in train_orgs:
                train_keys.append(m["key"])
            elif m["organoid_id"] in val_orgs:
                val_keys.append(m["key"])
            else:
                test_keys.append(m["key"])

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d sample groups.",
        len(train_keys), len(val_keys), len(test_keys),
    )

    window_bins = int(d["window_size_s"] * 1000.0 / d["bin_size_ms"])
    stride_bins = int(d["stride_size_s"] * 1000.0 / d["bin_size_ms"])

    train_ds = OrganoiMEADataset(h5_path, train_keys, window_bins, stride_bins,
                                  d["n_electrodes"], augment=True)
    val_ds   = OrganoiMEADataset(h5_path, val_keys,   window_bins, stride_bins,
                                  d["n_electrodes"], augment=False)
    test_ds  = OrganoiMEADataset(h5_path, test_keys,  window_bins, stride_bins,
                                  d["n_electrodes"], augment=False)

    loader_kwargs = dict(
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"],
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg.get("batch_size", 32),
                              shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
