"""
Tests for data downloader, preprocessor, avalanche extractor, and dataset.
All tests use synthetic data; no real MEA files required.
CPU-safe (no GPU needed).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data.downloader import generate_synthetic_dataset, _branching_process
from data.avalanche_extractor import AvalancheExtractor
from data.dataset import OrganoiMEADataset


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_h5(tmp_path_factory):
    """Generate a small synthetic HDF5 for testing."""
    tmp = tmp_path_factory.mktemp("data")
    path = str(tmp / "test_organoid.h5")
    generate_synthetic_dataset(
        output_path=path,
        n_organoids_per_genotype=2,
        recording_duration_s=30.0,
        n_electrodes=16,
        sampling_rate=20000,
        bin_size_ms=1.0,
        seed=0,
    )
    return path


# ─── Downloader Tests ─────────────────────────────────────────────────────────

class TestBranchingProcess:
    def test_shape(self):
        rng = np.random.default_rng(42)
        spikes = _branching_process(n_neurons=32, n_steps=1000, sigma=1.0,
                                    spontaneous_rate=0.01, rng=rng)
        assert spikes.shape == (32, 1000)
        assert spikes.dtype == bool

    def test_subcritical_lower_activity(self):
        rng = np.random.default_rng(0)
        sub = _branching_process(32, 2000, sigma=0.5, spontaneous_rate=0.005, rng=rng)
        rng = np.random.default_rng(0)
        sup = _branching_process(32, 2000, sigma=1.5, spontaneous_rate=0.005, rng=rng)
        assert sub.mean() < sup.mean()

    def test_synthetic_h5_created(self, synthetic_h5):
        assert Path(synthetic_h5).exists()
        assert Path(synthetic_h5).stat().st_size > 0


# ─── Avalanche Extractor Tests ────────────────────────────────────────────────

class TestAvalancheExtractor:
    @pytest.fixture
    def spike_train(self):
        rng = np.random.default_rng(42)
        spikes = _branching_process(64, 10000, sigma=1.0, spontaneous_rate=0.005, rng=rng)
        return spikes.astype(np.uint8)

    def test_extract_returns_avalanches(self, spike_train):
        ext = AvalancheExtractor(threshold=0, min_size=2)
        result = ext.extract(spike_train)
        assert len(result.sizes) > 0
        assert len(result.durations) > 0
        assert len(result.sizes) == len(result.durations)

    def test_branching_ratio_subcritical(self):
        rng = np.random.default_rng(1)
        spikes = _branching_process(64, 20000, sigma=0.5, spontaneous_rate=0.005, rng=rng)
        ext = AvalancheExtractor()
        result = ext.extract(spikes.astype(np.uint8))
        # Subcritical: sigma should be < 1
        assert result.branching_ratio < 1.2, f"Expected sigma<1.2, got {result.branching_ratio}"

    def test_branching_ratio_supercritical(self):
        rng = np.random.default_rng(2)
        spikes = _branching_process(64, 20000, sigma=1.5, spontaneous_rate=0.01, rng=rng)
        ext = AvalancheExtractor()
        result = ext.extract(spikes.astype(np.uint8))
        assert result.branching_ratio > 0.8, f"Expected sigma>0.8, got {result.branching_ratio}"

    def test_avalanche_sizes_positive(self, spike_train):
        ext = AvalancheExtractor()
        result = ext.extract(spike_train)
        assert (result.sizes >= ext.min_size).all()
        assert (result.durations > 0).all()

    def test_synchrony_index_range(self, spike_train):
        ext = AvalancheExtractor()
        si = ext.compute_synchrony_index(spike_train.astype(np.float32))
        assert -1.0 <= si <= 1.0


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestOrganoiMEADataset:
    def test_dataset_len_positive(self, synthetic_h5):
        import h5py
        with h5py.File(synthetic_h5, "r") as f:
            keys = sorted(f.keys())[:4]
        ds = OrganoiMEADataset(synthetic_h5, keys, window_size_bins=100, stride_bins=50,
                               n_electrodes=16)
        assert len(ds) > 0

    def test_item_shapes(self, synthetic_h5):
        import h5py
        with h5py.File(synthetic_h5, "r") as f:
            keys = list(f.keys())[:2]
        ds = OrganoiMEADataset(synthetic_h5, keys, window_size_bins=200, stride_bins=100,
                               n_electrodes=16)
        item = ds[0]
        assert item["spikes"].shape == (16, 200)
        assert item["criticality_label"].item() in (0, 1, 2)
        assert 0 <= item["genotype_id"].item() <= 3

    def test_augmentation_does_not_change_shape(self, synthetic_h5):
        import h5py
        with h5py.File(synthetic_h5, "r") as f:
            keys = list(f.keys())[:2]
        ds = OrganoiMEADataset(synthetic_h5, keys, window_size_bins=100, stride_bins=50,
                               n_electrodes=16, augment=True)
        item = ds[0]
        assert item["spikes"].shape == (16, 100)

    def test_sigma_to_label_boundaries(self):
        assert OrganoiMEADataset._sigma_to_label(0.5) == 0
        assert OrganoiMEADataset._sigma_to_label(0.95) == 1
        assert OrganoiMEADataset._sigma_to_label(1.05) == 1
        assert OrganoiMEADataset._sigma_to_label(1.5) == 2
