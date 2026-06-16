"""
Tests for CriticalityNet, SynchronyEncoder, and TranscriptionalMapper.
All tests run on CPU with minimal tensor sizes.
"""

import pytest
import torch

from models.criticality_net import CriticalityNet
from models.synchrony_encoder import SynchronyEncoder
from models.transcription_mapper import TranscriptionalMapper
from models.utils import count_parameters, memory_summary, set_seed


@pytest.fixture(autouse=True)
def seed():
    set_seed(42)


# ─── CriticalityNet Tests ─────────────────────────────────────────────────────

class TestCriticalityNet:
    B, C, T = 4, 16, 200

    @pytest.fixture
    def model(self):
        return CriticalityNet(n_electrodes=self.C, n_time_bins=self.T,
                              hidden_dim=32, n_tcn_layers=2, dropout=0.0)

    @pytest.fixture
    def spikes(self):
        return torch.randint(0, 2, (self.B, self.C, self.T)).float()

    def test_forward_output_keys(self, model, spikes):
        out = model(spikes)
        assert "criticality_logits" in out
        assert "branching_ratio" in out
        assert "genotype_logits" in out
        assert "features" in out

    def test_output_shapes(self, model, spikes):
        out = model(spikes)
        assert out["criticality_logits"].shape == (self.B, 3)
        assert out["branching_ratio"].shape == (self.B, 1)
        assert out["genotype_logits"].shape == (self.B, 4)

    def test_branching_ratio_range(self, model, spikes):
        out = model(spikes)
        sigma = out["branching_ratio"]
        assert (sigma >= 0.0).all()
        assert (sigma <= 2.0).all()

    def test_parameter_count_reasonable(self, model):
        n = count_parameters(model)
        assert 1_000 < n < 20_000_000, f"Param count unexpected: {n}"

    def test_memory_summary(self, model):
        summary = memory_summary(model, (self.C, self.T))
        assert summary["param_mb"] < 500

    def test_from_config(self):
        cfg = {"n_electrodes": self.C, "n_time_bins": self.T,
               "hidden_dim": 32, "n_tcn_layers": 2, "tcn_kernel_size": 3,
               "dropout": 0.0, "n_criticality_classes": 3,
               "n_genotype_classes": 4, "use_genotype_head": True,
               "use_branching_ratio_head": True}
        model = CriticalityNet.from_config(cfg)
        spikes = torch.zeros(2, self.C, self.T)
        out = model(spikes)
        assert out["criticality_logits"].shape[0] == 2

    def test_backward_runs(self, model, spikes):
        model.train()
        out = model(spikes)
        loss = out["criticality_logits"].mean() + out["branching_ratio"].mean()
        loss.backward()
        # Check at least one gradient exists
        for p in model.parameters():
            if p.grad is not None:
                return
        pytest.fail("No gradients computed.")

    def test_no_genotype_head(self):
        model = CriticalityNet(n_electrodes=self.C, n_time_bins=self.T,
                               hidden_dim=16, n_tcn_layers=2,
                               use_genotype_head=False, use_branching_ratio_head=False)
        spikes = torch.zeros(2, self.C, self.T)
        out = model(spikes)
        assert "genotype_logits" not in out
        assert "branching_ratio" not in out


# ─── SynchronyEncoder Tests ───────────────────────────────────────────────────

class TestSynchronyEncoder:
    B, C, T = 4, 16, 100

    @pytest.fixture
    def model(self):
        return SynchronyEncoder(n_electrodes=self.C, n_time_bins=self.T,
                                node_feat_dim=8, sync_dim=16, n_graph_layers=1)

    @pytest.fixture
    def spikes(self):
        return torch.rand(self.B, self.C, self.T)

    def test_output_shapes(self, model, spikes):
        out = model(spikes)
        assert out["sync_embedding"].shape == (self.B, 16)
        assert out["sync_index"].shape == (self.B, 1)

    def test_sync_index_range(self, model, spikes):
        out = model(spikes)
        assert (out["sync_index"] >= 0).all()
        assert (out["sync_index"] <= 1).all()


# ─── TranscriptionalMapper Tests ─────────────────────────────────────────────

class TestTranscriptionalMapper:
    B = 8

    @pytest.fixture
    def model(self):
        return TranscriptionalMapper(n_elec_features=8, n_gene_sets=16, latent_dim=8,
                                     hidden_dim=32, n_genotype_classes=4)

    @pytest.fixture
    def elec_feat(self):
        return torch.randn(self.B, 8)

    def test_forward_output_shapes(self, model, elec_feat):
        out = model(elec_feat)
        assert out["latent"].shape == (self.B, 8)
        assert out["pred_gene_scores"].shape == (self.B, 16)
        assert out["genotype_logits"].shape == (self.B, 4)
        assert out["recon_elec"].shape == (self.B, 8)


# ─── Loss Tests ───────────────────────────────────────────────────────────────

class TestCriticalityLoss:
    def test_loss_positive(self):
        from training.losses import CriticalityLoss
        loss_fn = CriticalityLoss()
        B = 4
        outputs = {
            "criticality_logits": torch.randn(B, 3),
            "branching_ratio": torch.sigmoid(torch.randn(B, 1)) * 2,
            "genotype_logits": torch.randn(B, 4),
        }
        crit_labels = torch.randint(0, 3, (B,))
        sigma_targets = torch.rand(B) * 2
        geno_labels = torch.randint(0, 4, (B,))
        total, loss_dict = loss_fn(outputs, crit_labels, sigma_targets, geno_labels)
        assert total.item() > 0
        assert "loss_criticality" in loss_dict
        assert "loss_sigma" in loss_dict
        assert "loss_genotype" in loss_dict
        assert "loss_consistency" in loss_dict
