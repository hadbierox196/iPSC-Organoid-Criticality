"""
Tests for power-law fitting, DFA, bifurcation detection, and statistics.
All CPU-safe with synthetic data.
"""

import numpy as np
import pytest

from analysis.power_law import PowerLawFitter, check_scaling_relation
from analysis.dfa import compute_dfa, interpret_hurst
from analysis.bifurcation import BifurcationDetector
from analysis.statistics import GenotypeStatistics


# ─── Power-Law Tests ─────────────────────────────────────────────────────────

class TestPowerLawFitter:
    @pytest.fixture
    def power_law_data(self):
        """Generate synthetic power-law distributed data (τ ≈ 1.5)."""
        rng = np.random.default_rng(0)
        xmin = 1
        tau = 1.5
        u = rng.uniform(0, 1, 1000)
        return xmin * (1.0 - u) ** (1.0 / (1.0 - tau))

    @pytest.fixture
    def avalanche_sizes(self):
        """Simulate critical avalanche size distribution."""
        rng = np.random.default_rng(42)
        # Power-law with τ ≈ 1.5, integer-valued
        sizes = []
        for _ in range(500):
            s = max(1, int(np.abs(rng.standard_cauchy() * 5) + 1))
            sizes.append(s)
        return np.array(sizes)

    def test_fit_returns_result(self, avalanche_sizes):
        fitter = PowerLawFitter(n_bootstrap=20, discrete=True)
        result = fitter.fit(avalanche_sizes)
        assert not np.isnan(result.exponent)
        assert result.exponent > 1.0
        assert result.n_tail > 0

    def test_exponent_in_plausible_range(self, power_law_data):
        fitter = PowerLawFitter(xmin_method=1.0, n_bootstrap=20, discrete=False)
        result = fitter.fit(power_law_data)
        # τ should be close to 1.5 for power-law with these params
        assert 1.0 < result.exponent < 4.0, f"Exponent {result.exponent} out of range"

    def test_ci_bounds_order(self, avalanche_sizes):
        fitter = PowerLawFitter(n_bootstrap=30)
        result = fitter.fit(avalanche_sizes)
        if not np.isnan(result.ci_lower):
            assert result.ci_lower < result.ci_upper

    def test_scaling_relation(self, avalanche_sizes):
        fitter = PowerLawFitter(n_bootstrap=20)
        size_res = fitter.fit(avalanche_sizes)
        dur_data = np.clip(avalanche_sizes * 0.7, 1, None)
        dur_res = fitter.fit(dur_data)
        scale = check_scaling_relation(size_res.exponent, dur_res.exponent, size_res, dur_res)
        assert "scaling_ratio" in scale

    def test_insufficient_data(self):
        fitter = PowerLawFitter()
        result = fitter.fit(np.array([1, 2, 3]))  # < 10 samples
        assert np.isnan(result.exponent)


# ─── DFA Tests ───────────────────────────────────────────────────────────────

class TestDFA:
    @pytest.fixture
    def white_noise(self):
        """White noise → H ≈ 0.5."""
        rng = np.random.default_rng(0)
        return rng.standard_normal(2000)

    @pytest.fixture
    def correlated_signal(self):
        """1/f noise → H ≈ 1.0."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        # Integrate to create 1/f-like signal
        return np.cumsum(x - x.mean())

    def test_returns_dict(self, white_noise):
        result = compute_dfa(white_noise)
        assert "H" in result
        assert "scales" in result
        assert "fluctuations" in result
        assert "r2" in result

    def test_white_noise_H_near_half(self, white_noise):
        result = compute_dfa(white_noise, scales=[8, 16, 32, 64, 128])
        H = result["H"]
        if not np.isnan(H):
            assert 0.2 < H < 0.85, f"White noise H={H} unexpected"

    def test_scales_and_fluctuations_same_length(self, white_noise):
        result = compute_dfa(white_noise)
        assert len(result["scales"]) == len(result["fluctuations"])

    def test_short_signal_returns_nan(self):
        result = compute_dfa(np.random.randn(10))
        assert np.isnan(result["H"]) or len(result["scales"]) == 0

    def test_interpret_hurst(self):
        assert "sub" in interpret_hurst(0.4).lower()
        assert "critical" in interpret_hurst(0.9).lower()
        assert "super" in interpret_hurst(1.3).lower() or "strong" in interpret_hurst(1.3).lower()


# ─── Bifurcation Tests ────────────────────────────────────────────────────────

class TestBifurcationDetector:
    divs = [0, 7, 14, 21, 28, 35, 42]
    sigma_control = np.array([0.3, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0])
    sigma_disc1   = np.array([0.3, 0.5, 0.7, 0.82, 0.88, 0.90, 0.93])

    def test_detect_critical_div_control(self):
        det = BifurcationDetector()
        result = det.analyze(self.divs, self.sigma_control)
        assert 14.0 <= result.div_critical <= 21.0, f"div_c={result.div_critical}"

    def test_disc1_no_crossing(self):
        det = BifurcationDetector()
        result = det.analyze(self.divs, self.sigma_disc1)
        # DISC1 never quite crosses 1.0 in our test data
        assert result.div_critical > 35.0 or np.isinf(result.div_critical)

    def test_ar1_trajectory_length(self):
        det = BifurcationDetector()
        result = det.analyze(self.divs, self.sigma_control)
        assert len(result.ar1_trajectory) == len(self.divs)

    def test_compare_genotypes(self):
        det = BifurcationDetector()
        r_ctrl = det.analyze(self.divs, self.sigma_control)
        r_d1   = det.analyze(self.divs, self.sigma_disc1)
        summary = det.compare_genotypes({"Control": r_ctrl, "DISC1": r_d1})
        assert "Control" in summary
        assert "DISC1" in summary
        assert summary["Control"]["div_critical"] < summary["DISC1"]["div_critical"]


# ─── Statistics Tests ─────────────────────────────────────────────────────────

class TestGenotypeStatistics:
    @pytest.fixture
    def groups(self):
        rng = np.random.default_rng(0)
        return {
            "Control": rng.normal(1.0, 0.05, 30),
            "DISC1":   rng.normal(0.85, 0.05, 30),
            "SHANK3":  rng.normal(1.15, 0.05, 30),
        }

    def test_omnibus_returns_result(self, groups):
        stats = GenotypeStatistics()
        results = stats.compare_groups(groups, metric_name="sigma")
        assert "omnibus" in results
        assert results["omnibus"].pvalue < 0.001

    def test_pairwise_returns_pairs(self, groups):
        stats = GenotypeStatistics()
        results = stats.compare_groups(groups)
        assert any("pairwise" in k for k in results)

    def test_summary_stats_complete(self, groups):
        stats = GenotypeStatistics()
        summ = stats.summary_stats(groups)
        for g in groups:
            assert g in summ
            assert "mean" in summ[g]
            assert "n" in summ[g]
