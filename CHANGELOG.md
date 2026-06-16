# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2024-01-01

### Added
- Initial repository scaffold with full MEA organoid criticality pipeline
- Multi-scale TCN model (CriticalityNet) for criticality state classification
- Power-law fitting via MLE (Clauset et al. 2009) with KS-test validation
- Detrended Fluctuation Analysis (DFA) for Hurst exponent estimation
- Branching process synthetic data generator (Brian2)
- Developmental bifurcation detection via change-point analysis
- Cross-genotype statistical comparison (DISC1, SHANK3, 22q11.2del, Control)
- Google Colab-optimized training loop (AMP, gradient accumulation, checkpointing)
- 4 production Jupyter notebooks (Setup, EDA, Training, Evaluation)
- Full pytest test suite (CPU-safe, synthetic data)
- GitHub Actions CI workflow (2-epoch dry run)
- JCNS-compatible figure export (84mm, 300 dpi, Arial)
