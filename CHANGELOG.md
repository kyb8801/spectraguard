# Changelog

## [0.2.0] - 2024-12-XX

### Added
- Uncertainty quantification via bootstrap resampling (`UncertaintyEstimator`)
- Adaptive weight calibration using variance, GMM, and supervised methods (`WeightCalibrator`)
- Adaptive baseline correction with multi-scale and hybrid approaches (`AdaptiveBaseline`)
- Cross-instrument transfer calibration with ICC computation (`InstrumentCalibrator`)
- Real-time streaming analyzer with SPC monitoring (`StreamingAnalyzer`, `SPC_Controller`)
- CLI tool with `analyze`, `batch`, and `benchmark` subcommands
- PyPI-ready package structure with `spectraguard` namespace

### Changed
- Updated arPLS baseline to support adaptive mode
- Improved fluorescence metric sensitivity (threshold 1.0 → 2.0, filter_size //3)
- Enhanced peak reproducibility with R² validation and SNR pre-check
- Fixed SNR metric for zero-signal spectra (returns 0.0 instead of 1.0)
- Fixed baseline metric for zero-range spectra (returns 0.0 instead of 1.0)
- Balanced benchmark scenarios (3 Bad + 3 Good = 50:50)

### Fixed
- `np.trapz` → `np.trapezoid` for NumPy 2.x compatibility
- All-zeros spectrum now correctly scores near 0 instead of ~54

## [0.1.0] - 2024-11-XX

### Added
- Initial 6-metric confidence scoring framework
- SNR, Baseline Stability, Peak Reproducibility, Resolution, Fluorescence, Hotspot Uniformity
- Multi-modality support (SERS, Raman, IR)
- Synthetic data generator
- Streamlit dashboard
- RRUFF real data analysis pipeline
- Publication figures and manuscript
