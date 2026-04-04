# SpectraGuard

[![CI](https://github.com/kyb8801/spectraguard/actions/workflows/ci.yml/badge.svg)](https://github.com/kyb8801/spectraguard/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Uncertainty-aware spectral quality assessment for SERS, Raman, and IR spectroscopy.**

SpectraGuard evaluates spectral quality through six physically interpretable metrics,
combined into a single Confidence Score (0-100) with bootstrap uncertainty quantification.
It is the only open-source tool that reports not just *what* your spectral quality score is,
but *how much you can trust it*.

## Features

- **6-Metric Confidence Score** — SNR, Baseline Stability, Peak Reproducibility, Spectral Resolution, Fluorescence Contamination, Hotspot Uniformity
- **Uncertainty Quantification** — Bootstrap-based confidence intervals for every score
- **Adaptive Weight Calibration** — Auto-optimize metric weights for your specific dataset
- **Adaptive Baseline Correction** — Multi-scale/hybrid arPLS with automatic parameter tuning
- **Cross-Instrument Transfer** — Calibrate QC scores across different spectrometers
- **Real-time Streaming** — SPC-based live quality monitoring with trend detection and alerts
- **Multi-Modality** — SERS, Raman, IR with modality-specific configurations
- **CLI** — Command-line interface for single and batch analysis

## Installation

```bash
pip install spectraguard
```

Or from source:

```bash
git clone https://github.com/kyb8801/spectraguard.git
cd spectraguard
pip install -e ".[dev]"
```

## Quick Start

### Basic Analysis

```python
from spectraguard import QualityAnalyzer
import numpy as np

analyzer = QualityAnalyzer(modality='raman')
result = analyzer.analyze(spectrum, wavenumber)
print(f"Quality: {result['confidence']:.1f} ({result['grade']})")
# Quality: 72.3 (Good)
```

### With Uncertainty

```python
analyzer = QualityAnalyzer(modality='raman', uncertainty=True)
result = analyzer.analyze(spectrum, wavenumber)
print(f"Quality: {result['confidence']:.1f} ± {result['confidence_std']:.1f}")
print(f"95% CI: {result['confidence_ci']}")
print(f"Grade stability: {result['grade_stability']:.0%}")
# Quality: 72.3 ± 2.1
# 95% CI: (68.2, 76.4)
# Grade stability: 94%
```

### Adaptive Weight Calibration

```python
analyzer = QualityAnalyzer(modality='raman')
weights = analyzer.calibrate_weights(reference_spectra, wavenumber, method='gmm')
# Weights auto-optimized for your data
result = analyzer.analyze(spectrum, wavenumber)
```

### Real-time Monitoring

```python
from spectraguard import StreamingAnalyzer

monitor = StreamingAnalyzer(modality='sers', alert_threshold=40.0)
monitor.on_alert(lambda a: print(f"ALERT: {a['type']}"))

for spectrum in spectrum_stream:
    result = monitor.push(spectrum, wavenumber)
    print(f"Score: {result['confidence']:.1f} | Trend: {result['trend']}")
```

### CLI

```bash
spectraguard analyze spectrum.csv --modality raman --uncertainty
spectraguard batch spectra_folder/ --output results.csv
```

## Supported Modalities

| Modality | SNR Ref | FWHM Range | Modality-Specific Metric |
|----------|---------|------------|--------------------------|
| SERS     | 50      | 10-50 cm-1 | Hotspot Uniformity       |
| Raman    | 100     | 5-30 cm-1  | -                        |
| IR       | 200     | 5-20 cm-1  | -                        |

## Examples

See the [examples/](examples/) directory for complete working examples:

- `01_basic_analysis.py` — Single spectrum analysis
- `02_uncertainty_analysis.py` — Bootstrap uncertainty estimation
- `03_batch_analysis.py` — Batch processing multiple spectra
- `04_streaming_qc.py` — Real-time monitoring with SPC
- `05_weight_calibration.py` — Data-driven weight calibration

## Architecture

```
spectraguard/
├── src/
│   ├── metrics/        # 6 individual metrics + uncertainty + adaptive baseline
│   ├── calibration/    # Cross-instrument transfer
│   ├── streaming/      # Real-time analyzer + SPC controller
│   ├── quality_analyzer.py
│   └── utils/          # Synthetic data generation
├── spectraguard/       # PyPI package wrapper + CLI
├── tests/              # 105 tests
├── examples/           # Usage examples
└── benchmarks/         # Comprehensive benchmark suite
```

## Citation

If you use SpectraGuard in your research, please cite:

```bibtex
@software{spectraguard2026,
  author = {Kim, Yongbeom},
  title = {SpectraGuard: Uncertainty-aware spectral quality assessment},
  year = {2026},
  url = {https://github.com/kyb8801/spectraguard}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
