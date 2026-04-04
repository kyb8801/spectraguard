"""
Core 5 Metrics - universal spectroscopy quality metrics.

These metrics apply to all spectroscopic modalities:
1. SNR
2. Baseline Stability
3. Peak Reproducibility
4. Spectral Resolution
5. Background Contamination (generalized from fluorescence)
"""

from ..snr import calculate_snr
from ..baseline import calculate_baseline
from ..peak_reproducibility import calculate_peak_reproducibility
from ..resolution import calculate_resolution
from .background_contamination import calculate_background_contamination

CORE_METRICS = {
    'snr': calculate_snr,
    'baseline': calculate_baseline,
    'peak': calculate_peak_reproducibility,
    'resolution': calculate_resolution,
    'background': calculate_background_contamination,
}
