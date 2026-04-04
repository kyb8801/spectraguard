"""
Core Metric 5: Background Contamination Index

Generalized version of Fluorescence Contamination for all spectroscopic modalities.
- SERS/Raman: fluorescence background
- IR: water vapor / CO2 absorption bands
- UV-Vis: scattering baseline

Compares broad (low-frequency) vs sharp (high-frequency) spectral components.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def calculate_background_contamination(spectrum: np.ndarray, wavenumber: np.ndarray,
                                       filter_size: int = 200,
                                       threshold: float = 1.0) -> float:
    """
    Calculate Background Contamination score.

    Parameters
    ----------
    spectrum : 1D array, intensity values
    wavenumber : 1D array, wavenumber axis
    filter_size : int, window for broad component extraction
    threshold : float, contamination ratio at which score = 0

    Returns
    -------
    float : score (0-1), higher = less contamination
    """
    filter_size = min(filter_size, len(spectrum) // 3)
    if filter_size < 3:
        filter_size = 3

    broad_component = uniform_filter1d(spectrum.astype(float), size=filter_size)
    peak_component = spectrum - broad_component

    std_broad = np.std(broad_component)
    std_peak = np.std(peak_component)

    if std_peak < 1e-10 and std_broad < 1e-10:
        return 0.0

    if std_peak < 1e-10:
        return 0.0

    contamination_ratio = std_broad / (std_peak + 1e-10)
    score = max(0.0, 1.0 - contamination_ratio / threshold)
    return float(score)
