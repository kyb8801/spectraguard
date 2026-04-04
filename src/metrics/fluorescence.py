"""
Metric 5: Fluorescence Contamination Index

Physical basis:
- Fluorescence produces broad background that overwhelms Raman/SERS peaks
- Detected by comparing broad (low-frequency) vs sharp (high-frequency) components
- broad_component = uniform_filter1d(spectrum, size=200)
- fluorescence_ratio = std(broad) / (std(peak_component) + eps)
- High ratio → strong fluorescence contamination

Weight in confidence score: 0.10
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def calculate_fluorescence(spectrum: np.ndarray, wavenumber: np.ndarray,
                           filter_size: int = 200,
                           threshold: float = 2.0) -> float:
    """
    Calculate Fluorescence Contamination score.

    Parameters
    ----------
    spectrum : 1D array, intensity values
    wavenumber : 1D array, wavenumber axis
    filter_size : int, window for broad component extraction
    threshold : float, fluorescence_ratio at which score becomes 0

    Returns
    -------
    float : fluorescence contamination score (0-1, higher = less contamination)
    """
    filter_size = min(filter_size, len(spectrum) // 3)
    if filter_size < 3:
        filter_size = 3

    broad_component = uniform_filter1d(spectrum.astype(float), size=filter_size)
    peak_component = spectrum - broad_component

    std_broad = np.std(broad_component)
    std_peak = np.std(peak_component)

    # Both zero → flat/empty spectrum
    if std_peak < 1e-10 and std_broad < 1e-10:
        return 0.0

    if std_peak < 1e-10:
        return 0.0  # No peaks, only broad → pure fluorescence

    fluorescence_ratio = std_broad / (std_peak + 1e-10)
    score = max(0.0, 1.0 - fluorescence_ratio / threshold)
    return float(score)


def get_fluorescence_ratio(spectrum: np.ndarray, filter_size: int = 200) -> float:
    """Return raw fluorescence ratio (for diagnostics)."""
    filter_size = min(filter_size, len(spectrum) // 2)
    if filter_size < 3:
        filter_size = 3
    broad = uniform_filter1d(spectrum.astype(float), size=filter_size)
    peak = spectrum - broad
    return float(np.std(broad) / (np.std(peak) + 1e-10))
