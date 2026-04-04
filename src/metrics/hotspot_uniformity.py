"""
Metric 6: Hotspot Uniformity Index

Physical basis:
- SERS enhancement is highly localized at "hotspots"
- Uniform hotspot distribution → consistent spectra across the substrate
- Single spectrum proxy: divide spectrum into N segments, compare total intensities
- Multi-point data: compare spectra from different spatial positions
- RSD < 10% = excellent, > 30% = poor

Weight in confidence score: 0.15
Note: This metric is SERS-specific. For other modalities, use "spectral consistency".
"""

import numpy as np


def calculate_hotspot_uniformity(spectrum: np.ndarray, wavenumber: np.ndarray,
                                 n_segments: int = 10,
                                 rsd_threshold: float = 30.0,
                                 multi_spectra: np.ndarray = None) -> float:
    """
    Calculate Hotspot Uniformity score.

    Single spectrum: spectral consistency proxy (segment intensity RSD)
    Multi-point: spatial uniformity across measurement positions

    Parameters
    ----------
    spectrum : 1D array, intensity values
    wavenumber : 1D array, wavenumber axis
    n_segments : int, number of segments for single-spectrum mode
    rsd_threshold : float, RSD (%) at which score becomes 0
    multi_spectra : 2D array (optional), spectra from different positions

    Returns
    -------
    float : uniformity score (0-1)
    """
    if multi_spectra is not None and len(multi_spectra) > 1:
        return _multi_point_uniformity(multi_spectra, rsd_threshold)

    return _single_spectrum_proxy(spectrum, n_segments, rsd_threshold)


def _single_spectrum_proxy(spectrum: np.ndarray, n_segments: int,
                           rsd_threshold: float) -> float:
    """Evaluate spectral consistency by comparing segment intensities."""
    n = len(spectrum)
    segment_size = n // n_segments

    if segment_size < 2:
        return 0.5

    segment_intensities = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else n
        segment = spectrum[start:end]
        segment_intensities.append(np.sum(np.abs(segment)))

    segment_intensities = np.array(segment_intensities)
    mean_int = np.mean(segment_intensities)

    if mean_int < 1e-10:
        return 0.0

    rsd = np.std(segment_intensities) / mean_int * 100
    score = max(0.0, 1.0 - rsd / rsd_threshold)
    return float(score)


def _multi_point_uniformity(multi_spectra: np.ndarray,
                            rsd_threshold: float) -> float:
    """Evaluate uniformity across multiple spatial positions."""
    total_intensities = np.sum(np.abs(multi_spectra), axis=1)
    mean_int = np.mean(total_intensities)

    if mean_int < 1e-10:
        return 0.0

    rsd = np.std(total_intensities) / mean_int * 100
    score = max(0.0, 1.0 - rsd / rsd_threshold)
    return float(score)
