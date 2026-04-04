"""
Metric 4: Spectral Resolution (FWHM)

Physical basis:
- SERS peaks typically have FWHM of 10-30 cm^-1
- Broader peaks indicate poor resolution or overlapping bands
- Lorentzian fit: L(x) = A * gamma^2 / ((x - x0)^2 + gamma^2), FWHM = 2*gamma
- Score: 1.0 at FWHM=10, 0.0 at FWHM>=50

Weight in confidence score: 0.15
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from lmfit.models import LorentzianModel


def calculate_resolution(spectrum: np.ndarray, wavenumber: np.ndarray,
                         fwhm_optimal: float = 10.0,
                         fwhm_worst: float = 50.0,
                         max_peaks: int = 10) -> float:
    """
    Calculate Spectral Resolution score based on average FWHM.

    Parameters
    ----------
    spectrum : 1D array, intensity values
    wavenumber : 1D array, wavenumber axis
    fwhm_optimal : float, FWHM considered perfect (cm^-1)
    fwhm_worst : float, FWHM considered unacceptable (cm^-1)
    max_peaks : int, maximum peaks to analyze

    Returns
    -------
    float : resolution score (0-1)
    """
    win = min(21, len(spectrum) // 4 * 2 + 1)
    if win < 5:
        win = 5
    smoothed = savgol_filter(spectrum, win, min(3, win - 1))

    height_threshold = np.max(smoothed) * 0.1
    peaks, properties = find_peaks(smoothed, height=height_threshold,
                                   distance=20, prominence=height_threshold * 0.3)

    if len(peaks) == 0:
        return 0.3  # Can't evaluate

    # Sort by intensity, take top peaks
    peak_heights = smoothed[peaks]
    sorted_idx = np.argsort(peak_heights)[::-1]
    peaks = peaks[sorted_idx[:max_peaks]]

    fwhm_values = []
    for peak_idx in peaks:
        peak_wn = wavenumber[peak_idx]
        fwhm = _fit_peak_fwhm(wavenumber, smoothed, peak_wn)
        if fwhm is not None and 1 < fwhm < 200:
            fwhm_values.append(fwhm)

    if not fwhm_values:
        return 0.3

    avg_fwhm = np.mean(fwhm_values)
    score = max(0.0, 1.0 - (avg_fwhm - fwhm_optimal) / (fwhm_worst - fwhm_optimal))
    return float(min(1.0, score))


def _fit_peak_fwhm(wavenumber: np.ndarray, spectrum: np.ndarray,
                   center: float, fit_width: float = 40) -> float:
    """Fit Lorentzian and extract FWHM."""
    mask = np.abs(wavenumber - center) < fit_width
    if mask.sum() < 5:
        return None

    x = wavenumber[mask]
    y = spectrum[mask]
    y_offset = np.min(y)
    y_shifted = y - y_offset

    model = LorentzianModel()
    params = model.guess(y_shifted, x=x)

    try:
        result = model.fit(y_shifted, params, x=x)
        if result.success:
            sigma = result.params['sigma'].value
            fwhm = 2 * sigma  # Lorentzian FWHM = 2*sigma in lmfit
            return abs(fwhm)
    except Exception:
        pass

    return None


def get_peak_fwhms(spectrum: np.ndarray, wavenumber: np.ndarray) -> list:
    """Return list of (peak_position, fwhm) tuples for all detected peaks."""
    win = min(21, len(spectrum) // 4 * 2 + 1)
    if win < 5:
        win = 5
    smoothed = savgol_filter(spectrum, win, min(3, win - 1))

    height_threshold = np.max(smoothed) * 0.1
    peaks, _ = find_peaks(smoothed, height=height_threshold, distance=20)

    results = []
    for peak_idx in peaks:
        peak_wn = wavenumber[peak_idx]
        fwhm = _fit_peak_fwhm(wavenumber, smoothed, peak_wn)
        if fwhm is not None:
            results.append((peak_wn, fwhm))

    return results
