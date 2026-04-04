"""
Metric 1: Signal-to-Noise Ratio (SNR)

Physical basis:
- SERS spectra with SNR > 50 are considered excellent quality
- Signal is measured from Savitzky-Golay filtered peak regions
- Noise is measured from peak-free regions (e.g., 2000-2200 cm^-1)
- SNR = max(|signal|) / std(noise)

Weight in confidence score: 0.25
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks


def calculate_snr(spectrum: np.ndarray, wavenumber: np.ndarray,
                  sg_window: int = 21, sg_polyorder: int = 3,
                  noise_region: tuple = (2000, 2200),
                  snr_reference: float = 50.0) -> float:
    """
    Calculate Signal-to-Noise Ratio score.

    Parameters
    ----------
    spectrum : 1D array, intensity values
    wavenumber : 1D array, wavenumber axis (cm^-1)
    sg_window : int, Savitzky-Golay window length
    sg_polyorder : int, Savitzky-Golay polynomial order
    noise_region : tuple, (min, max) wavenumber range for noise estimation
    snr_reference : float, SNR value considered excellent (score=1.0)

    Returns
    -------
    float : SNR score normalized to 0-1
    """
    # Ensure odd window
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window > len(spectrum):
        sg_window = max(5, len(spectrum) // 4 * 2 + 1)
    if sg_polyorder >= sg_window:
        sg_polyorder = sg_window - 1

    # Smooth spectrum
    smoothed = savgol_filter(spectrum, sg_window, sg_polyorder)

    # Signal: max peak intensity
    signal_max = np.max(np.abs(smoothed))

    # No signal (flat/zero spectrum) → score 0
    if signal_max < 1e-10:
        return 0.0

    # Noise region
    noise_mask = (wavenumber >= noise_region[0]) & (wavenumber <= noise_region[1])

    if noise_mask.sum() < 10:
        # Fallback: use residual between raw and smoothed as noise estimate
        residual = spectrum - smoothed
        noise_std = np.std(residual)
    else:
        noise_data = spectrum[noise_mask] - smoothed[noise_mask]
        noise_std = np.std(noise_data)
        if noise_std < 1e-10:
            # If noise region is too clean, use full residual
            noise_std = np.std(spectrum - smoothed)

    if noise_std < 1e-10:
        return 1.0  # essentially noiseless

    snr = signal_max / noise_std
    score = min(1.0, snr / snr_reference)
    return float(max(0.0, score))


def calculate_snr_raw(spectrum: np.ndarray, wavenumber: np.ndarray,
                      **kwargs) -> float:
    """Return raw SNR value (not normalized)."""
    sg_window = kwargs.get('sg_window', 21)
    sg_polyorder = kwargs.get('sg_polyorder', 3)
    noise_region = kwargs.get('noise_region', (2000, 2200))

    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window > len(spectrum):
        sg_window = max(5, len(spectrum) // 4 * 2 + 1)
    if sg_polyorder >= sg_window:
        sg_polyorder = sg_window - 1

    smoothed = savgol_filter(spectrum, sg_window, sg_polyorder)
    signal_max = np.max(np.abs(smoothed))

    if signal_max < 1e-10:
        return 0.0

    noise_mask = (wavenumber >= noise_region[0]) & (wavenumber <= noise_region[1])
    if noise_mask.sum() < 10:
        noise_std = np.std(spectrum - smoothed)
    else:
        noise_std = np.std(spectrum[noise_mask] - smoothed[noise_mask])
        if noise_std < 1e-10:
            noise_std = np.std(spectrum - smoothed)

    if noise_std < 1e-10:
        return float('inf')
    return float(signal_max / noise_std)
