"""
Metric 3: Peak Reproducibility

Physical basis:
- Reliable SERS spectra have symmetric, well-defined peaks
- Single spectrum mode: peak symmetry (left/right half-width ratio)
- Multi-spectrum mode: RSD of peak positions and intensities across repeats
- Target: RSD < 10% = excellent reproducibility

Weight in confidence score: 0.15
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from lmfit.models import LorentzianModel


def _fit_lorentzian_single(x: np.ndarray, y: np.ndarray,
                           center: float, width: int = 30):
    """Fit a single Lorentzian to a peak region."""
    # Extract region around peak
    mask = np.abs(x - center) < width
    if mask.sum() < 5:
        return None

    x_region = x[mask]
    y_region = y[mask]

    model = LorentzianModel()
    params = model.guess(y_region, x=x_region)
    try:
        result = model.fit(y_region, params, x=x_region)
        return result
    except Exception:
        return None


def calculate_peak_reproducibility(spectrum: np.ndarray, wavenumber: np.ndarray,
                                   spectra_group: np.ndarray = None,
                                   min_peaks: int = 3,
                                   peak_height_fraction: float = 0.1) -> float:
    """
    Calculate Peak Reproducibility score.

    Single spectrum: evaluates peak symmetry and Lorentzian fit quality.
    Multi-spectrum: evaluates RSD of peak positions/intensities.

    Parameters
    ----------
    spectrum : 1D array, primary spectrum
    wavenumber : 1D array, wavenumber axis
    spectra_group : 2D array (optional), repeat measurements for RSD calculation
    min_peaks : int, minimum peaks required for analysis
    peak_height_fraction : float, minimum peak height as fraction of max

    Returns
    -------
    float : reproducibility score (0-1)
    """
    if spectra_group is not None and len(spectra_group) > 1:
        return _multi_spectrum_mode(spectra_group, wavenumber, min_peaks)

    return _single_spectrum_mode(spectrum, wavenumber, min_peaks, peak_height_fraction)


def _single_spectrum_mode(spectrum, wavenumber, min_peaks, peak_height_fraction):
    """Evaluate peak symmetry and fit quality from a single spectrum."""
    # Smooth first
    win = min(21, len(spectrum) // 4 * 2 + 1)
    if win < 5:
        win = 5
    smoothed = savgol_filter(spectrum, win, min(3, win - 1))

    # Pre-check: if signal-to-noise is too low, peaks are unreliable
    noise_level = np.std(spectrum - smoothed)
    signal_level = np.max(np.abs(smoothed)) - np.median(np.abs(smoothed))
    if noise_level > 0 and signal_level / (noise_level + 1e-10) < 3.0:
        return 0.1  # SNR < 3 → cannot reliably identify peaks

    height_threshold = np.max(smoothed) * peak_height_fraction
    peaks, properties = find_peaks(smoothed, height=height_threshold,
                                   distance=20, prominence=height_threshold * 0.5)

    if len(peaks) < min_peaks:
        # Relax criteria
        peaks, _ = find_peaks(smoothed, height=height_threshold * 0.3, distance=10)

    if len(peaks) == 0:
        return 0.3  # Can't evaluate, return marginal

    symmetry_scores = []
    fit_scores = []

    for peak_idx in peaks[:10]:  # Limit to top 10 peaks
        peak_wn = wavenumber[peak_idx]
        result = _fit_lorentzian_single(wavenumber, smoothed, peak_wn)

        if result is not None and result.success:
            # Fit residual quality with R-squared validation
            residual_ratio = result.residual.std() / (np.max(result.best_fit) + 1e-10)
            fit_quality = max(0, 1 - residual_ratio * 8)

            # R-squared check
            ss_res = np.sum(result.residual**2)
            ss_tot = np.sum((result.data - np.mean(result.data))**2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
            fit_scores.append(fit_quality * max(0, r_squared))

            # Symmetry: compare left and right half-widths
            center = result.params['center'].value
            sigma = result.params['sigma'].value
            if sigma > 0:
                left_hw = center - (center - sigma)
                right_hw = (center + sigma) - center
                sym_ratio = min(left_hw, right_hw) / (max(left_hw, right_hw) + 1e-10)
                symmetry_scores.append(sym_ratio)

    scores = fit_scores + symmetry_scores
    if not scores:
        return 0.3

    return float(np.clip(np.mean(scores), 0, 1))


def _multi_spectrum_mode(spectra_group, wavenumber, min_peaks):
    """Evaluate RSD across repeat measurements."""
    win = min(21, len(wavenumber) // 4 * 2 + 1)
    if win < 5:
        win = 5

    all_peak_positions = []
    all_peak_intensities = []

    for spec in spectra_group:
        smoothed = savgol_filter(spec, win, min(3, win - 1))
        height_threshold = np.max(smoothed) * 0.1
        peaks, _ = find_peaks(smoothed, height=height_threshold, distance=20)

        if len(peaks) > 0:
            peak_wns = wavenumber[peaks]
            peak_ints = smoothed[peaks]
            all_peak_positions.append(peak_wns)
            all_peak_intensities.append(peak_ints)

    if len(all_peak_positions) < 2:
        return 0.5

    # Match peaks across spectra (nearest neighbor within tolerance)
    ref_positions = all_peak_positions[0]
    position_rsds = []
    intensity_rsds = []

    for ref_pos in ref_positions:
        matched_positions = [ref_pos]
        matched_intensities = []

        for i, positions in enumerate(all_peak_positions):
            if i == 0:
                idx = np.argmin(np.abs(positions - ref_pos))
                matched_intensities.append(all_peak_intensities[i][idx])
                continue
            diffs = np.abs(positions - ref_pos)
            if np.min(diffs) < 15:  # 15 cm^-1 tolerance
                idx = np.argmin(diffs)
                matched_positions.append(positions[idx])
                matched_intensities.append(all_peak_intensities[i][idx])

        if len(matched_positions) >= 2:
            pos_rsd = np.std(matched_positions) / (np.mean(matched_positions) + 1e-10) * 100
            position_rsds.append(pos_rsd)
        if len(matched_intensities) >= 2:
            int_rsd = np.std(matched_intensities) / (np.mean(matched_intensities) + 1e-10) * 100
            intensity_rsds.append(int_rsd)

    if not position_rsds and not intensity_rsds:
        return 0.5

    avg_rsd = np.mean(position_rsds + intensity_rsds)
    score = max(0.0, 1.0 - avg_rsd / 20.0)  # RSD 20% → score 0
    return float(score)
