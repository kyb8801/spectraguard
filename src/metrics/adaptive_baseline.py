"""
Adaptive baseline correction with auto-parameter selection.

Extends arPLS baseline estimation by automatically selecting parameters
based on spectrum characteristics (SNR, peak count, gradient variability,
broad feature ratio).

Methods:
- adaptive_arpls: Auto-select arPLS parameters from spectrum features
- multi_scale: Try multiple lambda values, pick best
- hybrid: Adaptive arPLS + peak masking + re-estimation
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Dict, Tuple, Literal
from src.metrics.baseline import arPLS


class AdaptiveBaseline:
    """Adaptive baseline correction with auto-parameter selection."""

    def __init__(self, method='adaptive_arpls'):
        """method: 'adaptive_arpls', 'multi_scale', or 'hybrid'"""
        self.method = method

    def estimate_parameters(self, spectrum, wavenumber):
        """
        Extract spectrum features and determine optimal arPLS parameters.

        Features:
        1. Spectrum length
        2. SNR estimate (SG residual)
        3. Peak count
        4. Gradient variability (std of 1st derivative)
        5. Broad feature ratio (low-freq energy / total energy)

        Rules:
        - High SNR + few peaks -> large lam (1e6-1e7)
        - Low SNR + many peaks -> smaller lam (1e4-1e5)
        - High broad ratio -> increase lam, decrease p
        - High gradient variability -> decrease lam

        Returns dict with: lam, p, max_iter, features
        """
        n = len(spectrum)
        win = min(21, n // 4 * 2 + 1)
        if win < 5:
            win = 5
        poly = min(3, win - 1)
        smoothed = savgol_filter(spectrum, win, poly)

        # Feature extraction
        noise_std = np.std(spectrum - smoothed)
        signal_max = np.max(np.abs(smoothed)) - np.median(np.abs(smoothed))
        snr_est = signal_max / (noise_std + 1e-10)

        height_thr = np.max(smoothed) * 0.1
        peaks, _ = find_peaks(smoothed, height=max(height_thr, 0), distance=10)
        n_peaks = len(peaks)

        gradient = np.gradient(spectrum)
        grad_variability = np.std(gradient) / (np.mean(np.abs(gradient)) + 1e-10)

        # Broad feature ratio via FFT
        fft_vals = np.abs(np.fft.rfft(spectrum - np.mean(spectrum)))
        n_freq = len(fft_vals)
        low_freq_cutoff = max(1, n_freq // 10)
        broad_ratio = np.sum(fft_vals[:low_freq_cutoff]) / (np.sum(fft_vals) + 1e-10)

        features = {
            'length': n,
            'snr_estimate': float(snr_est),
            'n_peaks': n_peaks,
            'gradient_variability': float(grad_variability),
            'broad_feature_ratio': float(broad_ratio),
        }

        # Parameter determination
        base_lam = 1e5
        if snr_est > 50:
            base_lam *= 10
        elif snr_est < 10:
            base_lam *= 0.1

        if n_peaks > 20:
            base_lam *= 2
        elif n_peaks < 3:
            base_lam *= 0.5

        if grad_variability > 2.0:
            base_lam *= 0.5

        if broad_ratio > 0.5:
            base_lam *= 2

        base_lam = np.clip(base_lam, 1e3, 1e8)

        p = 0.01
        if broad_ratio > 0.6:
            p = 0.05
        elif snr_est < 10:
            p = 0.02

        return {
            'lam': float(base_lam),
            'p': float(p),
            'max_iter': 100,
            'features': features,
        }

    def correct(self, spectrum, wavenumber):
        """
        Perform adaptive baseline correction.
        Returns: (corrected, baseline, info_dict)
        """
        if self.method == 'adaptive_arpls':
            params = self.estimate_parameters(spectrum, wavenumber)
            baseline = arPLS(spectrum, lam=params['lam'])
            corrected = spectrum - baseline
            return corrected, baseline, params
        elif self.method == 'multi_scale':
            baseline = self._multi_scale_baseline(spectrum)
            corrected = spectrum - baseline
            return corrected, baseline, {'method': 'multi_scale'}
        elif self.method == 'hybrid':
            return self._hybrid_correct(spectrum, wavenumber)
        else:
            baseline = arPLS(spectrum)
            return spectrum - baseline, baseline, {}

    def _multi_scale_baseline(self, spectrum):
        """Try multiple lam values, pick smoothest that stays below peaks."""
        lam_values = [1e3, 1e4, 1e5, 1e6, 1e7]
        baselines = []
        scores = []
        for lam in lam_values:
            bl = arPLS(spectrum, lam=lam)
            baselines.append(bl)
            # Score: penalize baseline going above spectrum, reward smoothness
            above_penalty = np.sum(np.maximum(bl - spectrum, 0))
            smoothness = np.std(np.diff(np.diff(bl)))
            score = -above_penalty - smoothness * 0.01
            scores.append(score)
        best_idx = np.argmax(scores)
        return baselines[best_idx]

    def _hybrid_correct(self, spectrum, wavenumber):
        """
        1. Adaptive arPLS for initial baseline
        2. Mask peak regions
        3. Re-estimate baseline using only non-peak regions
        """
        params = self.estimate_parameters(spectrum, wavenumber)
        baseline_init = arPLS(spectrum, lam=params['lam'])

        # Find peaks to create mask
        corrected_init = spectrum - baseline_init
        win = min(21, len(spectrum) // 4 * 2 + 1)
        if win < 5:
            win = 5
        smoothed = savgol_filter(corrected_init, win, min(3, win - 1))
        height_thr = np.max(smoothed) * 0.1
        peaks, properties = find_peaks(
            smoothed, height=max(height_thr, 0), distance=10
        )

        # Create peak mask (peaks +/- width in points)
        mask = np.ones(len(spectrum), dtype=bool)  # True = non-peak
        for p in peaks:
            width = 30  # default width in points
            lo = max(0, p - width)
            hi = min(len(spectrum), p + width)
            mask[lo:hi] = False

        # Re-estimate baseline using non-peak regions
        if mask.sum() > 50:
            # Interpolate baseline through non-peak regions
            from scipy.interpolate import interp1d

            x_nonpeak = np.where(mask)[0]
            y_nonpeak = spectrum[mask]
            # Smooth interpolation
            try:
                f = interp1d(
                    x_nonpeak, y_nonpeak,
                    kind='linear', fill_value='extrapolate',
                )
                baseline_interp = f(np.arange(len(spectrum)))
                # Apply arPLS to the interpolated baseline for smoothing
                baseline_final = arPLS(baseline_interp, lam=params['lam'] * 10)
            except Exception:
                baseline_final = baseline_init
        else:
            baseline_final = baseline_init

        corrected = spectrum - baseline_final
        params['method'] = 'hybrid'
        return corrected, baseline_final, params
