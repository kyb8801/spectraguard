"""
Cross-instrument calibration transfer for SERS measurements.

Provides:
- InstrumentProfile: builds a spectral characteristics profile from
  reference spectra (wavenumber shift, intensity scale, noise floor,
  spectral resolution).
- InstrumentCalibrator: registers instruments, builds transfer models,
  applies calibration, and computes ICC for inter-instrument agreement.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple


class InstrumentProfile:
    """Instrument spectral characteristics profile."""

    def __init__(self, name):
        self.name = name
        self.wavenumber_shift = 0.0
        self.intensity_scale = 1.0
        self.noise_floor = 0.0
        self.spectral_resolution = None
        self.n_reference_spectra = 0
        self._reference_peaks_measured = []

    def build_from_reference(self, spectra, wavenumber, reference_peaks=None):
        """
        Build profile from reference spectra with known peak positions.
        reference_peaks: list of known peak positions in cm-1 (e.g., [520.7] for Si)
        """
        self.n_reference_spectra = len(spectra)

        if reference_peaks and len(reference_peaks) > 0:
            # Detect actual peak positions and compute shift
            shifts = []
            for spec in spectra:
                win = min(21, len(spec) // 4 * 2 + 1)
                if win < 5:
                    win = 5
                smoothed = savgol_filter(spec, win, min(3, win - 1))
                for ref_peak in reference_peaks:
                    # Search near reference peak
                    search_mask = (
                        (wavenumber >= ref_peak - 20)
                        & (wavenumber <= ref_peak + 20)
                    )
                    if search_mask.sum() < 3:
                        continue
                    local_spec = smoothed[search_mask]
                    local_wn = wavenumber[search_mask]
                    peak_idx = np.argmax(local_spec)
                    measured_peak = local_wn[peak_idx]
                    shifts.append(measured_peak - ref_peak)
            if shifts:
                self.wavenumber_shift = float(np.median(shifts))

        # Estimate noise floor from residuals
        noise_levels = []
        for spec in spectra:
            win = min(21, len(spec) // 4 * 2 + 1)
            if win < 5:
                win = 5
            smoothed = savgol_filter(spec, win, min(3, win - 1))
            noise_levels.append(np.std(spec - smoothed))
        self.noise_floor = float(np.median(noise_levels))

        # Estimate intensity scale (relative to max=1)
        max_intensities = [np.max(spec) for spec in spectra]
        self.intensity_scale = float(np.median(max_intensities))

        # Estimate spectral resolution from peak FWHM
        fwhms = []
        for spec in spectra:
            win = min(21, len(spec) // 4 * 2 + 1)
            if win < 5:
                win = 5
            smoothed = savgol_filter(spec, win, min(3, win - 1))
            peaks_found, props = find_peaks(
                smoothed, height=np.max(smoothed) * 0.3, distance=20
            )
            for p in peaks_found[:5]:
                half_max = smoothed[p] / 2
                # Find FWHM
                left = p
                while left > 0 and smoothed[left] > half_max:
                    left -= 1
                right = p
                while right < len(smoothed) - 1 and smoothed[right] > half_max:
                    right += 1
                if right > left + 1:
                    fwhm = wavenumber[right] - wavenumber[left]
                    if 1 < fwhm < 100:
                        fwhms.append(fwhm)
        if fwhms:
            self.spectral_resolution = float(np.median(fwhms))


class InstrumentCalibrator:
    """Cross-instrument calibration transfer."""

    def __init__(self):
        self.source_profile = None
        self.target_profile = None
        self.transfer_model = None

    def register_instrument(self, name, spectra, wavenumber, reference_peaks=None):
        profile = InstrumentProfile(name)
        profile.build_from_reference(spectra, wavenumber, reference_peaks)
        return profile

    def build_transfer(self, source, target, paired_spectra=None):
        """Build transfer model between source and target instruments."""
        self.source_profile = source
        self.target_profile = target

        wn_correction = target.wavenumber_shift - source.wavenumber_shift
        int_correction = source.intensity_scale / (target.intensity_scale + 1e-10)

        self.transfer_model = {
            'method': 'profile',
            'wavenumber_correction': float(wn_correction),
            'intensity_correction': float(int_correction),
            'noise_adjustment': float(source.noise_floor - target.noise_floor),
        }

        if paired_spectra is not None and len(paired_spectra) > 3:
            self.transfer_model['method'] = 'paired'
            # Could add linear regression here for more precise correction

        return self.transfer_model

    def transfer_spectrum(self, spectrum, wavenumber):
        """Apply calibration transfer to a target instrument spectrum."""
        if self.transfer_model is None:
            raise ValueError("Call build_transfer() first")

        corrected_wn = wavenumber - self.transfer_model['wavenumber_correction']
        corrected_spec = spectrum * self.transfer_model['intensity_correction']

        return corrected_spec, corrected_wn

    def transfer_score(self, spectrum, wavenumber, scorer):
        """Score a target spectrum with source-instrument calibration."""
        corrected_spec, corrected_wn = self.transfer_spectrum(spectrum, wavenumber)
        return scorer.score(corrected_spec, corrected_wn)

    def compute_icc(self, source_scores, target_scores):
        """ICC(2,1) - Two-way random, single measures, absolute agreement."""
        a = np.array(source_scores)
        b = np.array(target_scores)
        n = len(a)
        k = 2

        data = np.column_stack([a, b])
        grand_mean = np.mean(data)
        row_means = np.mean(data, axis=1)
        col_means = np.mean(data, axis=0)

        ss_total = np.sum((data - grand_mean) ** 2)
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n - 1) if n > 1 else 0
        ms_cols = ss_cols / (k - 1) if k > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

        denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
        icc = (ms_rows - ms_error) / denom if denom > 0 else 0
        icc = float(np.clip(icc, 0, 1))

        if icc >= 0.9:
            interp = 'Excellent'
        elif icc >= 0.75:
            interp = 'Good'
        elif icc >= 0.5:
            interp = 'Moderate'
        else:
            interp = 'Poor'

        return {'icc': icc, 'interpretation': interp}
