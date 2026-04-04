"""
Unit tests for 6-Metric Confidence Score modules.

Tests use synthetic spectra with known properties to verify each metric.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.metrics.snr import calculate_snr, calculate_snr_raw
from src.metrics.baseline import calculate_baseline, arPLS
from src.metrics.peak_reproducibility import calculate_peak_reproducibility
from src.metrics.resolution import calculate_resolution
from src.metrics.fluorescence import calculate_fluorescence
from src.metrics.hotspot_uniformity import calculate_hotspot_uniformity
from src.metrics.confidence_score import ConfidenceScorer
from src.utils.synthetic_data import lorentzian


# --- Fixtures ---

@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def clean_spectrum(wavenumber):
    """High-quality spectrum: sharp peaks, no noise, flat baseline."""
    spec = np.zeros_like(wavenumber)
    for x0, amp, gamma in [(1000, 1.0, 8), (1400, 0.8, 7), (1600, 0.9, 9)]:
        spec += lorentzian(wavenumber, x0, amp, gamma)
    return spec


@pytest.fixture
def noisy_spectrum(wavenumber, clean_spectrum):
    """Very noisy spectrum with SNR ~5."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.2, len(wavenumber))
    return clean_spectrum + noise


@pytest.fixture
def pure_noise(wavenumber):
    """Pure random noise, no signal."""
    rng = np.random.default_rng(123)
    return rng.normal(0, 1.0, len(wavenumber))


@pytest.fixture
def fluorescent_spectrum(wavenumber, clean_spectrum):
    """Spectrum with strong fluorescence background."""
    x_norm = (wavenumber - 1500) / 500
    fluorescence = 3.0 * np.exp(-0.5 * x_norm**2)
    return clean_spectrum + fluorescence


# --- SNR Tests ---

class TestSNR:
    def test_clean_spectrum_high_snr(self, clean_spectrum, wavenumber):
        score = calculate_snr(clean_spectrum, wavenumber)
        assert score > 0.8, f"Clean spectrum should have high SNR score, got {score}"

    def test_noisy_spectrum_low_snr(self, noisy_spectrum, wavenumber):
        score = calculate_snr(noisy_spectrum, wavenumber)
        assert score < 0.8, f"Noisy spectrum should have lower SNR score, got {score}"

    def test_pure_noise_very_low(self, pure_noise, wavenumber):
        score = calculate_snr(pure_noise, wavenumber)
        assert score < 0.5, f"Pure noise should have low SNR score, got {score}"

    def test_score_range(self, clean_spectrum, wavenumber):
        score = calculate_snr(clean_spectrum, wavenumber)
        assert 0 <= score <= 1, f"Score must be in [0,1], got {score}"

    def test_known_snr(self, wavenumber):
        """Create spectrum with known SNR and verify measurement."""
        spec = np.zeros_like(wavenumber)
        spec += lorentzian(wavenumber, 1000, 1.0, 8)
        raw_snr = calculate_snr_raw(spec, wavenumber)
        assert raw_snr > 10, f"Clean Lorentzian should have high raw SNR, got {raw_snr}"


# --- Baseline Tests ---

class TestBaseline:
    def test_flat_baseline_high_score(self, clean_spectrum, wavenumber):
        score = calculate_baseline(clean_spectrum, wavenumber)
        assert score > 0.5, f"Flat baseline should score well, got {score}"

    def test_drifted_baseline(self, wavenumber):
        """Spectrum with strong polynomial drift."""
        x_norm = (wavenumber - wavenumber.min()) / (wavenumber.max() - wavenumber.min())
        drift = 2.0 * x_norm**2 + 1.5 * x_norm
        spec = drift + lorentzian(wavenumber, 1000, 0.5, 8)
        score = calculate_baseline(spec, wavenumber)
        # arPLS should handle this, so score shouldn't be terrible
        assert 0 <= score <= 1

    def test_arpls_output_shape(self, clean_spectrum):
        baseline = arPLS(clean_spectrum)
        assert baseline.shape == clean_spectrum.shape

    def test_score_range(self, clean_spectrum, wavenumber):
        score = calculate_baseline(clean_spectrum, wavenumber)
        assert 0 <= score <= 1


# --- Peak Reproducibility Tests ---

class TestPeakReproducibility:
    def test_clean_peaks_high_score(self, clean_spectrum, wavenumber):
        score = calculate_peak_reproducibility(clean_spectrum, wavenumber)
        assert score > 0.3, f"Clean peaks should have decent reproducibility, got {score}"

    def test_multi_spectrum_identical(self, clean_spectrum, wavenumber):
        """Identical spectra should have perfect reproducibility."""
        group = np.tile(clean_spectrum, (5, 1))
        score = calculate_peak_reproducibility(clean_spectrum, wavenumber,
                                               spectra_group=group)
        assert score > 0.8, f"Identical spectra should have high repro, got {score}"

    def test_multi_spectrum_varied(self, wavenumber):
        """Randomly varied spectra should have lower reproducibility."""
        rng = np.random.default_rng(42)
        group = []
        for _ in range(5):
            spec = np.zeros_like(wavenumber)
            for x0, amp, gamma in [(1000, 1.0, 8), (1400, 0.8, 7)]:
                spec += lorentzian(wavenumber, x0 + rng.normal(0, 10),
                                   amp * rng.uniform(0.5, 1.5), gamma)
            group.append(spec)
        group = np.array(group)
        score = calculate_peak_reproducibility(group[0], wavenumber,
                                               spectra_group=group)
        assert score < 0.9

    def test_score_range(self, clean_spectrum, wavenumber):
        score = calculate_peak_reproducibility(clean_spectrum, wavenumber)
        assert 0 <= score <= 1


# --- Resolution Tests ---

class TestResolution:
    def test_sharp_peaks_high_score(self, wavenumber):
        """Narrow Lorentzian (FWHM ~16 cm^-1) should score high."""
        spec = lorentzian(wavenumber, 1000, 1.0, 8)  # gamma=8 → FWHM≈16
        score = calculate_resolution(spec, wavenumber)
        assert score > 0.5, f"Sharp peaks should have high resolution score, got {score}"

    def test_broad_peaks_low_score(self, wavenumber):
        """Broad Lorentzian (FWHM ~80 cm^-1) should score low."""
        spec = lorentzian(wavenumber, 1000, 1.0, 40)  # gamma=40 → FWHM≈80
        score = calculate_resolution(spec, wavenumber)
        assert score < 0.5, f"Broad peaks should have low resolution score, got {score}"

    def test_score_range(self, clean_spectrum, wavenumber):
        score = calculate_resolution(clean_spectrum, wavenumber)
        assert 0 <= score <= 1


# --- Fluorescence Tests ---

class TestFluorescence:
    def test_clean_spectrum_high_score(self, clean_spectrum, wavenumber):
        score = calculate_fluorescence(clean_spectrum, wavenumber)
        assert score > 0.3, f"Clean spectrum should have low fluorescence, got {score}"

    def test_fluorescent_spectrum_low_score(self, fluorescent_spectrum, wavenumber):
        score = calculate_fluorescence(fluorescent_spectrum, wavenumber)
        assert score < 0.8, f"Fluorescent spectrum should score lower, got {score}"

    def test_score_range(self, clean_spectrum, wavenumber):
        score = calculate_fluorescence(clean_spectrum, wavenumber)
        assert 0 <= score <= 1


# --- Hotspot Uniformity Tests ---

class TestHotspotUniformity:
    def test_uniform_spectrum(self, wavenumber):
        """Constant spectrum → perfect uniformity."""
        spec = np.ones_like(wavenumber)
        score = calculate_hotspot_uniformity(spec, wavenumber)
        assert score > 0.9, f"Uniform spectrum should score high, got {score}"

    def test_non_uniform_spectrum(self, wavenumber):
        """Spectrum with one very intense region → low uniformity."""
        spec = np.ones_like(wavenumber) * 0.01
        spec[100:200] = 10.0
        score = calculate_hotspot_uniformity(spec, wavenumber)
        assert score < 0.8, f"Non-uniform spectrum should score lower, got {score}"

    def test_multi_point_identical(self, wavenumber, clean_spectrum):
        """Identical multi-point spectra → high uniformity."""
        multi = np.tile(clean_spectrum, (5, 1))
        score = calculate_hotspot_uniformity(clean_spectrum, wavenumber,
                                             multi_spectra=multi)
        assert score > 0.9

    def test_score_range(self, clean_spectrum, wavenumber):
        score = calculate_hotspot_uniformity(clean_spectrum, wavenumber)
        assert 0 <= score <= 1


# --- Integrated Confidence Score Tests ---

class TestConfidenceScore:
    def test_clean_spectrum_excellent(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        result = scorer.score(clean_spectrum, wavenumber)
        assert result['confidence'] > 40, f"Clean spectrum confidence too low: {result['confidence']}"
        assert result['grade'] in ('Excellent', 'Good', 'Marginal')
        assert all(0 <= v <= 1 for v in result['scores'].values())

    def test_pure_noise_poor(self, pure_noise, wavenumber):
        scorer = ConfidenceScorer()
        result = scorer.score(pure_noise, wavenumber)
        assert result['confidence'] < 60, f"Pure noise confidence too high: {result['confidence']}"

    def test_all_metrics_present(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        result = scorer.score(clean_spectrum, wavenumber)
        expected_keys = {'snr', 'baseline', 'peak', 'resolution',
                         'fluorescence', 'uniformity'}
        assert set(result['scores'].keys()) == expected_keys

    def test_weights_sum_to_one(self):
        scorer = ConfidenceScorer()
        assert abs(sum(scorer.weights.values()) - 1.0) < 1e-10

    def test_batch_scoring(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        spectra = np.tile(clean_spectrum, (3, 1))
        results = scorer.score_batch(spectra, wavenumber)
        assert len(results) == 3
        assert all('confidence' in r for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
