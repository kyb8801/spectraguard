import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.metrics.adaptive_baseline import AdaptiveBaseline
from src.metrics.baseline import calculate_baseline
from src.utils.synthetic_data import lorentzian


@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def clean_spectrum(wavenumber):
    spec = np.zeros_like(wavenumber)
    for x0, amp, gamma in [(1000, 1.0, 8), (1400, 0.8, 7)]:
        spec += lorentzian(wavenumber, x0, amp, gamma)
    return spec


class TestAdaptiveBaseline:
    def test_adaptive_returns_tuple(self, clean_spectrum, wavenumber):
        ab = AdaptiveBaseline(method='adaptive_arpls')
        corrected, baseline, info = ab.correct(clean_spectrum, wavenumber)
        assert len(corrected) == len(clean_spectrum)
        assert len(baseline) == len(clean_spectrum)
        assert 'lam' in info

    def test_parameter_estimation_reasonable(self, clean_spectrum, wavenumber):
        ab = AdaptiveBaseline()
        params = ab.estimate_parameters(clean_spectrum, wavenumber)
        assert 1e3 <= params['lam'] <= 1e8
        assert 0.001 <= params['p'] <= 0.1

    def test_multi_scale(self, clean_spectrum, wavenumber):
        ab = AdaptiveBaseline(method='multi_scale')
        corrected, baseline, info = ab.correct(clean_spectrum, wavenumber)
        assert len(corrected) == len(clean_spectrum)

    def test_hybrid_preserves_peaks(self, clean_spectrum, wavenumber):
        ab = AdaptiveBaseline(method='hybrid')
        corrected, baseline, info = ab.correct(clean_spectrum, wavenumber)
        # Peaks should still be visible after correction
        assert np.max(corrected) > 0.1

    def test_backward_compatible_baseline(self, clean_spectrum, wavenumber):
        score_fixed = calculate_baseline(clean_spectrum, wavenumber, adaptive=False)
        assert 0 <= score_fixed <= 1

    def test_adaptive_baseline_flag(self, clean_spectrum, wavenumber):
        score_adaptive = calculate_baseline(clean_spectrum, wavenumber, adaptive=True)
        assert 0 <= score_adaptive <= 1

    def test_drifted_spectrum(self, wavenumber):
        spec = np.zeros_like(wavenumber)
        for x0, amp, gamma in [(1000, 1.0, 8)]:
            spec += lorentzian(wavenumber, x0, amp, gamma)
        # Add strong drift
        spec += 0.5 * (wavenumber - 200) / 3000
        ab = AdaptiveBaseline(method='adaptive_arpls')
        corrected, baseline, info = ab.correct(spec, wavenumber)
        # After correction, drift should be reduced
        corrected_slope = np.polyfit(np.arange(len(corrected)), corrected, 1)[0]
        original_slope = np.polyfit(np.arange(len(spec)), spec, 1)[0]
        assert abs(corrected_slope) < abs(original_slope)
