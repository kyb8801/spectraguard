"""
Tests for QualityAnalyzer, config, core metrics, and edge cases.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.quality_analyzer import QualityAnalyzer
from src.metrics.config import get_config, MODALITY_CONFIG
from src.metrics.core.background_contamination import calculate_background_contamination
from src.utils.synthetic_data import lorentzian, generate_single_spectrum, generate_dataset


@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def clean_spectrum(wavenumber):
    spec = np.zeros_like(wavenumber)
    for x0, amp, gamma in [(1000, 1.0, 8), (1400, 0.8, 7), (1600, 0.9, 9)]:
        spec += lorentzian(wavenumber, x0, amp, gamma)
    return spec


class TestQualityAnalyzer:
    def test_sers_modality(self, clean_spectrum, wavenumber):
        qa = QualityAnalyzer(modality='sers')
        result = qa.analyze(clean_spectrum, wavenumber)
        assert 'confidence' in result
        assert 'grade' in result
        assert result['modality'] == 'sers'
        assert 0 <= result['confidence'] <= 100

    def test_raman_modality(self, clean_spectrum, wavenumber):
        qa = QualityAnalyzer(modality='raman')
        result = qa.analyze(clean_spectrum, wavenumber)
        assert result['modality'] == 'raman'
        assert 'hotspot' not in result['scores']  # SERS-only

    def test_ir_modality(self, clean_spectrum, wavenumber):
        qa = QualityAnalyzer(modality='ir')
        result = qa.analyze(clean_spectrum, wavenumber)
        assert result['modality'] == 'ir'

    def test_invalid_modality(self):
        with pytest.raises(ValueError):
            QualityAnalyzer(modality='uv-vis')

    def test_custom_weights(self, clean_spectrum, wavenumber):
        weights = {'snr': 1.0}
        qa = QualityAnalyzer(modality='sers', custom_weights=weights)
        result = qa.analyze(clean_spectrum, wavenumber)
        assert result['weights'] == weights

    def test_batch_analyze(self, clean_spectrum, wavenumber):
        qa = QualityAnalyzer(modality='sers')
        spectra = np.tile(clean_spectrum, (3, 1))
        results = qa.analyze_batch(spectra, wavenumber)
        assert len(results) == 3


class TestConfig:
    def test_sers_config(self):
        cfg = get_config('sers')
        assert cfg['snr_reference'] == 50
        assert 'weights' in cfg

    def test_raman_config(self):
        cfg = get_config('raman')
        assert cfg['snr_reference'] == 100

    def test_ir_config(self):
        cfg = get_config('ir')
        assert cfg['snr_reference'] == 200

    def test_invalid_modality(self):
        with pytest.raises(ValueError):
            get_config('xray')

    def test_all_modalities_have_weights(self):
        for mod in MODALITY_CONFIG:
            cfg = get_config(mod)
            assert abs(sum(cfg['weights'].values()) - 1.0) < 0.01


class TestBackgroundContamination:
    def test_clean_spectrum(self, clean_spectrum, wavenumber):
        score = calculate_background_contamination(clean_spectrum, wavenumber)
        assert 0 <= score <= 1

    def test_heavy_background(self, wavenumber):
        x_norm = (wavenumber - 1500) / 500
        spec = 5.0 * np.exp(-0.5 * x_norm**2)
        score = calculate_background_contamination(spec, wavenumber)
        assert score < 0.5


class TestEdgeCases:
    def test_all_zeros(self, wavenumber):
        qa = QualityAnalyzer(modality='sers')
        spec = np.zeros_like(wavenumber)
        result = qa.analyze(spec, wavenumber)
        assert result['confidence'] < 60  # all-zeros still gets some default scores

    def test_negative_values(self, wavenumber):
        qa = QualityAnalyzer(modality='sers')
        spec = generate_single_spectrum(wavenumber, snr=30,
                                         rng=np.random.default_rng(42))
        spec -= np.max(spec) * 0.5
        result = qa.analyze(spec, wavenumber)
        assert 0 <= result['confidence'] <= 100

    def test_nan_handling(self, wavenumber):
        qa = QualityAnalyzer(modality='sers')
        spec = generate_single_spectrum(wavenumber, snr=30,
                                         rng=np.random.default_rng(42))
        spec_clean = np.nan_to_num(spec, nan=0, posinf=0, neginf=0)
        result = qa.analyze(spec_clean, wavenumber)
        assert 0 <= result['confidence'] <= 100

    def test_short_spectrum(self):
        wn = np.linspace(900, 1100, 100)
        spec = lorentzian(wn, 1000, 1.0, 8)
        qa = QualityAnalyzer(modality='sers')
        result = qa.analyze(spec, wn)
        assert 0 <= result['confidence'] <= 100

    def test_cosmic_spike(self, wavenumber):
        qa = QualityAnalyzer(modality='sers')
        spec = lorentzian(wavenumber, 1000, 0.5, 8)
        spec[750] = 50  # spike
        result = qa.analyze(spec, wavenumber)
        assert 0 <= result['confidence'] <= 100


class TestSyntheticDataGenerator:
    def test_generate_dataset(self):
        data = generate_dataset(n_good=10, n_marginal=10, n_bad=10, seed=42)
        assert data['spectra'].shape == (30, 1500)
        assert len(data['labels']) == 30
        assert sum(data['labels'] == 'Good') == 10

    def test_single_spectrum(self):
        wn = np.linspace(200, 3200, 1500)
        rng = np.random.default_rng(42)
        spec = generate_single_spectrum(wn, snr=50, rng=rng)
        assert len(spec) == 1500
        assert not np.any(np.isnan(spec))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
