import sys, os, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.metrics.weight_calibrator import WeightCalibrator
from src.metrics.confidence_score import ConfidenceScorer
from src.utils.synthetic_data import generate_single_spectrum

@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)

@pytest.fixture
def sample_spectra(wavenumber):
    rng = np.random.default_rng(42)
    spectra = []
    labels = []
    for _ in range(15):
        spectra.append(generate_single_spectrum(wavenumber, snr=60, rng=rng))
        labels.append('Good')
    for _ in range(15):
        spectra.append(generate_single_spectrum(wavenumber, snr=8, fluorescence_level=0.5, rng=rng))
        labels.append('Bad')
    return spectra, labels

class TestWeightCalibrator:
    def test_variance_sum_to_one(self, sample_spectra, wavenumber):
        spectra, _ = sample_spectra
        scorer = ConfidenceScorer()
        wc = WeightCalibrator(method='variance')
        weights = wc.calibrate(spectra, wavenumber, scorer)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_gmm_method(self, sample_spectra, wavenumber):
        spectra, _ = sample_spectra
        scorer = ConfidenceScorer()
        wc = WeightCalibrator(method='gmm')
        weights = wc.calibrate(spectra, wavenumber, scorer)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in weights.values())

    def test_supervised_with_labels(self, sample_spectra, wavenumber):
        spectra, labels = sample_spectra
        scorer = ConfidenceScorer()
        wc = WeightCalibrator(method='supervised')
        weights = wc.calibrate(spectra, wavenumber, scorer, labels=labels)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_supervised_requires_labels(self, sample_spectra, wavenumber):
        spectra, _ = sample_spectra
        scorer = ConfidenceScorer()
        wc = WeightCalibrator(method='supervised')
        with pytest.raises(ValueError):
            wc.calibrate(spectra, wavenumber, scorer)

    def test_compare_with_default(self, sample_spectra, wavenumber):
        spectra, _ = sample_spectra
        scorer = ConfidenceScorer()
        wc = WeightCalibrator(method='variance')
        wc.calibrate(spectra, wavenumber, scorer)
        default_w = {'snr': 0.25, 'baseline': 0.20, 'peak': 0.15, 'resolution': 0.15, 'fluorescence': 0.10, 'uniformity': 0.15}
        comparison = wc.compare_with_default(spectra, wavenumber, scorer, default_w)
        assert 'improvement_pct' in comparison

    def test_generate_report(self, sample_spectra, wavenumber):
        spectra, _ = sample_spectra
        scorer = ConfidenceScorer()
        wc = WeightCalibrator(method='variance')
        wc.calibrate(spectra, wavenumber, scorer)
        report = wc.generate_report()
        assert 'snr' in report
        assert 'Weight' in report
