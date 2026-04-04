import sys, os, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.metrics.uncertainty import UncertaintyEstimator
from src.metrics.confidence_score import ConfidenceScorer
from src.utils.synthetic_data import lorentzian, generate_single_spectrum

@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)

@pytest.fixture
def clean_spectrum(wavenumber):
    spec = np.zeros_like(wavenumber)
    for x0, amp, gamma in [(1000, 1.0, 8), (1400, 0.8, 7), (1600, 0.9, 9)]:
        spec += lorentzian(wavenumber, x0, amp, gamma)
    return spec

class TestUncertaintyEstimator:
    def test_clean_spectrum_narrow_ci(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        ue = UncertaintyEstimator(n_bootstrap=50, seed=42)
        result = ue.estimate(clean_spectrum, wavenumber, scorer.score)
        ci_width = result['confidence_ci'][1] - result['confidence_ci'][0]
        assert ci_width < 10  # narrow CI for clean spectrum

    def test_noisy_spectrum_wide_ci(self, wavenumber):
        rng = np.random.default_rng(42)
        spec = generate_single_spectrum(wavenumber, snr=5, rng=rng)
        scorer = ConfidenceScorer()
        ue = UncertaintyEstimator(n_bootstrap=50, seed=42)
        result = ue.estimate(spec, wavenumber, scorer.score)
        ci_width = result['confidence_ci'][1] - result['confidence_ci'][0]
        assert ci_width > 3  # wider CI for noisy spectrum

    def test_grade_stability_clean(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        ue = UncertaintyEstimator(n_bootstrap=50, seed=42)
        result = ue.estimate(clean_spectrum, wavenumber, scorer.score)
        assert result['grade_stability'] > 0.8

    def test_noise_models(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        for model in ['gaussian', 'poisson', 'mixed']:
            ue = UncertaintyEstimator(n_bootstrap=20, noise_model=model, seed=42)
            result = ue.estimate(clean_spectrum, wavenumber, scorer.score)
            assert 'confidence_std' in result
            assert result['confidence_std'] >= 0

    def test_backward_compatibility(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        result = scorer.score(clean_spectrum, wavenumber)
        assert 'confidence_std' not in result  # no uncertainty by default

    def test_bootstrap_reproducibility(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        ue1 = UncertaintyEstimator(n_bootstrap=30, seed=42)
        ue2 = UncertaintyEstimator(n_bootstrap=30, seed=42)
        r1 = ue1.estimate(clean_spectrum, wavenumber, scorer.score)
        r2 = ue2.estimate(clean_spectrum, wavenumber, scorer.score)
        assert abs(r1['confidence'] - r2['confidence']) < 0.01

    def test_scores_ci_present(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        ue = UncertaintyEstimator(n_bootstrap=20, seed=42)
        result = ue.estimate(clean_spectrum, wavenumber, scorer.score)
        for m in ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']:
            assert m in result['scores_std']
            assert m in result['scores_ci']
            lo, hi = result['scores_ci'][m]
            assert lo <= hi

    def test_bootstrap_distribution_shape(self, clean_spectrum, wavenumber):
        scorer = ConfidenceScorer()
        ue = UncertaintyEstimator(n_bootstrap=50, seed=42)
        result = ue.estimate(clean_spectrum, wavenumber, scorer.score)
        assert len(result['bootstrap_distribution']) == 50
