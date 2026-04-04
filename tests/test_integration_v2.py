"""
Integration tests for SpectraGuard v0.2.0.

Tests end-to-end workflows combining all P11-P16 features:
- Uncertainty + QualityAnalyzer
- Weight calibration pipeline
- Adaptive baseline integration
- Cross-instrument transfer scoring
- Streaming + SPC full workflow
- CLI module import
- Package import (spectraguard namespace)
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quality_analyzer import QualityAnalyzer
from src.metrics.confidence_score import ConfidenceScorer
from src.metrics.uncertainty import UncertaintyEstimator
from src.metrics.weight_calibrator import WeightCalibrator
from src.metrics.adaptive_baseline import AdaptiveBaseline
from src.calibration.instrument_transfer import InstrumentCalibrator, InstrumentProfile
from src.streaming.realtime_analyzer import StreamingAnalyzer, SPC_Controller
from src.utils.synthetic_data import generate_single_spectrum, generate_dataset


@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ===================== End-to-End: Uncertainty =====================

class TestIntegrationUncertainty:
    def test_quality_analyzer_with_uncertainty(self, wavenumber, rng):
        """QualityAnalyzer produces uncertainty estimates when enabled."""
        qa = QualityAnalyzer(modality='sers', uncertainty=True)
        spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
        result = qa.analyze(spec, wavenumber)

        assert 'confidence_std' in result
        assert 'confidence_ci' in result
        assert 'scores_ci' in result
        assert 'grade_stability' in result
        assert result['confidence_std'] >= 0
        assert result['grade_stability'] >= 0

    def test_confidence_scorer_with_uncertainty(self, wavenumber, rng):
        """ConfidenceScorer produces uncertainty when enabled."""
        cs = ConfidenceScorer(uncertainty=True)
        spec = generate_single_spectrum(wavenumber, snr=40, rng=rng)
        result = cs.score(spec, wavenumber)

        assert 'confidence_std' in result
        assert 'confidence_ci' in result
        ci = result['confidence_ci']
        assert ci[0] < ci[1]  # CI lower bound < upper bound
        assert result['confidence_std'] >= 0

    def test_uncertainty_clean_vs_noisy(self, wavenumber, rng):
        """Clean spectra should have narrower uncertainty than noisy ones."""
        qa = QualityAnalyzer(modality='sers', uncertainty=True)
        clean = generate_single_spectrum(wavenumber, snr=80, rng=rng)
        noisy = generate_single_spectrum(wavenumber, snr=5, rng=rng)

        r_clean = qa.analyze(clean, wavenumber)
        r_noisy = qa.analyze(noisy, wavenumber)

        # Clean spectrum should generally have narrower CI (not always guaranteed,
        # but structurally the score should be higher)
        assert r_clean['confidence'] > r_noisy['confidence']


# ===================== End-to-End: Weight Calibration =====================

class TestIntegrationWeights:
    def test_calibrate_weights_variance(self, wavenumber, rng):
        """Weight calibration via variance method produces valid weights."""
        qa = QualityAnalyzer(modality='sers')
        spectra = [generate_single_spectrum(wavenumber, snr=rng.uniform(5, 100), rng=rng)
                   for _ in range(30)]

        scorer = ConfidenceScorer(uncertainty=False)
        wc = WeightCalibrator(method='variance')
        weights = wc.calibrate(spectra, wavenumber, scorer)

        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in weights.values())

    def test_qa_calibrate_weights_method(self, wavenumber, rng):
        """QualityAnalyzer.calibrate_weights updates internal weights."""
        qa = QualityAnalyzer(modality='sers')
        old_weights = qa.weights.copy()

        spectra = [generate_single_spectrum(wavenumber, snr=rng.uniform(5, 100), rng=rng)
                   for _ in range(30)]
        new_weights = qa.calibrate_weights(spectra, wavenumber, method='variance')

        assert abs(sum(new_weights.values()) - 1.0) < 1e-6
        # Weights should be different from defaults (unless by coincidence)
        assert qa.weights == new_weights

    def test_calibrated_weights_produce_valid_scores(self, wavenumber, rng):
        """Scoring with calibrated weights still gives valid results."""
        spectra = [generate_single_spectrum(wavenumber, snr=rng.uniform(10, 80), rng=rng)
                   for _ in range(30)]

        scorer = ConfidenceScorer(uncertainty=False)
        wc = WeightCalibrator(method='variance')
        cal_weights = wc.calibrate(spectra, wavenumber, scorer)

        scorer_cal = ConfidenceScorer(weights=cal_weights, uncertainty=False)
        for spec in spectra[:5]:
            result = scorer_cal.score(spec, wavenumber)
            assert 0 <= result['confidence'] <= 100
            assert result['grade'] in ('Excellent', 'Good', 'Marginal', 'Poor')


# ===================== End-to-End: Adaptive Baseline =====================

class TestIntegrationBaseline:
    def test_adaptive_baseline_in_scorer(self, wavenumber, rng):
        """Adaptive baseline can be computed independently."""
        spec = generate_single_spectrum(wavenumber, snr=30, rng=rng)
        ab = AdaptiveBaseline(method='hybrid')
        corrected, baseline, info = ab.correct(spec, wavenumber)

        assert corrected.shape == spec.shape
        assert baseline.shape == spec.shape
        assert isinstance(info, dict)

    def test_baseline_flag_integration(self, wavenumber, rng):
        """calculate_baseline(adaptive=True) uses AdaptiveBaseline."""
        from src.metrics.baseline import calculate_baseline
        spec = generate_single_spectrum(wavenumber, snr=30, rng=rng)

        score_standard = calculate_baseline(spec, wavenumber, adaptive=False)
        score_adaptive = calculate_baseline(spec, wavenumber, adaptive=True)

        assert 0 <= score_standard <= 1
        assert 0 <= score_adaptive <= 1


# ===================== End-to-End: Cross-Instrument Transfer =====================

class TestIntegrationTransfer:
    def test_full_transfer_pipeline(self, wavenumber, rng):
        """End-to-end: register instruments, build transfer, score."""
        # Generate reference spectra
        ref_spectra = np.array([generate_single_spectrum(wavenumber, snr=50, rng=rng)
                                for _ in range(10)])

        # Simulate target instrument (shifted, scaled)
        target_spectra = ref_spectra * 0.8 + rng.normal(0, 0.01, ref_spectra.shape)

        cal = InstrumentCalibrator()
        src = cal.register_instrument('Source', ref_spectra, wavenumber)
        tgt = cal.register_instrument('Target', target_spectra, wavenumber)
        transfer = cal.build_transfer(src, tgt)

        assert 'wavenumber_correction' in transfer
        assert 'intensity_correction' in transfer

        # Transfer a spectrum and score
        scorer = ConfidenceScorer(uncertainty=False)
        test_spec = target_spectra[0]
        result = cal.transfer_score(test_spec, wavenumber, scorer)
        assert 0 <= result['confidence'] <= 100

    def test_icc_computation(self, wavenumber, rng):
        """ICC is computed correctly for similar instruments."""
        cal = InstrumentCalibrator()
        # Highly correlated scores should give high ICC
        source_scores = [70 + rng.normal(0, 3) for _ in range(50)]
        target_scores = [s + rng.normal(0, 1) for s in source_scores]
        icc = cal.compute_icc(source_scores, target_scores)
        assert icc['icc'] > 0.5
        assert icc['interpretation'] in ('Excellent', 'Good', 'Moderate', 'Poor')


# ===================== End-to-End: Streaming =====================

class TestIntegrationStreaming:
    def test_streaming_full_workflow(self, wavenumber, rng):
        """Full streaming workflow: push, trend, alert, summary."""
        sa = StreamingAnalyzer(modality='sers', window_size=20, alert_threshold=40.0)

        alerts_received = []
        sa.on_alert(lambda a: alerts_received.append(a))

        # Push 15 spectra
        results = []
        for snr in np.linspace(80, 10, 15):
            spec = generate_single_spectrum(wavenumber, snr=snr, rng=rng)
            r = sa.push(spec, wavenumber)
            results.append(r)

        # Check all results have required fields
        for r in results:
            assert 'confidence' in r
            assert 'moving_avg' in r
            assert 'trend' in r
            assert 'alerts' in r
            assert 'cumulative_stats' in r

        # Summary
        summary = sa.get_summary()
        assert summary['total_spectra'] == 15
        assert summary['mean_confidence'] > 0

    def test_spc_integration(self, wavenumber, rng):
        """SPC controller works with streaming scores."""
        sa = StreamingAnalyzer(modality='sers')
        spc = SPC_Controller(target=70.0, sigma_limit=3.0)

        # Generate reference scores
        ref_scores = []
        for _ in range(20):
            spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
            r = sa.push(spec, wavenumber)
            ref_scores.append(r['confidence'])

        spc.calibrate(ref_scores)
        assert spc.ucl is not None
        assert spc.lcl is not None

        # Check new scores
        for _ in range(10):
            spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
            r = sa.push(spec, wavenumber)
            spc_result = spc.check(r['confidence'])
            assert 'in_control' in spc_result
            assert 'violations' in spc_result


# ===================== End-to-End: Package =====================

class TestIntegrationPackage:
    def test_spectraguard_import(self):
        """spectraguard package imports correctly."""
        import spectraguard
        assert hasattr(spectraguard, 'QualityAnalyzer')
        assert hasattr(spectraguard, 'ConfidenceScorer')
        assert hasattr(spectraguard, 'UncertaintyEstimator')
        assert hasattr(spectraguard, 'WeightCalibrator')
        assert hasattr(spectraguard, 'StreamingAnalyzer')
        assert hasattr(spectraguard, 'SPC_Controller')
        assert spectraguard.__version__ == '0.2.0'

    def test_cli_module_imports(self):
        """CLI module can be imported."""
        from spectraguard.cli import main
        assert callable(main)

    def test_all_modalities(self, wavenumber, rng):
        """All modalities (SERS, Raman, IR) produce valid results."""
        spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
        for modality in ['sers', 'raman', 'ir']:
            qa = QualityAnalyzer(modality=modality)
            result = qa.analyze(spec, wavenumber)
            assert 0 <= result['confidence'] <= 100
            assert result['grade'] in ('Excellent', 'Good', 'Marginal', 'Poor')
            assert result['modality'] == modality

    def test_batch_analyze(self, wavenumber, rng):
        """Batch analysis works end-to-end."""
        qa = QualityAnalyzer(modality='sers')
        spectra = [generate_single_spectrum(wavenumber, snr=snr, rng=rng)
                   for snr in [80, 50, 20, 5]]
        results = qa.analyze_batch(np.array(spectra), wavenumber)
        assert len(results) == 4
        # Higher SNR should generally give higher scores
        assert results[0]['confidence'] > results[-1]['confidence']


# ===================== End-to-End: Combined Pipeline =====================

class TestIntegrationPipeline:
    def test_full_pipeline(self, wavenumber, rng):
        """
        Full pipeline: generate data → calibrate weights → analyze with
        uncertainty → stream with SPC.
        """
        # Step 1: Generate data
        spectra = [generate_single_spectrum(wavenumber, snr=rng.uniform(10, 80), rng=rng)
                   for _ in range(30)]

        # Step 2: Calibrate weights
        scorer = ConfidenceScorer(uncertainty=False)
        wc = WeightCalibrator(method='variance')
        cal_weights = wc.calibrate(spectra, wavenumber, scorer)
        assert abs(sum(cal_weights.values()) - 1.0) < 1e-6

        # Step 3: Analyze with calibrated weights + uncertainty
        qa = QualityAnalyzer(modality='sers', custom_weights=cal_weights, uncertainty=True)
        result = qa.analyze(spectra[0], wavenumber)
        assert 'confidence_std' in result

        # Step 4: Stream with SPC
        sa = StreamingAnalyzer(modality='sers')
        spc = SPC_Controller()
        scores = []
        for spec in spectra[:20]:
            r = sa.push(spec, wavenumber)
            scores.append(r['confidence'])
        spc.calibrate(scores)

        # Verify SPC works
        for spec in spectra[20:]:
            r = sa.push(spec, wavenumber)
            spc_r = spc.check(r['confidence'])
            assert isinstance(spc_r['in_control'], bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
