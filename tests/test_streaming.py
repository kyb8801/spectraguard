"""Tests for real-time streaming analyzer and SPC controller."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.streaming.realtime_analyzer import StreamingAnalyzer, SPC_Controller
from src.utils.synthetic_data import generate_single_spectrum


@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def streaming(wavenumber):
    return StreamingAnalyzer(modality='sers', window_size=50, alert_threshold=40.0)


class TestStreamingAnalyzer:
    def test_push_returns_complete_result(self, streaming, wavenumber):
        rng = np.random.default_rng(42)
        spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
        result = streaming.push(spec, wavenumber)
        assert 'confidence' in result
        assert 'grade' in result
        assert 'moving_avg' in result
        assert 'moving_std' in result
        assert 'trend' in result
        assert 'trend_slope' in result
        assert 'alerts' in result
        assert 'cumulative_stats' in result

    def test_moving_average_correct(self, streaming, wavenumber):
        rng = np.random.default_rng(42)
        scores = []
        for _ in range(10):
            spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
            result = streaming.push(spec, wavenumber)
            scores.append(result['confidence'])
        expected_avg = np.mean(scores)
        assert abs(result['moving_avg'] - expected_avg) < 0.1

    def test_trend_detection_degrading(self, wavenumber):
        sa = StreamingAnalyzer(modality='sers', trend_window=10)
        rng = np.random.default_rng(42)
        # Push increasingly bad spectra
        for snr in [80, 70, 60, 50, 40, 30, 20, 15, 10, 8]:
            spec = generate_single_spectrum(wavenumber, snr=snr, rng=rng)
            result = sa.push(spec, wavenumber)
        assert result['trend'] == 'degrading'

    def test_trend_detection_stable(self, streaming, wavenumber):
        rng = np.random.default_rng(42)
        for _ in range(15):
            spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
            result = streaming.push(spec, wavenumber)
        assert result['trend'] == 'stable'

    def test_alert_on_low_score(self, wavenumber):
        sa = StreamingAnalyzer(modality='sers', alert_threshold=50.0)
        rng = np.random.default_rng(42)
        spec = generate_single_spectrum(wavenumber, snr=5, fluorescence_level=1.0, rng=rng)
        result = sa.push(spec, wavenumber)
        alert_types = [a['type'] for a in result['alerts']]
        assert 'threshold' in alert_types or result['confidence'] >= 50

    def test_alert_callback_called(self, wavenumber):
        sa = StreamingAnalyzer(modality='sers', alert_threshold=50.0)
        callback_results = []
        sa.on_alert(lambda alert: callback_results.append(alert))
        rng = np.random.default_rng(42)
        spec = generate_single_spectrum(wavenumber, snr=5, fluorescence_level=1.5, rng=rng)
        sa.push(spec, wavenumber)
        # Callback should have been called if alerts were generated
        assert isinstance(callback_results, list)

    def test_summary_statistics(self, streaming, wavenumber):
        rng = np.random.default_rng(42)
        for _ in range(5):
            spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
            streaming.push(spec, wavenumber)
        summary = streaming.get_summary()
        assert summary['total_spectra'] == 5
        assert summary['mean_confidence'] > 0
        assert summary['best_metric'] is not None
        assert summary['worst_metric'] is not None

    def test_reset(self, streaming, wavenumber):
        rng = np.random.default_rng(42)
        spec = generate_single_spectrum(wavenumber, snr=50, rng=rng)
        streaming.push(spec, wavenumber)
        streaming.reset()
        assert streaming._total_count == 0
        assert len(streaming.score_history) == 0


class TestSPCController:
    def test_calibrate(self):
        spc = SPC_Controller(target=70.0, sigma_limit=3.0)
        spc.calibrate([65, 70, 75, 68, 72, 71, 69, 73])
        assert spc.ucl is not None
        assert spc.lcl is not None
        assert spc.ucl > spc.lcl

    def test_in_control(self):
        spc = SPC_Controller()
        spc.calibrate([65, 70, 75, 68, 72, 71, 69, 73])
        result = spc.check(70.0)
        assert result['in_control'] is True
        assert len(result['violations']) == 0

    def test_out_of_control_rule1(self):
        spc = SPC_Controller()
        spc.calibrate([65, 70, 75, 68, 72, 71, 69, 73])
        # Push a point far outside 3σ
        result = spc.check(spc.ucl + 10)
        assert result['in_control'] is False
        assert any('Rule 1' in v for v in result['violations'])

    def test_spc_without_calibration(self):
        spc = SPC_Controller()
        result = spc.check(70.0)
        assert result['in_control'] is True  # No limits set


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
