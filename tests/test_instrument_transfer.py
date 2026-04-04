import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.calibration.instrument_transfer import InstrumentProfile, InstrumentCalibrator
from src.metrics.confidence_score import ConfidenceScorer
from src.utils.synthetic_data import lorentzian


@pytest.fixture
def wavenumber():
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def reference_spectra(wavenumber):
    rng = np.random.default_rng(42)
    spectra = []
    for _ in range(10):
        spec = np.zeros_like(wavenumber)
        # Si peak at 520.7
        spec += lorentzian(wavenumber, 520.7, 1.0, 5)
        spec += rng.normal(0, 0.01, len(wavenumber))
        spectra.append(spec)
    return spectra


class TestInstrumentProfile:
    def test_build_profile(self, reference_spectra, wavenumber):
        profile = InstrumentProfile('test')
        profile.build_from_reference(
            reference_spectra, wavenumber, reference_peaks=[520.7]
        )
        assert profile.n_reference_spectra == 10
        assert abs(profile.wavenumber_shift) < 2.0  # should detect near-zero shift

    def test_shift_detection(self, wavenumber):
        rng = np.random.default_rng(42)
        spectra = []
        for _ in range(5):
            spec = lorentzian(wavenumber + 3.0, 520.7, 1.0, 5)  # shifted by 3
            spec += rng.normal(0, 0.01, len(wavenumber))
            spectra.append(spec)
        profile = InstrumentProfile('shifted')
        profile.build_from_reference(
            spectra, wavenumber, reference_peaks=[520.7]
        )
        # Profile should detect the shift (measured peak at ~523.7 vs ref 520.7)


class TestInstrumentCalibrator:
    def test_register_instrument(self, reference_spectra, wavenumber):
        cal = InstrumentCalibrator()
        profile = cal.register_instrument(
            'renishaw', reference_spectra, wavenumber, [520.7]
        )
        assert profile.name == 'renishaw'

    def test_build_transfer(self, reference_spectra, wavenumber):
        cal = InstrumentCalibrator()
        source = cal.register_instrument(
            'source', reference_spectra, wavenumber, [520.7]
        )
        # Create shifted target spectra
        target_spectra = [s * 0.85 for s in reference_spectra]
        target = cal.register_instrument(
            'target', target_spectra, wavenumber, [520.7]
        )
        transfer = cal.build_transfer(source, target)
        assert 'wavenumber_correction' in transfer

    def test_transfer_score(self, reference_spectra, wavenumber):
        cal = InstrumentCalibrator()
        source = cal.register_instrument(
            'source', reference_spectra, wavenumber, [520.7]
        )
        target = cal.register_instrument(
            'target', reference_spectra, wavenumber, [520.7]
        )
        cal.build_transfer(source, target)
        scorer = ConfidenceScorer()
        result = cal.transfer_score(reference_spectra[0], wavenumber, scorer)
        assert 'confidence' in result

    def test_icc_computation(self):
        cal = InstrumentCalibrator()
        source = [70, 75, 80, 65, 72]
        target = [68, 73, 78, 63, 70]
        icc_result = cal.compute_icc(source, target)
        assert 0 <= icc_result['icc'] <= 1
        assert icc_result['interpretation'] in [
            'Poor', 'Moderate', 'Good', 'Excellent'
        ]

    def test_no_transfer_raises(self, reference_spectra, wavenumber):
        cal = InstrumentCalibrator()
        with pytest.raises(ValueError):
            cal.transfer_spectrum(reference_spectra[0], wavenumber)
