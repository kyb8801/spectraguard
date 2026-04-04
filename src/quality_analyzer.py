"""
SpectraGuard QualityAnalyzer - unified API for multi-modality spectrum QC.

Usage:
    from spectraguard import QualityAnalyzer
    qa = QualityAnalyzer(modality="sers")  # or "raman", "ir"
    report = qa.analyze(spectrum, wavenumber)
"""

import numpy as np
from .metrics.config import get_config
from .metrics.core import CORE_METRICS
from .metrics.modality.sers import SERS_METRICS


MODALITY_EXTRA_METRICS = {
    'sers': SERS_METRICS,
    'raman': {},
    'ir': {},
}

GRADE_THRESHOLDS = [
    (80, 'Excellent'),
    (60, 'Good'),
    (40, 'Marginal'),
    (0, 'Poor'),
]


class QualityAnalyzer:
    """
    Multi-modality spectrum quality analyzer.

    Parameters
    ----------
    modality : str, one of 'sers', 'raman', 'ir'
    custom_weights : dict (optional), override default weights
    """

    def __init__(self, modality: str = 'sers', custom_weights: dict = None,
                 uncertainty: bool = False):
        self.modality = modality.lower()
        self.config = get_config(self.modality)
        self.weights = custom_weights or self.config['weights']
        self.uncertainty = uncertainty
        self._uncertainty_estimator = None
        if self.uncertainty:
            from .metrics.uncertainty import UncertaintyEstimator
            self._uncertainty_estimator = UncertaintyEstimator()

        # Build metric functions
        self.metrics = {}
        for name, func in CORE_METRICS.items():
            if name in self.weights:
                self.metrics[name] = func

        extra = MODALITY_EXTRA_METRICS.get(self.modality, {})
        for name, func in extra.items():
            if name in self.weights:
                self.metrics[name] = func

    def analyze(self, spectrum: np.ndarray, wavenumber: np.ndarray,
                spectra_group: np.ndarray = None,
                multi_spectra: np.ndarray = None) -> dict:
        """
        Analyze a spectrum and return quality report.

        Returns
        -------
        dict with:
            'scores': per-metric scores (0-1)
            'confidence': overall score (0-100)
            'grade': quality grade string
            'modality': modality used
            'config': config used
        """
        scores = {}
        cfg = self.config

        for name, func in self.metrics.items():
            kwargs = {}
            if name == 'snr':
                kwargs['snr_reference'] = cfg.get('snr_reference', 50)
                kwargs['noise_region'] = cfg.get('noise_region', (2000, 2200))
            elif name == 'resolution':
                fwhm_range = cfg.get('fwhm_range', (10, 50))
                kwargs['fwhm_optimal'] = fwhm_range[0]
                kwargs['fwhm_worst'] = fwhm_range[1]
            elif name == 'peak':
                if spectra_group is not None:
                    kwargs['spectra_group'] = spectra_group
            elif name == 'hotspot':
                if multi_spectra is not None:
                    kwargs['multi_spectra'] = multi_spectra

            try:
                scores[name] = func(spectrum, wavenumber, **kwargs)
            except Exception as e:
                scores[name] = 0.0

        confidence = sum(
            self.weights.get(k, 0) * scores.get(k, 0) for k in self.weights
        ) * 100
        confidence = float(np.clip(confidence, 0, 100))

        grade = 'Poor'
        for threshold, label in GRADE_THRESHOLDS:
            if confidence >= threshold:
                grade = label
                break

        result = {
            'scores': scores,
            'confidence': confidence,
            'grade': grade,
            'modality': self.modality,
            'config': cfg,
            'weights': self.weights,
        }

        if self.uncertainty and self._uncertainty_estimator is not None:
            # Build a scorer_func compatible with UncertaintyEstimator
            def _scorer_func(spec, wn, **kw):
                base_qa = QualityAnalyzer(modality=self.modality,
                                          custom_weights=self.weights,
                                          uncertainty=False)
                return base_qa.analyze(spec, wn, **kw)
            unc = self._uncertainty_estimator.estimate(
                spectrum, wavenumber, _scorer_func,
                spectra_group=spectra_group, multi_spectra=multi_spectra,
            )
            result['confidence_std'] = unc['confidence_std']
            result['confidence_ci'] = unc['confidence_ci']
            result['scores_std'] = unc['scores_std']
            result['scores_ci'] = unc['scores_ci']
            result['grade_stability'] = unc['grade_stability']

        return result

    def calibrate_weights(self, spectra, wavenumber, method='variance', labels=None):
        """
        Calibrate metric weights using WeightCalibrator.

        Parameters
        ----------
        spectra : list of 1D arrays
        wavenumber : 1D array
        method : str, one of 'variance', 'gmm', 'supervised'
        labels : array-like (optional), required for 'supervised'

        Returns
        -------
        dict of calibrated weights
        """
        from .metrics.weight_calibrator import WeightCalibrator
        from .metrics.confidence_score import ConfidenceScorer
        wc = WeightCalibrator(method=method)
        scorer = ConfidenceScorer(weights=self.weights)
        calibrated = wc.calibrate(spectra, wavenumber, scorer, labels=labels)
        self.weights = calibrated
        return calibrated

    def analyze_batch(self, spectra: np.ndarray, wavenumber: np.ndarray) -> list:
        """Analyze multiple spectra."""
        return [self.analyze(spec, wavenumber) for spec in spectra]
