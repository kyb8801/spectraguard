"""
Integrated 6-Metric Confidence Score for SERS spectrum quality assessment.

Combines:
1. SNR (0.25)
2. Baseline Stability (0.20)
3. Peak Reproducibility (0.15)
4. Spectral Resolution (0.15)
5. Fluorescence Contamination (0.10)
6. Hotspot Uniformity (0.15)

Grades:
- 80+  : Excellent
- 60-80: Good
- 40-60: Marginal
- <40  : Poor
"""

import numpy as np
from .snr import calculate_snr
from .baseline import calculate_baseline
from .peak_reproducibility import calculate_peak_reproducibility
from .resolution import calculate_resolution
from .fluorescence import calculate_fluorescence
from .hotspot_uniformity import calculate_hotspot_uniformity


DEFAULT_WEIGHTS = {
    'snr': 0.25,
    'baseline': 0.20,
    'peak': 0.15,
    'resolution': 0.15,
    'fluorescence': 0.10,
    'uniformity': 0.15,
}

GRADE_THRESHOLDS = [
    (80, 'Excellent'),
    (60, 'Good'),
    (40, 'Marginal'),
    (0, 'Poor'),
]


class ConfidenceScorer:
    """Compute integrated confidence score for a spectrum."""

    def __init__(self, weights: dict = None, uncertainty: bool = False, **metric_kwargs):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.uncertainty = uncertainty
        self.metric_kwargs = metric_kwargs
        self._uncertainty_estimator = None
        if self.uncertainty:
            from .uncertainty import UncertaintyEstimator
            self._uncertainty_estimator = UncertaintyEstimator()

    def score(self, spectrum: np.ndarray, wavenumber: np.ndarray,
              spectra_group: np.ndarray = None,
              multi_spectra: np.ndarray = None) -> dict:
        """
        Calculate all metrics and integrated confidence score.

        Parameters
        ----------
        spectrum : 1D array, intensity values
        wavenumber : 1D array, wavenumber axis
        spectra_group : 2D array (optional), repeat measurements for peak repro
        multi_spectra : 2D array (optional), multi-point spectra for hotspot

        Returns
        -------
        dict with keys:
            'scores': {metric_name: float 0-1}
            'confidence': float 0-100
            'grade': str
            'weights': dict
        """
        scores = {}

        scores['snr'] = calculate_snr(
            spectrum, wavenumber,
            **{k: v for k, v in self.metric_kwargs.items()
               if k in ('sg_window', 'sg_polyorder', 'noise_region', 'snr_reference')}
        )
        scores['baseline'] = calculate_baseline(
            spectrum, wavenumber,
            **{k: v for k, v in self.metric_kwargs.items() if k in ('lam', 'p')}
        )
        scores['peak'] = calculate_peak_reproducibility(
            spectrum, wavenumber, spectra_group=spectra_group
        )
        scores['resolution'] = calculate_resolution(
            spectrum, wavenumber,
            **{k: v for k, v in self.metric_kwargs.items()
               if k in ('fwhm_optimal', 'fwhm_worst')}
        )
        scores['fluorescence'] = calculate_fluorescence(
            spectrum, wavenumber,
            **{k: v for k, v in self.metric_kwargs.items()
               if k in ('filter_size',)}
        )
        scores['uniformity'] = calculate_hotspot_uniformity(
            spectrum, wavenumber, multi_spectra=multi_spectra
        )

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
            'weights': self.weights.copy(),
        }

        if self.uncertainty and self._uncertainty_estimator is not None:
            # Create a scorer func without uncertainty to avoid recursion
            base_scorer = ConfidenceScorer(weights=self.weights, uncertainty=False,
                                          **self.metric_kwargs)
            unc = self._uncertainty_estimator.estimate(
                spectrum, wavenumber, base_scorer.score,
                spectra_group=spectra_group, multi_spectra=multi_spectra,
            )
            result['confidence_std'] = unc['confidence_std']
            result['confidence_ci'] = unc['confidence_ci']
            result['scores_std'] = unc['scores_std']
            result['scores_ci'] = unc['scores_ci']
            result['grade_stability'] = unc['grade_stability']
            result['bootstrap_distribution'] = unc['bootstrap_distribution']

        return result

    def score_batch(self, spectra: np.ndarray, wavenumber: np.ndarray) -> list:
        """Score multiple spectra."""
        return [self.score(spec, wavenumber) for spec in spectra]
