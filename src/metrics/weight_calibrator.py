"""
Weight calibration for spectral quality metrics.

Supports variance-based, GMM-based, and supervised (Random Forest)
methods for determining optimal metric weights.
"""

import numpy as np
from typing import Dict, List, Optional, Literal
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

METRIC_NAMES = ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']


class WeightCalibrator:
    def __init__(self, method='variance'):
        self.method = method
        self.calibrated_weights = None
        self.calibration_report = None

    def calibrate(self, spectra, wavenumber, scorer, labels=None):
        """Score all spectra and compute optimal weights."""
        # Score all spectra
        score_matrix = []
        for spec in spectra:
            result = scorer.score(spec, wavenumber)
            row = [result['scores'].get(m, 0) for m in METRIC_NAMES]
            score_matrix.append(row)
        score_matrix = np.array(score_matrix)

        if self.method == 'variance':
            raw_weights = self._variance_method(score_matrix)
        elif self.method == 'gmm':
            raw_weights = self._gmm_method(score_matrix)
        elif self.method == 'supervised':
            if labels is None:
                raise ValueError("labels required for supervised method")
            raw_weights = self._supervised_method(score_matrix, np.array(labels))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.calibrated_weights = {METRIC_NAMES[i]: float(raw_weights[i]) for i in range(len(METRIC_NAMES))}
        self._generate_report(score_matrix)
        return self.calibrated_weights

    def _variance_method(self, score_matrix):
        variances = np.var(score_matrix, axis=0)
        variances[variances < 1e-10] = 0
        total = variances.sum()
        if total < 1e-10:
            return np.ones(score_matrix.shape[1]) / score_matrix.shape[1]
        return variances / total

    def _gmm_method(self, score_matrix):
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(score_matrix)
        separations = np.abs(gmm.means_[0] - gmm.means_[1])
        total = separations.sum()
        if total < 1e-10:
            return np.ones(score_matrix.shape[1]) / score_matrix.shape[1]
        return separations / total

    def _supervised_method(self, score_matrix, labels):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(score_matrix, y)
        return rf.feature_importances_ / rf.feature_importances_.sum()

    def compare_with_default(self, spectra, wavenumber, scorer, default_weights):
        if self.calibrated_weights is None:
            raise ValueError("Call calibrate() first")

        # Compute scores with both weight sets
        default_scores = []
        calibrated_scores = []
        for spec in spectra:
            result = scorer.score(spec, wavenumber)
            s = result['scores']
            d_score = sum(default_weights.get(m, 0) * s.get(m, 0) for m in METRIC_NAMES) * 100
            c_score = sum(self.calibrated_weights.get(m, 0) * s.get(m, 0) for m in METRIC_NAMES) * 100
            default_scores.append(d_score)
            calibrated_scores.append(c_score)

        default_std = np.std(default_scores)
        calibrated_std = np.std(calibrated_scores)

        return {
            'default_weights': default_weights,
            'calibrated_weights': self.calibrated_weights,
            'default_std': float(default_std),
            'calibrated_std': float(calibrated_std),
            'improvement_pct': float((calibrated_std - default_std) / (default_std + 1e-10) * 100),
        }

    def _generate_report(self, score_matrix):
        lines = ["# Weight Calibration Report\n"]
        lines.append(f"Method: {self.method}")
        lines.append(f"Samples: {len(score_matrix)}\n")
        lines.append("| Metric | Weight |")
        lines.append("|--------|--------|")
        for m in METRIC_NAMES:
            w = self.calibrated_weights.get(m, 0)
            lines.append(f"| {m} | {w:.4f} |")
        self.calibration_report = "\n".join(lines)

    def generate_report(self):
        return self.calibration_report or "No calibration performed yet."
