"""
Bootstrap-based uncertainty estimation for spectral quality scores.

Estimates confidence intervals and grade stability by resampling
spectra with noise perturbations.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Tuple, Optional


class UncertaintyEstimator:
    def __init__(self, n_bootstrap=200, confidence_level=0.95, noise_model='gaussian', seed=42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.noise_model = noise_model
        self.rng = np.random.default_rng(seed)

    def estimate(self, spectrum, wavenumber, scorer_func, **kwargs):
        """
        Bootstrap resampling for score distribution.
        1. Estimate noise from SG filter residual
        2. n_bootstrap times: perturb spectrum, score it
        3. Compute stats for each metric + confidence

        Returns dict with:
        - confidence, confidence_std, confidence_ci (tuple)
        - scores, scores_std, scores_ci (dicts)
        - grade, grade_stability (float 0-1)
        - bootstrap_distribution (array)
        """
        # Get base score
        base_result = scorer_func(spectrum, wavenumber, **kwargs)
        noise_std = self._estimate_noise(spectrum)

        boot_confidences = []
        boot_scores = {m: [] for m in base_result['scores']}

        for _ in range(self.n_bootstrap):
            perturbed = self._perturb_spectrum(spectrum, noise_std)
            result = scorer_func(perturbed, wavenumber, **kwargs)
            boot_confidences.append(result['confidence'])
            for m in boot_scores:
                boot_scores[m].append(result['scores'][m])

        boot_confidences = np.array(boot_confidences)
        alpha = 1 - self.confidence_level
        ci_low = np.percentile(boot_confidences, 100 * alpha / 2)
        ci_high = np.percentile(boot_confidences, 100 * (1 - alpha / 2))

        # Grade stability
        def get_grade(c):
            if c >= 80:
                return 'Excellent'
            elif c >= 60:
                return 'Good'
            elif c >= 40:
                return 'Marginal'
            return 'Poor'

        final_grade = get_grade(np.mean(boot_confidences))
        grades = [get_grade(c) for c in boot_confidences]
        grade_stability = sum(1 for g in grades if g == final_grade) / len(grades)

        scores_std = {m: float(np.std(boot_scores[m])) for m in boot_scores}
        scores_ci = {}
        for m in boot_scores:
            vals = np.array(boot_scores[m])
            scores_ci[m] = (float(np.percentile(vals, 100 * alpha / 2)),
                            float(np.percentile(vals, 100 * (1 - alpha / 2))))

        return {
            'confidence': float(np.mean(boot_confidences)),
            'confidence_std': float(np.std(boot_confidences)),
            'confidence_ci': (float(ci_low), float(ci_high)),
            'scores': base_result['scores'],
            'scores_std': scores_std,
            'scores_ci': scores_ci,
            'grade': final_grade,
            'grade_stability': float(grade_stability),
            'bootstrap_distribution': boot_confidences,
        }

    def _estimate_noise(self, spectrum):
        n = len(spectrum)
        win = min(21, n // 4 * 2 + 1)
        if win < 5:
            win = 5
        poly = min(3, win - 1)
        smoothed = savgol_filter(spectrum, win, poly)
        return float(np.std(spectrum - smoothed))

    def _perturb_spectrum(self, spectrum, noise_std):
        if noise_std < 1e-10:
            noise_std = 1e-6
        if self.noise_model == 'gaussian':
            return spectrum + self.rng.normal(0, noise_std, len(spectrum))
        elif self.noise_model == 'poisson':
            poisson_scale = np.sqrt(np.abs(spectrum) + 1e-10)
            return spectrum + self.rng.normal(0, 1, len(spectrum)) * poisson_scale * noise_std
        elif self.noise_model == 'mixed':
            g = self.rng.normal(0, noise_std * 0.7, len(spectrum))
            p = self.rng.normal(0, 1, len(spectrum)) * np.sqrt(np.abs(spectrum) + 1e-10) * noise_std * 0.3
            return spectrum + g + p
        return spectrum + self.rng.normal(0, noise_std, len(spectrum))

    def estimate_batch(self, spectra, wavenumber, scorer_func, **kwargs):
        results = []
        for spec in spectra:
            results.append(self.estimate(spec, wavenumber, scorer_func, **kwargs))
        return results
