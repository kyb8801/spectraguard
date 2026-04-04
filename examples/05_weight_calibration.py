"""Example 5: Adaptive weight calibration from data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.metrics.weight_calibrator import WeightCalibrator
from src.metrics.confidence_score import ConfidenceScorer
from src.utils.synthetic_data import generate_single_spectrum

wavenumber = np.linspace(200, 3200, 1500)
rng = np.random.default_rng(42)

# Generate training spectra with varying quality
scorer = ConfidenceScorer(modality='sers')
spectra_scores = []
for _ in range(50):
    snr = rng.uniform(5, 100)
    spec = generate_single_spectrum(wavenumber, snr=snr, rng=rng)
    result = scorer.score(spec, wavenumber)
    spectra_scores.append(result['scores'])

# Calibrate weights using variance method
wc = WeightCalibrator()
new_weights = wc.calibrate(spectra_scores, method='variance')

print("Default weights vs. Calibrated weights (variance method):")
default_weights = {'snr': 0.25, 'baseline': 0.20, 'peak': 0.15,
                   'resolution': 0.15, 'fluorescence': 0.10, 'uniformity': 0.15}
print(f"{'Metric':>15} | {'Default':>8} | {'Calibrated':>10}")
print("-" * 40)
for m in default_weights:
    print(f"{m:>15} | {default_weights[m]:8.3f} | {new_weights.get(m, 0):10.3f}")

# Compare scoring
comparison = wc.compare_with_default(spectra_scores, new_weights)
print(f"\nComparison report:")
print(f"  Correlation: {comparison.get('correlation', 'N/A')}")
