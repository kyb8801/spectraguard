"""Example 2: Quality analysis with uncertainty quantification."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.quality_analyzer import QualityAnalyzer
from src.utils.synthetic_data import generate_single_spectrum

wavenumber = np.linspace(200, 3200, 1500)
rng = np.random.default_rng(42)
spectrum = generate_single_spectrum(wavenumber, snr=30, rng=rng)

# Enable uncertainty estimation (bootstrap resampling)
qa = QualityAnalyzer(modality='sers', uncertainty=True)
result = qa.analyze(spectrum, wavenumber)

print(f"Confidence: {result['confidence']:.1f} ± {result.get('confidence_std', 0):.1f}")
print(f"Grade: {result['grade']}")

if 'confidence_ci' in result:
    ci = result['confidence_ci']
    print(f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")

if 'grade_stability' in result:
    print(f"Grade Stability: {result['grade_stability']:.0%}")

if 'scores_ci' in result:
    print("\nMetric Confidence Intervals:")
    for metric, (lo, hi) in result['scores_ci'].items():
        print(f"  {metric:20s}: [{lo:.3f}, {hi:.3f}]")
