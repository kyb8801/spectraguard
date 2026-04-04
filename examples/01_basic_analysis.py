"""Example 1: Basic single-spectrum quality analysis."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.quality_analyzer import QualityAnalyzer
from src.utils.synthetic_data import generate_single_spectrum

# Generate a synthetic SERS spectrum
wavenumber = np.linspace(200, 3200, 1500)
rng = np.random.default_rng(42)
spectrum = generate_single_spectrum(wavenumber, snr=50, rng=rng)

# Analyze quality
qa = QualityAnalyzer(modality='sers')
result = qa.analyze(spectrum, wavenumber)

print(f"Confidence Score: {result['confidence']:.1f}")
print(f"Grade: {result['grade']}")
print("\nIndividual Metrics:")
for metric, score in result['scores'].items():
    print(f"  {metric:20s}: {score:.3f}")
