"""Example 3: Batch analysis of multiple spectra."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.quality_analyzer import QualityAnalyzer
from src.utils.synthetic_data import generate_single_spectrum

wavenumber = np.linspace(200, 3200, 1500)
rng = np.random.default_rng(42)

# Generate spectra with varying quality
snr_levels = [80, 50, 30, 15, 5]
spectra = [generate_single_spectrum(wavenumber, snr=snr, rng=rng) for snr in snr_levels]

# Batch analyze
qa = QualityAnalyzer(modality='sers')
results = qa.batch_analyze(spectra, wavenumber)

print(f"{'SNR':>5} | {'Confidence':>10} | {'Grade':>10}")
print("-" * 35)
for snr, result in zip(snr_levels, results):
    print(f"{snr:5d} | {result['confidence']:10.1f} | {result['grade']:>10}")
