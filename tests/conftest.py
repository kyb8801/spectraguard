"""Shared test fixtures for SpectraGuard test suite."""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.synthetic_data import generate_single_spectrum


@pytest.fixture
def wavenumber():
    """Standard wavenumber axis (200-3200 cm⁻¹, 1500 points)."""
    return np.linspace(200, 3200, 1500)


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def clean_spectrum(wavenumber, rng):
    """High-quality clean spectrum (SNR=80)."""
    return generate_single_spectrum(wavenumber, snr=80, rng=rng)


@pytest.fixture
def noisy_spectrum(wavenumber, rng):
    """Low-quality noisy spectrum (SNR=5)."""
    return generate_single_spectrum(wavenumber, snr=5, rng=rng)


@pytest.fixture
def fluorescent_spectrum(wavenumber, rng):
    """Spectrum with strong fluorescence background."""
    return generate_single_spectrum(wavenumber, snr=30, fluorescence_level=1.5, rng=rng)
