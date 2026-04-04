"""
Synthetic SERS spectrum generator for testing and validation.

Generates Lorentzian peak-based spectra with configurable:
- SNR (5-100)
- Baseline drift (polynomial)
- Hotspot RSD (5-40%)
- Fluorescence contamination
- Quality grades: Good, Marginal, Bad
"""

import numpy as np
from typing import Tuple, Optional


def lorentzian(x: np.ndarray, x0: float, amplitude: float, gamma: float) -> np.ndarray:
    """Single Lorentzian peak: L(x) = A * gamma^2 / ((x - x0)^2 + gamma^2)"""
    return amplitude * gamma**2 / ((x - x0)**2 + gamma**2)


# Common SERS peak positions and relative intensities for typical analytes
SERS_PEAK_LIBRARY = {
    'rhodamine6g': [
        (612, 0.8, 8), (774, 0.4, 10), (1185, 0.6, 9),
        (1311, 0.7, 8), (1363, 1.0, 7), (1510, 0.9, 8), (1650, 0.5, 10),
    ],
    'crystal_violet': [
        (440, 0.3, 10), (525, 0.4, 9), (725, 0.5, 8),
        (915, 0.9, 7), (1178, 1.0, 8), (1375, 0.7, 9), (1620, 0.8, 8),
    ],
    'generic': [
        (500, 0.5, 10), (800, 0.7, 12), (1000, 1.0, 8),
        (1200, 0.6, 9), (1400, 0.8, 10), (1600, 0.9, 7), (2900, 0.3, 15),
    ],
}


def generate_baseline(wavenumber: np.ndarray, drift_level: float = 0.1,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate polynomial baseline drift."""
    if rng is None:
        rng = np.random.default_rng()
    x_norm = (wavenumber - wavenumber.min()) / (wavenumber.max() - wavenumber.min())
    coeffs = rng.normal(0, drift_level, size=4)
    baseline = (coeffs[0] + coeffs[1] * x_norm + coeffs[2] * x_norm**2
                + coeffs[3] * x_norm**3)
    return baseline


def generate_fluorescence(wavenumber: np.ndarray, intensity: float = 0.5,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate broad fluorescence background (Gaussian envelope)."""
    if rng is None:
        rng = np.random.default_rng()
    center = rng.uniform(800, 2000)
    width = rng.uniform(400, 1000)
    x_norm = (wavenumber - center) / width
    return intensity * np.exp(-0.5 * x_norm**2)


def generate_single_spectrum(
    wavenumber: np.ndarray,
    snr: float = 30.0,
    baseline_drift: float = 0.1,
    hotspot_rsd: float = 10.0,
    fluorescence_level: float = 0.0,
    analyte: str = 'generic',
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a single synthetic SERS spectrum.

    Parameters
    ----------
    wavenumber : array-like, wavenumber axis (cm^-1)
    snr : float, target signal-to-noise ratio
    baseline_drift : float, baseline drift magnitude (0-1)
    hotspot_rsd : float, hotspot intensity variation (%)
    fluorescence_level : float, fluorescence contamination (0-1)
    analyte : str, peak profile to use
    rng : numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    peaks = SERS_PEAK_LIBRARY.get(analyte, SERS_PEAK_LIBRARY['generic'])

    # Build clean signal from Lorentzian peaks
    signal = np.zeros_like(wavenumber, dtype=float)
    for x0, amp, gamma in peaks:
        # Apply hotspot variation
        amp_varied = amp * (1 + rng.normal(0, hotspot_rsd / 100.0))
        gamma_varied = gamma * (1 + rng.normal(0, 0.05))
        signal += lorentzian(wavenumber, x0, max(amp_varied, 0.01), abs(gamma_varied) + 1)

    # Add baseline drift
    baseline = generate_baseline(wavenumber, drift_level=baseline_drift, rng=rng)

    # Add fluorescence
    fluor = generate_fluorescence(wavenumber, intensity=fluorescence_level, rng=rng)

    # Combine clean spectrum
    clean = signal + baseline + fluor

    # Add noise based on target SNR
    signal_max = np.max(np.abs(signal))
    noise_std = signal_max / snr if snr > 0 else 0.0
    noise = rng.normal(0, noise_std, size=len(wavenumber))

    return clean + noise


def generate_dataset(
    n_good: int = 1000,
    n_marginal: int = 1000,
    n_bad: int = 1000,
    wavenumber_range: Tuple[float, float] = (200, 3200),
    n_points: int = 1500,
    seed: int = 42,
) -> dict:
    """
    Generate a full synthetic SERS dataset with Good/Marginal/Bad quality grades.

    Returns dict with keys:
    - 'wavenumber': 1D array
    - 'spectra': 2D array (n_total, n_points)
    - 'labels': 1D array of grade labels
    - 'params': list of dicts with generation parameters
    """
    rng = np.random.default_rng(seed)
    wavenumber = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)

    grade_configs = {
        'Good': {
            'snr': (30, 100), 'baseline_drift': (0.0, 0.05),
            'hotspot_rsd': (5, 12), 'fluorescence': (0.0, 0.1),
        },
        'Marginal': {
            'snr': (15, 30), 'baseline_drift': (0.05, 0.2),
            'hotspot_rsd': (12, 25), 'fluorescence': (0.1, 0.4),
        },
        'Bad': {
            'snr': (5, 15), 'baseline_drift': (0.2, 0.5),
            'hotspot_rsd': (25, 40), 'fluorescence': (0.3, 0.8),
        },
    }

    counts = {'Good': n_good, 'Marginal': n_marginal, 'Bad': n_bad}
    analytes = list(SERS_PEAK_LIBRARY.keys())

    all_spectra = []
    all_labels = []
    all_params = []

    for grade, n in counts.items():
        cfg = grade_configs[grade]
        for _ in range(n):
            snr = rng.uniform(*cfg['snr'])
            drift = rng.uniform(*cfg['baseline_drift'])
            rsd = rng.uniform(*cfg['hotspot_rsd'])
            fluor = rng.uniform(*cfg['fluorescence'])
            analyte = rng.choice(analytes)

            spectrum = generate_single_spectrum(
                wavenumber, snr=snr, baseline_drift=drift,
                hotspot_rsd=rsd, fluorescence_level=fluor,
                analyte=analyte, rng=rng,
            )
            all_spectra.append(spectrum)
            all_labels.append(grade)
            all_params.append({
                'snr': snr, 'baseline_drift': drift,
                'hotspot_rsd': rsd, 'fluorescence': fluor,
                'analyte': analyte,
            })

    return {
        'wavenumber': wavenumber,
        'spectra': np.array(all_spectra),
        'labels': np.array(all_labels),
        'params': all_params,
    }


if __name__ == '__main__':
    import os
    data = generate_dataset(n_good=1000, n_marginal=1000, n_bad=1000)
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, 'synthetic_sers.npz'),
        wavenumber=data['wavenumber'],
        spectra=data['spectra'],
        labels=data['labels'],
    )
    print(f"Generated {len(data['labels'])} spectra → data/raw/synthetic_sers.npz")
    for grade in ['Good', 'Marginal', 'Bad']:
        print(f"  {grade}: {np.sum(data['labels'] == grade)}")
