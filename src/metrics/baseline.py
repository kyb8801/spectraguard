"""
Metric 2: Baseline Stability

Physical basis:
- Good SERS spectra have flat, stable baselines
- arPLS (asymmetrically reweighted penalized least squares) estimates baseline
- Residual variability after baseline removal indicates instability
- baseline_score = max(0, 1 - baseline_residual * 3)

Weight in confidence score: 0.20
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def arPLS(y: np.ndarray, lam: float = 1e5, ratio: float = 1e-6,
          itermax: int = 100) -> np.ndarray:
    """
    Asymmetrically Reweighted Penalized Least Squares baseline estimation.

    Parameters
    ----------
    y : 1D array, spectrum
    lam : float, smoothness parameter (larger = smoother baseline)
    ratio : float, convergence criterion
    itermax : int, maximum iterations

    Returns
    -------
    z : 1D array, estimated baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.T)
    w = np.ones(L)

    for _ in range(itermax):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            break
        m = np.mean(dn)
        s = np.std(dn)
        if s < 1e-10:
            break
        w_new = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-10) < ratio:
            break
        w = w_new

    return z


def calculate_baseline(spectrum: np.ndarray, wavenumber: np.ndarray,
                       lam: float = 1e5, p: float = 0.01,
                       adaptive: bool = False) -> float:
    """
    Calculate Baseline Stability score.

    Parameters
    ----------
    spectrum : 1D array, intensity values
    wavenumber : 1D array, wavenumber axis
    lam : float, arPLS smoothness parameter
    p : float, arPLS asymmetry parameter (unused, kept for API compat)
    adaptive : bool, if True use AdaptiveBaseline hybrid method

    Returns
    -------
    float : baseline stability score (0-1)
    """
    if adaptive:
        from src.metrics.adaptive_baseline import AdaptiveBaseline
        ab = AdaptiveBaseline(method='hybrid')
        corrected, baseline, info = ab.correct(spectrum, wavenumber)
        spec_range = np.max(spectrum) - np.min(spectrum)
        if spec_range < 1e-10:
            return 0.0
        baseline_residual = np.std(corrected) / spec_range
        score = max(0.0, 1.0 - baseline_residual * 3)
        return float(score)

    baseline = arPLS(spectrum, lam=lam)
    corrected = spectrum - baseline

    spec_range = np.max(spectrum) - np.min(spectrum)
    if spec_range < 1e-10:
        return 0.0  # No signal → cannot evaluate baseline

    baseline_residual = np.std(corrected) / spec_range
    score = max(0.0, 1.0 - baseline_residual * 3)
    return float(score)


def get_baseline(spectrum: np.ndarray, lam: float = 1e5) -> np.ndarray:
    """Return the estimated baseline (for visualization)."""
    return arPLS(spectrum, lam=lam)
