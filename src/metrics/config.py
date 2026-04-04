"""
Modality-specific configuration for SpectraGuard.

Supports: SERS, Raman, IR (extensible to UV-Vis, etc.)
"""

MODALITY_CONFIG = {
    'sers': {
        'weights': {
            'snr': 0.25, 'baseline': 0.20, 'peak': 0.15,
            'resolution': 0.15, 'background': 0.10, 'hotspot': 0.15,
        },
        'snr_reference': 50,
        'fwhm_range': (10, 50),  # cm^-1
        'wavenumber_range': (200, 3200),
        'noise_region': (2000, 2200),
        'modality_metrics': ['hotspot'],
    },
    'raman': {
        'weights': {
            'snr': 0.30, 'baseline': 0.25, 'peak': 0.20,
            'resolution': 0.15, 'background': 0.10,
        },
        'snr_reference': 100,
        'fwhm_range': (5, 30),
        'wavenumber_range': (100, 4000),
        'noise_region': (2000, 2200),
        'modality_metrics': [],
    },
    'ir': {
        'weights': {
            'snr': 0.30, 'baseline': 0.25, 'peak': 0.20,
            'resolution': 0.15, 'background': 0.10,
        },
        'snr_reference': 200,
        'fwhm_range': (5, 20),
        'wavenumber_range': (400, 4000),
        'noise_region': (2500, 2700),
        'modality_metrics': [],
    },
}


def get_config(modality: str) -> dict:
    """Get configuration for a specific modality."""
    modality = modality.lower()
    if modality not in MODALITY_CONFIG:
        raise ValueError(f"Unknown modality '{modality}'. "
                         f"Available: {list(MODALITY_CONFIG.keys())}")
    return MODALITY_CONFIG[modality].copy()
