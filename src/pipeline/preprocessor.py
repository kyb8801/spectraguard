"""
RRUFF Raman Data Preprocessor

Parses RRUFF .txt files (Processed only) and unifies into a standard format.
Saves unified spectra to data/processed/unified_spectra.pkl

Format per spectrum:
    {
        'wavenumber': np.ndarray,
        'intensity': np.ndarray,
        'source': 'RRUFF',
        'sample_id': str,
        'metadata': {
            'mineral_name': str,
            'laser_wavelength': str,
            'rruff_id': str,
            'filename': str,
        }
    }
"""

import os
import re
import pickle
import numpy as np
from pathlib import Path


def parse_rruff_file(filepath: str) -> dict:
    """
    Parse a single RRUFF .txt file.

    Parameters
    ----------
    filepath : str
        Path to the RRUFF .txt file.

    Returns
    -------
    dict or None
        Parsed spectrum data, or None if parsing fails.
    """
    metadata = {}
    wavenumbers = []
    intensities = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Header lines start with ##
                if line.startswith('##'):
                    # Parse key=value from header
                    match = re.match(r'^##(.+?)=(.+)$', line)
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        metadata[key] = value
                else:
                    # Data line: "wavenumber , intensity"
                    parts = line.split(',')
                    if len(parts) == 2:
                        try:
                            wn = float(parts[0].strip())
                            inten = float(parts[1].strip())
                            wavenumbers.append(wn)
                            intensities.append(inten)
                        except ValueError:
                            continue
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

    if len(wavenumbers) < 50:
        return None

    wavenumber = np.array(wavenumbers, dtype=np.float64)
    intensity = np.array(intensities, dtype=np.float64)

    # Check for NaN/Inf
    valid = np.isfinite(wavenumber) & np.isfinite(intensity)
    if valid.sum() < 50:
        return None
    wavenumber = wavenumber[valid]
    intensity = intensity[valid]

    # Extract metadata fields
    mineral_name = metadata.get('NAMES', 'Unknown')
    rruff_id = metadata.get('RRUFFID', 'Unknown')
    laser_wl = metadata.get('RAMAN WAVELENGTH', 'Unknown')
    filename = os.path.basename(filepath)

    sample_id = f"{mineral_name}_{rruff_id}_{laser_wl}nm"

    return {
        'wavenumber': wavenumber,
        'intensity': intensity,
        'source': 'RRUFF',
        'sample_id': sample_id,
        'metadata': {
            'mineral_name': mineral_name,
            'laser_wavelength': laser_wl,
            'rruff_id': rruff_id,
            'filename': filename,
        }
    }


def load_rruff_spectra(spectra_dir: str, max_spectra: int = 2000,
                       processed_only: bool = True) -> list:
    """
    Load RRUFF spectra from directory.

    Parameters
    ----------
    spectra_dir : str
        Directory containing RRUFF .txt files.
    max_spectra : int
        Maximum number of spectra to load.
    processed_only : bool
        If True, only load "Processed" files (not RAW).

    Returns
    -------
    list of dict
        List of parsed spectrum dictionaries.
    """
    txt_files = sorted(Path(spectra_dir).glob('*.txt'))

    if processed_only:
        txt_files = [f for f in txt_files if 'Processed' in f.name]

    print(f"Found {len(txt_files)} {'Processed ' if processed_only else ''}files")

    if len(txt_files) > max_spectra:
        # Sample evenly across the file list
        indices = np.linspace(0, len(txt_files) - 1, max_spectra, dtype=int)
        txt_files = [txt_files[i] for i in indices]

    spectra = []
    errors = 0
    for i, fpath in enumerate(txt_files):
        if (i + 1) % 200 == 0:
            print(f"  Parsed {i + 1}/{len(txt_files)} files...")

        result = parse_rruff_file(str(fpath))
        if result is not None:
            spectra.append(result)
        else:
            errors += 1

    print(f"Successfully loaded {len(spectra)} spectra ({errors} errors)")
    return spectra


def save_unified(spectra: list, output_path: str):
    """Save unified spectra list to pickle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(spectra, f)
    print(f"Saved {len(spectra)} spectra to {output_path}")


def load_unified(path: str) -> list:
    """Load unified spectra from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    spectra_dir = os.path.join(project_root, 'data', 'raw', 'rruff', 'spectra')
    output_path = os.path.join(project_root, 'data', 'processed', 'unified_spectra.pkl')

    spectra = load_rruff_spectra(spectra_dir, max_spectra=2000, processed_only=True)
    save_unified(spectra, output_path)

    # Print summary
    if spectra:
        minerals = set(s['metadata']['mineral_name'] for s in spectra)
        lasers = set(s['metadata']['laser_wavelength'] for s in spectra)
        lengths = [len(s['wavenumber']) for s in spectra]
        print(f"\nSummary:")
        print(f"  Unique minerals: {len(minerals)}")
        print(f"  Laser wavelengths: {lasers}")
        print(f"  Spectrum lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
