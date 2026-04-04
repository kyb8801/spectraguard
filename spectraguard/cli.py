"""
SpectraGuard CLI.

Usage:
    spectraguard analyze spectrum.csv
    spectraguard analyze spectrum.csv --modality raman --uncertainty
    spectraguard batch spectra_folder/ --output results.csv
    spectraguard benchmark --dataset synthetic
"""

import argparse
import sys
import os
import json
import csv
import numpy as np

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(
        description='SpectraGuard: Spectral Quality Assessment',
        prog='spectraguard'
    )
    subparsers = parser.add_subparsers(dest='command')

    # analyze
    ap = subparsers.add_parser('analyze', help='Analyze a single spectrum')
    ap.add_argument('file', help='Spectrum file (CSV/TXT, columns: wavenumber,intensity)')
    ap.add_argument('--modality', choices=['sers', 'raman', 'ir'], default='sers')
    ap.add_argument('--uncertainty', action='store_true')
    ap.add_argument('--output', '-o', help='Output JSON file')
    ap.add_argument('--format', choices=['json', 'table', 'brief'], default='table')

    # batch
    bp = subparsers.add_parser('batch', help='Batch analyze spectrum files')
    bp.add_argument('directory', help='Directory with spectrum files')
    bp.add_argument('--modality', choices=['sers', 'raman', 'ir'], default='sers')
    bp.add_argument('--output', '-o', default='results.csv')
    bp.add_argument('--uncertainty', action='store_true')

    # benchmark
    benchp = subparsers.add_parser('benchmark', help='Run benchmark')
    benchp.add_argument('--dataset', choices=['rruff', 'synthetic'], default='synthetic')
    benchp.add_argument('--n', type=int, default=100, help='Number of spectra')
    benchp.add_argument('--output', '-o', default='benchmark_report.md')

    args = parser.parse_args()

    if args.command == 'analyze':
        run_analyze(args)
    elif args.command == 'batch':
        run_batch(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    else:
        parser.print_help()


def _load_spectrum(filepath):
    """Load spectrum from CSV/TXT (wavenumber, intensity columns)."""
    # Try comma-separated first (most common), then whitespace
    for delimiter in [',', None]:
        try:
            data = np.genfromtxt(filepath, delimiter=delimiter, comments='#',
                                skip_header=0, invalid_raise=False)
            # Remove rows with NaN (from header lines)
            data = data[~np.isnan(data).any(axis=1)]
            if data.ndim == 2 and data.shape[1] >= 2 and len(data) > 0:
                return data[:, 0], data[:, 1]
        except Exception:
            continue
    raise ValueError("File must have 2 columns: wavenumber, intensity")


def run_analyze(args):
    """Analyze a single spectrum file."""
    from src.quality_analyzer import QualityAnalyzer

    try:
        wavenumber, intensity = _load_spectrum(args.file)
    except Exception as e:
        print(f"Error loading {args.file}: {e}")
        sys.exit(1)

    qa = QualityAnalyzer(modality=args.modality, uncertainty=args.uncertainty)
    result = qa.analyze(intensity, wavenumber)

    if args.format == 'brief':
        unc_str = f" ± {result.get('confidence_std', 0):.1f}" if args.uncertainty else ""
        print(f"{result['confidence']:.1f}{unc_str} ({result['grade']})")

    elif args.format == 'table':
        print(f"\n{'='*50}")
        print(f"  SpectraGuard Analysis — {args.modality.upper()}")
        print(f"{'='*50}")
        print(f"  File: {args.file}")
        print(f"  Points: {len(wavenumber)}")
        print(f"  Range: {wavenumber.min():.0f} – {wavenumber.max():.0f} cm⁻¹")
        print(f"{'─'*50}")
        for m, v in result['scores'].items():
            ci_str = ""
            if args.uncertainty and 'scores_ci' in result:
                lo, hi = result['scores_ci'][m]
                ci_str = f"  [{lo:.3f}, {hi:.3f}]"
            print(f"  {m:20s}: {v:.3f}{ci_str}")
        print(f"{'─'*50}")
        unc_str = f" ± {result.get('confidence_std', 0):.1f}" if args.uncertainty else ""
        print(f"  Confidence: {result['confidence']:.1f}{unc_str}")
        print(f"  Grade: {result['grade']}")
        if args.uncertainty and 'grade_stability' in result:
            print(f"  Grade Stability: {result['grade_stability']:.0%}")
            ci = result.get('confidence_ci', (0, 0))
            print(f"  95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
        print(f"{'='*50}\n")

    elif args.format == 'json':
        # Convert numpy types for JSON
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.float64, np.float32)):
                out[k] = float(v)
            elif isinstance(v, dict):
                out[k] = {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv
                          for kk, vv in v.items()}
            else:
                out[k] = v
        print(json.dumps(out, indent=2, default=str))

    if args.output:
        out = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Saved to {args.output}")


def run_batch(args):
    """Batch analyze directory of spectrum files."""
    from src.quality_analyzer import QualityAnalyzer
    import glob

    files = sorted(glob.glob(os.path.join(args.directory, '*.csv')) +
                   glob.glob(os.path.join(args.directory, '*.txt')))

    if not files:
        print(f"No CSV/TXT files found in {args.directory}")
        sys.exit(1)

    qa = QualityAnalyzer(modality=args.modality, uncertainty=args.uncertainty)

    results = []
    for f in files:
        try:
            wn, intensity = _load_spectrum(f)
            result = qa.analyze(intensity, wn)
            row = {'file': os.path.basename(f), 'confidence': result['confidence'],
                   'grade': result['grade']}
            row.update(result['scores'])
            if args.uncertainty:
                row['confidence_std'] = result.get('confidence_std', 0)
                row['grade_stability'] = result.get('grade_stability', 0)
            results.append(row)
            print(f"  {os.path.basename(f)}: {result['confidence']:.1f} ({result['grade']})")
        except Exception as e:
            print(f"  {os.path.basename(f)}: ERROR - {e}")

    # Write CSV
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output} ({len(results)} spectra)")


def run_benchmark(args):
    """Run benchmark on synthetic data."""
    from src.quality_analyzer import QualityAnalyzer
    from src.utils.synthetic_data import generate_single_spectrum

    wn = np.linspace(200, 3200, 1500)
    rng = np.random.default_rng(42)
    qa = QualityAnalyzer(modality='sers')
    n = args.n

    grades = {'Excellent': 0, 'Good': 0, 'Marginal': 0, 'Poor': 0}
    scores = []

    print(f"Benchmarking on {n} synthetic spectra...")
    for i in range(n):
        snr = rng.uniform(5, 100)
        spec = generate_single_spectrum(wn, snr=snr, rng=rng)
        result = qa.analyze(spec, wn)
        grades[result['grade']] += 1
        scores.append(result['confidence'])

    report = f"""# SpectraGuard Benchmark Report

## Dataset: {args.dataset} ({n} spectra)

## Score Distribution
- Mean: {np.mean(scores):.1f}
- Std: {np.std(scores):.1f}
- Min: {np.min(scores):.1f}
- Max: {np.max(scores):.1f}

## Grade Distribution
| Grade | Count | Percentage |
|-------|-------|------------|
"""
    for g, c in grades.items():
        report += f"| {g} | {c} | {c/n*100:.1f}% |\n"

    with open(args.output, 'w') as f:
        f.write(report)
    print(f"Report saved to {args.output}")
    print(f"Mean score: {np.mean(scores):.1f}, Grades: {grades}")


if __name__ == '__main__':
    main()
