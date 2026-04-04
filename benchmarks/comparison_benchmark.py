"""
SpectraGuard Comprehensive Benchmark Suite (P17).

Compares all v0.2.0 features:
1. Base vs uncertainty-aware scoring
2. Default vs calibrated weights
3. Standard vs adaptive baseline
4. Cross-instrument transfer effectiveness
5. Streaming throughput
6. End-to-end quality grading accuracy

Generates 6 benchmark figures and a markdown report.
"""

import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*spsolve.*')
warnings.filterwarnings('ignore', message='.*overflow.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quality_analyzer import QualityAnalyzer
from src.metrics.confidence_score import ConfidenceScorer
from src.metrics.uncertainty import UncertaintyEstimator
from src.metrics.weight_calibrator import WeightCalibrator
from src.metrics.adaptive_baseline import AdaptiveBaseline
from src.calibration.instrument_transfer import InstrumentCalibrator
from src.streaming.realtime_analyzer import StreamingAnalyzer, SPC_Controller
from src.utils.synthetic_data import generate_single_spectrum, generate_dataset


# --- Configuration ---
SEED = 42
N_SPECTRA = 200
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'benchmark')
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
FONT = 'DejaVu Sans'


def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.1,
    })


def generate_test_spectra(n=N_SPECTRA, seed=SEED):
    """Generate test spectra with known quality levels."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(200, 3200, 1500)

    spectra = []
    labels = []
    snrs = []
    for _ in range(n):
        snr = rng.uniform(5, 100)
        fluor = rng.uniform(0, 0.5) if rng.random() < 0.3 else 0.0
        spec = generate_single_spectrum(wn, snr=snr, fluorescence_level=fluor, rng=rng)
        spectra.append(spec)
        snrs.append(snr)
        if snr > 40:
            labels.append('Good')
        elif snr > 15:
            labels.append('Marginal')
        else:
            labels.append('Bad')

    return wn, np.array(spectra), np.array(labels), np.array(snrs)


# ======================= Benchmark 1: Base vs Uncertainty =======================
def benchmark_1_uncertainty(wn, spectra, labels):
    """Compare base scoring with uncertainty-aware scoring."""
    print("  Benchmark 1: Base vs Uncertainty scoring...")
    qa_base = QualityAnalyzer(modality='sers', uncertainty=False)
    qa_unc = QualityAnalyzer(modality='sers', uncertainty=True)

    base_scores = []
    unc_scores = []
    unc_stds = []
    unc_grades_stable = []

    n = min(20, len(spectra))  # Limit for speed (uncertainty is slow)
    for i in range(n):
        r_base = qa_base.analyze(spectra[i], wn)
        base_scores.append(r_base['confidence'])

        r_unc = qa_unc.analyze(spectra[i], wn)
        unc_scores.append(r_unc['confidence'])
        unc_stds.append(r_unc.get('confidence_std', 0))
        unc_grades_stable.append(r_unc.get('grade_stability', 1.0))

    # Figure 1: Error bar comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(n)
    axes[0].errorbar(x, unc_scores, yerr=unc_stds, fmt='o', color=COLORS[0],
                     markersize=3, alpha=0.7, label='With uncertainty')
    axes[0].scatter(x, base_scores, s=10, color=COLORS[1], alpha=0.7, label='Base')
    axes[0].set_xlabel('Spectrum index')
    axes[0].set_ylabel('Confidence Score')
    axes[0].set_title('Base vs Uncertainty-Aware Scoring')
    axes[0].legend()
    axes[0].set_ylim(0, 105)

    axes[1].hist(unc_stds, bins=20, color=COLORS[2], edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Confidence Std Dev')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Uncertainty')
    axes[1].axvline(np.mean(unc_stds), color=COLORS[1], linestyle='--',
                    label=f'Mean={np.mean(unc_stds):.1f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_b1_uncertainty.png'))
    plt.close()

    return {
        'mean_base': float(np.mean(base_scores)),
        'mean_unc': float(np.mean(unc_scores)),
        'mean_std': float(np.mean(unc_stds)),
        'mean_grade_stability': float(np.mean(unc_grades_stable)),
        'correlation': float(np.corrcoef(base_scores, unc_scores)[0, 1]),
    }


# ======================= Benchmark 2: Default vs Calibrated Weights =======================
def benchmark_2_weights(wn, spectra, labels):
    """Compare default weights vs variance-calibrated weights."""
    print("  Benchmark 2: Default vs Calibrated weights...")
    scorer_default = ConfidenceScorer(uncertainty=False)

    # Score with defaults
    default_scores = []
    score_dicts = []
    for spec in spectra:
        r = scorer_default.score(spec, wn)
        default_scores.append(r['confidence'])
        score_dicts.append(r['scores'])

    # Calibrate weights
    wc = WeightCalibrator(method='variance')
    cal_weights = wc.calibrate(spectra, wn, scorer_default)

    # Re-score with calibrated weights
    scorer_cal = ConfidenceScorer(weights=cal_weights, uncertainty=False)
    cal_scores = []
    for spec in spectra:
        r = scorer_cal.score(spec, wn)
        cal_scores.append(r['confidence'])

    # Figure 2: Weight comparison + score scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Weight bars
    default_w = {'snr': 0.25, 'baseline': 0.20, 'peak': 0.15,
                 'resolution': 0.15, 'fluorescence': 0.10, 'uniformity': 0.15}
    metrics = list(default_w.keys())
    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x - w/2, [default_w[m] for m in metrics], w, label='Default',
                color=COLORS[0], edgecolor='white')
    axes[0].bar(x + w/2, [cal_weights.get(m, 0) for m in metrics], w, label='Calibrated',
                color=COLORS[1], edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=30, ha='right')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Default vs Calibrated Weights')
    axes[0].legend()

    # Score scatter
    axes[1].scatter(default_scores, cal_scores, s=10, alpha=0.5, color=COLORS[2])
    lims = [0, 105]
    axes[1].plot(lims, lims, '--', color='gray', alpha=0.5)
    axes[1].set_xlabel('Default Score')
    axes[1].set_ylabel('Calibrated Score')
    axes[1].set_title(f'Score Comparison (r={np.corrcoef(default_scores, cal_scores)[0,1]:.3f})')
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_b2_weights.png'))
    plt.close()

    return {
        'default_mean': float(np.mean(default_scores)),
        'calibrated_mean': float(np.mean(cal_scores)),
        'correlation': float(np.corrcoef(default_scores, cal_scores)[0, 1]),
        'calibrated_weights': cal_weights,
    }


# ======================= Benchmark 3: Standard vs Adaptive Baseline =======================
def benchmark_3_baseline(wn, spectra, labels):
    """Compare standard arPLS vs adaptive baseline correction."""
    print("  Benchmark 3: Standard vs Adaptive baseline...")
    from src.metrics.baseline import calculate_baseline

    standard_scores = []
    adaptive_scores = []

    for spec in spectra:
        s = calculate_baseline(spec, wn, adaptive=False)
        a = calculate_baseline(spec, wn, adaptive=True)
        standard_scores.append(s)
        adaptive_scores.append(a)

    # Figure 3: Baseline comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(standard_scores, adaptive_scores, s=10, alpha=0.5, color=COLORS[3])
    axes[0].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    axes[0].set_xlabel('Standard arPLS Score')
    axes[0].set_ylabel('Adaptive Baseline Score')
    r = np.corrcoef(standard_scores, adaptive_scores)[0, 1]
    axes[0].set_title(f'Baseline Method Comparison (r={r:.3f})')

    diff = np.array(adaptive_scores) - np.array(standard_scores)
    axes[1].hist(diff, bins=30, color=COLORS[4], edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_xlabel('Score Difference (Adaptive - Standard)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Improvement Distribution (Mean={np.mean(diff):.3f})')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_b3_baseline.png'))
    plt.close()

    return {
        'standard_mean': float(np.mean(standard_scores)),
        'adaptive_mean': float(np.mean(adaptive_scores)),
        'mean_improvement': float(np.mean(diff)),
        'pct_improved': float(np.mean(diff > 0.01) * 100),
        'correlation': float(r),
    }


# ======================= Benchmark 4: Cross-Instrument Transfer =======================
def benchmark_4_transfer(wn, spectra):
    """Benchmark cross-instrument transfer calibration."""
    print("  Benchmark 4: Cross-instrument transfer...")
    rng = np.random.default_rng(SEED)

    # Simulate two instruments with different characteristics
    n_ref = 20
    ref_spectra_source = spectra[:n_ref]

    # Target instrument: shifted wavenumber, different intensity scale, more noise
    wn_shift = 2.5  # cm-1
    int_scale = 0.7
    noise_extra = 0.02
    ref_spectra_target = []
    for spec in ref_spectra_source:
        shifted = np.interp(wn, wn + wn_shift, spec) * int_scale
        shifted += rng.normal(0, noise_extra * np.max(spec), len(wn))
        ref_spectra_target.append(shifted)
    ref_spectra_target = np.array(ref_spectra_target)

    # Build calibrator
    cal = InstrumentCalibrator()
    source_profile = cal.register_instrument('Source', ref_spectra_source, wn,
                                              reference_peaks=[1000, 1400])
    target_profile = cal.register_instrument('Target', ref_spectra_target, wn,
                                              reference_peaks=[1000, 1400])
    cal.build_transfer(source_profile, target_profile)

    # Score test spectra on both instruments
    scorer = ConfidenceScorer(uncertainty=False)
    source_scores = []
    target_raw_scores = []
    target_cal_scores = []

    n_test = min(100, len(spectra) - n_ref)
    for i in range(n_ref, n_ref + n_test):
        spec_source = spectra[i]
        spec_target = np.interp(wn, wn + wn_shift, spec_source) * int_scale
        spec_target += rng.normal(0, noise_extra * np.max(spec_source), len(wn))

        r_source = scorer.score(spec_source, wn)
        r_target_raw = scorer.score(spec_target, wn)
        r_target_cal = cal.transfer_score(spec_target, wn, scorer)

        source_scores.append(r_source['confidence'])
        target_raw_scores.append(r_target_raw['confidence'])
        target_cal_scores.append(r_target_cal['confidence'])

    # ICC
    icc_raw = cal.compute_icc(source_scores, target_raw_scores)
    icc_cal = cal.compute_icc(source_scores, target_cal_scores)

    # Figure 4: Transfer calibration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(source_scores, target_raw_scores, s=15, alpha=0.5,
                    color=COLORS[1], label=f'Raw (ICC={icc_raw["icc"]:.3f})')
    axes[0].scatter(source_scores, target_cal_scores, s=15, alpha=0.5,
                    color=COLORS[2], label=f'Calibrated (ICC={icc_cal["icc"]:.3f})')
    axes[0].plot([0, 100], [0, 100], '--', color='gray', alpha=0.5)
    axes[0].set_xlabel('Source Instrument Score')
    axes[0].set_ylabel('Target Instrument Score')
    axes[0].set_title('Cross-Instrument Agreement')
    axes[0].legend()

    # Bland-Altman
    mean_raw = (np.array(source_scores) + np.array(target_raw_scores)) / 2
    diff_raw = np.array(source_scores) - np.array(target_raw_scores)
    mean_cal = (np.array(source_scores) + np.array(target_cal_scores)) / 2
    diff_cal = np.array(source_scores) - np.array(target_cal_scores)

    axes[1].scatter(mean_raw, diff_raw, s=10, alpha=0.4, color=COLORS[1], label='Raw')
    axes[1].scatter(mean_cal, diff_cal, s=10, alpha=0.4, color=COLORS[2], label='Calibrated')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_xlabel('Mean Score')
    axes[1].set_ylabel('Difference (Source - Target)')
    axes[1].set_title('Bland-Altman Plot')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_b4_transfer.png'))
    plt.close()

    return {
        'icc_raw': icc_raw['icc'],
        'icc_calibrated': icc_cal['icc'],
        'icc_interpretation_raw': icc_raw['interpretation'],
        'icc_interpretation_cal': icc_cal['interpretation'],
        'mean_bias_raw': float(np.mean(diff_raw)),
        'mean_bias_cal': float(np.mean(diff_cal)),
    }


# ======================= Benchmark 5: Streaming Throughput =======================
def benchmark_5_streaming(wn, spectra):
    """Benchmark streaming analyzer throughput."""
    print("  Benchmark 5: Streaming throughput...")
    sa = StreamingAnalyzer(modality='sers', window_size=50, alert_threshold=40.0)

    # Warm-up
    sa.push(spectra[0], wn)
    sa.reset()

    # Throughput test
    n = min(100, len(spectra))
    start = time.perf_counter()
    scores = []
    trends = []
    alert_counts = []
    for i in range(n):
        result = sa.push(spectra[i], wn)
        scores.append(result['confidence'])
        trends.append(result['trend'])
        alert_counts.append(len(result['alerts']))
    elapsed = time.perf_counter() - start
    throughput = n / elapsed

    summary = sa.get_summary()

    # SPC test
    spc = SPC_Controller(target=70.0, sigma_limit=3.0)
    ref_scores = scores[:20]
    spc.calibrate(ref_scores)
    spc_results = [spc.check(s) for s in scores[20:]]
    ooc_count = sum(1 for r in spc_results if not r['in_control'])

    # Figure 5: Streaming dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Score time series
    axes[0, 0].plot(scores, color=COLORS[0], linewidth=0.8)
    ma_window = 10
    if len(scores) >= ma_window:
        ma = np.convolve(scores, np.ones(ma_window)/ma_window, mode='valid')
        axes[0, 0].plot(range(ma_window-1, len(scores)), ma, color=COLORS[1], linewidth=2,
                        label=f'{ma_window}-pt MA')
    axes[0, 0].set_xlabel('Spectrum #')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_title('Streaming Scores')
    axes[0, 0].legend()

    # SPC chart
    if spc.ucl is not None:
        axes[0, 1].plot(scores[20:], color=COLORS[0], linewidth=0.8)
        axes[0, 1].axhline(spc.ucl, color=COLORS[1], linestyle='--', label=f'UCL={spc.ucl:.1f}')
        axes[0, 1].axhline(spc.lcl, color=COLORS[1], linestyle='--', label=f'LCL={spc.lcl:.1f}')
        axes[0, 1].axhline(spc._mean, color=COLORS[2], linestyle='-', alpha=0.5,
                           label=f'CL={spc._mean:.1f}')
        ooc_idx = [i for i, r in enumerate(spc_results) if not r['in_control']]
        ooc_vals = [scores[20+i] for i in ooc_idx]
        axes[0, 1].scatter(ooc_idx, ooc_vals, color='red', s=30, zorder=5, label='OOC')
        axes[0, 1].set_xlabel('Spectrum #')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('SPC Control Chart')
        axes[0, 1].legend(fontsize=8)

    # Alert distribution
    alert_total = sum(alert_counts)
    axes[1, 0].bar(['Threshold', 'Sudden Drop', 'Decline', 'Metric'], [0, 0, 0, 0],
                    color=COLORS[:4])
    # Count actual alert types
    alert_type_counts = defaultdict(int)
    for result_alerts in sa.alerts:
        alert_type_counts[result_alerts.get('type', 'unknown')] += 1
    types = ['threshold', 'sudden_drop', 'consecutive_decline', 'metric_anomaly']
    type_labels = ['Threshold', 'Sudden Drop', 'Decline', 'Metric Anomaly']
    counts = [alert_type_counts.get(t, 0) for t in types]
    axes[1, 0].bar(type_labels, counts, color=COLORS[:4], edgecolor='white')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Alert Distribution (Total: {len(sa.alerts)})')

    # Throughput summary
    axes[1, 1].text(0.5, 0.7, f'Throughput: {throughput:.1f} spectra/sec',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.5, f'Total spectra: {n}',
                    ha='center', va='center', fontsize=11,
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.35, f'Elapsed: {elapsed:.2f}s',
                    ha='center', va='center', fontsize=11,
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.15, f'OOC points: {ooc_count}/{len(spc_results)}',
                    ha='center', va='center', fontsize=11,
                    transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_b5_streaming.png'))
    plt.close()

    return {
        'throughput_per_sec': float(throughput),
        'total_time_sec': float(elapsed),
        'total_spectra': n,
        'total_alerts': len(sa.alerts),
        'ooc_count': ooc_count,
        'grade_distribution': summary['grade_distribution'],
    }


# ======================= Benchmark 6: End-to-End Grading Accuracy =======================
def benchmark_6_grading(wn, spectra, labels, snrs):
    """End-to-end quality grading accuracy."""
    print("  Benchmark 6: Grading accuracy...")
    qa = QualityAnalyzer(modality='sers')

    pred_grades = []
    scores = []
    for spec in spectra:
        r = qa.analyze(spec, wn)
        pred_grades.append(r['grade'])
        scores.append(r['confidence'])

    # Map grades to binary (Good vs not-Good)
    true_good = (labels == 'Good')
    pred_good = np.array([g in ('Excellent', 'Good') for g in pred_grades])

    tp = np.sum(true_good & pred_good)
    fp = np.sum(~true_good & pred_good)
    fn = np.sum(true_good & ~pred_good)
    tn = np.sum(~true_good & ~pred_good)

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Figure 6: Grading performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Score vs SNR
    for label, color in zip(['Good', 'Marginal', 'Bad'], COLORS[:3]):
        mask = labels == label
        axes[0].scatter(snrs[mask], np.array(scores)[mask], s=10, alpha=0.5,
                        color=color, label=label)
    axes[0].set_xlabel('Input SNR')
    axes[0].set_ylabel('Confidence Score')
    axes[0].set_title('Score vs SNR by True Quality')
    axes[0].legend()

    # Grade distribution
    grade_order = ['Excellent', 'Good', 'Marginal', 'Poor']
    grade_counts = {g: pred_grades.count(g) for g in grade_order}
    axes[1].bar(grade_order, [grade_counts[g] for g in grade_order],
                color=COLORS[:4], edgecolor='white')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Predicted Grade Distribution')

    # Confusion-like: Score distribution by true label
    for label, color in zip(['Good', 'Marginal', 'Bad'], COLORS[:3]):
        mask = labels == label
        axes[2].hist(np.array(scores)[mask], bins=20, alpha=0.5, color=color,
                     label=label, edgecolor='white')
    axes[2].set_xlabel('Confidence Score')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Score Distribution by True Quality')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_b6_grading.png'))
    plt.close()

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'grade_distribution': grade_counts,
        'mean_score': float(np.mean(scores)),
    }


# ======================= Report Generation =======================
def generate_report(results):
    """Generate markdown benchmark report."""
    report = f"""# SpectraGuard v0.2.0 Benchmark Report

## Overview
Comprehensive benchmark of all SpectraGuard features on {N_SPECTRA} synthetic spectra.

---

## 1. Uncertainty Quantification
| Metric | Value |
|--------|-------|
| Base mean score | {results['uncertainty']['mean_base']:.1f} |
| Uncertainty-aware mean | {results['uncertainty']['mean_unc']:.1f} |
| Mean uncertainty (std) | {results['uncertainty']['mean_std']:.1f} |
| Mean grade stability | {results['uncertainty']['mean_grade_stability']:.1%} |
| Base-Unc correlation | {results['uncertainty']['correlation']:.3f} |

## 2. Weight Calibration
| Metric | Value |
|--------|-------|
| Default weight mean score | {results['weights']['default_mean']:.1f} |
| Calibrated weight mean score | {results['weights']['calibrated_mean']:.1f} |
| Correlation | {results['weights']['correlation']:.3f} |

### Calibrated Weights
| Metric | Default | Calibrated |
|--------|---------|------------|
"""
    default_w = {'snr': 0.25, 'baseline': 0.20, 'peak': 0.15,
                 'resolution': 0.15, 'fluorescence': 0.10, 'uniformity': 0.15}
    for m in default_w:
        cw = results['weights']['calibrated_weights'].get(m, 0)
        report += f"| {m} | {default_w[m]:.3f} | {cw:.3f} |\n"

    report += f"""
## 3. Adaptive Baseline
| Metric | Value |
|--------|-------|
| Standard arPLS mean | {results['baseline']['standard_mean']:.3f} |
| Adaptive mean | {results['baseline']['adaptive_mean']:.3f} |
| Mean improvement | {results['baseline']['mean_improvement']:.3f} |
| % spectra improved | {results['baseline']['pct_improved']:.1f}% |
| Correlation | {results['baseline']['correlation']:.3f} |

## 4. Cross-Instrument Transfer
| Metric | Value |
|--------|-------|
| ICC (raw) | {results['transfer']['icc_raw']:.3f} ({results['transfer']['icc_interpretation_raw']}) |
| ICC (calibrated) | {results['transfer']['icc_calibrated']:.3f} ({results['transfer']['icc_interpretation_cal']}) |
| Mean bias (raw) | {results['transfer']['mean_bias_raw']:.1f} |
| Mean bias (calibrated) | {results['transfer']['mean_bias_cal']:.1f} |

## 5. Streaming Performance
| Metric | Value |
|--------|-------|
| Throughput | {results['streaming']['throughput_per_sec']:.1f} spectra/sec |
| Total time | {results['streaming']['total_time_sec']:.2f}s |
| Total alerts | {results['streaming']['total_alerts']} |
| SPC OOC points | {results['streaming']['ooc_count']} |

### Grade Distribution (Streaming)
"""
    for g, c in results['streaming']['grade_distribution'].items():
        report += f"| {g} | {c} |\n"

    report += f"""
## 6. Grading Accuracy
| Metric | Value |
|--------|-------|
| Accuracy | {results['grading']['accuracy']:.1%} |
| Precision | {results['grading']['precision']:.1%} |
| Recall | {results['grading']['recall']:.1%} |
| F1 Score | {results['grading']['f1']:.1%} |
| Mean confidence | {results['grading']['mean_score']:.1f} |

---

*Generated by SpectraGuard v0.2.0 benchmark suite*
"""
    return report


def main():
    print("SpectraGuard v0.2.0 Comprehensive Benchmark")
    print("=" * 50)

    setup()
    wn, spectra, labels, snrs = generate_test_spectra()
    print(f"Generated {len(spectra)} test spectra")
    print(f"  Good: {np.sum(labels == 'Good')}, Marginal: {np.sum(labels == 'Marginal')}, "
          f"Bad: {np.sum(labels == 'Bad')}")

    results = {}

    results['uncertainty'] = benchmark_1_uncertainty(wn, spectra, labels)
    results['weights'] = benchmark_2_weights(wn, spectra, labels)
    results['baseline'] = benchmark_3_baseline(wn, spectra, labels)
    results['transfer'] = benchmark_4_transfer(wn, spectra)
    results['streaming'] = benchmark_5_streaming(wn, spectra)
    results['grading'] = benchmark_6_grading(wn, spectra, labels, snrs)

    # Generate report
    report = generate_report(results)
    report_path = os.path.join(OUTPUT_DIR, 'benchmark_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  Grading F1: {results['grading']['f1']:.1%}")
    print(f"  ICC (calibrated): {results['transfer']['icc_calibrated']:.3f}")
    print(f"  Throughput: {results['streaming']['throughput_per_sec']:.1f} spectra/sec")
    print(f"  Uncertainty mean std: {results['uncertainty']['mean_std']:.1f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
