"""
SpectraGuard Simulation Suite
5 comprehensive simulations for validating the 6-metric confidence scoring system.
"""

import sys
import os
import warnings
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161']

from src.utils.synthetic_data import generate_single_spectrum, lorentzian, SERS_PEAK_LIBRARY
from src.metrics.confidence_score import ConfidenceScorer
from src.metrics.snr import calculate_snr
from src.metrics.baseline import calculate_baseline
from src.metrics.peak_reproducibility import calculate_peak_reproducibility
from src.metrics.resolution import calculate_resolution
from src.metrics.fluorescence import calculate_fluorescence
from src.metrics.hotspot_uniformity import calculate_hotspot_uniformity

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'simulation')
os.makedirs(RESULTS_DIR, exist_ok=True)

WAVENUMBER = np.linspace(200, 3200, 1500)
METRIC_NAMES = ['SNR', 'Baseline', 'Peak Repro', 'Resolution', 'Fluorescence', 'Uniformity']
METRIC_KEYS = ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']


def safe_score(scorer, spectrum, wavenumber, **kwargs):
    """Score a spectrum with full error handling."""
    spec = np.copy(spectrum).astype(float)
    # Handle NaN/Inf
    if np.any(np.isnan(spec)) or np.any(np.isinf(spec)):
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        result = scorer.score(spec, wavenumber, **kwargs)
        return result
    except Exception:
        return {
            'scores': {k: 0.0 for k in METRIC_KEYS},
            'confidence': 0.0,
            'grade': 'Error',
            'weights': {},
        }


def safe_metric(func, spectrum, wavenumber, **kwargs):
    """Call a single metric function safely."""
    spec = np.copy(spectrum).astype(float)
    if np.any(np.isnan(spec)) or np.any(np.isinf(spec)):
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        val = func(spec, wavenumber, **kwargs)
        if val is None or np.isnan(val) or np.isinf(val):
            return 0.0
        return float(np.clip(val, 0, 1))
    except Exception:
        return 0.0


# ============================================================================
# SIMULATION 1: Sensitivity Analysis
# ============================================================================
def simulation_1_sensitivity():
    print("=" * 60)
    print("SIMULATION 1: Sensitivity Analysis")
    print("=" * 60)

    scorer = ConfidenceScorer()
    rng = np.random.default_rng(42)
    n_repeats = 20

    # Parameter degradation sweeps
    params = {
        'SNR': [100, 80, 60, 40, 20, 10, 5],
        'Baseline Drift': [0, 0.05, 0.1, 0.2, 0.5, 1.0],
        'Peak Width (gamma)': [5, 7.5, 10, 15, 25, 40],
        'Hotspot RSD (%)': [0, 5, 10, 15, 20, 30, 40],
        'Fluorescence': [0, 0.1, 0.3, 0.5, 1.0, 2.0],
    }

    # Which metric should respond most to each parameter
    matching_metric_idx = [0, 1, 3, 5, 4]  # SNR, Baseline, Resolution, Uniformity, Fluorescence

    param_names = list(params.keys())
    all_results = {}

    for pi, pname in enumerate(param_names):
        values = params[pname]
        results_for_param = {mk: [] for mk in METRIC_KEYS}

        for val in values:
            scores_accum = {mk: [] for mk in METRIC_KEYS}
            for _ in range(n_repeats):
                seed = rng.integers(0, 2**31)
                local_rng = np.random.default_rng(seed)

                if pname == 'SNR':
                    spec = generate_single_spectrum(WAVENUMBER, snr=val, baseline_drift=0,
                                                   hotspot_rsd=0, fluorescence_level=0, rng=local_rng)
                elif pname == 'Baseline Drift':
                    spec = generate_single_spectrum(WAVENUMBER, snr=100, baseline_drift=val,
                                                   hotspot_rsd=0, fluorescence_level=0, rng=local_rng)
                elif pname == 'Peak Width (gamma)':
                    # Custom spectrum with specified gamma
                    signal = np.zeros_like(WAVENUMBER, dtype=float)
                    for x0, amp, _ in SERS_PEAK_LIBRARY['generic']:
                        signal += lorentzian(WAVENUMBER, x0, amp, val)
                    noise_std = np.max(signal) / 100.0
                    spec = signal + local_rng.normal(0, noise_std, len(WAVENUMBER))
                elif pname == 'Hotspot RSD (%)':
                    spec = generate_single_spectrum(WAVENUMBER, snr=100, baseline_drift=0,
                                                   hotspot_rsd=val, fluorescence_level=0, rng=local_rng)
                elif pname == 'Fluorescence':
                    spec = generate_single_spectrum(WAVENUMBER, snr=100, baseline_drift=0,
                                                   hotspot_rsd=0, fluorescence_level=val, rng=local_rng)

                result = safe_score(scorer, spec, WAVENUMBER)
                for mk in METRIC_KEYS:
                    scores_accum[mk].append(result['scores'][mk])

            for mk in METRIC_KEYS:
                results_for_param[mk].append(np.mean(scores_accum[mk]))

        all_results[pname] = results_for_param

    # Plot 5x6 matrix
    fig, axes = plt.subplots(5, 6, figsize=(16, 12), sharex=False, sharey=True)
    fig.suptitle('Sensitivity Analysis: Single-Parameter Degradation', fontsize=14, fontweight='bold')

    for pi, pname in enumerate(param_names):
        values = params[pname]
        for mi, mk in enumerate(METRIC_KEYS):
            ax = axes[pi, mi]
            color = '#D55E00' if mi == matching_metric_idx[pi] else '#999999'
            lw = 2.0 if mi == matching_metric_idx[pi] else 1.0
            ax.plot(range(len(values)), all_results[pname][mk], 'o-',
                    color=color, linewidth=lw, markersize=4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values], fontsize=6, rotation=45)
            if pi == 0:
                ax.set_title(METRIC_NAMES[mi], fontsize=9)
            if mi == 0:
                ax.set_ylabel(pname, fontsize=8)
            if pi == len(param_names) - 1:
                ax.set_xlabel('Parameter value', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, 'sensitivity_matrix.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# SIMULATION 2: Cross-Instrument Transfer
# ============================================================================
def simulation_2_cross_instrument():
    print("=" * 60)
    print("SIMULATION 2: Cross-Instrument Transfer")
    print("=" * 60)

    scorer = ConfidenceScorer()
    rng = np.random.default_rng(123)

    instruments = {
        'A (Renishaw)': {'shift': 0, 'scale': 1.0, 'noise_floor': 0.01, 'broadening': 0},
        'B (Horiba)': {'shift': 2.5, 'scale': 0.85, 'noise_floor': 0.02, 'broadening': 3},
        'C (Bruker)': {'shift': -1.8, 'scale': 1.15, 'noise_floor': 0.015, 'broadening': 1},
    }

    n_spectra = 200

    def apply_instrument(spectrum, wn, inst):
        spec = spectrum.copy()
        # Intensity scaling
        spec *= inst['scale']
        # Add instrument noise floor
        spec += rng.normal(0, inst['noise_floor'], len(spec))
        # Resolution broadening (convolve with Gaussian)
        if inst['broadening'] > 0:
            from scipy.ndimage import gaussian_filter1d
            sigma_pts = inst['broadening'] / (wn[1] - wn[0])
            spec = gaussian_filter1d(spec, sigma=sigma_pts)
        return spec

    def correct_spectrum(spectrum, wn, inst_params, ref_params):
        """Simple correction: undo scale and shift."""
        spec = spectrum.copy()
        # Intensity rescaling
        if inst_params['scale'] != 0:
            spec /= inst_params['scale'] / ref_params['scale']
        return spec

    # Generate base spectra
    base_spectra = []
    for i in range(n_spectra):
        spec = generate_single_spectrum(WAVENUMBER, snr=60, baseline_drift=0.02,
                                       hotspot_rsd=5, fluorescence_level=0.05,
                                       rng=np.random.default_rng(rng.integers(0, 2**31)))
        base_spectra.append(spec)
    base_spectra = np.array(base_spectra)

    # Score for each instrument before/after correction
    inst_names = list(instruments.keys())
    ref_inst = instruments['A (Renishaw)']

    scores_before = {name: [] for name in inst_names}
    scores_after = {name: [] for name in inst_names}

    for name, inst in instruments.items():
        for i in range(n_spectra):
            # Shift wavenumber for this instrument
            wn_shifted = WAVENUMBER + inst['shift']
            spec_inst = apply_instrument(base_spectra[i], WAVENUMBER, inst)

            # Score before correction (use original wavenumber axis)
            res_before = safe_score(scorer, spec_inst, WAVENUMBER)
            scores_before[name].append(res_before['confidence'])

            # Correct
            spec_corrected = correct_spectrum(spec_inst, WAVENUMBER, inst, ref_inst)
            res_after = safe_score(scorer, spec_corrected, WAVENUMBER)
            scores_after[name].append(res_after['confidence'])

    # --- Violin plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (data_dict, title) in enumerate([
        (scores_before, 'Before Correction'),
        (scores_after, 'After Correction'),
    ]):
        ax = axes[ax_idx]
        positions = np.arange(len(inst_names))
        parts = ax.violinplot([data_dict[n] for n in inst_names],
                              positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS[i])
            pc.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(inst_names, fontsize=9)
        ax.set_ylabel('Confidence Score')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    # Calculate ICC (simplified: one-way random)
    def calc_icc(groups):
        """One-way random ICC(1,1)."""
        k = len(groups)
        ns = [len(g) for g in groups]
        n = ns[0]
        grand_mean = np.mean(np.concatenate(groups))
        msb = n * np.sum([(np.mean(g) - grand_mean)**2 for g in groups]) / (k - 1)
        msw = np.sum([np.sum((g - np.mean(g))**2) for g in groups]) / (k * (n - 1))
        if (msb + (k - 1) * msw) < 1e-10:
            return 1.0
        icc = (msb - msw) / (msb + (k - 1) * msw)
        return float(np.clip(icc, -1, 1))

    groups_before = [np.array(scores_before[n]) for n in inst_names]
    groups_after = [np.array(scores_after[n]) for n in inst_names]
    icc_before = calc_icc(groups_before)
    icc_after = calc_icc(groups_after)

    fig.suptitle(f'Cross-Instrument Transfer  |  ICC: Before={icc_before:.3f}, After={icc_after:.3f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, 'cross_instrument_violin.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")

    # --- Bland-Altman plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ref_scores = np.array(scores_before['A (Renishaw)'])

    for ax_idx, name in enumerate(['B (Horiba)', 'C (Bruker)']):
        ax = axes[ax_idx]
        test_scores = np.array(scores_before[name])
        mean_vals = (ref_scores + test_scores) / 2
        diff_vals = ref_scores - test_scores
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        ax.scatter(mean_vals, diff_vals, alpha=0.3, s=10, color=COLORS[ax_idx + 1])
        ax.axhline(mean_diff, color='k', linestyle='-', linewidth=1, label=f'Mean: {mean_diff:.1f}')
        ax.axhline(mean_diff + 1.96 * std_diff, color='r', linestyle='--', linewidth=0.8,
                   label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.1f}')
        ax.axhline(mean_diff - 1.96 * std_diff, color='r', linestyle='--', linewidth=0.8,
                   label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.1f}')
        ax.set_xlabel('Mean of A & ' + name)
        ax.set_ylabel('Difference (A - ' + name + ')')
        ax.set_title(f'Bland-Altman: A vs {name}', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'cross_instrument_blandaltman.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# SIMULATION 3: Concentration-Response
# ============================================================================
def simulation_3_concentration():
    print("=" * 60)
    print("SIMULATION 3: Concentration-Response")
    print("=" * 60)

    scorer = ConfidenceScorer()
    rng = np.random.default_rng(456)

    # 8 log-spaced concentrations from 1nM to 100uM
    concentrations = np.logspace(-9, -4, 8)  # 1e-9 to 1e-4
    conc_labels = ['1nM', '10nM', '100nM', '1uM', '3uM', '10uM', '30uM', '100uM']
    n_per_conc = 100

    mean_scores = []
    std_scores = []
    all_metric_means = {mk: [] for mk in METRIC_KEYS}

    for ci, conc in enumerate(concentrations):
        # Model concentration effects
        log_conc = np.log10(conc)  # -9 to -4
        # Normalize to 0-1 range
        conc_norm = (log_conc - (-9)) / ((-4) - (-9))  # 0 to 1

        # Low conc: low SNR + low amplitude
        # High conc: peak broadening + saturation
        snr = 5 + 90 * np.clip(conc_norm * 2, 0, 1)  # 5 to 95
        amplitude_scale = 0.1 + 0.9 * np.clip(conc_norm * 1.5, 0, 1)
        # Broadening at high concentration (saturation)
        gamma_extra = max(0, (conc_norm - 0.6) * 30)  # broadening kicks in above 60%
        fluorescence = max(0, (conc_norm - 0.7) * 1.0)  # fluorescence at high conc

        conc_scores = []
        conc_metrics = {mk: [] for mk in METRIC_KEYS}

        for _ in range(n_per_conc):
            seed = rng.integers(0, 2**31)
            local_rng = np.random.default_rng(seed)

            # Build spectrum with concentration-dependent parameters
            signal = np.zeros_like(WAVENUMBER, dtype=float)
            for x0, amp, gamma in SERS_PEAK_LIBRARY['generic']:
                amp_scaled = amp * amplitude_scale * (1 + local_rng.normal(0, 0.05))
                gamma_eff = gamma + gamma_extra
                signal += lorentzian(WAVENUMBER, x0, max(amp_scaled, 0.01), gamma_eff)

            noise_std = np.max(signal) / snr if snr > 0 else 0
            noise = local_rng.normal(0, noise_std, len(WAVENUMBER))
            spec = signal + noise

            # Add fluorescence at high conc
            if fluorescence > 0:
                from src.utils.synthetic_data import generate_fluorescence
                fluor = generate_fluorescence(WAVENUMBER, intensity=fluorescence, rng=local_rng)
                spec += fluor

            result = safe_score(scorer, spec, WAVENUMBER)
            conc_scores.append(result['confidence'])
            for mk in METRIC_KEYS:
                conc_metrics[mk].append(result['scores'][mk])

        mean_scores.append(np.mean(conc_scores))
        std_scores.append(np.std(conc_scores))
        for mk in METRIC_KEYS:
            all_metric_means[mk].append(np.mean(conc_metrics[mk]))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: confidence score vs concentration
    ax1.errorbar(range(len(concentrations)), mean_scores, yerr=std_scores,
                 fmt='o-', color=COLORS[0], linewidth=2, markersize=6, capsize=4)
    ax1.fill_between(range(len(concentrations)),
                     np.array(mean_scores) - np.array(std_scores),
                     np.array(mean_scores) + np.array(std_scores),
                     alpha=0.2, color=COLORS[0])
    ax1.set_xticks(range(len(concentrations)))
    ax1.set_xticklabels(conc_labels, rotation=45)
    ax1.set_xlabel('Concentration')
    ax1.set_ylabel('Confidence Score')
    ax1.set_title('Concentration-Response Curve', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Right: individual metrics
    for mi, mk in enumerate(METRIC_KEYS):
        ax2.plot(range(len(concentrations)), all_metric_means[mk],
                 'o-', color=COLORS[mi], label=METRIC_NAMES[mi], linewidth=1.5, markersize=4)
    ax2.set_xticks(range(len(concentrations)))
    ax2.set_xticklabels(conc_labels, rotation=45)
    ax2.set_xlabel('Concentration')
    ax2.set_ylabel('Metric Score (0-1)')
    ax2.set_title('Individual Metrics vs Concentration', fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'concentration_response.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# SIMULATION 4: Comparative Benchmark
# ============================================================================
def simulation_4_benchmark():
    print("=" * 60)
    print("SIMULATION 4: Comparative Benchmark")
    print("=" * 60)

    from sklearn.metrics import roc_curve, auc as sklearn_auc

    scorer = ConfidenceScorer()
    rng = np.random.default_rng(789)
    n_per_scenario = 100

    # Generate balanced scenarios: 3 Bad + 3 Good = 300:300
    scenarios = {}
    ground_truth = {}

    # Bad scenarios
    # S1: High SNR + heavy fluorescence → Bad
    s1 = [generate_single_spectrum(WAVENUMBER, snr=80, baseline_drift=0.02,
          hotspot_rsd=5, fluorescence_level=1.5,
          rng=np.random.default_rng(rng.integers(0, 2**31))) for _ in range(n_per_scenario)]
    scenarios['S1: High SNR\n+ Fluor'] = np.array(s1)
    ground_truth['S1: High SNR\n+ Fluor'] = np.zeros(n_per_scenario)

    # S2: Low SNR + clean peaks → Bad
    s2 = [generate_single_spectrum(WAVENUMBER, snr=8, baseline_drift=0,
          hotspot_rsd=0, fluorescence_level=0,
          rng=np.random.default_rng(rng.integers(0, 2**31))) for _ in range(n_per_scenario)]
    scenarios['S2: Low SNR\n+ Clean'] = np.array(s2)
    ground_truth['S2: Low SNR\n+ Clean'] = np.zeros(n_per_scenario)

    # S3: High hotspot RSD → Bad
    s3 = [generate_single_spectrum(WAVENUMBER, snr=60, baseline_drift=0.02,
          hotspot_rsd=40, fluorescence_level=0.05,
          rng=np.random.default_rng(rng.integers(0, 2**31))) for _ in range(n_per_scenario)]
    scenarios['S3: High\nHotspot RSD'] = np.array(s3)
    ground_truth['S3: High\nHotspot RSD'] = np.zeros(n_per_scenario)

    # Good scenarios
    # S4: Good quality
    s4 = [generate_single_spectrum(WAVENUMBER, snr=60, baseline_drift=0.02,
          hotspot_rsd=5, fluorescence_level=0.05,
          rng=np.random.default_rng(rng.integers(0, 2**31))) for _ in range(n_per_scenario)]
    scenarios['S4: Good\nQuality'] = np.array(s4)
    ground_truth['S4: Good\nQuality'] = np.ones(n_per_scenario)

    # S5: Moderate good
    s5 = [generate_single_spectrum(WAVENUMBER, snr=40, baseline_drift=0.05,
          hotspot_rsd=8, fluorescence_level=0.1,
          rng=np.random.default_rng(rng.integers(0, 2**31))) for _ in range(n_per_scenario)]
    scenarios['S5: Moderate\nGood'] = np.array(s5)
    ground_truth['S5: Moderate\nGood'] = np.ones(n_per_scenario)

    # S6: Very good quality
    s6 = [generate_single_spectrum(WAVENUMBER, snr=80, baseline_drift=0.01,
          hotspot_rsd=3, fluorescence_level=0.02,
          rng=np.random.default_rng(rng.integers(0, 2**31))) for _ in range(n_per_scenario)]
    scenarios['S6: Very\nGood'] = np.array(s6)
    ground_truth['S6: Very\nGood'] = np.ones(n_per_scenario)

    # Combine all (300 Bad + 300 Good = balanced)
    all_spectra = np.vstack([scenarios[k] for k in scenarios])
    all_labels = np.concatenate([ground_truth[k] for k in ground_truth])

    ref_spectrum = generate_single_spectrum(WAVENUMBER, snr=100, baseline_drift=0,
                                           hotspot_rsd=0, fluorescence_level=0,
                                           rng=np.random.default_rng(999))

    # Get continuous scores for all methods
    print("  Computing scores for all methods...")
    scores_all = {'SNR-only': [], 'HQI proxy': [], 'Expert sim.': [], 'SpectraGuard': []}

    for spec in all_spectra:
        s = np.nan_to_num(spec, nan=0.0)
        # SNR-only
        scores_all['SNR-only'].append(safe_metric(calculate_snr, s, WAVENUMBER))
        # HQI
        corr = np.corrcoef(s, ref_spectrum)[0, 1]
        scores_all['HQI proxy'].append(corr if not np.isnan(corr) else 0)
        # Expert (mean of metric scores)
        result = safe_score(scorer, s, WAVENUMBER)
        sc = result['scores']
        scores_all['Expert sim.'].append(np.mean(list(sc.values())))
        # SpectraGuard
        scores_all['SpectraGuard'].append(result['confidence'] / 100.0)

    for k in scores_all:
        scores_all[k] = np.array(scores_all[k])

    # Optimal thresholds via ROC
    thresholds_opt = {}
    for mname, cont in scores_all.items():
        fpr, tpr, thrs = roc_curve(all_labels, cont)
        # Youden's J = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresholds_opt[mname] = thrs[best_idx] if best_idx < len(thrs) else 0.5

    # Predictions using fixed reasonable thresholds
    fixed_thresholds = {'SNR-only': 0.6, 'HQI proxy': 0.7, 'Expert sim.': 0.5, 'SpectraGuard': 0.55}
    methods_pred = {}
    for mname in scores_all:
        thr = fixed_thresholds[mname]
        methods_pred[mname] = (scores_all[mname] >= thr).astype(int)

    # Calculate metrics
    def calc_metrics(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-10)
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    results = {}
    for mname, preds in methods_pred.items():
        results[mname] = calc_metrics(all_labels, preds)
        print(f"  {mname}: Acc={results[mname]['accuracy']:.3f}, "
              f"Prec={results[mname]['precision']:.3f}, "
              f"Rec={results[mname]['recall']:.3f}, "
              f"F1={results[mname]['f1']:.3f}")

    # --- ROC curves (sklearn) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for mi, mname in enumerate(scores_all.keys()):
        fpr, tpr, _ = roc_curve(all_labels, scores_all[mname])
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[mi], linewidth=2,
                label=f'{mname} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Method Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'benchmark_roc.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")

    # --- Grouped bar chart ---
    fig, ax = plt.subplots(figsize=(9, 5))
    metric_names_bench = ['Accuracy', 'Precision', 'Recall', 'F1']
    method_names = list(results.keys())
    x = np.arange(len(metric_names_bench))
    width = 0.18

    for mi, mname in enumerate(method_names):
        vals = [results[mname][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
        bars = ax.bar(x + mi * width - 1.5 * width, vals, width,
                      label=mname, color=COLORS[mi], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names_bench)
    ax.set_ylabel('Score')
    ax.set_title('Comparative Benchmark: Method Performance', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'benchmark_bars.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# SIMULATION 5: Edge Cases / Robustness
# ============================================================================
def simulation_5_edge_cases():
    print("=" * 60)
    print("SIMULATION 5: Edge Cases / Robustness")
    print("=" * 60)

    scorer = ConfidenceScorer()
    rng = np.random.default_rng(321)

    edge_cases = {}

    # (a) Pure noise
    edge_cases['Pure noise'] = rng.normal(0, 0.1, len(WAVENUMBER))

    # (b) Single cosmic ray spike
    spec_cosmic = rng.normal(0, 0.01, len(WAVENUMBER))
    spec_cosmic[750] = 100.0  # huge spike
    edge_cases['Cosmic ray spike'] = spec_cosmic

    # (c) Baseline >> peaks (heavy fluorescence)
    signal = np.zeros_like(WAVENUMBER, dtype=float)
    for x0, amp, gamma in SERS_PEAK_LIBRARY['generic']:
        signal += lorentzian(WAVENUMBER, x0, amp * 0.01, gamma)
    from src.utils.synthetic_data import generate_fluorescence
    fluor = generate_fluorescence(WAVENUMBER, intensity=5.0, rng=rng)
    edge_cases['Heavy fluorescence'] = signal + fluor

    # (d) Heavily overlapping peaks
    signal_overlap = np.zeros_like(WAVENUMBER, dtype=float)
    for center in [1000, 1010, 1020, 1030, 1040]:
        signal_overlap += lorentzian(WAVENUMBER, center, 1.0, 15)
    noise = rng.normal(0, 0.01, len(WAVENUMBER))
    edge_cases['Overlapping peaks'] = signal_overlap + noise

    # (e) Negative intensities
    spec_neg = generate_single_spectrum(WAVENUMBER, snr=50, rng=rng)
    spec_neg -= np.max(spec_neg) * 1.5  # shift everything negative
    edge_cases['Negative intensities'] = spec_neg

    # (f) Short wavenumber range (200 cm-1 span)
    wn_short = np.linspace(1000, 1200, 100)
    spec_short_signal = np.zeros_like(wn_short, dtype=float)
    spec_short_signal += lorentzian(wn_short, 1100, 1.0, 8)
    spec_short_signal += rng.normal(0, 0.02, len(wn_short))
    edge_cases['Short range (200 cm-1)'] = (spec_short_signal, wn_short)

    # (g) All zeros
    edge_cases['All zeros'] = np.zeros(len(WAVENUMBER))

    # (h) Contains NaN values
    spec_nan = generate_single_spectrum(WAVENUMBER, snr=50, rng=rng)
    spec_nan[100:110] = np.nan
    spec_nan[500] = np.inf
    edge_cases['Contains NaN/Inf'] = spec_nan

    # Evaluate each
    results_table = []
    case_names = []
    heatmap_data = []

    for case_name, data in edge_cases.items():
        case_names.append(case_name)
        row = {}
        error_msg = None

        if isinstance(data, tuple):
            spec, wn = data
        else:
            spec, wn = data, WAVENUMBER

        try:
            result = safe_score(scorer, spec, wn)
            for mk, mn in zip(METRIC_KEYS, METRIC_NAMES):
                row[mn] = result['scores'][mk]
            row['Confidence'] = result['confidence']
            row['Grade'] = result['grade']
        except Exception as e:
            error_msg = str(e)
            for mn in METRIC_NAMES:
                row[mn] = 0.0
            row['Confidence'] = 0.0
            row['Grade'] = 'Error'

        row['Error'] = error_msg
        results_table.append(row)
        heatmap_data.append([row[mn] for mn in METRIC_NAMES])

        status = f"  {case_name}: Confidence={row['Confidence']:.1f}, Grade={row['Grade']}"
        if error_msg:
            status += f" [ERROR: {error_msg}]"
        print(status)

    # Save markdown report
    md_lines = ['# Edge Cases Report\n']
    md_lines.append('| Case | ' + ' | '.join(METRIC_NAMES) + ' | Confidence | Grade | Error |')
    md_lines.append('|' + '---|' * (len(METRIC_NAMES) + 3))
    for i, case_name in enumerate(case_names):
        row = results_table[i]
        vals = ' | '.join([f'{row[mn]:.3f}' for mn in METRIC_NAMES])
        err = row['Error'] if row['Error'] else '-'
        md_lines.append(f'| {case_name} | {vals} | {row["Confidence"]:.1f} | {row["Grade"]} | {err} |')

    md_path = os.path.join(RESULTS_DIR, 'edge_cases_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"  Saved: {md_path}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    hm = np.array(heatmap_data)
    im = ax.imshow(hm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names, fontsize=9)

    # Annotate cells
    for i in range(len(case_names)):
        for j in range(len(METRIC_NAMES)):
            val = hm[i, j]
            color = 'white' if val < 0.3 or val > 0.8 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score (0-1)')
    ax.set_title('Edge Cases: Metric Scores Heatmap', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'edge_cases_heatmap.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    print("\nSpectraGuard Simulation Suite")
    print("=" * 60)

    simulation_1_sensitivity()
    simulation_2_cross_instrument()
    simulation_3_concentration()
    simulation_4_benchmark()
    simulation_5_edge_cases()

    print("\n" + "=" * 60)
    print("All simulations complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)
