"""
Exploratory Data Analysis (EDA) pipeline for SERS spectra.

Generates comprehensive analysis including:
1. Basic statistics
2. Spectrum visualization (overlay, heatmap, mean±std)
3. Noise characteristics (SG filter, FFT, SNR distribution)
4. Baseline analysis (arPLS fitting)
5. Peak analysis (detection, FWHM, reproducibility)
6. Correlation analysis (6-metric correlation, PCA)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics.snr import calculate_snr
from src.metrics.baseline import calculate_baseline, get_baseline
from src.metrics.peak_reproducibility import calculate_peak_reproducibility
from src.metrics.resolution import calculate_resolution
from src.metrics.fluorescence import calculate_fluorescence
from src.metrics.hotspot_uniformity import calculate_hotspot_uniformity
from src.metrics.confidence_score import ConfidenceScorer


def run_eda(data_path: str, results_dir: str):
    """Run full EDA pipeline."""
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Load data
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    wavenumber = data['wavenumber']
    spectra = data['spectra']
    labels = data['labels']

    grades = ['Good', 'Marginal', 'Bad']
    grade_colors = {'Good': '#2ecc71', 'Marginal': '#f39c12', 'Bad': '#e74c3c'}

    report_lines = ["# SpectraGuard EDA Report\n"]

    # ===== 1. Basic Statistics =====
    print("1. Basic statistics...")
    report_lines.append("## 1. Basic Statistics\n")
    report_lines.append(f"- Total spectra: {len(spectra)}")
    for g in grades:
        report_lines.append(f"- {g}: {np.sum(labels == g)}")
    report_lines.append(f"- Wavenumber range: {wavenumber[0]:.1f} - {wavenumber[-1]:.1f} cm⁻¹")
    report_lines.append(f"- Points per spectrum: {len(wavenumber)}")
    report_lines.append(f"- Intensity range: [{np.min(spectra):.4f}, {np.max(spectra):.4f}]")
    report_lines.append(f"- Mean intensity: {np.mean(spectra):.4f}")
    report_lines.append(f"- NaN values: {np.sum(np.isnan(spectra))}")
    report_lines.append(f"- Inf values: {np.sum(np.isinf(spectra))}\n")

    # ===== 2. Spectrum Visualization =====
    print("2. Spectrum visualization...")

    # 2a. Representative overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, g in enumerate(grades):
        mask = labels == g
        idx = np.where(mask)[0]
        for j in range(min(10, len(idx))):
            axes[i].plot(wavenumber, spectra[idx[j]], alpha=0.5, linewidth=0.7)
        axes[i].set_title(f'{g} (n={mask.sum()})', fontsize=14)
        axes[i].set_xlabel('Wavenumber (cm⁻¹)')
        axes[i].set_ylabel('Intensity')
        axes[i].invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'spectra_overlay.png'), dpi=150)
    plt.close()

    # 2b. Mean ± std
    fig, ax = plt.subplots(figsize=(12, 5))
    for g in grades:
        mask = labels == g
        mean_spec = np.mean(spectra[mask], axis=0)
        std_spec = np.std(spectra[mask], axis=0)
        ax.plot(wavenumber, mean_spec, label=g, color=grade_colors[g], linewidth=1.5)
        ax.fill_between(wavenumber, mean_spec - std_spec, mean_spec + std_spec,
                        alpha=0.2, color=grade_colors[g])
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title('Mean ± Std Spectra by Quality Grade')
    ax.legend()
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mean_std_spectra.png'), dpi=150)
    plt.close()

    # 2c. Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    # Sort by grade
    sort_idx = np.argsort(labels)
    sorted_spectra = spectra[sort_idx]
    im = ax.imshow(sorted_spectra[:500], aspect='auto', cmap='viridis',
                   extent=[wavenumber[-1], wavenumber[0], 500, 0])
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Sample Index')
    ax.set_title('Spectrum Heatmap (first 500 samples, sorted by grade)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'spectrum_heatmap.png'), dpi=150)
    plt.close()

    # ===== 3. Noise Characteristics =====
    print("3. Noise characteristics...")

    # 3a. SG filter before/after
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, g in enumerate(grades):
        mask = labels == g
        idx = np.where(mask)[0][0]
        raw = spectra[idx]
        smoothed = savgol_filter(raw, 21, 3)
        axes[i].plot(wavenumber, raw, alpha=0.5, label='Raw', linewidth=0.7)
        axes[i].plot(wavenumber, smoothed, label='SG Filtered', linewidth=1.2)
        axes[i].set_title(f'{g} - SG Filter')
        axes[i].legend()
        axes[i].invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sg_filter_comparison.png'), dpi=150)
    plt.close()

    # 3b. Noise FFT
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, g in enumerate(grades):
        mask = labels == g
        idx = np.where(mask)[0][0]
        noise = spectra[idx] - savgol_filter(spectra[idx], 21, 3)
        N = len(noise)
        yf = np.abs(fft(noise))[:N // 2]
        xf = fftfreq(N, d=(wavenumber[1] - wavenumber[0]))[:N // 2]
        axes[i].plot(xf[1:], yf[1:], linewidth=0.7)
        axes[i].set_title(f'{g} - Noise FFT')
        axes[i].set_xlabel('Frequency')
        axes[i].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'noise_fft.png'), dpi=150)
    plt.close()

    # 3c. SNR distribution
    print("   Computing SNR distribution (sampling 300 spectra)...")
    snr_scores = {g: [] for g in grades}
    for g in grades:
        mask = labels == g
        idx = np.where(mask)[0]
        sample_idx = idx[np.linspace(0, len(idx) - 1, min(100, len(idx)), dtype=int)]
        for si in sample_idx:
            s = calculate_snr(spectra[si], wavenumber)
            snr_scores[g].append(s)

    fig, ax = plt.subplots(figsize=(8, 5))
    data_for_hist = [snr_scores[g] for g in grades]
    ax.hist(data_for_hist, bins=30, label=grades,
            color=[grade_colors[g] for g in grades], alpha=0.7, stacked=False)
    ax.set_xlabel('SNR Score')
    ax.set_ylabel('Count')
    ax.set_title('SNR Score Distribution by Grade')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'snr_distribution.png'), dpi=150)
    plt.close()

    # ===== 4. Baseline Characteristics =====
    print("4. Baseline analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, g in enumerate(grades):
        mask = labels == g
        idx = np.where(mask)[0][0]
        spec = spectra[idx]
        bl = get_baseline(spec)
        corrected = spec - bl
        axes[i].plot(wavenumber, spec, alpha=0.5, label='Original', linewidth=0.7)
        axes[i].plot(wavenumber, bl, label='Baseline', linewidth=1.2, color='red')
        axes[i].plot(wavenumber, corrected, label='Corrected', linewidth=1, color='green')
        axes[i].set_title(f'{g} - Baseline')
        axes[i].legend(fontsize=8)
        axes[i].invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'baseline_correction.png'), dpi=150)
    plt.close()

    # ===== 5. Peak Analysis =====
    print("5. Peak analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, g in enumerate(grades):
        mask = labels == g
        idx = np.where(mask)[0][0]
        spec = spectra[idx]
        smoothed = savgol_filter(spec, 21, 3)
        height_threshold = np.max(smoothed) * 0.1
        peaks, props = find_peaks(smoothed, height=height_threshold, distance=20)
        axes[i].plot(wavenumber, smoothed, linewidth=1)
        if len(peaks) > 0:
            axes[i].plot(wavenumber[peaks], smoothed[peaks], 'rv', markersize=8)
            for p in peaks[:5]:
                axes[i].annotate(f'{wavenumber[p]:.0f}', (wavenumber[p], smoothed[p]),
                                textcoords="offset points", xytext=(0, 10), fontsize=7)
        axes[i].set_title(f'{g} - Peak Detection ({len(peaks)} peaks)')
        axes[i].invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'peak_detection.png'), dpi=150)
    plt.close()

    # ===== 6. Correlation & PCA =====
    print("6. Computing 6-metric scores for correlation analysis (sampling 600)...")

    scorer = ConfidenceScorer()
    metric_names = ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']
    all_scores = {m: [] for m in metric_names}
    all_confidence = []
    all_grade_labels = []

    for g in grades:
        mask = labels == g
        idx = np.where(mask)[0]
        sample_idx = idx[np.linspace(0, len(idx) - 1, min(200, len(idx)), dtype=int)]
        for si in sample_idx:
            result = scorer.score(spectra[si], wavenumber)
            for m in metric_names:
                all_scores[m].append(result['scores'][m])
            all_confidence.append(result['confidence'])
            all_grade_labels.append(g)

    # 6a. Correlation matrix
    import pandas as pd
    score_df = pd.DataFrame(all_scores)
    score_df['confidence'] = all_confidence
    score_df['grade'] = all_grade_labels

    fig, ax = plt.subplots(figsize=(8, 7))
    corr = score_df[metric_names + ['confidence']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title('6-Metric Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metric_correlation.png'), dpi=150)
    plt.close()

    # 6b. PCA
    X = score_df[metric_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for g in grades:
        mask_g = np.array(all_grade_labels) == g
        ax.scatter(X_pca[mask_g, 0], X_pca[mask_g, 1],
                   label=g, color=grade_colors[g], alpha=0.6, s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA of 6-Metric Scores by Quality Grade')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pca_quality_grades.png'), dpi=150)
    plt.close()

    # 6c. Boxplots per metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, m in enumerate(metric_names):
        ax = axes[i // 3, i % 3]
        data = [score_df[score_df['grade'] == g][m].values for g in grades]
        bp = ax.boxplot(data, labels=grades, patch_artist=True)
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(grade_colors[grades[j]])
            patch.set_alpha(0.7)
        ax.set_title(m.upper())
        ax.set_ylabel('Score')
    plt.suptitle('Metric Score Distributions by Quality Grade', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metric_boxplots.png'), dpi=150)
    plt.close()

    # Confidence score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in grades:
        mask_g = np.array(all_grade_labels) == g
        conf_vals = np.array(all_confidence)[mask_g]
        ax.hist(conf_vals, bins=25, alpha=0.6, label=g, color=grade_colors[g])
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Score Distribution by Grade')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confidence_distribution.png'), dpi=150)
    plt.close()

    # ===== Write Report =====
    report_lines.append("## 2. Spectrum Visualization")
    report_lines.append("- ![Overlay](spectra_overlay.png)")
    report_lines.append("- ![Mean±Std](mean_std_spectra.png)")
    report_lines.append("- ![Heatmap](spectrum_heatmap.png)\n")

    report_lines.append("## 3. Noise Characteristics")
    report_lines.append("- ![SG Filter](sg_filter_comparison.png)")
    report_lines.append("- ![FFT](noise_fft.png)")
    report_lines.append("- ![SNR Distribution](snr_distribution.png)\n")

    report_lines.append("## 4. Baseline Analysis")
    report_lines.append("- ![Baseline](baseline_correction.png)\n")

    report_lines.append("## 5. Peak Analysis")
    report_lines.append("- ![Peaks](peak_detection.png)\n")

    report_lines.append("## 6. Correlation & PCA")
    report_lines.append("- ![Correlation](metric_correlation.png)")
    report_lines.append("- ![PCA](pca_quality_grades.png)")
    report_lines.append("- ![Boxplots](metric_boxplots.png)")
    report_lines.append("- ![Confidence](confidence_distribution.png)\n")

    # Summary stats
    report_lines.append("## Summary Statistics\n")
    report_lines.append("| Grade | Mean Confidence | Std |")
    report_lines.append("|-------|----------------|-----|")
    for g in grades:
        mask_g = np.array(all_grade_labels) == g
        conf_vals = np.array(all_confidence)[mask_g]
        report_lines.append(f"| {g} | {np.mean(conf_vals):.1f} | {np.std(conf_vals):.1f} |")

    report_lines.append("\n## Per-Metric Means\n")
    report_lines.append("| Metric | Good | Marginal | Bad |")
    report_lines.append("|--------|------|----------|-----|")
    for m in metric_names:
        vals = []
        for g in grades:
            mask_g = np.array(all_grade_labels) == g
            vals.append(f"{np.mean(score_df[m].values[mask_g]):.3f}")
        report_lines.append(f"| {m} | {' | '.join(vals)} |")

    report_path = os.path.join(results_dir, 'eda_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\nEDA complete! Report saved to {report_path}")
    print(f"Plots saved to {results_dir}/")


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(base_dir, 'data', 'raw', 'synthetic_sers.npz')
    results_dir = os.path.join(base_dir, 'results')

    if not os.path.exists(data_path):
        print("Generating synthetic data first...")
        from src.utils.synthetic_data import generate_dataset
        data = generate_dataset()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez_compressed(data_path,
                            wavenumber=data['wavenumber'],
                            spectra=data['spectra'],
                            labels=data['labels'])
        print(f"Saved to {data_path}")

    run_eda(data_path, results_dir)
