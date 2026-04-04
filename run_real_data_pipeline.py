"""
Full Real Data Analysis Pipeline for SpectraGuard.

Steps:
1. Parse RRUFF data and save unified format
2. Quality auto-labeling with ConfidenceScorer + GMM
3. Synthetic vs Real comparison (KDE plots)
4. Weight recalibration analysis

Usage:
    ./venv/Scripts/python run_real_data_pipeline.py
"""

import os
import sys
import pickle
import warnings
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

# Project imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pipeline.preprocessor import load_rruff_spectra, save_unified, load_unified
from src.metrics.confidence_score import ConfidenceScorer, DEFAULT_WEIGHTS
from src.metrics.snr import calculate_snr
from src.metrics.baseline import calculate_baseline
from src.metrics.peak_reproducibility import calculate_peak_reproducibility
from src.metrics.resolution import calculate_resolution
from src.metrics.fluorescence import calculate_fluorescence
from src.metrics.hotspot_uniformity import calculate_hotspot_uniformity

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
SPECTRA_DIR = os.path.join(project_root, 'data', 'raw', 'rruff', 'spectra')
UNIFIED_PATH = os.path.join(project_root, 'data', 'processed', 'unified_spectra.pkl')
SYNTHETIC_PATH = os.path.join(project_root, 'data', 'raw', 'synthetic_sers.npz')
RESULTS_DIR = os.path.join(project_root, 'results', 'real_data')
LABELED_PATH = os.path.join(project_root, 'data', 'processed', 'labeled_spectra.pkl')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(UNIFIED_PATH), exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'Good': '#2ecc71', 'Marginal': '#f39c12', 'Bad': '#e74c3c'}

report_lines = []

def add_report(text):
    report_lines.append(text)
    print(text)


# ============================================================
# STEP 1: Parse RRUFF Data
# ============================================================
add_report("# SpectraGuard Real Data Analysis Report\n")
add_report("## 1. Data Loading & Preprocessing\n")

if os.path.exists(UNIFIED_PATH):
    add_report("Loading cached unified spectra...")
    spectra_list = load_unified(UNIFIED_PATH)
    add_report(f"Loaded {len(spectra_list)} cached spectra.")
else:
    add_report("Parsing RRUFF Processed files...")
    spectra_list = load_rruff_spectra(SPECTRA_DIR, max_spectra=2000, processed_only=True)
    save_unified(spectra_list, UNIFIED_PATH)

# Summary stats
minerals = set(s['metadata']['mineral_name'] for s in spectra_list)
lasers = set(s['metadata']['laser_wavelength'] for s in spectra_list)
lengths = [len(s['wavenumber']) for s in spectra_list]

add_report(f"- Total spectra loaded: {len(spectra_list)}")
add_report(f"- Unique minerals: {len(minerals)}")
add_report(f"- Laser wavelengths: {sorted(lasers)}")
add_report(f"- Spectrum lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
add_report("")

# ============================================================
# STEP 2: Quality Auto-labeling
# ============================================================
add_report("## 2. Quality Scoring & Auto-labeling\n")

scorer = ConfidenceScorer()

all_scores = []
all_metric_scores = []
metric_names = ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']

add_report("Scoring all spectra (this may take a while)...")
for i, spec_data in enumerate(spectra_list):
    if (i + 1) % 200 == 0:
        print(f"  Scored {i + 1}/{len(spectra_list)}...")

    wn = spec_data['wavenumber']
    intensity = spec_data['intensity']

    try:
        result = scorer.score(intensity, wn)
        all_scores.append(result['confidence'])
        all_metric_scores.append(result['scores'])
        spec_data['quality_result'] = result
    except Exception as e:
        # Fallback: assign 0
        all_scores.append(0.0)
        all_metric_scores.append({m: 0.0 for m in metric_names})
        spec_data['quality_result'] = {
            'scores': {m: 0.0 for m in metric_names},
            'confidence': 0.0,
            'grade': 'Poor',
            'weights': DEFAULT_WEIGHTS.copy()
        }

all_scores = np.array(all_scores)

add_report(f"Score statistics:")
add_report(f"  Mean: {np.mean(all_scores):.1f}")
add_report(f"  Median: {np.median(all_scores):.1f}")
add_report(f"  Std: {np.std(all_scores):.1f}")
add_report(f"  Min: {np.min(all_scores):.1f}, Max: {np.max(all_scores):.1f}")
add_report(f"  Q25: {np.percentile(all_scores, 25):.1f}, Q75: {np.percentile(all_scores, 75):.1f}")
add_report("")

# --- Score distribution plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(all_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(80, color='green', linestyle='--', linewidth=2, label='Excellent (80)')
axes[0].axvline(60, color='orange', linestyle='--', linewidth=2, label='Good (60)')
axes[0].axvline(40, color='red', linestyle='--', linewidth=2, label='Marginal (40)')
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Count')
axes[0].set_title('RRUFF Real Data - Score Distribution')
axes[0].legend()

# KDE
try:
    kde = gaussian_kde(all_scores[np.isfinite(all_scores)])
    x_range = np.linspace(0, 100, 500)
    axes[1].plot(x_range, kde(x_range), color='steelblue', linewidth=2)
    axes[1].fill_between(x_range, kde(x_range), alpha=0.3, color='steelblue')
except Exception:
    axes[1].hist(all_scores, bins=50, density=True, alpha=0.7)
axes[1].axvline(80, color='green', linestyle='--', linewidth=2)
axes[1].axvline(60, color='orange', linestyle='--', linewidth=2)
axes[1].axvline(40, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Confidence Score')
axes[1].set_ylabel('Density')
axes[1].set_title('Score Distribution (KDE)')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'score_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- GMM clustering (k=3) ---
add_report("### GMM Auto-clustering (k=3)\n")
scores_2d = all_scores.reshape(-1, 1)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(scores_2d)
gmm_labels = gmm.predict(scores_2d)

# Sort clusters by mean score
cluster_means = []
for c in range(3):
    mask = gmm_labels == c
    cluster_means.append((c, np.mean(all_scores[mask])))
cluster_means.sort(key=lambda x: x[1], reverse=True)

# Map: highest mean -> Good, middle -> Marginal, lowest -> Bad
label_map = {}
quality_names = ['Good', 'Marginal', 'Bad']
for rank, (cluster_id, mean_score) in enumerate(cluster_means):
    label_map[cluster_id] = quality_names[rank]
    add_report(f"  Cluster {cluster_id}: mean={mean_score:.1f}, label={quality_names[rank]}, "
               f"count={np.sum(gmm_labels == cluster_id)}")

quality_labels = np.array([label_map[l] for l in gmm_labels])

# Find GMM boundaries
gmm_boundaries = []
for i in range(len(cluster_means) - 1):
    c1, m1 = cluster_means[i]
    c2, m2 = cluster_means[i + 1]
    # Find crossover point
    x_test = np.linspace(min(m1, m2), max(m1, m2), 1000).reshape(-1, 1)
    probs = gmm.predict_proba(x_test)
    # Find where the two cluster probabilities cross
    p1 = probs[:, c1]
    p2 = probs[:, c2]
    crossover_idx = np.argmin(np.abs(p1 - p2))
    boundary = x_test[crossover_idx, 0]
    gmm_boundaries.append(boundary)

add_report(f"\n  GMM-derived boundaries: {[f'{b:.1f}' for b in sorted(gmm_boundaries, reverse=True)]}")

# Check if 80/60/40 thresholds work
original_thresholds = [80, 60, 40]
add_report(f"  Original thresholds: {original_thresholds}")

# Apply original thresholds
orig_good = np.sum(all_scores >= 80)
orig_marginal = np.sum((all_scores >= 60) & (all_scores < 80))
orig_bad = np.sum((all_scores >= 40) & (all_scores < 60))
orig_poor = np.sum(all_scores < 40)

add_report(f"\n  With original 80/60/40 thresholds:")
add_report(f"    Excellent (>=80): {orig_good} ({100*orig_good/len(all_scores):.1f}%)")
add_report(f"    Good (60-80): {orig_marginal} ({100*orig_marginal/len(all_scores):.1f}%)")
add_report(f"    Marginal (40-60): {orig_bad} ({100*orig_bad/len(all_scores):.1f}%)")
add_report(f"    Poor (<40): {orig_poor} ({100*orig_poor/len(all_scores):.1f}%)")

# Propose new thresholds based on GMM if needed
sorted_boundaries = sorted(gmm_boundaries, reverse=True)
if len(sorted_boundaries) >= 2:
    proposed_good_threshold = round(sorted_boundaries[0], 0)
    proposed_marginal_threshold = round(sorted_boundaries[1], 0)
else:
    proposed_good_threshold = round(sorted_boundaries[0], 0) if sorted_boundaries else 80
    proposed_marginal_threshold = 40

# Check if thresholds need adjustment
threshold_diff = abs(proposed_good_threshold - 80) + abs(proposed_marginal_threshold - 60)
if threshold_diff > 10:
    add_report(f"\n  **Proposed new thresholds (GMM-based):**")
    add_report(f"    Good: >= {proposed_good_threshold:.0f}")
    add_report(f"    Marginal: {proposed_marginal_threshold:.0f} - {proposed_good_threshold:.0f}")
    add_report(f"    Bad: < {proposed_marginal_threshold:.0f}")
    use_proposed = True
else:
    add_report(f"\n  Original thresholds (80/60/40) are adequate for this data distribution.")
    proposed_good_threshold = 80
    proposed_marginal_threshold = 60
    use_proposed = False

add_report("")

# GMM visualization
fig, ax = plt.subplots(figsize=(10, 5))
for label_name, color in COLORS.items():
    mask = quality_labels == label_name
    if mask.sum() > 0:
        ax.hist(all_scores[mask], bins=30, alpha=0.6, label=f'{label_name} (n={mask.sum()})',
                color=color, edgecolor='white')
for b in gmm_boundaries:
    ax.axvline(b, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(80, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Original: 80')
ax.axvline(60, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Original: 60')
ax.axvline(40, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Original: 40')
ax.set_xlabel('Confidence Score')
ax.set_ylabel('Count')
ax.set_title('GMM Auto-clustering (k=3) with Boundaries')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gmm_clustering.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save labeled data
for i, spec_data in enumerate(spectra_list):
    spec_data['quality_label'] = quality_labels[i]
    spec_data['confidence_score'] = all_scores[i]

with open(LABELED_PATH, 'wb') as f:
    pickle.dump(spectra_list, f)
add_report(f"Saved labeled spectra to {LABELED_PATH}\n")


# ============================================================
# STEP 3: Synthetic vs Real Comparison
# ============================================================
add_report("## 3. Synthetic vs Real Comparison\n")

# Load synthetic data
synth_data = np.load(SYNTHETIC_PATH, allow_pickle=True)
synth_keys = list(synth_data.keys())
add_report(f"Synthetic data keys: {synth_keys}")

synth_wavenumber = synth_data['wavenumber']
synth_spectra = synth_data['spectra']
synth_labels = synth_data['labels'] if 'labels' in synth_data else None

add_report(f"Synthetic: {synth_spectra.shape[0]} spectra, {len(synth_wavenumber)} points")
add_report(f"Real RRUFF: {len(spectra_list)} spectra\n")

# Score synthetic data
add_report("Scoring synthetic spectra...")
synth_scores_all = []
synth_metric_scores = []

for i in range(synth_spectra.shape[0]):
    if (i + 1) % 200 == 0:
        print(f"  Scored synthetic {i + 1}/{synth_spectra.shape[0]}...")
    try:
        result = scorer.score(synth_spectra[i], synth_wavenumber)
        synth_scores_all.append(result['confidence'])
        synth_metric_scores.append(result['scores'])
    except Exception:
        synth_scores_all.append(0.0)
        synth_metric_scores.append({m: 0.0 for m in metric_names})

synth_scores_all = np.array(synth_scores_all)

# Build metric arrays for comparison
real_metrics = {m: np.array([s.get(m, 0.0) for s in all_metric_scores]) for m in metric_names}
synth_metrics = {m: np.array([s.get(m, 0.0) for s in synth_metric_scores]) for m in metric_names}

# KDE comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

metric_diffs = {}

for idx, m in enumerate(metric_names):
    ax = axes[idx]
    real_vals = real_metrics[m]
    synth_vals = synth_metrics[m]

    # Compute KS-like difference (mean absolute difference)
    mean_diff = abs(np.mean(real_vals) - np.mean(synth_vals))
    metric_diffs[m] = mean_diff

    x_range = np.linspace(0, 1, 200)

    try:
        if np.std(real_vals) > 1e-6:
            kde_real = gaussian_kde(real_vals)
            ax.plot(x_range, kde_real(x_range), color='blue', linewidth=2, label='Real (RRUFF)')
            ax.fill_between(x_range, kde_real(x_range), alpha=0.2, color='blue')
    except Exception:
        ax.hist(real_vals, bins=30, density=True, alpha=0.3, color='blue', label='Real (RRUFF)')

    try:
        if np.std(synth_vals) > 1e-6:
            kde_synth = gaussian_kde(synth_vals)
            ax.plot(x_range, kde_synth(x_range), color='red', linewidth=2, label='Synthetic')
            ax.fill_between(x_range, kde_synth(x_range), alpha=0.2, color='red')
    except Exception:
        ax.hist(synth_vals, bins=30, density=True, alpha=0.3, color='red', label='Synthetic')

    ax.set_title(f'{m.upper()} (diff={mean_diff:.3f})')
    ax.set_xlabel('Score (0-1)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle('Metric Distributions: Real (RRUFF) vs Synthetic', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'real_vs_synthetic_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Report which metrics differ most
sorted_diffs = sorted(metric_diffs.items(), key=lambda x: x[1], reverse=True)
add_report("### Metric differences (Real vs Synthetic):\n")
add_report("| Metric | Mean Diff | Real Mean | Synth Mean |")
add_report("|--------|-----------|-----------|------------|")
for m, diff in sorted_diffs:
    add_report(f"| {m} | {diff:.3f} | {np.mean(real_metrics[m]):.3f} | {np.mean(synth_metrics[m]):.3f} |")

add_report(f"\n**Most different metric:** {sorted_diffs[0][0]} (diff={sorted_diffs[0][1]:.3f})")
add_report(f"**Least different metric:** {sorted_diffs[-1][0]} (diff={sorted_diffs[-1][1]:.3f})")
add_report("")


# ============================================================
# STEP 4: Weight Recalibration
# ============================================================
add_report("## 4. Weight Recalibration Analysis\n")

# Build metric matrix for real data
metric_matrix = np.column_stack([real_metrics[m] for m in metric_names])

# Correlation analysis
add_report("### 6-Metric Correlation Matrix (Real Data)\n")
corr_matrix = np.corrcoef(metric_matrix.T)

add_report("| | " + " | ".join(metric_names) + " |")
add_report("|---" * (len(metric_names) + 1) + "|")
for i, m in enumerate(metric_names):
    row = f"| {m} |"
    for j in range(len(metric_names)):
        row += f" {corr_matrix[i, j]:.2f} |"
    add_report(row)
add_report("")

# Correlation plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(metric_names)))
ax.set_yticks(range(len(metric_names)))
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.set_yticklabels(metric_names)
for i in range(len(metric_names)):
    for j in range(len(metric_names)):
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                fontsize=10, color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
plt.colorbar(im)
ax.set_title('6-Metric Correlation Matrix (Real RRUFF Data)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metric_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()

# Metric variance analysis - metrics with more variance are more discriminative
metric_stds = {m: np.std(real_metrics[m]) for m in metric_names}
add_report("### Metric Discriminative Power (Std Dev)\n")
for m in metric_names:
    add_report(f"  {m}: std={metric_stds[m]:.3f}")
add_report("")

# Recalibrate weights based on discriminative power and low redundancy
# Strategy: upweight metrics with high std (discriminative) and low correlation with others
# Penalize highly correlated pairs to reduce redundancy

avg_abs_corr = {}
for i, m in enumerate(metric_names):
    others = [abs(corr_matrix[i, j]) for j in range(len(metric_names)) if j != i]
    avg_abs_corr[m] = np.mean(others)

add_report("### Average |Correlation| with Other Metrics\n")
for m in metric_names:
    add_report(f"  {m}: avg|r|={avg_abs_corr[m]:.3f}")
add_report("")

# Compute recalibrated weights
# Weight ~ std * (1 - avg_abs_corr) to reward discriminative, non-redundant metrics
raw_weights = {}
for m in metric_names:
    raw_weights[m] = metric_stds[m] * (1 - avg_abs_corr[m])

# Normalize to sum = 1
total = sum(raw_weights.values())
if total > 0:
    recalibrated_weights = {m: w / total for m, w in raw_weights.items()}
else:
    recalibrated_weights = DEFAULT_WEIGHTS.copy()

add_report("### Weight Comparison\n")
add_report("| Metric | Original | Recalibrated | Change |")
add_report("|--------|----------|--------------|--------|")
for m in metric_names:
    orig = DEFAULT_WEIGHTS.get(m, 0.0)
    new = recalibrated_weights[m]
    change = new - orig
    direction = "+" if change > 0 else ""
    add_report(f"| {m} | {orig:.2f} | {new:.3f} | {direction}{change:.3f} |")
add_report("")

# Apply recalibrated weights and compare score distributions
recalib_scorer = ConfidenceScorer(weights=recalibrated_weights)
recalib_scores = []
for i, spec_data in enumerate(spectra_list):
    wn = spec_data['wavenumber']
    intensity = spec_data['intensity']
    try:
        result = recalib_scorer.score(intensity, wn)
        recalib_scores.append(result['confidence'])
    except Exception:
        recalib_scores.append(0.0)

recalib_scores = np.array(recalib_scores)

# Before/After comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='white', label='Original')
axes[0].axvline(np.mean(all_scores), color='navy', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(all_scores):.1f}')
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Original Weights')
axes[0].legend()

axes[1].hist(recalib_scores, bins=50, alpha=0.7, color='coral', edgecolor='white', label='Recalibrated')
axes[1].axvline(np.mean(recalib_scores), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(recalib_scores):.1f}')
axes[1].set_xlabel('Confidence Score')
axes[1].set_ylabel('Count')
axes[1].set_title('Recalibrated Weights')
axes[1].legend()

plt.suptitle('Score Distribution: Before vs After Weight Recalibration', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'weight_recalibration_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Overlay KDE comparison
fig, ax = plt.subplots(figsize=(10, 5))
x_range = np.linspace(0, 100, 500)
try:
    kde_orig = gaussian_kde(all_scores[np.isfinite(all_scores)])
    ax.plot(x_range, kde_orig(x_range), color='steelblue', linewidth=2, label='Original Weights')
    ax.fill_between(x_range, kde_orig(x_range), alpha=0.2, color='steelblue')
except Exception:
    pass
try:
    kde_recalib = gaussian_kde(recalib_scores[np.isfinite(recalib_scores)])
    ax.plot(x_range, kde_recalib(x_range), color='coral', linewidth=2, label='Recalibrated Weights')
    ax.fill_between(x_range, kde_recalib(x_range), alpha=0.2, color='coral')
except Exception:
    pass
ax.set_xlabel('Confidence Score')
ax.set_ylabel('Density')
ax.set_title('Score Distribution Comparison')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'score_kde_before_after.png'), dpi=150, bbox_inches='tight')
plt.close()

# Per-metric box plots
fig, axes = plt.subplots(1, 6, figsize=(20, 5))
for idx, m in enumerate(metric_names):
    ax = axes[idx]
    data_to_plot = [real_metrics[m], synth_metrics[m]]
    bp = ax.boxplot(data_to_plot, labels=['Real', 'Synth'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax.set_title(m.upper(), fontsize=11)
    ax.set_ylabel('Score')
plt.suptitle('Per-Metric Box Plots: Real vs Synthetic', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metric_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Summary & Recommendations
# ============================================================
add_report("## 5. Summary & Recommendations\n")

add_report(f"### Data Overview")
add_report(f"- Loaded {len(spectra_list)} real RRUFF Raman spectra (Processed only)")
add_report(f"- Compared against {synth_spectra.shape[0]} synthetic SERS spectra\n")

add_report(f"### Scoring Results")
add_report(f"- Real data mean confidence: {np.mean(all_scores):.1f} (std={np.std(all_scores):.1f})")
add_report(f"- Synthetic data mean confidence: {np.mean(synth_scores_all):.1f} (std={np.std(synth_scores_all):.1f})\n")

add_report(f"### GMM Clustering")
for label_name in ['Good', 'Marginal', 'Bad']:
    n = np.sum(quality_labels == label_name)
    pct = 100 * n / len(quality_labels)
    add_report(f"- {label_name}: {n} spectra ({pct:.1f}%)")
add_report("")

add_report(f"### Top Differentiating Metrics (Real vs Synthetic)")
for m, diff in sorted_diffs[:3]:
    add_report(f"- {m}: mean diff = {diff:.3f}")
add_report("")

add_report(f"### Recalibrated Weights")
for m in metric_names:
    orig = DEFAULT_WEIGHTS.get(m, 0.0)
    new = recalibrated_weights[m]
    add_report(f"- {m}: {orig:.2f} -> {new:.3f}")
add_report("")

add_report("### Output Files")
add_report(f"- Unified spectra: {UNIFIED_PATH}")
add_report(f"- Labeled spectra: {LABELED_PATH}")
add_report(f"- Score distribution: {os.path.join(RESULTS_DIR, 'score_distribution.png')}")
add_report(f"- GMM clustering: {os.path.join(RESULTS_DIR, 'gmm_clustering.png')}")
add_report(f"- Real vs Synthetic: {os.path.join(RESULTS_DIR, 'real_vs_synthetic_comparison.png')}")
add_report(f"- Metric correlation: {os.path.join(RESULTS_DIR, 'metric_correlation.png')}")
add_report(f"- Weight recalibration: {os.path.join(RESULTS_DIR, 'weight_recalibration_comparison.png')}")
add_report(f"- Metric box plots: {os.path.join(RESULTS_DIR, 'metric_boxplots.png')}")

# Save report
report_path = os.path.join(RESULTS_DIR, 'analysis_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n{'='*60}")
print(f"Pipeline complete! Report saved to {report_path}")
print(f"{'='*60}")
