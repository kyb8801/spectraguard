"""
Validation pipeline for 6-Metric Confidence Score.

1. Synthetic data validation (Good>75, Bad<40)
2. ML classification comparison (RF vs threshold)
3. Cross-instrument simulation
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics.confidence_score import ConfidenceScorer
from src.utils.synthetic_data import generate_dataset


def run_validation(results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    scorer = ConfidenceScorer()
    metric_names = ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']
    grades_order = ['Good', 'Marginal', 'Bad']
    grade_colors = {'Good': '#2ecc71', 'Marginal': '#f39c12', 'Bad': '#e74c3c'}

    # ===== 1. Synthetic Data Validation =====
    print("1. Synthetic data validation...")
    data = generate_dataset(n_good=100, n_marginal=100, n_bad=100, seed=99)
    wavenumber = data['wavenumber']
    spectra = data['spectra']
    labels = data['labels']

    all_results = []
    for i, spec in enumerate(spectra):
        result = scorer.score(spec, wavenumber)
        result['true_label'] = labels[i]
        all_results.append(result)

    # Check thresholds
    good_scores = [r['confidence'] for r in all_results if r['true_label'] == 'Good']
    bad_scores = [r['confidence'] for r in all_results if r['true_label'] == 'Bad']

    print(f"  Good mean: {np.mean(good_scores):.1f} (target > 75)")
    print(f"  Bad mean: {np.mean(bad_scores):.1f} (target < 40)")

    # Boxplot per grade
    fig, ax = plt.subplots(figsize=(8, 6))
    data_box = [[r['confidence'] for r in all_results if r['true_label'] == g]
                for g in grades_order]
    bp = ax.boxplot(data_box, labels=grades_order, patch_artist=True)
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(grade_colors[grades_order[j]])
        patch.set_alpha(0.7)
    ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='Good threshold (75)')
    ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Bad threshold (40)')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Confidence Score Validation by Quality Grade')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'validation_boxplot.png'), dpi=150)
    plt.close()

    # Per-metric boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, m in enumerate(metric_names):
        ax = axes[i // 3, i % 3]
        data_m = [[r['scores'][m] for r in all_results if r['true_label'] == g]
                  for g in grades_order]
        bp = ax.boxplot(data_m, labels=grades_order, patch_artist=True)
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(grade_colors[grades_order[j]])
            patch.set_alpha(0.7)
        ax.set_title(m.upper())
    plt.suptitle('Per-Metric Score Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'validation_metrics.png'), dpi=150)
    plt.close()

    # ===== 2. ML Classification Comparison =====
    print("2. ML classification comparison...")

    X = np.array([[r['scores'][m] for m in metric_names] for r in all_results])
    le = LabelEncoder()
    y = le.fit_transform([r['true_label'] for r in all_results])

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    y_pred_rf = cross_val_predict(rf, X, y, cv=5)

    # Threshold-based
    y_pred_thresh = []
    for r in all_results:
        if r['confidence'] >= 75:
            y_pred_thresh.append('Good')
        elif r['confidence'] >= 45:
            y_pred_thresh.append('Marginal')
        else:
            y_pred_thresh.append('Bad')
    y_pred_thresh = le.transform(y_pred_thresh)

    thresh_acc = np.mean(y_pred_thresh == y)
    rf_acc = np.mean(cv_scores)

    print(f"  Threshold accuracy: {thresh_acc:.3f}")
    print(f"  Random Forest CV accuracy: {rf_acc:.3f} ± {np.std(cv_scores):.3f}")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_pred, title in [
        (axes[0], y_pred_thresh, f'Threshold (acc={thresh_acc:.3f})'),
        (axes[1], y_pred_rf, f'Random Forest (acc={rf_acc:.3f})'),
    ]:
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=150)
    plt.close()

    # Feature importance
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(metric_names)), importances[sorted_idx],
           color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels([metric_names[i] for i in sorted_idx], rotation=30)
    ax.set_ylabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=150)
    plt.close()

    # ===== 3. Cross-instrument Simulation =====
    print("3. Cross-instrument simulation...")

    rng = np.random.default_rng(42)
    good_mask = labels == 'Good'
    good_spectra = spectra[good_mask][:50]

    # Simulate instrument variations
    shifts = [-3, -1, 0, 1, 3]  # cm^-1
    scale_factors = [0.85, 0.95, 1.0, 1.05, 1.15]
    noise_levels = [0.0, 0.01, 0.02, 0.05]

    original_scores = []
    shifted_scores = {s: [] for s in shifts}

    for spec in good_spectra:
        orig = scorer.score(spec, wavenumber)
        original_scores.append(orig['confidence'])

        for shift in shifts:
            shifted_wn = wavenumber + shift
            result = scorer.score(spec, shifted_wn)
            shifted_scores[shift].append(result['confidence'])

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(shifts)))
    data_shift = [shifted_scores[s] for s in shifts]
    bp = ax.boxplot(data_shift, labels=[f'{s:+d}' for s in shifts], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax.set_xlabel('Wavenumber Shift (cm⁻¹)')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Cross-Instrument: Wavenumber Shift Effect')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_instrument_shift.png'), dpi=150)
    plt.close()

    # Scale factor effect
    scaled_scores = {s: [] for s in scale_factors}
    for spec in good_spectra:
        for scale in scale_factors:
            result = scorer.score(spec * scale, wavenumber)
            scaled_scores[scale].append(result['confidence'])

    fig, ax = plt.subplots(figsize=(8, 5))
    data_scale = [scaled_scores[s] for s in scale_factors]
    bp = ax.boxplot(data_scale, labels=[f'×{s:.2f}' for s in scale_factors],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    ax.set_xlabel('Intensity Scale Factor')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Cross-Instrument: Intensity Scaling Effect')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_instrument_scale.png'), dpi=150)
    plt.close()

    print("\nValidation complete!")
    print(f"Results saved to {results_dir}/")


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    run_validation(os.path.join(base_dir, 'results'))
