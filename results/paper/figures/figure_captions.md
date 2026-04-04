# Figure Captions

## Figure 1: SpectraGuard Framework Overview
Schematic diagram of the SpectraGuard quality assessment framework. A raw SERS spectrum is evaluated by six complementary metrics—Signal-to-Noise Ratio, Baseline Stability, Peak Reproducibility, Spectral Resolution, Fluorescence Contamination Index, and Hotspot Uniformity Index—each normalized to a 0–1 scale. The metrics are combined through a weighted sum to produce an integrated Confidence Score (0–100) with corresponding quality grades (Excellent ≥ 80, Good 60–79, Marginal 40–59, Poor < 40).

## Figure 2: Sensitivity Matrix
Response of each metric (columns) to systematic degradation of individual quality parameters (rows). The highlighted metric (colored line) corresponds to the parameter's primary target; non-target metrics (gray dashed) show limited cross-sensitivity. Parameters varied: SNR (100 → 5), baseline drift (0 → 1.0), peak width γ (5 → 40 cm⁻¹), hotspot RSD (0 → 40%), and fluorescence level (0 → 2.0). Each point represents the mean of 20 independently generated spectra.

## Figure 3: Cross-Instrument Transfer Analysis
(a) Confidence Score distributions for three virtual instruments (Renishaw, Horiba, Bruker) before calibration correction. (b) Distributions after wavenumber shift and intensity scale correction. (c) Bland-Altman plot comparing Renishaw vs. Horiba scores before (blue) and after (green) correction, with mean difference lines shown as dashed horizontals.

## Figure 4: Concentration-Dependent Spectral Quality
Mean Confidence Score (±1 SD, n = 100 per concentration) as a function of analyte concentration. Background colors indicate quality grade regions. The characteristic inverse-U shape reflects noise-dominated spectra at low concentrations and saturation/broadening effects at high concentrations, with optimal SERS performance in the 100 nM – 10 μM range.

## Figure 5: Comparative Benchmark
(a) ROC curves for four quality assessment methods across four test scenarios. SpectraGuard achieves the highest AUC by leveraging multi-dimensional quality assessment. (b) Per-scenario accuracy comparison showing that single-metric methods (SNR-only, HQI) fail in scenarios involving multi-dimensional quality degradation.

## Figure 6: Real Data Validation
(a) Radar chart showing mean metric profiles for Good, Marginal, and Bad quality grades. (b) KDE comparison of SNR score distributions between synthetic and real spectral data. (c) PCA projection of 6-metric feature space showing distribution overlap between synthetic training data and real validation data.

## Figure S1: Edge Case Robustness
Heatmap of metric scores (0–1) across eight edge case scenarios. All cases were processed without computational errors. Pathological inputs (pure noise, all zeros, NaN contamination) correctly receive near-zero scores, while challenging but valid spectra (negative baseline shift, overlapping peaks) receive appropriately modulated scores.
