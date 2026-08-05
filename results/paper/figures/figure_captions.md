# Figure Captions

> **Data origin.** Figures 1–5 and S1 are generated from **synthetic spectra** produced by
> `src/pipeline/generate_figures.py`. Only Figure 6 uses real measured spectra (RRUFF).
> No figure in this folder has been submitted to or published in any journal.

## Figure 1: SpectraGuard Framework Overview
Schematic diagram of the SpectraGuard quality assessment framework. A raw SERS spectrum is evaluated by six complementary metrics—Signal-to-Noise Ratio, Baseline Stability, Peak Reproducibility, Spectral Resolution, Fluorescence Contamination Index, and Hotspot Uniformity Index—each normalized to a 0–1 scale. The metrics are combined through a weighted sum to produce an integrated Confidence Score (0–100) with corresponding quality grades (Excellent ≥ 80, Good 60–79, Marginal 40–59, Poor < 40).

## Figure 2: Sensitivity Matrix
Response of each metric (columns) to systematic degradation of individual quality parameters (rows). The highlighted metric (colored line) corresponds to the parameter's primary target; non-target metrics (gray dashed) show limited cross-sensitivity. Parameters varied: SNR (100 → 5), baseline drift (0 → 1.0), peak width γ (5 → 40 cm⁻¹), hotspot RSD (0 → 40%), and fluorescence level (0 → 2.0). Each point represents the mean of 20 independently generated **synthetic** spectra.

## Figure 3: Cross-Instrument Transfer Analysis
(a) Confidence Score distributions for three **virtual** instruments (parameterized after published Renishaw, Horiba and Bruker specifications; no physical instrument was measured) before calibration correction. (b) Distributions after wavenumber shift and intensity scale correction. (c) Bland-Altman plot comparing virtual-Renishaw vs. virtual-Horiba scores before (blue) and after (green) correction, with mean difference lines shown as dashed horizontals.

## Figure 4: Concentration-Dependent Spectral Quality
Mean Confidence Score (±1 SD, n = 100 **synthetic** spectra per concentration) as a function of nominal analyte concentration in the generator model. Background colors indicate quality grade regions. The inverse-U shape is a property of the generator's noise and saturation terms; it reproduces the qualitative trend reported for SERS but is not an independent measurement of it.

## Figure 5: Metric Complementarity Illustration (**not** a benchmark)
(a) ROC curves and (b) per-scenario accuracy for four scoring rules applied to four synthetic degradation scenarios.

**This comparison is circular by construction and must not be quoted as evidence that SpectraGuard outperforms any real method or any human analyst.** Three things make it circular:

1. The ground-truth label is assigned by the synthetic generator from its own parameters (SNR, baseline drift, hotspot RSD, fluorescence level).
2. SpectraGuard scores those same parameters, so it is being graded against its own inputs.
3. The comparison rules are deliberately handicapped: "SNR-only" and "HQI" see one dimension each, and **"Expert heuristic (sim.)" is not a human expert** — it is the mean of three hand-written linear rules over SNR and HQI plus `uniform(-0.1, 0.1)` noise (see `generate_figures.py`). No analyst scored these spectra.

What the figure legitimately shows: single-dimension rules cannot separate degradations along dimensions they do not measure, which is the motivation for a multi-metric score. It does not measure how well that score works. A real benchmark needs human-labeled spectra and independently implemented published baselines; neither exists here yet.

## Figure 6: Real Data Validation
(a) Radar chart showing mean metric profiles for Good, Marginal, and Bad quality grades. (b) KDE comparison of SNR score distributions between synthetic and real spectral data. (c) PCA projection of 6-metric feature space showing distribution overlap between synthetic and real data.

Real spectra are from the public RRUFF database. Grades are **auto-labeled** by `ConfidenceScorer` + GMM (`run_real_data_pipeline.py`), not by human annotation — so (a) describes how the tool clusters real data, not how well it agrees with expert judgement.

## Figure S1: Edge Case Robustness
Heatmap of metric scores (0–1) across eight edge case scenarios. All cases were processed without computational errors. Pathological inputs (pure noise, all zeros, NaN contamination) correctly receive near-zero scores, while challenging but valid spectra (negative baseline shift, overlapping peaks) receive appropriately modulated scores. This is a robustness / no-crash check, not an accuracy result.
