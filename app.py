"""
SpectraGuard Dashboard - Streamlit-based SERS spectrum QC analyzer.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.metrics.confidence_score import ConfidenceScorer
from src.metrics.snr import calculate_snr_raw
from src.metrics.baseline import get_baseline
from src.metrics.resolution import get_peak_fwhms
from src.metrics.fluorescence import get_fluorescence_ratio


st.set_page_config(page_title="SpectraGuard", page_icon="🔬", layout="wide")

st.title("SpectraGuard - SERS Spectrum Quality Analyzer")
st.markdown("Upload a spectrum (CSV/TXT) to get a 6-Metric Confidence Score.")

METRIC_DESCRIPTIONS = {
    'snr': 'Signal-to-Noise Ratio',
    'baseline': 'Baseline Stability',
    'peak': 'Peak Reproducibility',
    'resolution': 'Spectral Resolution (FWHM)',
    'fluorescence': 'Fluorescence Contamination',
    'uniformity': 'Hotspot Uniformity',
}

GRADE_COLORS = {
    'Excellent': '#2ecc71',
    'Good': '#3498db',
    'Marginal': '#f39c12',
    'Poor': '#e74c3c',
}


def parse_spectrum_file(uploaded_file) -> tuple:
    """Parse CSV or TXT file into wavenumber and intensity arrays."""
    content = uploaded_file.read().decode('utf-8')
    lines = content.strip().split('\n')

    # Detect separator
    sep = ','
    if '\t' in lines[0]:
        sep = '\t'
    elif ';' in lines[0]:
        sep = ';'

    # Skip comment lines
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]

    # Try to detect header
    try:
        float(data_lines[0].split(sep)[0])
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        data_lines = data_lines[1:]

    wavenumber = []
    intensity = []
    for line in data_lines:
        parts = line.strip().split(sep)
        if len(parts) >= 2:
            try:
                wavenumber.append(float(parts[0]))
                intensity.append(float(parts[1]))
            except ValueError:
                continue
        elif len(parts) == 1:
            try:
                intensity.append(float(parts[0]))
            except ValueError:
                continue

    if not wavenumber and intensity:
        wavenumber = list(range(len(intensity)))

    return np.array(wavenumber), np.array(intensity)


# Sidebar
st.sidebar.header("Settings")
use_demo = st.sidebar.checkbox("Use demo spectrum", value=True)

if use_demo:
    from src.utils.synthetic_data import generate_single_spectrum
    demo_quality = st.sidebar.selectbox("Demo quality", ["Good", "Marginal", "Bad"])
    rng = np.random.default_rng(42)
    wavenumber = np.linspace(200, 3200, 1500)

    params = {
        'Good': dict(snr=50, baseline_drift=0.02, hotspot_rsd=8, fluorescence_level=0.05),
        'Marginal': dict(snr=20, baseline_drift=0.15, hotspot_rsd=18, fluorescence_level=0.25),
        'Bad': dict(snr=8, baseline_drift=0.35, hotspot_rsd=30, fluorescence_level=0.5),
    }
    intensity = generate_single_spectrum(wavenumber, **params[demo_quality], rng=rng)
    st.sidebar.success(f"Loaded {demo_quality} demo spectrum")
else:
    uploaded = st.sidebar.file_uploader("Upload spectrum (CSV/TXT)",
                                        type=['csv', 'txt', 'tsv'])
    if uploaded is not None:
        wavenumber, intensity = parse_spectrum_file(uploaded)
        st.sidebar.success(f"Loaded {len(wavenumber)} points")
    else:
        st.info("Please upload a spectrum file or enable demo mode.")
        st.stop()

# Score
scorer = ConfidenceScorer()
result = scorer.score(intensity, wavenumber)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Spectrum plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavenumber, y=intensity, mode='lines',
                             name='Spectrum', line=dict(color='#3498db')))
    bl = get_baseline(intensity)
    fig.add_trace(go.Scatter(x=wavenumber, y=bl, mode='lines',
                             name='Baseline', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title='Spectrum',
        xaxis_title='Wavenumber (cm⁻¹)',
        yaxis_title='Intensity',
        xaxis=dict(autorange='reversed'),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Confidence score display
    grade = result['grade']
    conf = result['confidence']
    st.markdown(f"### Confidence Score")
    st.markdown(
        f"<div style='text-align:center; padding:20px; background-color:{GRADE_COLORS[grade]}20; "
        f"border-radius:10px; border:2px solid {GRADE_COLORS[grade]}'>"
        f"<h1 style='color:{GRADE_COLORS[grade]}; margin:0'>{conf:.1f}</h1>"
        f"<h3 style='color:{GRADE_COLORS[grade]}; margin:0'>{grade}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Radar chart
    categories = list(METRIC_DESCRIPTIONS.keys())
    values = [result['scores'][m] for m in categories]
    values.append(values[0])  # close the polygon

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=[METRIC_DESCRIPTIONS[m] for m in categories] + [METRIC_DESCRIPTIONS[categories[0]]],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db'),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=350,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Detailed metrics
st.markdown("### Detailed Metric Scores")
metric_cols = st.columns(3)
for i, (key, desc) in enumerate(METRIC_DESCRIPTIONS.items()):
    with metric_cols[i % 3]:
        score = result['scores'][key]
        color = '#2ecc71' if score > 0.7 else '#f39c12' if score > 0.4 else '#e74c3c'
        st.markdown(f"**{desc}**")
        st.progress(score)
        st.markdown(f"<span style='color:{color}'>{score:.3f}</span>",
                    unsafe_allow_html=True)

# Additional diagnostics
with st.expander("Diagnostics"):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        raw_snr = calculate_snr_raw(intensity, wavenumber)
        st.metric("Raw SNR", f"{raw_snr:.1f}")
        fluor_ratio = get_fluorescence_ratio(intensity)
        st.metric("Fluorescence Ratio", f"{fluor_ratio:.3f}")
    with col_d2:
        fwhms = get_peak_fwhms(intensity, wavenumber)
        if fwhms:
            avg_fwhm = np.mean([f[1] for f in fwhms])
            st.metric("Avg FWHM", f"{avg_fwhm:.1f} cm⁻¹")
            st.metric("Peaks Detected", len(fwhms))
        else:
            st.metric("Peaks Detected", 0)

# PDF report (simple text download)
report_text = f"""SpectraGuard Quality Report
===========================
Confidence Score: {conf:.1f} ({grade})

Metric Scores:
"""
for key, desc in METRIC_DESCRIPTIONS.items():
    report_text += f"  {desc}: {result['scores'][key]:.3f}\n"

report_text += f"""
Weights: {result['weights']}
Wavenumber range: {wavenumber[0]:.1f} - {wavenumber[-1]:.1f} cm⁻¹
Points: {len(wavenumber)}
"""

st.download_button(
    "Download Report (TXT)",
    data=report_text,
    file_name="spectraguard_report.txt",
    mime="text/plain",
)
