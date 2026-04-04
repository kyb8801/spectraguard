"""Example 4: Real-time streaming quality control with SPC."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.streaming.realtime_analyzer import StreamingAnalyzer, SPC_Controller
from src.utils.synthetic_data import generate_single_spectrum

wavenumber = np.linspace(200, 3200, 1500)
rng = np.random.default_rng(42)

# Set up streaming analyzer with alert threshold
sa = StreamingAnalyzer(modality='sers', window_size=20, alert_threshold=40.0)

# Register alert callback
def on_alert(alert):
    print(f"  ⚠ ALERT [{alert['severity']}]: {alert['message']}")

sa.on_alert(on_alert)

# Simulate streaming spectra (gradual degradation)
print("Streaming 20 spectra with degrading quality...\n")
for i, snr in enumerate(np.linspace(80, 5, 20)):
    spec = generate_single_spectrum(wavenumber, snr=snr, rng=rng)
    result = sa.push(spec, wavenumber)
    print(f"  [{i+1:2d}] SNR={snr:5.1f} → Score={result['confidence']:5.1f} "
          f"({result['grade']:10s}) Trend={result['trend']}")

# Print summary
print("\n--- Summary ---")
summary = sa.get_summary()
print(f"Total spectra: {summary['total_spectra']}")
print(f"Mean confidence: {summary['mean_confidence']:.1f}")
print(f"Grade distribution: {summary['grade_distribution']}")
print(f"Best metric: {summary['best_metric']}")
print(f"Worst metric: {summary['worst_metric']}")

# SPC Controller demo
print("\n--- SPC Controller ---")
spc = SPC_Controller(target=70.0, sigma_limit=3.0)
spc.calibrate([65, 70, 75, 68, 72, 71, 69, 73])  # Calibrate from reference data
print(f"Control limits: LCL={spc.lcl:.1f}, UCL={spc.ucl:.1f}")

for score in [70, 72, 68, 50, 30]:
    result = spc.check(score)
    status = "IN CONTROL" if result['in_control'] else "OUT OF CONTROL"
    print(f"  Score={score:.0f} → {status}")
    for v in result['violations']:
        print(f"    - {v}")
