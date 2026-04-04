"""SpectraGuard: Uncertainty-aware spectral quality assessment."""

__version__ = "0.2.0"

import sys
import os

# Add project root so src/ imports work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.quality_analyzer import QualityAnalyzer
from src.metrics.confidence_score import ConfidenceScorer
from src.metrics.uncertainty import UncertaintyEstimator
from src.metrics.weight_calibrator import WeightCalibrator
from src.streaming.realtime_analyzer import StreamingAnalyzer, SPC_Controller

# Lazy import for calibration (may not exist yet)
def _lazy_import_calibrator():
    from src.calibration.instrument_transfer import InstrumentCalibrator
    return InstrumentCalibrator

__all__ = [
    "QualityAnalyzer",
    "ConfidenceScorer",
    "UncertaintyEstimator",
    "WeightCalibrator",
    "StreamingAnalyzer",
    "SPC_Controller",
]
