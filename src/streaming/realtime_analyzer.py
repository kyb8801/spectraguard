"""
Real-time streaming spectral quality analyzer.

Provides continuous QC monitoring with trend detection, SPC alerts,
and moving statistics for factory/field deployment.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Callable, List, Tuple
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.quality_analyzer import QualityAnalyzer


class StreamingAnalyzer:
    """Real-time streaming spectrum QC analyzer."""

    def __init__(self, modality: str = 'sers', window_size: int = 50,
                 alert_threshold: float = 40.0, trend_window: int = 20):
        self.analyzer = QualityAnalyzer(modality=modality)
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.trend_window = trend_window

        self.score_history: deque = deque(maxlen=window_size)
        self.metric_history: Dict[str, deque] = {
            m: deque(maxlen=window_size)
            for m in ['snr', 'baseline', 'peak', 'resolution', 'fluorescence', 'uniformity']
        }
        self.timestamps: deque = deque(maxlen=window_size)
        self.alerts: List[Dict] = []
        self._callbacks: List[Callable] = []
        self._total_count = 0
        self._start_time: Optional[datetime] = None
        self._grade_counts: Dict[str, int] = {
            'Excellent': 0, 'Good': 0, 'Marginal': 0, 'Poor': 0
        }

    def push(self, spectrum: np.ndarray, wavenumber: np.ndarray,
             timestamp: Optional[datetime] = None) -> Dict:
        """
        Analyze one new spectrum and update state.

        Returns dict with: confidence, grade, scores, moving_avg, moving_std,
        trend, trend_slope, alerts, cumulative_stats
        """
        if self._start_time is None:
            self._start_time = datetime.now()

        ts = timestamp or datetime.now()
        self._total_count += 1

        # Analyze
        result = self.analyzer.analyze(spectrum, wavenumber)
        conf = result['confidence']
        grade = result['grade']

        # Update history
        self.score_history.append(conf)
        self.timestamps.append(ts)
        for m in self.metric_history:
            self.metric_history[m].append(result['scores'].get(m, 0))
        self._grade_counts[grade] = self._grade_counts.get(grade, 0) + 1

        # Moving stats
        scores_arr = np.array(self.score_history)
        moving_avg = float(np.mean(scores_arr))
        moving_std = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0

        # Trend detection
        trend, trend_slope = self._detect_trend()

        # Alert checking
        new_alerts = self._check_alerts(result)
        self.alerts.extend(new_alerts)
        for alert in new_alerts:
            for cb in self._callbacks:
                cb(alert)

        return {
            'confidence': conf,
            'grade': grade,
            'scores': result['scores'],
            'moving_avg': moving_avg,
            'moving_std': moving_std,
            'trend': trend,
            'trend_slope': trend_slope,
            'alerts': new_alerts,
            'cumulative_stats': self._get_cumulative_stats(),
        }

    def _detect_trend(self) -> Tuple[str, float]:
        """Detect trend from recent scores using linear regression slope."""
        n = min(self.trend_window, len(self.score_history))
        if n < 3:
            return 'stable', 0.0

        recent = list(self.score_history)[-n:]
        x = np.arange(n)
        slope = np.polyfit(x, recent, 1)[0]

        if slope < -0.5:
            return 'degrading', float(slope)
        elif slope > 0.5:
            return 'improving', float(slope)
        return 'stable', float(slope)

    def _check_alerts(self, result: Dict) -> List[Dict]:
        """Check alert conditions."""
        alerts = []
        conf = result['confidence']

        # 1. Absolute threshold
        if conf < self.alert_threshold:
            alerts.append({
                'type': 'threshold',
                'message': f'Confidence {conf:.1f} below threshold {self.alert_threshold}',
                'severity': 'critical',
                'timestamp': datetime.now(),
            })

        # 2. Sudden drop
        if len(self.score_history) >= 2:
            prev = list(self.score_history)[-2]
            if conf < prev - 20:
                alerts.append({
                    'type': 'sudden_drop',
                    'message': f'Sudden drop: {prev:.1f} → {conf:.1f}',
                    'severity': 'warning',
                    'timestamp': datetime.now(),
                })

        # 3. Consecutive decline
        if len(self.score_history) >= 5:
            recent5 = list(self.score_history)[-5:]
            if all(recent5[i] > recent5[i+1] for i in range(4)):
                alerts.append({
                    'type': 'consecutive_decline',
                    'message': '5 consecutive declining scores',
                    'severity': 'warning',
                    'timestamp': datetime.now(),
                })

        # 4. Individual metric anomaly
        for m, val in result['scores'].items():
            if val < 0.1:
                alerts.append({
                    'type': 'metric_anomaly',
                    'message': f'Metric {m} critically low: {val:.3f}',
                    'severity': 'warning',
                    'timestamp': datetime.now(),
                })

        return alerts

    def on_alert(self, callback: Callable):
        """Register alert callback."""
        self._callbacks.append(callback)

    def _get_cumulative_stats(self) -> Dict:
        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        return {
            'total_spectra': self._total_count,
            'mean_confidence': float(np.mean(self.score_history)) if self.score_history else 0,
            'std_confidence': float(np.std(self.score_history)) if len(self.score_history) > 1 else 0,
            'grade_distribution': dict(self._grade_counts),
            'alert_count': len(self.alerts),
            'uptime_seconds': elapsed,
            'throughput': self._total_count / elapsed if elapsed > 0 else 0,
        }

    def get_summary(self) -> Dict:
        """Get full summary statistics."""
        stats = self._get_cumulative_stats()

        # Best/worst metric
        if self.metric_history['snr']:
            metric_means = {m: float(np.mean(self.metric_history[m]))
                          for m in self.metric_history}
            stats['best_metric'] = max(metric_means, key=metric_means.get)
            stats['worst_metric'] = min(metric_means, key=metric_means.get)
        else:
            stats['best_metric'] = None
            stats['worst_metric'] = None

        return stats

    def reset(self):
        """Reset all state."""
        self.score_history.clear()
        for m in self.metric_history:
            self.metric_history[m].clear()
        self.timestamps.clear()
        self.alerts.clear()
        self._total_count = 0
        self._start_time = None
        self._grade_counts = {'Excellent': 0, 'Good': 0, 'Marginal': 0, 'Poor': 0}


class SPC_Controller:
    """Statistical Process Control for quality monitoring."""

    def __init__(self, target: float = 70.0, sigma_limit: float = 3.0):
        self.target = target
        self.sigma_limit = sigma_limit
        self.ucl: Optional[float] = None
        self.lcl: Optional[float] = None
        self._mean: float = target
        self._std: float = 10.0
        self._history: deque = deque(maxlen=100)

    def calibrate(self, reference_scores: list):
        """Set control limits from reference data."""
        self._mean = float(np.mean(reference_scores))
        self._std = float(np.std(reference_scores))
        self.ucl = self._mean + self.sigma_limit * self._std
        self.lcl = self._mean - self.sigma_limit * self._std

    def check(self, score: float) -> Dict:
        """
        Check SPC rules (Western Electric rules).
        Returns dict with: in_control (bool), violations (list of str)
        """
        self._history.append(score)
        violations = []

        if self.ucl is None:
            return {'in_control': True, 'violations': [], 'score': score}

        # Rule 1: One point beyond 3σ
        if score > self.ucl or score < self.lcl:
            violations.append(f'Rule 1: Point ({score:.1f}) outside 3σ limits [{self.lcl:.1f}, {self.ucl:.1f}]')

        # Rule 2: 2 of 3 consecutive points beyond 2σ (same side)
        sigma2_upper = self._mean + 2 * self._std
        sigma2_lower = self._mean - 2 * self._std
        if len(self._history) >= 3:
            recent3 = list(self._history)[-3:]
            above_2s = sum(1 for s in recent3 if s > sigma2_upper)
            below_2s = sum(1 for s in recent3 if s < sigma2_lower)
            if above_2s >= 2:
                violations.append('Rule 2: 2 of 3 points above 2σ')
            if below_2s >= 2:
                violations.append('Rule 2: 2 of 3 points below 2σ')

        # Rule 3: 4 of 5 consecutive beyond 1σ (same side)
        sigma1_upper = self._mean + self._std
        sigma1_lower = self._mean - self._std
        if len(self._history) >= 5:
            recent5 = list(self._history)[-5:]
            above_1s = sum(1 for s in recent5 if s > sigma1_upper)
            below_1s = sum(1 for s in recent5 if s < sigma1_lower)
            if above_1s >= 4:
                violations.append('Rule 3: 4 of 5 points above 1σ')
            if below_1s >= 4:
                violations.append('Rule 3: 4 of 5 points below 1σ')

        # Rule 4: 8 consecutive on same side of center
        if len(self._history) >= 8:
            recent8 = list(self._history)[-8:]
            if all(s > self._mean for s in recent8):
                violations.append('Rule 4: 8 consecutive above center')
            elif all(s < self._mean for s in recent8):
                violations.append('Rule 4: 8 consecutive below center')

        return {
            'in_control': len(violations) == 0,
            'violations': violations,
            'score': score,
            'ucl': self.ucl,
            'lcl': self.lcl,
            'center': self._mean,
        }
