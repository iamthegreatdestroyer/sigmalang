"""
Anomaly Detection for SigmaLang - Phase 7 Track 12

Detects compression performance anomalies by comparing real-time metrics
against rolling statistical baselines. Feeds into the optimization daemon's
self-healing actions.

Detection Methods:
    1. Z-Score: Flag values >3 standard deviations from mean
    2. Percentage Drop: Flag values dropping >N% from rolling median
    3. Threshold Breach: Flag values exceeding absolute limits
    4. Trend Detection: Flag sustained directional drift

Usage:
    from sigmalang.core.anomaly_detector import MetricsAnomalyDetector

    detector = MetricsAnomalyDetector()

    # Record samples over time
    detector.record('compression_ratio', 15.2)
    detector.record('compression_ratio', 14.8)
    detector.record('compression_ratio', 8.1)  # Sudden drop

    # Check for anomalies
    anomalies = detector.check('compression_ratio')
    for a in anomalies:
        print(f"[{a['severity']}] {a['message']}")
"""

import math
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnomalyThresholds:
    """Thresholds for anomaly detection."""

    # Z-score anomaly (standard deviations from mean)
    zscore_warning: float = 2.0
    zscore_critical: float = 3.0

    # Percentage drop from rolling median
    drop_warning_pct: float = 15.0
    drop_critical_pct: float = 30.0

    # Minimum samples before anomaly detection activates
    min_samples: int = 10

    # Rolling window size
    window_size: int = 100

    # Trend detection: consecutive samples in same direction
    trend_warning_count: int = 10  # 10 consecutive drops = warning


# =============================================================================
# Metric Time Series
# =============================================================================

@dataclass
class MetricSample:
    """A single metric sample with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)


class MetricTimeSeries:
    """Rolling time series for a single metric."""

    def __init__(self, max_size: int = 100):
        self.samples: deque = deque(maxlen=max_size)
        self.name: str = ""

    def add(self, value: float) -> None:
        """Add a sample."""
        self.samples.append(MetricSample(value=value))

    @property
    def values(self) -> List[float]:
        return [s.value for s in self.samples]

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.value for s in self.samples) / len(self.samples)

    @property
    def median(self) -> float:
        if not self.samples:
            return 0.0
        sorted_vals = sorted(s.value for s in self.samples)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        return sorted_vals[n // 2]

    @property
    def std(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        m = self.mean
        variance = sum((s.value - m) ** 2 for s in self.samples) / (len(self.samples) - 1)
        return math.sqrt(variance)

    @property
    def latest(self) -> Optional[float]:
        if self.samples:
            return self.samples[-1].value
        return None

    def recent_trend(self, n: int = 10) -> Optional[str]:
        """
        Detect trend in last N samples.

        Returns 'rising', 'falling', or None.
        """
        if len(self.samples) < n:
            return None

        recent = [s.value for s in list(self.samples)[-n:]]
        rising = 0
        falling = 0
        for i in range(1, len(recent)):
            if recent[i] > recent[i - 1]:
                rising += 1
            elif recent[i] < recent[i - 1]:
                falling += 1

        threshold = n * 0.7  # 70% in same direction
        if rising >= threshold:
            return 'rising'
        if falling >= threshold:
            return 'falling'
        return None


# =============================================================================
# Anomaly Detector
# =============================================================================

class MetricsAnomalyDetector:
    """
    Statistical anomaly detection for compression and performance metrics.

    Maintains rolling baselines per metric and flags values that deviate
    significantly from expected behavior.
    """

    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        self.thresholds = thresholds or AnomalyThresholds()
        self._series: Dict[str, MetricTimeSeries] = {}
        self._absolute_thresholds: Dict[str, Dict[str, float]] = {}

    def set_absolute_threshold(
        self,
        metric: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        """Set absolute min/max thresholds for a metric."""
        self._absolute_thresholds[metric] = {}
        if min_value is not None:
            self._absolute_thresholds[metric]['min'] = min_value
        if max_value is not None:
            self._absolute_thresholds[metric]['max'] = max_value

    def _get_series(self, metric: str) -> MetricTimeSeries:
        """Get or create a time series for a metric."""
        if metric not in self._series:
            self._series[metric] = MetricTimeSeries(
                max_size=self.thresholds.window_size
            )
            self._series[metric].name = metric
        return self._series[metric]

    def record(self, metric: str, value: float) -> None:
        """Record a metric sample."""
        series = self._get_series(metric)
        series.add(value)

    def check(self, metric: str) -> List[Dict[str, Any]]:
        """
        Check a metric for anomalies.

        Returns list of anomaly dicts with: severity, type, message, value, threshold
        """
        series = self._get_series(metric)
        anomalies = []

        if series.count < self.thresholds.min_samples:
            return []  # Not enough data

        latest = series.latest
        if latest is None:
            return []

        # 1. Z-Score check
        std = series.std
        if std > 0:
            zscore = abs(latest - series.mean) / std

            if zscore > self.thresholds.zscore_critical:
                anomalies.append({
                    'severity': 'critical',
                    'type': 'zscore',
                    'metric': metric,
                    'value': latest,
                    'zscore': round(zscore, 2),
                    'mean': round(series.mean, 4),
                    'std': round(std, 4),
                    'message': f"{metric}: value {latest:.4f} is {zscore:.1f} "
                               f"std devs from mean {series.mean:.4f}"
                })
            elif zscore > self.thresholds.zscore_warning:
                anomalies.append({
                    'severity': 'warning',
                    'type': 'zscore',
                    'metric': metric,
                    'value': latest,
                    'zscore': round(zscore, 2),
                    'message': f"{metric}: value {latest:.4f} is {zscore:.1f} "
                               f"std devs from mean {series.mean:.4f}"
                })

        # 2. Percentage drop from median
        median = series.median
        if median > 0:
            drop_pct = (median - latest) / median * 100

            if drop_pct > self.thresholds.drop_critical_pct:
                anomalies.append({
                    'severity': 'critical',
                    'type': 'drop',
                    'metric': metric,
                    'value': latest,
                    'median': round(median, 4),
                    'drop_pct': round(drop_pct, 1),
                    'message': f"{metric}: dropped {drop_pct:.1f}% from "
                               f"median {median:.4f} to {latest:.4f}"
                })
            elif drop_pct > self.thresholds.drop_warning_pct:
                anomalies.append({
                    'severity': 'warning',
                    'type': 'drop',
                    'metric': metric,
                    'value': latest,
                    'median': round(median, 4),
                    'drop_pct': round(drop_pct, 1),
                    'message': f"{metric}: dropped {drop_pct:.1f}% from "
                               f"median {median:.4f}"
                })

        # 3. Absolute threshold breach
        if metric in self._absolute_thresholds:
            limits = self._absolute_thresholds[metric]
            if 'min' in limits and latest < limits['min']:
                anomalies.append({
                    'severity': 'critical',
                    'type': 'threshold_breach',
                    'metric': metric,
                    'value': latest,
                    'threshold': limits['min'],
                    'message': f"{metric}: {latest:.4f} below minimum {limits['min']:.4f}"
                })
            if 'max' in limits and latest > limits['max']:
                anomalies.append({
                    'severity': 'critical',
                    'type': 'threshold_breach',
                    'metric': metric,
                    'value': latest,
                    'threshold': limits['max'],
                    'message': f"{metric}: {latest:.4f} above maximum {limits['max']:.4f}"
                })

        # 4. Trend detection
        trend = series.recent_trend(self.thresholds.trend_warning_count)
        if trend == 'falling':
            anomalies.append({
                'severity': 'warning',
                'type': 'trend',
                'metric': metric,
                'trend': 'falling',
                'message': f"{metric}: sustained downward trend detected "
                           f"({self.thresholds.trend_warning_count} consecutive drops)"
            })

        return anomalies

    def check_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Check all tracked metrics for anomalies."""
        results = {}
        for metric in self._series:
            anomalies = self.check(metric)
            if anomalies:
                results[metric] = anomalies
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = {}
        for metric, series in self._series.items():
            summary[metric] = {
                'count': series.count,
                'mean': round(series.mean, 4),
                'median': round(series.median, 4),
                'std': round(series.std, 4),
                'latest': series.latest,
                'trend': series.recent_trend(),
            }
        return summary


# =============================================================================
# Pre-configured Detector for SigmaLang
# =============================================================================

def create_sigmalang_detector() -> MetricsAnomalyDetector:
    """
    Create a pre-configured anomaly detector for SigmaLang metrics.

    Sets appropriate thresholds for compression ratio, latency,
    codebook utilization, and error rates.
    """
    detector = MetricsAnomalyDetector()

    # Compression ratio should not drop below 5x
    detector.set_absolute_threshold('compression_ratio', min_value=3.0)

    # Encoding latency should not exceed 100ms
    detector.set_absolute_threshold('encoding_latency_ms', max_value=100.0)

    # Error rate should stay below 5%
    detector.set_absolute_threshold('error_rate', max_value=0.05)

    # Codebook utilization should stay below 95%
    detector.set_absolute_threshold('codebook_utilization', max_value=0.95)

    return detector
