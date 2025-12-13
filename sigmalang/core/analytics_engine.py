"""
Analytics & Visualization Engine for SigmaLang.

This module provides comprehensive performance monitoring, analytics,
and visualization capabilities for the analogy processing pipeline.

Features:
- Performance metrics collection with time-series tracking
- Resource utilization monitoring (CPU, memory, throughput)
- Configurable analytics dashboards
- Visualization utilities for performance analysis
- Anomaly detection in performance patterns
- Export capabilities (JSON, CSV, Prometheus format)

Architecture:
    MetricsCollector -> MetricsAggregator -> AnalyticsDashboard
           |                    |                    |
           v                    v                    v
    TimeSeriesStore     AggregateStore      Visualizer

Example:
    >>> from core.analytics_engine import (
    ...     MetricsCollector,
    ...     AnalyticsDashboard,
    ...     create_analytics_engine
    ... )
    >>> 
    >>> # Create analytics engine
    >>> engine = create_analytics_engine()
    >>> 
    >>> # Record metrics
    >>> engine.record_latency("process_query", 15.5)
    >>> engine.record_throughput("queries_per_second", 100)
    >>> 
    >>> # Get dashboard
    >>> dashboard = engine.get_dashboard()
    >>> print(dashboard.summary())
"""

from __future__ import annotations

import time
import json
import threading
import statistics
import logging
from dataclasses import dataclass, field
from typing import (
    Dict, List, Any, Optional, Union, 
    Callable, Tuple, TypeVar, Generic
)
from enum import Enum, auto
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = auto()     # Monotonically increasing counter
    GAUGE = auto()       # Value that can go up or down
    HISTOGRAM = auto()   # Distribution of values
    TIMER = auto()       # Duration measurements
    RATE = auto()        # Rate of events per time unit


class AggregationType(Enum):
    """Types of aggregations for metrics."""
    SUM = auto()
    MEAN = auto()
    MEDIAN = auto()
    MIN = auto()
    MAX = auto()
    COUNT = auto()
    P50 = auto()   # 50th percentile
    P95 = auto()   # 95th percentile
    P99 = auto()   # 99th percentile
    STDDEV = auto()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = auto()
    CSV = auto()
    PROMETHEUS = auto()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'type': self.metric_type.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricPoint':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            value=data['value'],
            timestamp=data.get('timestamp', time.time()),
            labels=data.get('labels', {}),
            metric_type=MetricType[data.get('type', 'GAUGE')]
        )


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time window."""
    name: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    stddev_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    start_time: float
    end_time: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'count': self.count,
            'sum': self.sum_value,
            'min': self.min_value,
            'max': self.max_value,
            'mean': self.mean_value,
            'median': self.median_value,
            'stddev': self.stddev_value,
            'p50': self.p50_value,
            'p95': self.p95_value,
            'p99': self.p99_value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'labels': self.labels
        }


@dataclass
class Alert:
    """Performance alert."""
    metric_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'severity': self.severity.name,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    metric_name: str
    condition: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    threshold: float
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: float = 60.0
    
    def check(self, value: float) -> bool:
        """Check if value triggers alert."""
        if self.condition == 'gt':
            return value > self.threshold
        elif self.condition == 'lt':
            return value < self.threshold
        elif self.condition == 'gte':
            return value >= self.threshold
        elif self.condition == 'lte':
            return value <= self.threshold
        elif self.condition == 'eq':
            return value == self.threshold
        return False


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_bytes: int = 0
    active_threads: int = 0
    open_files: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_bytes': self.memory_bytes,
            'active_threads': self.active_threads,
            'open_files': self.open_files
        }


@dataclass
class DashboardConfig:
    """Configuration for analytics dashboard."""
    name: str = "SigmaLang Analytics"
    refresh_interval_seconds: float = 5.0
    retention_hours: float = 24.0
    enabled_metrics: List[str] = field(default_factory=list)
    alert_rules: List[AlertRule] = field(default_factory=list)
    aggregation_window_seconds: float = 60.0


# ============================================================================
# TIME SERIES STORE
# ============================================================================

class TimeSeriesStore:
    """
    In-memory time series storage for metrics.
    
    Features:
    - Configurable retention period
    - Automatic cleanup of old data
    - Thread-safe operations
    - Efficient range queries
    
    Example:
        >>> store = TimeSeriesStore(retention_hours=24)
        >>> store.add("latency", 15.5)
        >>> store.add("latency", 20.3)
        >>> points = store.get_range("latency", time.time() - 3600, time.time())
    """
    
    def __init__(
        self, 
        retention_hours: float = 24.0,
        max_points_per_metric: int = 100000
    ):
        """
        Initialize time series store.
        
        Args:
            retention_hours: How long to keep data
            max_points_per_metric: Maximum points per metric (prevents memory bloat)
        """
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        self._data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_points_per_metric)
        )
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def add(
        self, 
        name: str, 
        value: float, 
        timestamp: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ) -> None:
        """
        Add a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
            labels: Optional labels
            metric_type: Type of metric
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp or time.time(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        with self._lock:
            self._data[name].append(point)
            
            # Periodic cleanup
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired()
    
    def get_range(
        self, 
        name: str, 
        start_time: float, 
        end_time: float
    ) -> List[MetricPoint]:
        """
        Get metric points in time range.
        
        Args:
            name: Metric name
            start_time: Range start
            end_time: Range end
            
        Returns:
            List of MetricPoint in range
        """
        with self._lock:
            if name not in self._data:
                return []
            
            return [
                p for p in self._data[name]
                if start_time <= p.timestamp <= end_time
            ]
    
    def get_latest(self, name: str, count: int = 1) -> List[MetricPoint]:
        """
        Get latest metric points.
        
        Args:
            name: Metric name
            count: Number of points to get
            
        Returns:
            List of latest MetricPoint
        """
        with self._lock:
            if name not in self._data:
                return []
            
            data = list(self._data[name])
            return data[-count:] if len(data) >= count else data
    
    def get_all_metrics(self) -> List[str]:
        """Get all metric names."""
        with self._lock:
            return list(self._data.keys())
    
    def get_count(self, name: str) -> int:
        """Get count of points for metric."""
        with self._lock:
            return len(self._data.get(name, []))
    
    def _cleanup_expired(self) -> int:
        """Remove expired data points."""
        cutoff = time.time() - (self.retention_hours * 3600)
        removed = 0
        
        for name in list(self._data.keys()):
            original_len = len(self._data[name])
            
            # Remove old points from the front
            while self._data[name] and self._data[name][0].timestamp < cutoff:
                self._data[name].popleft()
                removed += 1
            
            # Remove empty series
            if not self._data[name]:
                del self._data[name]
        
        self._last_cleanup = time.time()
        return removed
    
    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._data.clear()


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """
    Collects and stores performance metrics.
    
    Features:
    - Multiple metric types (counter, gauge, histogram, timer)
    - Automatic aggregation
    - Thread-safe collection
    - Built-in resource monitoring
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_counter("requests_total", 1)
        >>> collector.record_gauge("active_connections", 50)
        >>> collector.record_timer("request_latency_ms", 15.5)
    """
    
    def __init__(
        self, 
        store: Optional[TimeSeriesStore] = None,
        enable_resource_monitoring: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            store: Optional time series store
            enable_resource_monitoring: Enable resource monitoring
        """
        self.store = store or TimeSeriesStore()
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Counters (monotonically increasing)
        self._counters: Dict[str, float] = defaultdict(float)
        self._counter_lock = threading.Lock()
        
        # Resource monitoring
        self._resource_history: deque = deque(maxlen=1000)
        self._last_resource_check = 0.0
        self._resource_check_interval = 1.0
        
        # Histogram buckets
        self._histogram_buckets: Dict[str, List[float]] = defaultdict(list)
    
    def record_counter(
        self, 
        name: str, 
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record counter metric (monotonically increasing).
        
        Args:
            name: Counter name
            value: Value to add (default 1)
            labels: Optional labels
        """
        with self._counter_lock:
            self._counters[name] += value
            current_value = self._counters[name]
        
        self.store.add(name, current_value, labels=labels, metric_type=MetricType.COUNTER)
    
    def record_gauge(
        self, 
        name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record gauge metric (can go up or down).
        
        Args:
            name: Gauge name
            value: Current value
            labels: Optional labels
        """
        self.store.add(name, value, labels=labels, metric_type=MetricType.GAUGE)
    
    def record_histogram(
        self, 
        name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record histogram metric (distribution of values).
        
        Args:
            name: Histogram name
            value: Observed value
            labels: Optional labels
        """
        self.store.add(name, value, labels=labels, metric_type=MetricType.HISTOGRAM)
        
        # Also track in histogram buckets for percentile calculations
        self._histogram_buckets[name].append(value)
        
        # Limit bucket size
        if len(self._histogram_buckets[name]) > 10000:
            self._histogram_buckets[name] = self._histogram_buckets[name][-10000:]
    
    def record_timer(
        self, 
        name: str, 
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record timer metric (duration measurement).
        
        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            labels: Optional labels
        """
        self.store.add(name, duration_ms, labels=labels, metric_type=MetricType.TIMER)
    
    def record_rate(
        self, 
        name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record rate metric (events per time unit).
        
        Args:
            name: Rate name
            value: Rate value
            labels: Optional labels
        """
        self.store.add(name, value, labels=labels, metric_type=MetricType.RATE)
    
    def time_operation(self, name: str) -> 'TimerContext':
        """
        Context manager for timing operations.
        
        Args:
            name: Timer name
            
        Returns:
            TimerContext that records duration on exit
        """
        return TimerContext(self, name)
    
    def get_counter_value(self, name: str) -> float:
        """Get current counter value."""
        with self._counter_lock:
            return self._counters.get(name, 0.0)
    
    def collect_resource_usage(self) -> ResourceUsage:
        """
        Collect current resource usage.
        
        Returns:
            ResourceUsage snapshot
        """
        usage = ResourceUsage(
            timestamp=time.time(),
            active_threads=threading.active_count()
        )
        
        # Try to get memory info if psutil is available
        try:
            import psutil
            process = psutil.Process()
            usage.cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            usage.memory_bytes = memory_info.rss
            usage.memory_percent = process.memory_percent()
            usage.open_files = len(process.open_files())
        except ImportError:
            pass  # psutil not available
        except Exception as e:
            logger.debug(f"Resource collection error: {e}")
        
        self._resource_history.append(usage)
        return usage
    
    def get_resource_history(
        self, 
        last_n: Optional[int] = None
    ) -> List[ResourceUsage]:
        """
        Get resource usage history.
        
        Args:
            last_n: Number of entries to return (None for all)
            
        Returns:
            List of ResourceUsage
        """
        history = list(self._resource_history)
        if last_n is not None:
            return history[-last_n:]
        return history
    
    def get_histogram_percentile(
        self, 
        name: str, 
        percentile: float
    ) -> Optional[float]:
        """
        Get percentile from histogram.
        
        Args:
            name: Histogram name
            percentile: Percentile (0-100)
            
        Returns:
            Percentile value or None
        """
        if name not in self._histogram_buckets or not self._histogram_buckets[name]:
            return None
        
        data = sorted(self._histogram_buckets[name])
        index = int(len(data) * percentile / 100)
        index = min(index, len(data) - 1)
        return data[index]


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time = 0.0
    
    def __enter__(self) -> 'TimerContext':
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.time() - self.start_time) * 1000
        self.collector.record_timer(self.name, duration_ms)


# ============================================================================
# METRICS AGGREGATOR
# ============================================================================

class MetricsAggregator:
    """
    Aggregates metrics over time windows.
    
    Features:
    - Configurable aggregation windows
    - Multiple aggregation types
    - Statistical calculations
    - Trend detection
    
    Example:
        >>> aggregator = MetricsAggregator(store)
        >>> agg = aggregator.aggregate("latency", window_seconds=60)
        >>> print(f"Mean latency: {agg.mean_value}ms")
    """
    
    def __init__(self, store: TimeSeriesStore):
        """
        Initialize aggregator.
        
        Args:
            store: Time series store to aggregate from
        """
        self.store = store
    
    def aggregate(
        self, 
        name: str, 
        window_seconds: float = 60.0,
        end_time: Optional[float] = None
    ) -> Optional[AggregatedMetric]:
        """
        Aggregate metric over time window.
        
        Args:
            name: Metric name
            window_seconds: Aggregation window
            end_time: End of window (defaults to now)
            
        Returns:
            AggregatedMetric or None if no data
        """
        end_time = end_time or time.time()
        start_time = end_time - window_seconds
        
        points = self.store.get_range(name, start_time, end_time)
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        return AggregatedMetric(
            name=name,
            count=len(values),
            sum_value=sum(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=statistics.mean(values),
            median_value=statistics.median(values),
            stddev_value=statistics.stdev(values) if len(values) > 1 else 0.0,
            p50_value=self._percentile(values, 50),
            p95_value=self._percentile(values, 95),
            p99_value=self._percentile(values, 99),
            start_time=start_time,
            end_time=end_time
        )
    
    def aggregate_multiple(
        self, 
        names: List[str], 
        window_seconds: float = 60.0
    ) -> Dict[str, Optional[AggregatedMetric]]:
        """
        Aggregate multiple metrics.
        
        Args:
            names: Metric names
            window_seconds: Aggregation window
            
        Returns:
            Dict mapping name to AggregatedMetric
        """
        return {
            name: self.aggregate(name, window_seconds)
            for name in names
        }
    
    def calculate_rate_of_change(
        self, 
        name: str, 
        window_seconds: float = 60.0
    ) -> Optional[float]:
        """
        Calculate rate of change over window.
        
        Args:
            name: Metric name
            window_seconds: Window for calculation
            
        Returns:
            Rate of change per second
        """
        end_time = time.time()
        start_time = end_time - window_seconds
        
        points = self.store.get_range(name, start_time, end_time)
        
        if len(points) < 2:
            return None
        
        first = points[0]
        last = points[-1]
        
        time_diff = last.timestamp - first.timestamp
        if time_diff <= 0:
            return None
        
        value_diff = last.value - first.value
        return value_diff / time_diff
    
    def detect_anomaly(
        self, 
        name: str, 
        threshold_stddev: float = 2.0,
        window_seconds: float = 300.0
    ) -> Optional[Tuple[bool, float, float]]:
        """
        Detect anomaly in latest value.
        
        Args:
            name: Metric name
            threshold_stddev: Number of std devs for anomaly
            window_seconds: Historical window
            
        Returns:
            Tuple of (is_anomaly, latest_value, z_score) or None
        """
        agg = self.aggregate(name, window_seconds)
        if agg is None or agg.stddev_value == 0:
            return None
        
        latest = self.store.get_latest(name, 1)
        if not latest:
            return None
        
        latest_value = latest[0].value
        z_score = (latest_value - agg.mean_value) / agg.stddev_value
        
        is_anomaly = abs(z_score) > threshold_stddev
        return (is_anomaly, latest_value, z_score)
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

class AnalyticsDashboard:
    """
    Analytics dashboard for performance visualization.
    
    Features:
    - Real-time metrics display
    - Alert management
    - Export capabilities
    - Summary generation
    
    Example:
        >>> dashboard = AnalyticsDashboard(collector, aggregator, config)
        >>> print(dashboard.summary())
        >>> alerts = dashboard.check_alerts()
    """
    
    def __init__(
        self, 
        collector: MetricsCollector,
        aggregator: MetricsAggregator,
        config: Optional[DashboardConfig] = None
    ):
        """
        Initialize dashboard.
        
        Args:
            collector: Metrics collector
            aggregator: Metrics aggregator
            config: Dashboard configuration
        """
        self.collector = collector
        self.aggregator = aggregator
        self.config = config or DashboardConfig()
        
        # Alert state
        self._alerts: List[Alert] = []
        self._last_alert_time: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate dashboard summary.
        
        Returns:
            Summary dictionary
        """
        metrics = self.collector.store.get_all_metrics()
        window = self.config.aggregation_window_seconds
        
        summary = {
            'dashboard_name': self.config.name,
            'timestamp': time.time(),
            'metrics_count': len(metrics),
            'active_alerts': len(self._alerts),
            'aggregations': {}
        }
        
        for metric_name in metrics:
            agg = self.aggregator.aggregate(metric_name, window)
            if agg:
                summary['aggregations'][metric_name] = agg.to_dict()
        
        return summary
    
    def check_alerts(self) -> List[Alert]:
        """
        Check all alert rules and return triggered alerts.
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        current_time = time.time()
        
        for rule in self.config.alert_rules:
            # Check cooldown
            last_alert = self._last_alert_time.get(rule.metric_name, 0)
            if current_time - last_alert < rule.cooldown_seconds:
                continue
            
            # Get latest value
            latest = self.collector.store.get_latest(rule.metric_name, 1)
            if not latest:
                continue
            
            value = latest[0].value
            
            if rule.check(value):
                alert = Alert(
                    metric_name=rule.metric_name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        value=value, 
                        threshold=rule.threshold
                    ),
                    value=value,
                    threshold=rule.threshold
                )
                
                triggered.append(alert)
                self._last_alert_time[rule.metric_name] = current_time
                
                with self._lock:
                    self._alerts.append(alert)
        
        return triggered
    
    def get_alerts(
        self, 
        severity: Optional[AlertSeverity] = None,
        since: Optional[float] = None
    ) -> List[Alert]:
        """
        Get alerts, optionally filtered.
        
        Args:
            severity: Filter by severity
            since: Only alerts after this timestamp
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self._alerts)
        
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def clear_alerts(self) -> int:
        """Clear all alerts and return count cleared."""
        with self._lock:
            count = len(self._alerts)
            self._alerts.clear()
            return count
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.config.alert_rules.append(rule)
    
    def export(
        self, 
        format: ExportFormat = ExportFormat.JSON,
        metrics: Optional[List[str]] = None,
        window_seconds: float = 3600.0
    ) -> str:
        """
        Export metrics data.
        
        Args:
            format: Export format
            metrics: Specific metrics to export (None for all)
            window_seconds: Time window for export
            
        Returns:
            Exported data as string
        """
        end_time = time.time()
        start_time = end_time - window_seconds
        
        all_metrics = metrics or self.collector.store.get_all_metrics()
        
        if format == ExportFormat.JSON:
            return self._export_json(all_metrics, start_time, end_time)
        elif format == ExportFormat.CSV:
            return self._export_csv(all_metrics, start_time, end_time)
        elif format == ExportFormat.PROMETHEUS:
            return self._export_prometheus(all_metrics)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(
        self, 
        metrics: List[str], 
        start_time: float, 
        end_time: float
    ) -> str:
        """Export as JSON."""
        data = {
            'export_time': time.time(),
            'start_time': start_time,
            'end_time': end_time,
            'metrics': {}
        }
        
        for name in metrics:
            points = self.collector.store.get_range(name, start_time, end_time)
            data['metrics'][name] = [p.to_dict() for p in points]
        
        return json.dumps(data, indent=2)
    
    def _export_csv(
        self, 
        metrics: List[str], 
        start_time: float, 
        end_time: float
    ) -> str:
        """Export as CSV."""
        lines = ["timestamp,metric_name,value,type"]
        
        for name in metrics:
            points = self.collector.store.get_range(name, start_time, end_time)
            for p in points:
                lines.append(
                    f"{p.timestamp},{p.name},{p.value},{p.metric_type.name}"
                )
        
        return "\n".join(lines)
    
    def _export_prometheus(self, metrics: List[str]) -> str:
        """Export in Prometheus format."""
        lines = []
        
        for name in metrics:
            latest = self.collector.store.get_latest(name, 1)
            if latest:
                point = latest[0]
                metric_name = name.replace('.', '_').replace('-', '_')
                
                # Format labels
                labels_str = ""
                if point.labels:
                    label_pairs = [
                        f'{k}="{v}"' for k, v in point.labels.items()
                    ]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(
                    f"# TYPE {metric_name} {point.metric_type.name.lower()}"
                )
                lines.append(f"{metric_name}{labels_str} {point.value}")
        
        return "\n".join(lines)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status.
        
        Returns:
            Health status dictionary
        """
        critical_alerts = len([
            a for a in self._alerts 
            if a.severity == AlertSeverity.CRITICAL
        ])
        
        warning_alerts = len([
            a for a in self._alerts 
            if a.severity == AlertSeverity.WARNING
        ])
        
        if critical_alerts > 0:
            status = "critical"
        elif warning_alerts > 0:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'total_metrics': len(self.collector.store.get_all_metrics()),
            'last_check': time.time()
        }


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """
    Unified analytics engine combining all components.
    
    This is the main entry point for analytics functionality.
    
    Example:
        >>> engine = AnalyticsEngine()
        >>> engine.record_latency("query_processing", 15.5)
        >>> engine.record_throughput("queries_per_second", 100)
        >>> summary = engine.get_summary()
    """
    
    def __init__(
        self, 
        config: Optional[DashboardConfig] = None,
        retention_hours: float = 24.0
    ):
        """
        Initialize analytics engine.
        
        Args:
            config: Dashboard configuration
            retention_hours: Data retention period
        """
        self.store = TimeSeriesStore(retention_hours=retention_hours)
        self.collector = MetricsCollector(store=self.store)
        self.aggregator = MetricsAggregator(store=self.store)
        self.dashboard = AnalyticsDashboard(
            collector=self.collector,
            aggregator=self.aggregator,
            config=config
        )
    
    def record_latency(
        self, 
        name: str, 
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record latency metric."""
        self.collector.record_timer(name, latency_ms, labels)
    
    def record_throughput(
        self, 
        name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record throughput metric."""
        self.collector.record_rate(name, value, labels)
    
    def record_counter(
        self, 
        name: str, 
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record counter metric."""
        self.collector.record_counter(name, value, labels)
    
    def record_gauge(
        self, 
        name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record gauge metric."""
        self.collector.record_gauge(name, value, labels)
    
    def time_operation(self, name: str) -> TimerContext:
        """Get context manager for timing operations."""
        return self.collector.time_operation(name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dashboard summary."""
        return self.dashboard.summary()
    
    def get_aggregation(
        self, 
        name: str, 
        window_seconds: float = 60.0
    ) -> Optional[AggregatedMetric]:
        """Get aggregated metric."""
        return self.aggregator.aggregate(name, window_seconds)
    
    def check_alerts(self) -> List[Alert]:
        """Check and return triggered alerts."""
        return self.dashboard.check_alerts()
    
    def export(
        self, 
        format: ExportFormat = ExportFormat.JSON,
        window_seconds: float = 3600.0
    ) -> str:
        """Export metrics data."""
        return self.dashboard.export(format, window_seconds=window_seconds)
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return self.dashboard.get_health_status()
    
    def add_alert_rule(
        self,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: str = "Metric {value} exceeded threshold {threshold}"
    ) -> None:
        """Add an alert rule."""
        rule = AlertRule(
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            message_template=message
        )
        self.dashboard.add_alert_rule(rule)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_analytics_engine(
    retention_hours: float = 24.0,
    aggregation_window: float = 60.0
) -> AnalyticsEngine:
    """
    Create a configured analytics engine.
    
    Args:
        retention_hours: Data retention period
        aggregation_window: Default aggregation window
        
    Returns:
        Configured AnalyticsEngine
    """
    config = DashboardConfig(
        aggregation_window_seconds=aggregation_window
    )
    return AnalyticsEngine(config=config, retention_hours=retention_hours)


def create_default_alert_rules() -> List[AlertRule]:
    """
    Create default alert rules for common metrics.
    
    Returns:
        List of default AlertRule
    """
    return [
        AlertRule(
            metric_name="query_latency_ms",
            condition="gt",
            threshold=1000.0,
            severity=AlertSeverity.WARNING,
            message_template="Query latency {value}ms exceeds {threshold}ms"
        ),
        AlertRule(
            metric_name="query_latency_ms",
            condition="gt",
            threshold=5000.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical: Query latency {value}ms exceeds {threshold}ms"
        ),
        AlertRule(
            metric_name="error_rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.WARNING,
            message_template="Error rate {value} exceeds {threshold}"
        ),
        AlertRule(
            metric_name="memory_percent",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Memory usage {value}% exceeds {threshold}%"
        )
    ]


# Export all public symbols
__all__ = [
    # Enums
    'MetricType',
    'AggregationType',
    'AlertSeverity',
    'ExportFormat',
    
    # Data classes
    'MetricPoint',
    'AggregatedMetric',
    'Alert',
    'AlertRule',
    'ResourceUsage',
    'DashboardConfig',
    
    # Core classes
    'TimeSeriesStore',
    'MetricsCollector',
    'TimerContext',
    'MetricsAggregator',
    'AnalyticsDashboard',
    'AnalyticsEngine',
    
    # Convenience functions
    'create_analytics_engine',
    'create_default_alert_rules'
]
