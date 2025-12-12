"""
Test suite for Analytics & Visualization Engine.

Tests cover:
- MetricPoint creation and serialization
- TimeSeriesStore operations
- MetricsCollector functionality
- MetricsAggregator calculations
- AnalyticsDashboard features
- AnalyticsEngine integration
- Alert system functionality
- Export formats
"""

import pytest
import time
import json
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.analytics_engine import (
    # Enums
    MetricType,
    AggregationType,
    AlertSeverity,
    ExportFormat,
    
    # Data classes
    MetricPoint,
    AggregatedMetric,
    Alert,
    AlertRule,
    ResourceUsage,
    DashboardConfig,
    
    # Core classes
    TimeSeriesStore,
    MetricsCollector,
    TimerContext,
    MetricsAggregator,
    AnalyticsDashboard,
    AnalyticsEngine,
    
    # Convenience functions
    create_analytics_engine,
    create_default_alert_rules
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def time_series_store():
    """Create a TimeSeriesStore for testing."""
    return TimeSeriesStore(retention_hours=1.0, max_points_per_metric=1000)


@pytest.fixture
def metrics_collector(time_series_store):
    """Create a MetricsCollector for testing."""
    return MetricsCollector(store=time_series_store)


@pytest.fixture
def metrics_aggregator(time_series_store):
    """Create a MetricsAggregator for testing."""
    return MetricsAggregator(store=time_series_store)


@pytest.fixture
def analytics_dashboard(metrics_collector, metrics_aggregator):
    """Create an AnalyticsDashboard for testing."""
    return AnalyticsDashboard(
        collector=metrics_collector,
        aggregator=metrics_aggregator
    )


@pytest.fixture
def analytics_engine():
    """Create an AnalyticsEngine for testing."""
    return create_analytics_engine(retention_hours=1.0)


# ============================================================================
# TEST ENUMS
# ============================================================================

class TestMetricType:
    """Tests for MetricType enum."""
    
    def test_metric_type_values(self):
        """Test that all metric types exist."""
        assert MetricType.COUNTER is not None
        assert MetricType.GAUGE is not None
        assert MetricType.HISTOGRAM is not None
        assert MetricType.TIMER is not None
        assert MetricType.RATE is not None
    
    def test_metric_type_count(self):
        """Test number of metric types."""
        assert len(MetricType) == 5


class TestAggregationType:
    """Tests for AggregationType enum."""
    
    def test_aggregation_type_values(self):
        """Test that all aggregation types exist."""
        assert AggregationType.SUM is not None
        assert AggregationType.MEAN is not None
        assert AggregationType.MEDIAN is not None
        assert AggregationType.P95 is not None
        assert AggregationType.P99 is not None


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""
    
    def test_alert_severity_values(self):
        """Test that all severity levels exist."""
        assert AlertSeverity.INFO is not None
        assert AlertSeverity.WARNING is not None
        assert AlertSeverity.CRITICAL is not None


class TestExportFormat:
    """Tests for ExportFormat enum."""
    
    def test_export_format_values(self):
        """Test that all export formats exist."""
        assert ExportFormat.JSON is not None
        assert ExportFormat.CSV is not None
        assert ExportFormat.PROMETHEUS is not None


# ============================================================================
# TEST METRIC POINT
# ============================================================================

class TestMetricPoint:
    """Tests for MetricPoint data class."""
    
    def test_metric_point_creation(self):
        """Test basic metric point creation."""
        point = MetricPoint(name="test_metric", value=42.0)
        
        assert point.name == "test_metric"
        assert point.value == 42.0
        assert point.timestamp > 0
        assert point.metric_type == MetricType.GAUGE
    
    def test_metric_point_with_labels(self):
        """Test metric point with labels."""
        labels = {"env": "test", "host": "localhost"}
        point = MetricPoint(
            name="labeled_metric",
            value=100.0,
            labels=labels
        )
        
        assert point.labels == labels
    
    def test_metric_point_with_type(self):
        """Test metric point with custom type."""
        point = MetricPoint(
            name="counter_metric",
            value=1.0,
            metric_type=MetricType.COUNTER
        )
        
        assert point.metric_type == MetricType.COUNTER
    
    def test_metric_point_to_dict(self):
        """Test conversion to dictionary."""
        point = MetricPoint(
            name="test",
            value=50.0,
            timestamp=1000.0,
            labels={"key": "value"},
            metric_type=MetricType.TIMER
        )
        
        d = point.to_dict()
        
        assert d['name'] == "test"
        assert d['value'] == 50.0
        assert d['timestamp'] == 1000.0
        assert d['labels'] == {"key": "value"}
        assert d['type'] == "TIMER"
    
    def test_metric_point_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'name': 'restored',
            'value': 25.0,
            'timestamp': 2000.0,
            'labels': {'a': 'b'},
            'type': 'HISTOGRAM'
        }
        
        point = MetricPoint.from_dict(data)
        
        assert point.name == 'restored'
        assert point.value == 25.0
        assert point.timestamp == 2000.0
        assert point.labels == {'a': 'b'}
        assert point.metric_type == MetricType.HISTOGRAM


# ============================================================================
# TEST AGGREGATED METRIC
# ============================================================================

class TestAggregatedMetric:
    """Tests for AggregatedMetric data class."""
    
    def test_aggregated_metric_creation(self):
        """Test aggregated metric creation."""
        agg = AggregatedMetric(
            name="latency",
            count=100,
            sum_value=1500.0,
            min_value=5.0,
            max_value=50.0,
            mean_value=15.0,
            median_value=14.0,
            stddev_value=8.0,
            p50_value=14.0,
            p95_value=45.0,
            p99_value=49.0,
            start_time=1000.0,
            end_time=2000.0
        )
        
        assert agg.name == "latency"
        assert agg.count == 100
        assert agg.mean_value == 15.0
    
    def test_aggregated_metric_to_dict(self):
        """Test conversion to dictionary."""
        agg = AggregatedMetric(
            name="test",
            count=10,
            sum_value=100.0,
            min_value=1.0,
            max_value=20.0,
            mean_value=10.0,
            median_value=9.0,
            stddev_value=5.0,
            p50_value=9.0,
            p95_value=18.0,
            p99_value=19.0,
            start_time=0.0,
            end_time=60.0
        )
        
        d = agg.to_dict()
        
        assert 'name' in d
        assert 'count' in d
        assert 'mean' in d
        assert 'p95' in d


# ============================================================================
# TEST ALERT
# ============================================================================

class TestAlert:
    """Tests for Alert data class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            metric_name="latency",
            severity=AlertSeverity.WARNING,
            message="Latency too high",
            value=1500.0,
            threshold=1000.0
        )
        
        assert alert.metric_name == "latency"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.value == 1500.0
    
    def test_alert_to_dict(self):
        """Test alert conversion to dictionary."""
        alert = Alert(
            metric_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Critical!",
            value=100.0,
            threshold=50.0
        )
        
        d = alert.to_dict()
        
        assert d['metric_name'] == "test"
        assert d['severity'] == "CRITICAL"
        assert d['value'] == 100.0


# ============================================================================
# TEST ALERT RULE
# ============================================================================

class TestAlertRule:
    """Tests for AlertRule data class."""
    
    def test_alert_rule_greater_than(self):
        """Test greater than condition."""
        rule = AlertRule(
            metric_name="latency",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            message_template="Value {value} > {threshold}"
        )
        
        assert rule.check(150.0) is True
        assert rule.check(100.0) is False
        assert rule.check(50.0) is False
    
    def test_alert_rule_less_than(self):
        """Test less than condition."""
        rule = AlertRule(
            metric_name="throughput",
            condition="lt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            message_template="Low throughput"
        )
        
        assert rule.check(5.0) is True
        assert rule.check(10.0) is False
        assert rule.check(15.0) is False
    
    def test_alert_rule_greater_equal(self):
        """Test greater than or equal condition."""
        rule = AlertRule(
            metric_name="errors",
            condition="gte",
            threshold=5.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Errors >= {threshold}"
        )
        
        assert rule.check(5.0) is True
        assert rule.check(6.0) is True
        assert rule.check(4.0) is False
    
    def test_alert_rule_equal(self):
        """Test equal condition."""
        rule = AlertRule(
            metric_name="status",
            condition="eq",
            threshold=0.0,
            severity=AlertSeverity.INFO,
            message_template="Status is zero"
        )
        
        assert rule.check(0.0) is True
        assert rule.check(1.0) is False


# ============================================================================
# TEST RESOURCE USAGE
# ============================================================================

class TestResourceUsage:
    """Tests for ResourceUsage data class."""
    
    def test_resource_usage_creation(self):
        """Test resource usage creation."""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_bytes=1024000,
            active_threads=10
        )
        
        assert usage.cpu_percent == 50.0
        assert usage.memory_percent == 60.0
        assert usage.active_threads == 10
    
    def test_resource_usage_to_dict(self):
        """Test conversion to dictionary."""
        usage = ResourceUsage()
        d = usage.to_dict()
        
        assert 'timestamp' in d
        assert 'cpu_percent' in d
        assert 'memory_percent' in d


# ============================================================================
# TEST TIME SERIES STORE
# ============================================================================

class TestTimeSeriesStore:
    """Tests for TimeSeriesStore."""
    
    def test_store_creation(self):
        """Test store creation."""
        store = TimeSeriesStore(retention_hours=24.0)
        
        assert store.retention_hours == 24.0
        assert len(store.get_all_metrics()) == 0
    
    def test_add_and_get_latest(self, time_series_store):
        """Test adding and retrieving latest points."""
        time_series_store.add("test_metric", 10.0)
        time_series_store.add("test_metric", 20.0)
        time_series_store.add("test_metric", 30.0)
        
        latest = time_series_store.get_latest("test_metric", 2)
        
        assert len(latest) == 2
        assert latest[0].value == 20.0
        assert latest[1].value == 30.0
    
    def test_get_range(self, time_series_store):
        """Test getting points in time range."""
        now = time.time()
        
        # Add points at different times
        time_series_store.add("metric", 10.0, timestamp=now - 100)
        time_series_store.add("metric", 20.0, timestamp=now - 50)
        time_series_store.add("metric", 30.0, timestamp=now)
        
        # Get range
        points = time_series_store.get_range("metric", now - 75, now + 1)
        
        assert len(points) == 2
        assert points[0].value == 20.0
        assert points[1].value == 30.0
    
    def test_get_nonexistent_metric(self, time_series_store):
        """Test getting points for nonexistent metric."""
        latest = time_series_store.get_latest("nonexistent", 10)
        assert latest == []
        
        range_points = time_series_store.get_range("nonexistent", 0, time.time())
        assert range_points == []
    
    def test_get_all_metrics(self, time_series_store):
        """Test getting all metric names."""
        time_series_store.add("metric_a", 1.0)
        time_series_store.add("metric_b", 2.0)
        time_series_store.add("metric_c", 3.0)
        
        metrics = time_series_store.get_all_metrics()
        
        assert len(metrics) == 3
        assert "metric_a" in metrics
        assert "metric_b" in metrics
        assert "metric_c" in metrics
    
    def test_get_count(self, time_series_store):
        """Test getting point count."""
        for i in range(5):
            time_series_store.add("counted", float(i))
        
        assert time_series_store.get_count("counted") == 5
        assert time_series_store.get_count("nonexistent") == 0
    
    def test_clear(self, time_series_store):
        """Test clearing store."""
        time_series_store.add("metric1", 1.0)
        time_series_store.add("metric2", 2.0)
        
        time_series_store.clear()
        
        assert len(time_series_store.get_all_metrics()) == 0
    
    def test_with_labels(self, time_series_store):
        """Test adding metrics with labels."""
        labels = {"env": "test", "region": "us-west"}
        time_series_store.add(
            "labeled_metric", 
            42.0, 
            labels=labels,
            metric_type=MetricType.COUNTER
        )
        
        latest = time_series_store.get_latest("labeled_metric", 1)
        
        assert len(latest) == 1
        assert latest[0].labels == labels
        assert latest[0].metric_type == MetricType.COUNTER
    
    def test_max_points_limit(self):
        """Test that store respects max points limit."""
        store = TimeSeriesStore(max_points_per_metric=10)
        
        # Add more than max points
        for i in range(20):
            store.add("limited", float(i))
        
        # Should only have max points
        assert store.get_count("limited") == 10
        
        # Should have latest values
        latest = store.get_latest("limited", 10)
        assert latest[0].value == 10.0  # First remaining
        assert latest[-1].value == 19.0  # Last added
    
    def test_thread_safety(self, time_series_store):
        """Test thread-safe operations."""
        errors = []
        
        def add_points(metric_name: str, count: int):
            try:
                for i in range(count):
                    time_series_store.add(metric_name, float(i))
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=add_points, args=(f"metric_{i}", 100))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(time_series_store.get_all_metrics()) == 5


# ============================================================================
# TEST METRICS COLLECTOR
# ============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_record_counter(self, metrics_collector):
        """Test recording counter metric."""
        metrics_collector.record_counter("requests_total", 1)
        metrics_collector.record_counter("requests_total", 1)
        metrics_collector.record_counter("requests_total", 1)
        
        value = metrics_collector.get_counter_value("requests_total")
        assert value == 3.0
    
    def test_record_gauge(self, metrics_collector):
        """Test recording gauge metric."""
        metrics_collector.record_gauge("temperature", 25.5)
        metrics_collector.record_gauge("temperature", 26.0)
        
        latest = metrics_collector.store.get_latest("temperature", 1)
        assert latest[0].value == 26.0
    
    def test_record_histogram(self, metrics_collector):
        """Test recording histogram metric."""
        for value in [10, 20, 30, 40, 50]:
            metrics_collector.record_histogram("response_time", float(value))
        
        # Check percentiles
        p50 = metrics_collector.get_histogram_percentile("response_time", 50)
        assert p50 is not None
        assert 20 <= p50 <= 40
    
    def test_record_timer(self, metrics_collector):
        """Test recording timer metric."""
        metrics_collector.record_timer("process_duration", 150.5)
        
        latest = metrics_collector.store.get_latest("process_duration", 1)
        assert latest[0].value == 150.5
        assert latest[0].metric_type == MetricType.TIMER
    
    def test_record_rate(self, metrics_collector):
        """Test recording rate metric."""
        metrics_collector.record_rate("events_per_second", 1000.0)
        
        latest = metrics_collector.store.get_latest("events_per_second", 1)
        assert latest[0].value == 1000.0
        assert latest[0].metric_type == MetricType.RATE
    
    def test_time_operation_context_manager(self, metrics_collector):
        """Test timing operation with context manager."""
        with metrics_collector.time_operation("timed_op"):
            time.sleep(0.05)  # 50ms
        
        latest = metrics_collector.store.get_latest("timed_op", 1)
        assert len(latest) == 1
        assert latest[0].value >= 50.0  # At least 50ms
    
    def test_counter_with_labels(self, metrics_collector):
        """Test counter with labels."""
        metrics_collector.record_counter(
            "http_requests",
            1,
            labels={"method": "GET", "status": "200"}
        )
        
        latest = metrics_collector.store.get_latest("http_requests", 1)
        assert latest[0].labels["method"] == "GET"
    
    def test_collect_resource_usage(self, metrics_collector):
        """Test resource usage collection."""
        usage = metrics_collector.collect_resource_usage()
        
        assert usage.timestamp > 0
        assert usage.active_threads > 0
    
    def test_get_resource_history(self, metrics_collector):
        """Test resource history."""
        # Collect several samples
        for _ in range(5):
            metrics_collector.collect_resource_usage()
        
        history = metrics_collector.get_resource_history(last_n=3)
        assert len(history) == 3
    
    def test_histogram_percentile_nonexistent(self, metrics_collector):
        """Test getting percentile for nonexistent histogram."""
        result = metrics_collector.get_histogram_percentile("nonexistent", 50)
        assert result is None


# ============================================================================
# TEST TIMER CONTEXT
# ============================================================================

class TestTimerContext:
    """Tests for TimerContext."""
    
    def test_timer_context_records_duration(self, metrics_collector):
        """Test that timer context records correct duration."""
        with TimerContext(metrics_collector, "test_timer") as ctx:
            time.sleep(0.01)  # 10ms
        
        latest = metrics_collector.store.get_latest("test_timer", 1)
        assert len(latest) == 1
        assert latest[0].value >= 10.0  # At least 10ms
    
    def test_timer_context_start_time(self, metrics_collector):
        """Test that start time is recorded."""
        ctx = TimerContext(metrics_collector, "ctx_test")
        ctx.__enter__()
        
        assert ctx.start_time > 0


# ============================================================================
# TEST METRICS AGGREGATOR
# ============================================================================

class TestMetricsAggregator:
    """Tests for MetricsAggregator."""
    
    def test_aggregate_basic(self, time_series_store, metrics_aggregator):
        """Test basic aggregation."""
        # Add test data
        for value in [10.0, 20.0, 30.0, 40.0, 50.0]:
            time_series_store.add("test_metric", value)
        
        agg = metrics_aggregator.aggregate("test_metric", window_seconds=60)
        
        assert agg is not None
        assert agg.count == 5
        assert agg.sum_value == 150.0
        assert agg.min_value == 10.0
        assert agg.max_value == 50.0
        assert agg.mean_value == 30.0
    
    def test_aggregate_empty(self, metrics_aggregator):
        """Test aggregation of empty metric."""
        agg = metrics_aggregator.aggregate("nonexistent", window_seconds=60)
        assert agg is None
    
    def test_aggregate_multiple(self, time_series_store, metrics_aggregator):
        """Test aggregating multiple metrics."""
        time_series_store.add("metric_a", 10.0)
        time_series_store.add("metric_b", 20.0)
        
        results = metrics_aggregator.aggregate_multiple(
            ["metric_a", "metric_b", "metric_c"],
            window_seconds=60
        )
        
        assert results["metric_a"] is not None
        assert results["metric_b"] is not None
        assert results["metric_c"] is None  # Doesn't exist
    
    def test_calculate_rate_of_change(self, time_series_store, metrics_aggregator):
        """Test rate of change calculation."""
        now = time.time()
        
        # Add increasing values over time
        time_series_store.add("increasing", 100.0, timestamp=now - 60)
        time_series_store.add("increasing", 200.0, timestamp=now)
        
        rate = metrics_aggregator.calculate_rate_of_change(
            "increasing", 
            window_seconds=120
        )
        
        assert rate is not None
        # Should be approximately 100 / 60 ≈ 1.67 per second
        assert 1.0 < rate < 2.5
    
    def test_rate_of_change_insufficient_data(self, metrics_aggregator):
        """Test rate of change with insufficient data."""
        rate = metrics_aggregator.calculate_rate_of_change("nonexistent")
        assert rate is None
    
    def test_detect_anomaly(self, time_series_store, metrics_aggregator):
        """Test anomaly detection."""
        # Add normal values
        for i in range(100):
            time_series_store.add("normal_metric", 50.0 + (i % 10))
        
        # Add an anomaly
        time_series_store.add("normal_metric", 200.0)  # Way outside normal range
        
        result = metrics_aggregator.detect_anomaly(
            "normal_metric",
            threshold_stddev=2.0,
            window_seconds=3600
        )
        
        assert result is not None
        is_anomaly, latest_value, z_score = result
        assert is_anomaly is True
        assert latest_value == 200.0
        assert abs(z_score) > 2.0
    
    def test_percentile_calculation(self, time_series_store, metrics_aggregator):
        """Test percentile calculations in aggregation."""
        # Add data
        for i in range(100):
            time_series_store.add("percentile_test", float(i))
        
        agg = metrics_aggregator.aggregate("percentile_test", window_seconds=60)
        
        assert agg is not None
        assert agg.p50_value >= 40 and agg.p50_value <= 60
        assert agg.p95_value >= 90
        assert agg.p99_value >= 95


# ============================================================================
# TEST ANALYTICS DASHBOARD
# ============================================================================

class TestAnalyticsDashboard:
    """Tests for AnalyticsDashboard."""
    
    def test_summary(self, analytics_dashboard, metrics_collector):
        """Test dashboard summary generation."""
        # Add some metrics
        metrics_collector.record_gauge("test_gauge", 100.0)
        metrics_collector.record_counter("test_counter", 5)
        
        summary = analytics_dashboard.summary()
        
        assert 'dashboard_name' in summary
        assert 'timestamp' in summary
        assert 'metrics_count' in summary
        assert summary['metrics_count'] >= 2
    
    def test_check_alerts_triggers(self, analytics_dashboard, metrics_collector):
        """Test that alerts are triggered correctly."""
        # Add alert rule
        rule = AlertRule(
            metric_name="high_latency",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            message_template="Latency {value} > {threshold}"
        )
        analytics_dashboard.add_alert_rule(rule)
        
        # Add metric that triggers alert
        metrics_collector.record_gauge("high_latency", 150.0)
        
        alerts = analytics_dashboard.check_alerts()
        
        assert len(alerts) == 1
        assert alerts[0].metric_name == "high_latency"
        assert alerts[0].severity == AlertSeverity.WARNING
    
    def test_check_alerts_no_trigger(self, analytics_dashboard, metrics_collector):
        """Test that alerts don't trigger for normal values."""
        rule = AlertRule(
            metric_name="normal_metric",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            message_template="Alert!"
        )
        analytics_dashboard.add_alert_rule(rule)
        
        metrics_collector.record_gauge("normal_metric", 50.0)
        
        alerts = analytics_dashboard.check_alerts()
        assert len(alerts) == 0
    
    def test_alert_cooldown(self, analytics_dashboard, metrics_collector):
        """Test alert cooldown prevents spam."""
        rule = AlertRule(
            metric_name="spammy_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            message_template="Alert!",
            cooldown_seconds=60.0  # 60 second cooldown
        )
        analytics_dashboard.add_alert_rule(rule)
        
        # First alert
        metrics_collector.record_gauge("spammy_metric", 20.0)
        alerts1 = analytics_dashboard.check_alerts()
        
        # Second alert immediately after - should be suppressed
        metrics_collector.record_gauge("spammy_metric", 30.0)
        alerts2 = analytics_dashboard.check_alerts()
        
        assert len(alerts1) == 1
        assert len(alerts2) == 0  # Cooldown active
    
    def test_get_alerts_filtered(self, analytics_dashboard, metrics_collector):
        """Test getting filtered alerts."""
        # Add rules for different severities
        analytics_dashboard.add_alert_rule(AlertRule(
            metric_name="warning_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            message_template="Warning!"
        ))
        analytics_dashboard.add_alert_rule(AlertRule(
            metric_name="critical_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical!"
        ))
        
        # Trigger both
        metrics_collector.record_gauge("warning_metric", 20.0)
        metrics_collector.record_gauge("critical_metric", 20.0)
        analytics_dashboard.check_alerts()
        
        # Get only critical
        critical_alerts = analytics_dashboard.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_clear_alerts(self, analytics_dashboard, metrics_collector):
        """Test clearing alerts."""
        rule = AlertRule(
            metric_name="clearable",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            message_template="Alert!"
        )
        analytics_dashboard.add_alert_rule(rule)
        
        metrics_collector.record_gauge("clearable", 20.0)
        analytics_dashboard.check_alerts()
        
        cleared = analytics_dashboard.clear_alerts()
        
        assert cleared == 1
        assert len(analytics_dashboard.get_alerts()) == 0
    
    def test_export_json(self, analytics_dashboard, metrics_collector):
        """Test JSON export."""
        metrics_collector.record_gauge("export_test", 42.0)
        
        exported = analytics_dashboard.export(
            format=ExportFormat.JSON,
            window_seconds=60
        )
        
        data = json.loads(exported)
        assert 'export_time' in data
        assert 'metrics' in data
        assert 'export_test' in data['metrics']
    
    def test_export_csv(self, analytics_dashboard, metrics_collector):
        """Test CSV export."""
        metrics_collector.record_gauge("csv_test", 100.0)
        
        exported = analytics_dashboard.export(
            format=ExportFormat.CSV,
            window_seconds=60
        )
        
        lines = exported.split('\n')
        assert lines[0] == "timestamp,metric_name,value,type"
        assert len(lines) >= 2
        assert "csv_test" in lines[1]
    
    def test_export_prometheus(self, analytics_dashboard, metrics_collector):
        """Test Prometheus format export."""
        metrics_collector.record_gauge("prom_test", 75.0)
        
        exported = analytics_dashboard.export(
            format=ExportFormat.PROMETHEUS,
            window_seconds=60
        )
        
        assert "prom_test" in exported
        assert "# TYPE" in exported
    
    def test_get_health_status_healthy(self, analytics_dashboard):
        """Test health status when healthy."""
        health = analytics_dashboard.get_health_status()
        
        assert health['status'] == 'healthy'
        assert health['critical_alerts'] == 0
    
    def test_get_health_status_warning(self, analytics_dashboard, metrics_collector):
        """Test health status with warnings."""
        rule = AlertRule(
            metric_name="health_warning",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            message_template="Warning!"
        )
        analytics_dashboard.add_alert_rule(rule)
        metrics_collector.record_gauge("health_warning", 20.0)
        analytics_dashboard.check_alerts()
        
        health = analytics_dashboard.get_health_status()
        
        assert health['status'] == 'warning'
        assert health['warning_alerts'] == 1
    
    def test_get_health_status_critical(self, analytics_dashboard, metrics_collector):
        """Test health status with critical alerts."""
        rule = AlertRule(
            metric_name="health_critical",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical!"
        )
        analytics_dashboard.add_alert_rule(rule)
        metrics_collector.record_gauge("health_critical", 20.0)
        analytics_dashboard.check_alerts()
        
        health = analytics_dashboard.get_health_status()
        
        assert health['status'] == 'critical'
        assert health['critical_alerts'] == 1


# ============================================================================
# TEST ANALYTICS ENGINE
# ============================================================================

class TestAnalyticsEngine:
    """Tests for AnalyticsEngine."""
    
    def test_engine_creation(self, analytics_engine):
        """Test engine creation."""
        assert analytics_engine.store is not None
        assert analytics_engine.collector is not None
        assert analytics_engine.aggregator is not None
        assert analytics_engine.dashboard is not None
    
    def test_record_latency(self, analytics_engine):
        """Test recording latency."""
        analytics_engine.record_latency("query_time", 25.5)
        
        latest = analytics_engine.store.get_latest("query_time", 1)
        assert latest[0].value == 25.5
    
    def test_record_throughput(self, analytics_engine):
        """Test recording throughput."""
        analytics_engine.record_throughput("requests_per_sec", 500)
        
        latest = analytics_engine.store.get_latest("requests_per_sec", 1)
        assert latest[0].value == 500
    
    def test_record_counter(self, analytics_engine):
        """Test recording counter."""
        analytics_engine.record_counter("total_requests", 1)
        analytics_engine.record_counter("total_requests", 1)
        
        value = analytics_engine.collector.get_counter_value("total_requests")
        assert value == 2.0
    
    def test_record_gauge(self, analytics_engine):
        """Test recording gauge."""
        analytics_engine.record_gauge("active_users", 150)
        
        latest = analytics_engine.store.get_latest("active_users", 1)
        assert latest[0].value == 150
    
    def test_time_operation(self, analytics_engine):
        """Test timing operation."""
        with analytics_engine.time_operation("timed_task"):
            time.sleep(0.01)
        
        latest = analytics_engine.store.get_latest("timed_task", 1)
        assert latest[0].value >= 10.0
    
    def test_get_summary(self, analytics_engine):
        """Test getting summary."""
        analytics_engine.record_gauge("summary_test", 42)
        
        summary = analytics_engine.get_summary()
        
        assert 'dashboard_name' in summary
        assert 'aggregations' in summary
    
    def test_get_aggregation(self, analytics_engine):
        """Test getting aggregation."""
        for i in range(10):
            analytics_engine.record_latency("agg_test", float(i * 10))
        
        agg = analytics_engine.get_aggregation("agg_test", window_seconds=60)
        
        assert agg is not None
        assert agg.count == 10
        assert agg.mean_value == 45.0
    
    def test_check_alerts(self, analytics_engine):
        """Test checking alerts."""
        analytics_engine.add_alert_rule(
            metric_name="alert_test",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING
        )
        
        analytics_engine.record_gauge("alert_test", 150)
        alerts = analytics_engine.check_alerts()
        
        assert len(alerts) == 1
    
    def test_export(self, analytics_engine):
        """Test export functionality."""
        analytics_engine.record_gauge("export_metric", 42)
        
        json_export = analytics_engine.export(ExportFormat.JSON)
        assert "export_metric" in json_export
    
    def test_get_health(self, analytics_engine):
        """Test getting health status."""
        health = analytics_engine.get_health()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'warning', 'critical']
    
    def test_add_alert_rule(self, analytics_engine):
        """Test adding alert rule."""
        analytics_engine.add_alert_rule(
            metric_name="custom_metric",
            condition="lt",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL,
            message="Low value alert"
        )
        
        assert len(analytics_engine.dashboard.config.alert_rules) >= 1


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_analytics_engine(self):
        """Test creating analytics engine."""
        engine = create_analytics_engine(retention_hours=12.0)
        
        assert engine is not None
        assert engine.store.retention_hours == 12.0
    
    def test_create_analytics_engine_with_aggregation_window(self):
        """Test creating engine with custom aggregation window."""
        engine = create_analytics_engine(aggregation_window=120.0)
        
        assert engine.dashboard.config.aggregation_window_seconds == 120.0
    
    def test_create_default_alert_rules(self):
        """Test creating default alert rules."""
        rules = create_default_alert_rules()
        
        assert len(rules) >= 4
        
        # Check for expected rules
        rule_names = [r.metric_name for r in rules]
        assert "query_latency_ms" in rule_names
        assert "error_rate" in rule_names


# ============================================================================
# TEST INTEGRATION SCENARIOS
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests for analytics engine."""
    
    def test_full_metrics_workflow(self):
        """Test complete metrics collection and analysis workflow."""
        engine = create_analytics_engine()
        
        # Simulate a workload
        for i in range(50):
            engine.record_latency("api_latency", 10.0 + (i % 20))
            engine.record_counter("api_requests")
            engine.record_throughput("api_rps", 100 + (i % 50))
        
        # Get aggregation
        agg = engine.get_aggregation("api_latency", window_seconds=60)
        assert agg is not None
        assert agg.count == 50
        
        # Get summary
        summary = engine.get_summary()
        assert summary['metrics_count'] >= 3
        
        # Export
        export = engine.export(ExportFormat.JSON)
        data = json.loads(export)
        assert "api_latency" in data['metrics']
    
    def test_alerting_workflow(self):
        """Test alerting workflow."""
        engine = create_analytics_engine()
        
        # Add custom alert
        engine.add_alert_rule(
            metric_name="latency",
            condition="gt",
            threshold=50.0,
            severity=AlertSeverity.WARNING,
            message="Latency spike: {value}ms"
        )
        
        # Normal values - no alerts
        engine.record_latency("latency", 30.0)
        alerts = engine.check_alerts()
        assert len(alerts) == 0
        
        # High value - should alert
        engine.record_latency("latency", 100.0)
        alerts = engine.check_alerts()
        assert len(alerts) == 1
        assert "spike" in alerts[0].message
    
    def test_multi_metric_analysis(self):
        """Test analysis across multiple related metrics."""
        engine = create_analytics_engine()
        
        # Record related metrics
        for i in range(30):
            latency = 10.0 + (i * 2)
            throughput = 100 - (i * 2)
            errors = 1 if i > 25 else 0
            
            engine.record_latency("response_time", latency)
            engine.record_throughput("requests_per_sec", throughput)
            engine.record_counter("error_count", errors)
        
        # Analyze
        latency_agg = engine.get_aggregation("response_time")
        throughput_agg = engine.get_aggregation("requests_per_sec")
        
        assert latency_agg is not None
        assert throughput_agg is not None
        
        # As latency increased, throughput decreased
        assert latency_agg.mean_value > latency_agg.min_value
    
    def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection."""
        engine = create_analytics_engine()
        errors = []
        
        def record_metrics(thread_id: int):
            try:
                for i in range(100):
                    engine.record_latency(f"thread_{thread_id}_latency", float(i))
                    engine.record_counter(f"thread_{thread_id}_count")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=record_metrics, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Verify all metrics were recorded
        all_metrics = engine.store.get_all_metrics()
        assert len(all_metrics) >= 10  # 5 threads * 2 metrics each
    
    def test_export_formats_consistency(self):
        """Test that all export formats contain same data."""
        engine = create_analytics_engine()
        
        # Add test data
        engine.record_gauge("consistency_test", 42.0)
        engine.record_counter("count_test", 5)
        
        # Export all formats
        json_export = engine.export(ExportFormat.JSON)
        csv_export = engine.export(ExportFormat.CSV)
        prom_export = engine.export(ExportFormat.PROMETHEUS)
        
        # All should contain our test metrics
        assert "consistency_test" in json_export
        assert "consistency_test" in csv_export
        assert "consistency_test" in prom_export
