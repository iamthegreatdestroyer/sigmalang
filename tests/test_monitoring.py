"""
Tests for Phase 3: Monitoring & Observability

Comprehensive test suite for metrics, health checks, logging, and tracing.
"""

import time
import threading
from unittest.mock import patch, MagicMock

import pytest

from core.monitoring import (
    # Metric types
    MetricType,
    MetricValue,
    # Metric classes
    Counter,
    Gauge,
    Histogram,
    # Registry
    MetricsRegistry,
    get_registry,
    # Predefined metrics
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ENCODE_COUNT,
    # Health checks
    HealthCheckResult,
    HealthChecker,
    get_health_checker,
    # Logging
    StructuredLogger,
    configure_logging,
    # Tracing
    Span,
    Tracer,
    get_tracer,
    # Decorators
    timed,
    counted,
    traced,
)


# =============================================================================
# Counter Tests
# =============================================================================

class TestCounter:
    """Tests for Counter metric."""
    
    def test_counter_basic(self):
        """Test basic counter operations."""
        counter = Counter("test_counter", "Test counter")
        
        counter.inc()
        assert counter.get() == 1.0
        
        counter.inc(5.0)
        assert counter.get() == 6.0
    
    def test_counter_labels(self):
        """Test counter with labels."""
        counter = Counter("labeled_counter", labels=["method", "status"])
        
        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="200")
        counter.inc(2.0, method="GET", status="200")
        
        assert counter.get(method="GET", status="200") == 3.0
        assert counter.get(method="POST", status="200") == 1.0
        assert counter.get(method="GET", status="500") == 0.0
    
    def test_counter_negative_rejected(self):
        """Test counter rejects negative values."""
        counter = Counter("test_counter")
        
        with pytest.raises(ValueError):
            counter.inc(-1.0)
    
    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        counter = Counter("thread_counter")
        threads = []
        
        def increment():
            for _ in range(1000):
                counter.inc()
        
        for _ in range(10):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert counter.get() == 10000.0
    
    def test_counter_collect(self):
        """Test collecting counter metrics."""
        counter = Counter("collect_counter", labels=["status"])
        counter.inc(status="ok")
        counter.inc(2.0, status="error")
        
        metrics = counter.collect()
        
        assert len(metrics) == 2
        assert all(m.metric_type == MetricType.COUNTER for m in metrics)


# =============================================================================
# Gauge Tests
# =============================================================================

class TestGauge:
    """Tests for Gauge metric."""
    
    def test_gauge_basic(self):
        """Test basic gauge operations."""
        gauge = Gauge("test_gauge")
        
        gauge.set(10.0)
        assert gauge.get() == 10.0
        
        gauge.inc(5.0)
        assert gauge.get() == 15.0
        
        gauge.dec(3.0)
        assert gauge.get() == 12.0
    
    def test_gauge_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("labeled_gauge", labels=["region"])
        
        gauge.set(100.0, region="us-east")
        gauge.set(200.0, region="us-west")
        
        assert gauge.get(region="us-east") == 100.0
        assert gauge.get(region="us-west") == 200.0
    
    def test_gauge_track_inprogress(self):
        """Test tracking in-progress operations."""
        gauge = Gauge("inprogress_gauge")
        
        assert gauge.get() == 0.0
        
        with gauge.track_inprogress():
            assert gauge.get() == 1.0
        
        assert gauge.get() == 0.0
    
    def test_gauge_track_inprogress_exception(self):
        """Test in-progress tracking handles exceptions."""
        gauge = Gauge("exception_gauge")
        
        try:
            with gauge.track_inprogress():
                assert gauge.get() == 1.0
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert gauge.get() == 0.0
    
    def test_gauge_collect(self):
        """Test collecting gauge metrics."""
        gauge = Gauge("collect_gauge")
        gauge.set(42.0)
        
        metrics = gauge.collect()
        
        assert len(metrics) == 1
        assert metrics[0].value == 42.0
        assert metrics[0].metric_type == MetricType.GAUGE


# =============================================================================
# Histogram Tests
# =============================================================================

class TestHistogram:
    """Tests for Histogram metric."""
    
    def test_histogram_basic(self):
        """Test basic histogram operations."""
        histogram = Histogram("test_histogram")
        
        histogram.observe(0.1)
        histogram.observe(0.5)
        histogram.observe(1.0)
        
        metrics = histogram.collect()
        
        # Should have bucket, sum, and count metrics
        assert len(metrics) > 0
        assert any("_sum" in m.name for m in metrics)
        assert any("_count" in m.name for m in metrics)
    
    def test_histogram_custom_buckets(self):
        """Test histogram with custom buckets."""
        histogram = Histogram(
            "custom_histogram",
            buckets=(0.1, 0.5, 1.0, 5.0)
        )
        
        histogram.observe(0.05)  # Goes in 0.1 bucket
        histogram.observe(0.3)   # Goes in 0.5 bucket
        histogram.observe(0.8)   # Goes in 1.0 bucket
        
        metrics = histogram.collect()
        bucket_metrics = [m for m in metrics if "_bucket" in m.name]
        
        assert len(bucket_metrics) == 4  # 4 custom buckets
    
    def test_histogram_time_context(self):
        """Test histogram timing context manager."""
        histogram = Histogram("timing_histogram")
        
        with histogram.time():
            time.sleep(0.01)
        
        metrics = histogram.collect()
        sum_metric = next(m for m in metrics if "_sum" in m.name)
        
        assert sum_metric.value >= 0.01
    
    def test_histogram_labels(self):
        """Test histogram with labels."""
        histogram = Histogram("labeled_histogram", labels=["endpoint"])
        
        histogram.observe(0.1, endpoint="/api/v1/encode")
        histogram.observe(0.2, endpoint="/api/v1/decode")
        
        metrics = histogram.collect()
        
        assert len(metrics) > 0


# =============================================================================
# MetricsRegistry Tests
# =============================================================================

class TestMetricsRegistry:
    """Tests for MetricsRegistry."""
    
    def test_registry_counter(self):
        """Test getting counter from registry."""
        registry = MetricsRegistry()
        
        counter1 = registry.counter("test_counter", "Description")
        counter2 = registry.counter("test_counter")
        
        assert counter1 is counter2
    
    def test_registry_gauge(self):
        """Test getting gauge from registry."""
        registry = MetricsRegistry()
        
        gauge = registry.gauge("test_gauge")
        gauge.set(100)
        
        assert registry.gauge("test_gauge").get() == 100
    
    def test_registry_histogram(self):
        """Test getting histogram from registry."""
        registry = MetricsRegistry()
        
        histogram = registry.histogram("test_histogram")
        histogram.observe(0.5)
        
        metrics = registry.histogram("test_histogram").collect()
        assert len(metrics) > 0
    
    def test_registry_collect_all(self):
        """Test collecting all metrics."""
        registry = MetricsRegistry()
        
        registry.counter("counter1").inc()
        registry.gauge("gauge1").set(42)
        registry.histogram("hist1").observe(0.5)
        
        all_metrics = registry.collect_all()
        
        assert len(all_metrics) > 0
    
    def test_registry_to_prometheus(self):
        """Test exporting to Prometheus format."""
        registry = MetricsRegistry()
        
        registry.counter("http_requests_total").inc(status="200")
        registry.gauge("cpu_usage").set(0.75)
        
        output = registry.to_prometheus()
        
        assert "http_requests_total" in output
        assert "cpu_usage" in output
        assert "status=" in output
    
    def test_registry_uptime(self):
        """Test getting uptime."""
        registry = MetricsRegistry()
        
        time.sleep(0.01)
        uptime = registry.get_uptime()
        
        assert uptime >= 0.01


class TestGlobalRegistry:
    """Tests for global registry."""
    
    def test_get_registry_singleton(self):
        """Test global registry is singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        assert registry1 is registry2
    
    def test_predefined_metrics(self):
        """Test predefined metrics exist."""
        REQUEST_COUNT.inc(method="GET", endpoint="/test", status="200")
        REQUEST_LATENCY.observe(0.1, method="GET", endpoint="/test")
        ENCODE_COUNT.inc(status="success")
        
        # Should not raise
        assert REQUEST_COUNT.get(method="GET", endpoint="/test", status="200") >= 1


# =============================================================================
# HealthChecker Tests
# =============================================================================

class TestHealthChecker:
    """Tests for HealthChecker."""
    
    def test_health_checker_register(self):
        """Test registering health checks."""
        checker = HealthChecker()
        
        checker.register("test", lambda: HealthCheckResult(name="test", healthy=True))
        
        result = checker.check("test")
        
        assert result.healthy is True
        assert result.name == "test"
    
    def test_health_checker_unregister(self):
        """Test unregistering health checks."""
        checker = HealthChecker()
        
        checker.register("temp", lambda: HealthCheckResult(name="temp", healthy=True))
        checker.unregister("temp")
        
        result = checker.check("temp")
        
        assert result.healthy is False
        assert "not found" in result.message.lower()
    
    def test_health_checker_check_all(self):
        """Test checking all health checks."""
        checker = HealthChecker()
        
        checker.register("check1", lambda: HealthCheckResult(name="check1", healthy=True))
        checker.register("check2", lambda: HealthCheckResult(name="check2", healthy=True))
        
        results = checker.check_all()
        
        assert len(results) == 2
        assert all(r.healthy for r in results)
    
    def test_health_checker_is_healthy(self):
        """Test overall health check."""
        checker = HealthChecker()
        
        checker.register("good", lambda: HealthCheckResult(name="good", healthy=True))
        assert checker.is_healthy() is True
        
        checker.register("bad", lambda: HealthCheckResult(name="bad", healthy=False))
        assert checker.is_healthy() is False
    
    def test_health_checker_handles_exception(self):
        """Test health check handles exceptions."""
        checker = HealthChecker()
        
        def failing_check():
            raise RuntimeError("Check failed")
        
        checker.register("failing", failing_check)
        result = checker.check("failing")
        
        assert result.healthy is False
        assert "Check failed" in result.message
    
    def test_health_checker_latency(self):
        """Test health check measures latency."""
        checker = HealthChecker()
        
        def slow_check():
            time.sleep(0.01)
            return HealthCheckResult(name="slow", healthy=True)
        
        checker.register("slow", slow_check)
        result = checker.check("slow")
        
        assert result.latency_ms >= 10


class TestGlobalHealthChecker:
    """Tests for global health checker."""
    
    def test_get_health_checker_singleton(self):
        """Test global health checker is singleton."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        
        assert checker1 is checker2
    
    def test_default_self_check(self):
        """Test default self check exists."""
        checker = get_health_checker()
        result = checker.check("self")
        
        assert result.healthy is True


# =============================================================================
# StructuredLogger Tests
# =============================================================================

class TestStructuredLogger:
    """Tests for StructuredLogger."""
    
    def test_logger_info(self):
        """Test info logging."""
        logger = StructuredLogger("test")
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Test message", key="value")
            
            mock_info.assert_called_once()
            logged = mock_info.call_args[0][0]
            
            assert "Test message" in logged
            assert "INFO" in logged
    
    def test_logger_error(self):
        """Test error logging."""
        logger = StructuredLogger("test")
        
        with patch.object(logger.logger, 'error') as mock_error:
            logger.error("Error occurred", error_code=500)
            
            mock_error.assert_called_once()
            logged = mock_error.call_args[0][0]
            
            assert "Error occurred" in logged
            assert "500" in logged
    
    def test_logger_set_context(self):
        """Test setting persistent context."""
        logger = StructuredLogger("test")
        logger.set_context(request_id="123", user="test")
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Message")
            
            logged = mock_info.call_args[0][0]
            
            assert "123" in logged
            assert "test" in logged
    
    def test_logger_clear_context(self):
        """Test clearing context."""
        logger = StructuredLogger("test")
        logger.set_context(key="value")
        logger.clear_context()
        
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Message")
            
            logged = mock_info.call_args[0][0]
            
            assert "key" not in logged or "value" not in logged
    
    def test_logger_with_context(self):
        """Test temporary context manager."""
        logger = StructuredLogger("test")
        
        with patch.object(logger.logger, 'info') as mock_info:
            with logger.with_context(temp_key="temp_value"):
                logger.info("Inside context")
            
            logger.info("Outside context")
            
            calls = mock_info.call_args_list
            
            assert "temp_value" in calls[0][0][0]
            assert "temp_value" not in calls[1][0][0]


# =============================================================================
# Tracer Tests
# =============================================================================

class TestSpan:
    """Tests for Span."""
    
    def test_span_basic(self):
        """Test basic span operations."""
        span = Span(
            trace_id="trace123",
            span_id="span456",
            parent_id=None,
            name="test_operation",
            start_time=time.time()
        )
        
        assert span.name == "test_operation"
        assert span.status == "OK"
    
    def test_span_end(self):
        """Test ending a span."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=time.time()
        )
        
        time.sleep(0.01)
        span.end()
        
        assert span.end_time is not None
        assert span.duration_ms() >= 10
    
    def test_span_attributes(self):
        """Test span attributes."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=time.time()
        )
        
        span.set_attribute("http.method", "GET")
        span.set_attribute("http.url", "/api/v1/encode")
        
        assert span.attributes["http.method"] == "GET"
        assert span.attributes["http.url"] == "/api/v1/encode"
    
    def test_span_events(self):
        """Test span events."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=time.time()
        )
        
        span.add_event("cache_hit", {"key": "user:123"})
        
        assert len(span.events) == 1
        assert span.events[0]["name"] == "cache_hit"
    
    def test_span_set_error(self):
        """Test setting span error."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=time.time()
        )
        
        span.set_error(ValueError("Test error"))
        
        assert span.status == "ERROR"
        assert span.attributes["error.type"] == "ValueError"


class TestTracer:
    """Tests for Tracer."""
    
    def test_tracer_start_span(self):
        """Test starting a span."""
        tracer = Tracer("test_service")
        
        span = tracer.start_span("operation")
        
        assert span.name == "operation"
        assert span.trace_id is not None
        assert span.span_id is not None
    
    def test_tracer_nested_spans(self):
        """Test nested spans have same trace_id."""
        tracer = Tracer("test_service")
        
        parent = tracer.start_span("parent")
        child = tracer.start_span("child")
        
        assert child.trace_id == parent.trace_id
        assert child.parent_id == parent.span_id
        
        tracer.end_span(child)
        tracer.end_span(parent)
    
    def test_tracer_span_context_manager(self):
        """Test span context manager."""
        tracer = Tracer("test_service")
        
        with tracer.span("operation", {"key": "value"}) as span:
            assert span.name == "operation"
            assert span.attributes["key"] == "value"
        
        assert span.end_time is not None
    
    def test_tracer_span_exception(self):
        """Test span handles exceptions."""
        tracer = Tracer("test_service")
        
        try:
            with tracer.span("failing") as span:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert span.status == "ERROR"
        assert span.attributes["error.type"] == "ValueError"
    
    def test_tracer_get_spans(self):
        """Test getting recorded spans."""
        tracer = Tracer("test_service")
        tracer.clear_spans()
        
        with tracer.span("op1"):
            pass
        
        with tracer.span("op2"):
            pass
        
        spans = tracer.get_spans()
        
        assert len(spans) == 2
    
    def test_tracer_clear_spans(self):
        """Test clearing spans."""
        tracer = Tracer("test_service")
        
        with tracer.span("op"):
            pass
        
        tracer.clear_spans()
        
        assert len(tracer.get_spans()) == 0


class TestGlobalTracer:
    """Tests for global tracer."""
    
    def test_get_tracer_singleton(self):
        """Test global tracer is singleton."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        
        assert tracer1 is tracer2


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    """Tests for monitoring decorators."""
    
    def test_timed_decorator(self):
        """Test timed decorator."""
        @timed("test_function_duration")
        def slow_function():
            time.sleep(0.01)
            return "result"
        
        result = slow_function()
        
        assert result == "result"
    
    def test_counted_decorator(self):
        """Test counted decorator."""
        registry = get_registry()
        
        @counted("test_function_calls")
        def counted_function():
            return "called"
        
        for _ in range(5):
            counted_function()
        
        counter = registry.counter("test_function_calls")
        assert counter.get() >= 5
    
    def test_traced_decorator(self):
        """Test traced decorator."""
        tracer = get_tracer()
        tracer.clear_spans()
        
        @traced("test_traced_function")
        def traced_function():
            return "traced"
        
        result = traced_function()
        
        assert result == "traced"
        
        spans = tracer.get_spans()
        assert any(s.name == "test_traced_function" for s in spans)
    
    def test_traced_decorator_with_exception(self):
        """Test traced decorator handles exceptions."""
        tracer = get_tracer()
        tracer.clear_spans()
        
        @traced("failing_function")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        spans = tracer.get_spans()
        error_span = next(s for s in spans if s.name == "failing_function")
        
        assert error_span.status == "ERROR"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
