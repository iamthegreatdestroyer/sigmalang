"""
ΣLANG Monitoring & Observability

Prometheus metrics, health checks, structured logging, and OpenTelemetry tracing.
"""

import time
import threading
import logging
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Metric Types
# =============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


# =============================================================================
# Counter
# =============================================================================

class Counter:
    """A counter that only goes up."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only increase")
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + value
    
    def get(self, **labels) -> float:
        """Get current counter value."""
        label_key = self._label_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))
    
    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(
                    name=self.name,
                    value=value,
                    labels=dict(labels),
                    metric_type=MetricType.COUNTER
                )
                for labels, value in self._values.items()
            ]


# =============================================================================
# Gauge
# =============================================================================

class Gauge:
    """A gauge that can go up and down."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def set(self, value: float, **labels) -> None:
        """Set the gauge value."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] = value
    
    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + value
    
    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement the gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) - value
    
    def get(self, **labels) -> float:
        """Get current gauge value."""
        label_key = self._label_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))
    
    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(
                    name=self.name,
                    value=value,
                    labels=dict(labels),
                    metric_type=MetricType.GAUGE
                )
                for labels, value in self._values.items()
            ]
    
    @contextmanager
    def track_inprogress(self, **labels):
        """Context manager to track in-progress operations."""
        self.inc(**labels)
        try:
            yield
        finally:
            self.dec(**labels)


# =============================================================================
# Histogram
# =============================================================================

class Histogram:
    """A histogram for measuring distributions."""
    
    DEFAULT_BUCKETS = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    )
    
    def __init__(
        self, 
        name: str, 
        description: str = "", 
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: Dict[tuple, Dict[float, int]] = {}
        self._sums: Dict[tuple, float] = {}
        self._totals: Dict[tuple, int] = {}
        self._lock = threading.Lock()
    
    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        label_key = self._label_key(labels)
        with self._lock:
            if label_key not in self._counts:
                self._counts[label_key] = {b: 0 for b in self.buckets}
                self._sums[label_key] = 0.0
                self._totals[label_key] = 0
            
            self._sums[label_key] += value
            self._totals[label_key] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
    
    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))
    
    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        metrics = []
        with self._lock:
            for labels, counts in self._counts.items():
                label_dict = dict(labels)
                # Bucket counts
                for bucket, count in counts.items():
                    metrics.append(MetricValue(
                        name=f"{self.name}_bucket",
                        value=count,
                        labels={**label_dict, "le": str(bucket)},
                        metric_type=MetricType.HISTOGRAM
                    ))
                # Sum
                metrics.append(MetricValue(
                    name=f"{self.name}_sum",
                    value=self._sums.get(labels, 0.0),
                    labels=label_dict,
                    metric_type=MetricType.HISTOGRAM
                ))
                # Count
                metrics.append(MetricValue(
                    name=f"{self.name}_count",
                    value=self._totals.get(labels, 0),
                    labels=label_dict,
                    metric_type=MetricType.HISTOGRAM
                ))
        return metrics
    
    @contextmanager
    def time(self, **labels):
        """Context manager to time an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, **labels)


# =============================================================================
# Metrics Registry
# =============================================================================

class MetricsRegistry:
    """Central registry for all metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram]] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def counter(self, name: str, description: str = "", labels: Optional[List[str]] = None) -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description, labels)
            return self._metrics[name]
    
    def gauge(self, name: str, description: str = "", labels: Optional[List[str]] = None) -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description, labels)
            return self._metrics[name]
    
    def histogram(
        self, 
        name: str, 
        description: str = "", 
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None
    ) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, labels, buckets)
            return self._metrics[name]
    
    def collect_all(self) -> List[MetricValue]:
        """Collect all metrics."""
        all_metrics = []
        with self._lock:
            for metric in self._metrics.values():
                all_metrics.extend(metric.collect())
        return all_metrics
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        for metric in self.collect_all():
            label_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
            if label_str:
                lines.append(f'{metric.name}{{{label_str}}} {metric.value}')
            else:
                lines.append(f'{metric.name} {metric.value}')
        return "\n".join(lines)
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self._start_time


# Global registry
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _registry


# =============================================================================
# Pre-defined Metrics
# =============================================================================

# Request metrics
REQUEST_COUNT = _registry.counter(
    "sigmalang_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = _registry.histogram(
    "sigmalang_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"]
)

REQUEST_IN_PROGRESS = _registry.gauge(
    "sigmalang_requests_in_progress",
    "Number of requests currently being processed",
    ["method", "endpoint"]
)

# Encoding metrics
ENCODE_COUNT = _registry.counter(
    "sigmalang_encode_total",
    "Total number of encode operations",
    ["status"]
)

ENCODE_LATENCY = _registry.histogram(
    "sigmalang_encode_duration_seconds",
    "Encode operation latency",
    []
)

# Analogy metrics
ANALOGY_COUNT = _registry.counter(
    "sigmalang_analogy_total",
    "Total number of analogy operations",
    ["type", "status"]
)

ANALOGY_LATENCY = _registry.histogram(
    "sigmalang_analogy_duration_seconds",
    "Analogy operation latency",
    ["type"]
)

# Search metrics
SEARCH_COUNT = _registry.counter(
    "sigmalang_search_total",
    "Total number of search operations",
    ["mode", "status"]
)

SEARCH_LATENCY = _registry.histogram(
    "sigmalang_search_duration_seconds",
    "Search operation latency",
    ["mode"]
)

# Cache metrics
CACHE_HITS = _registry.counter(
    "sigmalang_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"]
)

CACHE_MISSES = _registry.counter(
    "sigmalang_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"]
)

# Error metrics
ERROR_COUNT = _registry.counter(
    "sigmalang_errors_total",
    "Total number of errors",
    ["error_type", "endpoint"]
)


# =============================================================================
# Health Checks
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Manages health checks for all components."""
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, check_fn: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[name] = check_fn
    
    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
    
    def check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        with self._lock:
            check_fn = self._checks.get(name)
        
        if check_fn is None:
            return HealthCheckResult(name=name, healthy=False, message="Check not found")
        
        start = time.perf_counter()
        try:
            result = check_fn()
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=name,
                healthy=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(e)
            )
    
    def check_all(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        with self._lock:
            names = list(self._checks.keys())
        return [self.check(name) for name in names]
    
    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(r.healthy for r in self.check_all())
    
    def is_ready(self) -> bool:
        """Check if service is ready to accept requests."""
        return self.is_healthy()


# Global health checker
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    return _health_checker


# Register default health checks
def _check_self() -> HealthCheckResult:
    """Self health check."""
    return HealthCheckResult(name="self", healthy=True, message="OK")


_health_checker.register("self", _check_self)


# =============================================================================
# Structured Logging
# =============================================================================

class StructuredLogger:
    """Structured JSON logging with context."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._context: Dict[str, Any] = {}
        self._local = threading.local()
    
    def _format_message(self, level: str, message: str, **kwargs) -> str:
        """Format log message as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "level": level,
            "logger": self.name,
            "message": message,
            **self._context,
            **getattr(self._local, 'context', {}),
            **kwargs
        }
        return json.dumps(log_data, default=str)
    
    def set_context(self, **kwargs) -> None:
        """Set persistent context for all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()
    
    @contextmanager
    def with_context(self, **kwargs):
        """Temporary context for a block of code."""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        old_context = self._local.context.copy()
        self._local.context.update(kwargs)
        try:
            yield
        finally:
            self._local.context = old_context
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message("DEBUG", message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(self._format_message("INFO", message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message("WARNING", message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(self._format_message("ERROR", message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message("CRITICAL", message, **kwargs))
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        import traceback
        kwargs['traceback'] = traceback.format_exc()
        self.logger.error(self._format_message("ERROR", message, **kwargs))


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    output: str = "stdout"
) -> None:
    """Configure logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if format_type == "json":
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(output)
    
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)


# =============================================================================
# Tracing
# =============================================================================

@dataclass
class Span:
    """A single span in a trace."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    
    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_error(self, error: Exception) -> None:
        """Mark span as error."""
        self.status = "ERROR"
        self.set_attribute("error.type", type(error).__name__)
        self.set_attribute("error.message", str(error))
    
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Simple tracer for distributed tracing."""
    
    def __init__(self, service_name: str = "sigmalang"):
        self.service_name = service_name
        self._local = threading.local()
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._enabled = True
    
    def _generate_id(self) -> str:
        """Generate a random trace/span ID."""
        import os
        return os.urandom(8).hex()
    
    @property
    def current_span(self) -> Optional[Span]:
        """Get the current span."""
        stack = getattr(self._local, 'span_stack', [])
        return stack[-1] if stack else None
    
    def start_span(self, name: str, parent: Optional[Span] = None) -> Span:
        """Start a new span."""
        if not hasattr(self._local, 'span_stack'):
            self._local.span_stack = []
        
        parent = parent or self.current_span
        trace_id = parent.trace_id if parent else self._generate_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_id=parent.span_id if parent else None,
            name=name,
            start_time=time.time()
        )
        
        self._local.span_stack.append(span)
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()
        
        if hasattr(self._local, 'span_stack') and self._local.span_stack:
            if self._local.span_stack[-1] is span:
                self._local.span_stack.pop()
        
        with self._lock:
            self._spans.append(span)
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for creating spans."""
        s = self.start_span(name)
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, v)
        try:
            yield s
        except Exception as e:
            s.set_error(e)
            raise
        finally:
            self.end_span(s)
    
    def get_spans(self) -> List[Span]:
        """Get all recorded spans."""
        with self._lock:
            return list(self._spans)
    
    def clear_spans(self) -> None:
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()


# Global tracer
_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the global tracer."""
    return _tracer


# =============================================================================
# Decorators
# =============================================================================

def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            histogram = _registry.histogram(metric_name, f"Duration of {func.__name__}")
            with histogram.time(**(labels or {})):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def counted(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            counter = _registry.counter(metric_name, f"Calls to {func.__name__}")
            counter.inc(**(labels or {}))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def traced(name: Optional[str] = None):
    """Decorator to trace function execution."""
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _tracer.span(span_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
