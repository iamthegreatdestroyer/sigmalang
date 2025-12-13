"""
Production Hardening Module for SigmaLang.

This module provides production-ready utilities for building resilient systems:
- CircuitBreaker: Prevents cascade failures with automatic recovery
- RateLimiter: Token bucket and sliding window rate limiting
- HealthCheck: System health monitoring and reporting
- GracefulShutdown: Clean shutdown with resource cleanup
- RetryPolicy: Configurable retry strategies with backoff
- Bulkhead: Resource isolation for fault containment

Phase 2A.5 Task 5 - Production Hardening
"""

from __future__ import annotations

import asyncio
import threading
import time
import logging
import signal
import weakref
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# ENUMS
# =============================================================================


class CircuitState(Enum):
    """States for the circuit breaker."""
    
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing recovery


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    
    TOKEN_BUCKET = auto()       # Token bucket algorithm
    SLIDING_WINDOW = auto()     # Sliding window counter
    FIXED_WINDOW = auto()       # Fixed window counter
    LEAKY_BUCKET = auto()       # Leaky bucket algorithm


class HealthStatus(Enum):
    """Health check status levels."""
    
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


class RetryStrategy(Enum):
    """Retry backoff strategies."""
    
    CONSTANT = auto()       # Fixed delay between retries
    LINEAR = auto()         # Linearly increasing delay
    EXPONENTIAL = auto()    # Exponentially increasing delay
    FIBONACCI = auto()      # Fibonacci sequence delays


class ShutdownPhase(Enum):
    """Phases of graceful shutdown."""
    
    RUNNING = auto()
    DRAINING = auto()
    STOPPING = auto()
    STOPPED = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time in open state before half-open
    half_open_max_calls: int = 3        # Max calls in half-open state
    exclude_exceptions: Tuple[type, ...] = ()  # Exceptions that don't trip


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    
    requests_per_second: float = 10.0
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    window_size_seconds: float = 1.0


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    retry_on: Tuple[type, ...] = (Exception,)


@dataclass
class ShutdownConfig:
    """Configuration for graceful shutdown."""
    
    timeout_seconds: float = 30.0
    drain_timeout_seconds: float = 10.0
    force_after_timeout: bool = True
    signals: List[signal.Signals] = field(
        default_factory=lambda: [signal.SIGTERM, signal.SIGINT]
    )


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascade failures by detecting repeated failures and
    temporarily stopping requests to failing services.
    
    Example:
        >>> breaker = CircuitBreaker("api-service")
        >>> @breaker
        ... def call_api():
        ...     return requests.get("http://api.example.com")
        >>> result = call_api()  # Protected by circuit breaker
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        
        self._lock = threading.RLock()
        self._listeners: List[Callable[[CircuitState, CircuitState], None]] = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    @property
    def success_count(self) -> int:
        """Get current success count in half-open state."""
        return self._success_count
    
    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return
        
        self._state = new_state
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.OPEN:
            self._last_failure_time = time.time()
        
        logger.info(
            f"Circuit '{self.name}' transitioned from {old_state.name} to {new_state.name}"
        )
        
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in circuit breaker listener: {e}")
    
    def add_listener(
        self,
        listener: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Add a state change listener."""
        self._listeners.append(listener)
    
    def remove_listener(
        self,
        listener: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """Remove a state change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self, exc: Exception) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if isinstance(exc, self.config.exclude_exceptions):
            return
        
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
    
    def execute(self, func: Callable[[], T]) -> T:
        """Execute a function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerError(
                f"Circuit '{self.name}' is {self._state.name}"
            )
        
        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect a function with circuit breaker."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(lambda: func(*args, **kwargs))
        return wrapper
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "half_open_calls": self._half_open_calls,
            }


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """
    Rate limiter with multiple strategies.
    
    Supports token bucket, sliding window, fixed window, and leaky bucket
    algorithms for flexible rate limiting.
    
    Example:
        >>> limiter = RateLimiter(requests_per_second=10, burst_size=20)
        >>> @limiter
        ... def api_call():
        ...     return "response"
        >>> api_call()  # Rate limited
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
        config: Optional[RateLimitConfig] = None
    ):
        if config:
            self.requests_per_second = config.requests_per_second
            self.burst_size = config.burst_size
            self.strategy = config.strategy
            self._window_size = config.window_size_seconds
        else:
            self.requests_per_second = requests_per_second
            self.burst_size = burst_size
            self.strategy = strategy
            self._window_size = 1.0
        
        self._lock = threading.Lock()
        
        # Token bucket state
        self._tokens = float(self.burst_size)
        self._last_refill = time.time()
        
        # Sliding window state
        self._request_times: Deque[float] = deque()
        
        # Fixed window state
        self._window_start = time.time()
        self._window_count = 0
        
        # Leaky bucket state
        self._bucket_level = 0.0
        self._last_leak = time.time()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.requests_per_second
        )
        self._last_refill = now
    
    def _check_token_bucket(self) -> Tuple[bool, float]:
        """Check token bucket rate limit."""
        self._refill_tokens()
        
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True, 0.0
        else:
            wait_time = (1.0 - self._tokens) / self.requests_per_second
            return False, wait_time
    
    def _check_sliding_window(self) -> Tuple[bool, float]:
        """Check sliding window rate limit."""
        now = time.time()
        window_start = now - self._window_size
        
        # Remove old requests
        while self._request_times and self._request_times[0] < window_start:
            self._request_times.popleft()
        
        max_requests = int(self.requests_per_second * self._window_size)
        
        if len(self._request_times) < max_requests:
            self._request_times.append(now)
            return True, 0.0
        else:
            oldest = self._request_times[0]
            wait_time = oldest + self._window_size - now
            return False, max(0.0, wait_time)
    
    def _check_fixed_window(self) -> Tuple[bool, float]:
        """Check fixed window rate limit."""
        now = time.time()
        
        if now - self._window_start >= self._window_size:
            self._window_start = now
            self._window_count = 0
        
        max_requests = int(self.requests_per_second * self._window_size)
        
        if self._window_count < max_requests:
            self._window_count += 1
            return True, 0.0
        else:
            wait_time = self._window_start + self._window_size - now
            return False, max(0.0, wait_time)
    
    def _check_leaky_bucket(self) -> Tuple[bool, float]:
        """Check leaky bucket rate limit."""
        now = time.time()
        elapsed = now - self._last_leak
        
        # Leak from bucket
        leaked = elapsed * self.requests_per_second
        self._bucket_level = max(0.0, self._bucket_level - leaked)
        self._last_leak = now
        
        # Use tolerance-based comparison to handle floating point precision
        # A request uses 1.0 unit of bucket space
        if self._bucket_level + 1.0 <= self.burst_size + 1e-9:
            self._bucket_level += 1.0
            return True, 0.0
        else:
            wait_time = 1.0 / self.requests_per_second
            return False, wait_time
    
    def acquire(self, blocking: bool = False, timeout: float = 0.0) -> bool:
        """
        Acquire permission to proceed.
        
        Args:
            blocking: If True, wait until permission is granted
            timeout: Maximum time to wait (0 = no limit)
        
        Returns:
            True if permission granted, False otherwise
        
        Raises:
            RateLimitExceededError: If blocking=False and limit exceeded
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    allowed, wait_time = self._check_token_bucket()
                elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    allowed, wait_time = self._check_sliding_window()
                elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                    allowed, wait_time = self._check_fixed_window()
                else:  # LEAKY_BUCKET
                    allowed, wait_time = self._check_leaky_bucket()
                
                if allowed:
                    return True
                
                if not blocking:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded",
                        retry_after=wait_time
                    )
            
            # Check timeout
            if timeout > 0 and time.time() - start_time >= timeout:
                return False
            
            # Wait a bit before retrying
            time.sleep(min(wait_time, 0.1))
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate limit a function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.acquire()
            return func(*args, **kwargs)
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            stats = {
                "strategy": self.strategy.name,
                "requests_per_second": self.requests_per_second,
                "burst_size": self.burst_size,
            }
            
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self._refill_tokens()
                stats["available_tokens"] = self._tokens
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                now = time.time()
                window_start = now - self._window_size
                while self._request_times and self._request_times[0] < window_start:
                    self._request_times.popleft()
                stats["requests_in_window"] = len(self._request_times)
            elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                stats["window_count"] = self._window_count
            else:
                stats["bucket_level"] = self._bucket_level
            
            return stats


# =============================================================================
# HEALTH CHECK
# =============================================================================


class HealthChecker(ABC):
    """Abstract base class for health checks."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the health check."""
        pass
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass


class CallableHealthCheck(HealthChecker):
    """Health check based on a callable function."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        description: str = ""
    ):
        self._name = name
        self._check_func = check_func
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            result = self._check_func()
            duration_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                message=self._description if result else f"{self._name} check failed",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration_ms,
            )


class HealthCheckRegistry:
    """
    Registry for managing health checks.
    
    Aggregates multiple health checks and provides overall system health.
    
    Example:
        >>> registry = HealthCheckRegistry()
        >>> registry.register(CallableHealthCheck("db", check_db_connection))
        >>> registry.register(CallableHealthCheck("cache", check_cache))
        >>> status = registry.check_all()
    """
    
    def __init__(self):
        self._checks: Dict[str, HealthChecker] = {}
        self._lock = threading.Lock()
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    def register(self, checker: HealthChecker) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[checker.name] = checker
    
    def unregister(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                if name in self._last_results:
                    del self._last_results[name]
                return True
            return False
    
    def check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        with self._lock:
            if name not in self._checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check '{name}' not found",
                )
            
            checker = self._checks[name]
        
        result = checker.check()
        
        with self._lock:
            self._last_results[name] = result
        
        return result
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        with self._lock:
            checkers = list(self._checks.values())
        
        results = {}
        for checker in checkers:
            result = checker.check()
            results[checker.name] = result
        
        with self._lock:
            self._last_results.update(results)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.check_all()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN
    
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get the last cached health check results."""
        with self._lock:
            return dict(self._last_results)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of health status."""
        results = self.check_all()
        
        return {
            "overall_status": self.get_overall_status().name,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": result.status.name,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                }
                for name, result in results.items()
            },
        }


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================


class GracefulShutdown:
    """
    Graceful shutdown handler for clean resource cleanup.
    
    Manages shutdown signals and ensures resources are properly cleaned up
    before the process terminates.
    
    Example:
        >>> shutdown = GracefulShutdown()
        >>> shutdown.register_cleanup(cleanup_database)
        >>> shutdown.register_cleanup(close_connections)
        >>> # On SIGTERM or SIGINT, cleanup functions are called in order
    """
    
    _instance: Optional[GracefulShutdown] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> GracefulShutdown:
        """Singleton pattern for shutdown handler."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[ShutdownConfig] = None):
        if self._initialized:
            return
        
        self.config = config or ShutdownConfig()
        self._phase = ShutdownPhase.RUNNING
        self._cleanup_handlers: List[Tuple[int, Callable[[], None]]] = []
        self._active_requests = 0
        self._shutdown_event = threading.Event()
        self._drain_event = threading.Event()
        
        self._lock = threading.RLock()
        self._original_handlers: Dict[signal.Signals, Any] = {}
        
        self._initialized = True
    
    @property
    def phase(self) -> ShutdownPhase:
        """Get current shutdown phase."""
        return self._phase
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._phase != ShutdownPhase.RUNNING
    
    def register_cleanup(
        self,
        handler: Callable[[], None],
        priority: int = 100
    ) -> None:
        """
        Register a cleanup handler.
        
        Args:
            handler: Cleanup function to call during shutdown
            priority: Lower values are called first (default: 100)
        """
        with self._lock:
            self._cleanup_handlers.append((priority, handler))
            self._cleanup_handlers.sort(key=lambda x: x[0])
    
    def unregister_cleanup(self, handler: Callable[[], None]) -> bool:
        """Unregister a cleanup handler."""
        with self._lock:
            for i, (_, h) in enumerate(self._cleanup_handlers):
                if h == handler:
                    del self._cleanup_handlers[i]
                    return True
            return False
    
    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        for sig in self.config.signals:
            try:
                self._original_handlers[sig] = signal.signal(
                    sig,
                    self._signal_handler
                )
            except (ValueError, OSError) as e:
                logger.warning(f"Could not install handler for {sig}: {e}")
    
    def uninstall_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError) as e:
                logger.warning(f"Could not restore handler for {sig}: {e}")
        self._original_handlers.clear()
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.initiate_shutdown()
    
    def initiate_shutdown(self) -> None:
        """Initiate the shutdown sequence."""
        with self._lock:
            if self._phase != ShutdownPhase.RUNNING:
                return
            
            self._phase = ShutdownPhase.DRAINING
            logger.info("Entering drain phase")
        
        # Start drain timer in background
        threading.Thread(target=self._drain_phase, daemon=True).start()
    
    def _drain_phase(self) -> None:
        """Wait for active requests to complete."""
        start_time = time.time()
        
        while True:
            with self._lock:
                if self._active_requests == 0:
                    break
            
            if time.time() - start_time > self.config.drain_timeout_seconds:
                logger.warning(
                    f"Drain timeout exceeded, {self._active_requests} requests still active"
                )
                break
            
            time.sleep(0.1)
        
        self._drain_event.set()
        self._execute_shutdown()
    
    def _execute_shutdown(self) -> None:
        """Execute cleanup handlers."""
        with self._lock:
            self._phase = ShutdownPhase.STOPPING
            handlers = list(self._cleanup_handlers)
        
        logger.info(f"Executing {len(handlers)} cleanup handlers")
        
        for priority, handler in handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}")
        
        with self._lock:
            self._phase = ShutdownPhase.STOPPED
        
        self._shutdown_event.set()
        logger.info("Shutdown complete")
    
    @contextmanager
    def track_request(self) -> Iterator[None]:
        """Context manager to track active requests during shutdown."""
        with self._lock:
            if self._phase != ShutdownPhase.RUNNING:
                raise RuntimeError("Server is shutting down")
            self._active_requests += 1
        
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete."""
        return self._shutdown_event.wait(
            timeout=timeout or self.config.timeout_seconds
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        with self._lock:
            return {
                "phase": self._phase.name,
                "active_requests": self._active_requests,
                "handlers_registered": len(self._cleanup_handlers),
            }
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.uninstall_signal_handlers()
                cls._instance = None


# =============================================================================
# RETRY POLICY
# =============================================================================


class RetryPolicy:
    """
    Retry policy with configurable backoff strategies.
    
    Implements various backoff strategies for retrying failed operations.
    
    Example:
        >>> policy = RetryPolicy(max_retries=3, strategy=RetryStrategy.EXPONENTIAL)
        >>> @policy
        ... def flaky_operation():
        ...     return call_external_service()
        >>> result = flaky_operation()  # Automatically retried on failure
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        retry_on: Tuple[type, ...] = (Exception,),
        config: Optional[RetryConfig] = None
    ):
        if config:
            self.max_retries = config.max_retries
            self.base_delay = config.base_delay_seconds
            self.max_delay = config.max_delay_seconds
            self.strategy = config.strategy
            self.jitter = config.jitter
            self.retry_on = config.retry_on
        else:
            self.max_retries = max_retries
            self.base_delay = base_delay_seconds
            self.max_delay = max_delay_seconds
            self.strategy = strategy
            self.jitter = jitter
            self.retry_on = retry_on
        
        self._fibonacci_cache = [1, 1]
    
    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        if self.strategy == RetryStrategy.CONSTANT:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        else:  # FIBONACCI
            delay = self.base_delay * self._fibonacci(attempt)
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(
                self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            )
        return self._fibonacci_cache[n]
    
    def execute(self, func: Callable[[], T]) -> T:
        """Execute a function with retry logic."""
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except Exception as e:
                if not isinstance(e, self.retry_on):
                    raise
                
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
        
        raise last_exception  # type: ignore
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply retry policy to a function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(lambda: func(*args, **kwargs))
        return wrapper


# =============================================================================
# BULKHEAD
# =============================================================================


class BulkheadError(Exception):
    """Raised when bulkhead rejects a request."""
    pass


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.
    
    Limits concurrent execution to isolate failures and prevent
    resource exhaustion.
    
    Example:
        >>> bulkhead = Bulkhead("db-pool", max_concurrent=10, max_wait=5.0)
        >>> @bulkhead
        ... def query_database():
        ...     return execute_query()
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait_seconds: float = 0.0
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait = max_wait_seconds
        
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._rejected_count = 0
        self._lock = threading.Lock()
    
    @contextmanager
    def acquire(self) -> Iterator[None]:
        """Acquire a slot in the bulkhead."""
        acquired = self._semaphore.acquire(
            blocking=self.max_wait > 0,
            timeout=self.max_wait if self.max_wait > 0 else None
        )
        
        if not acquired:
            with self._lock:
                self._rejected_count += 1
            raise BulkheadError(
                f"Bulkhead '{self.name}' is full ({self.max_concurrent} concurrent)"
            )
        
        with self._lock:
            self._active_count += 1
        
        try:
            yield
        finally:
            with self._lock:
                self._active_count -= 1
            self._semaphore.release()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply bulkhead to a function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with self.acquire():
                return func(*args, **kwargs)
        return wrapper
    
    @property
    def active_count(self) -> int:
        """Number of currently active executions."""
        return self._active_count
    
    @property
    def rejected_count(self) -> int:
        """Number of rejected requests."""
        return self._rejected_count
    
    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "available_slots": self.available_slots,
            "rejected_count": self._rejected_count,
        }


# =============================================================================
# TIMEOUT HANDLER
# =============================================================================


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


class TimeoutHandler:
    """
    Timeout handler for operations.
    
    Example:
        >>> with TimeoutHandler(5.0):
        ...     long_running_operation()
    """
    
    def __init__(self, timeout_seconds: float):
        self.timeout = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def execute(self, func: Callable[[], T]) -> T:
        """Execute a function with timeout."""
        future = self._executor.submit(func)
        try:
            return future.result(timeout=self.timeout)
        except Exception as e:
            if "timed out" in str(e).lower():
                raise TimeoutError(
                    f"Operation timed out after {self.timeout}s"
                ) from e
            raise
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply timeout to a function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(lambda: func(*args, **kwargs))
        return wrapper


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 30.0,
    **kwargs: Any
) -> CircuitBreaker:
    """Create a circuit breaker with common defaults."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
        **kwargs
    )
    return CircuitBreaker(name, config)


def create_rate_limiter(
    requests_per_second: float = 10.0,
    burst_size: int = 20,
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
) -> RateLimiter:
    """Create a rate limiter with common defaults."""
    return RateLimiter(
        requests_per_second=requests_per_second,
        burst_size=burst_size,
        strategy=strategy
    )


def create_health_registry() -> HealthCheckRegistry:
    """Create a new health check registry."""
    return HealthCheckRegistry()


def create_retry_policy(
    max_retries: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    **kwargs: Any
) -> RetryPolicy:
    """Create a retry policy with common defaults."""
    return RetryPolicy(
        max_retries=max_retries,
        strategy=strategy,
        **kwargs
    )


def create_bulkhead(
    name: str,
    max_concurrent: int = 10,
    max_wait_seconds: float = 0.0
) -> Bulkhead:
    """Create a bulkhead with common defaults."""
    return Bulkhead(
        name=name,
        max_concurrent=max_concurrent,
        max_wait_seconds=max_wait_seconds
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "CircuitState",
    "RateLimitStrategy",
    "HealthStatus",
    "RetryStrategy",
    "ShutdownPhase",
    # Configs
    "CircuitBreakerConfig",
    "RateLimitConfig",
    "HealthCheckResult",
    "RetryConfig",
    "ShutdownConfig",
    # Exceptions
    "CircuitBreakerError",
    "RateLimitExceededError",
    "BulkheadError",
    "TimeoutError",
    # Core classes
    "CircuitBreaker",
    "RateLimiter",
    "HealthChecker",
    "CallableHealthCheck",
    "HealthCheckRegistry",
    "GracefulShutdown",
    "RetryPolicy",
    "Bulkhead",
    "TimeoutHandler",
    # Convenience functions
    "create_circuit_breaker",
    "create_rate_limiter",
    "create_health_registry",
    "create_retry_policy",
    "create_bulkhead",
]
