"""
Comprehensive tests for Production Hardening Module.

Tests for CircuitBreaker, RateLimiter, HealthCheck, GracefulShutdown,
RetryPolicy, Bulkhead, and TimeoutHandler.

Phase 2A.5 Task 5 - Production Hardening Tests
"""

import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from sigmalang.core.production_hardening import (
    # Enums
    CircuitState,
    RateLimitStrategy,
    HealthStatus,
    RetryStrategy,
    ShutdownPhase,
    # Configs
    CircuitBreakerConfig,
    RateLimitConfig,
    HealthCheckResult,
    RetryConfig,
    ShutdownConfig,
    # Exceptions
    CircuitBreakerError,
    RateLimitExceededError,
    BulkheadError,
    TimeoutError,
    # Core classes
    CircuitBreaker,
    RateLimiter,
    HealthChecker,
    CallableHealthCheck,
    HealthCheckRegistry,
    GracefulShutdown,
    RetryPolicy,
    Bulkhead,
    TimeoutHandler,
    # Convenience functions
    create_circuit_breaker,
    create_rate_limiter,
    create_health_registry,
    create_retry_policy,
    create_bulkhead,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestCircuitState:
    """Tests for CircuitState enum."""
    
    def test_all_states_exist(self):
        """Test that all circuit states are defined."""
        assert hasattr(CircuitState, "CLOSED")
        assert hasattr(CircuitState, "OPEN")
        assert hasattr(CircuitState, "HALF_OPEN")
    
    def test_states_are_unique(self):
        """Test that all states have unique values."""
        states = [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
        assert len(set(states)) == len(states)


class TestRateLimitStrategy:
    """Tests for RateLimitStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test that all strategies are defined."""
        assert hasattr(RateLimitStrategy, "TOKEN_BUCKET")
        assert hasattr(RateLimitStrategy, "SLIDING_WINDOW")
        assert hasattr(RateLimitStrategy, "FIXED_WINDOW")
        assert hasattr(RateLimitStrategy, "LEAKY_BUCKET")


class TestHealthStatus:
    """Tests for HealthStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test that all health statuses are defined."""
        assert hasattr(HealthStatus, "HEALTHY")
        assert hasattr(HealthStatus, "DEGRADED")
        assert hasattr(HealthStatus, "UNHEALTHY")
        assert hasattr(HealthStatus, "UNKNOWN")


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test that all retry strategies are defined."""
        assert hasattr(RetryStrategy, "CONSTANT")
        assert hasattr(RetryStrategy, "LINEAR")
        assert hasattr(RetryStrategy, "EXPONENTIAL")
        assert hasattr(RetryStrategy, "FIBONACCI")


class TestShutdownPhase:
    """Tests for ShutdownPhase enum."""
    
    def test_all_phases_exist(self):
        """Test that all shutdown phases are defined."""
        assert hasattr(ShutdownPhase, "RUNNING")
        assert hasattr(ShutdownPhase, "DRAINING")
        assert hasattr(ShutdownPhase, "STOPPING")
        assert hasattr(ShutdownPhase, "STOPPED")


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.exclude_exceptions == ()
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=60.0,
            exclude_exceptions=(ValueError,)
        )
        assert config.failure_threshold == 10
        assert config.timeout_seconds == 60.0
        assert config.exclude_exceptions == (ValueError,)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20
        assert config.strategy == RateLimitStrategy.TOKEN_BUCKET
        assert config.window_size_seconds == 1.0


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""
    
    def test_creation(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "OK"
    
    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY
        )
        assert isinstance(result.timestamp, datetime)


class TestRetryConfig:
    """Tests for RetryConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.jitter is True


class TestShutdownConfig:
    """Tests for ShutdownConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ShutdownConfig()
        assert config.timeout_seconds == 30.0
        assert config.drain_timeout_seconds == 10.0
        assert config.force_after_timeout is True


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    def test_creation(self):
        """Test creating a circuit breaker."""
        breaker = CircuitBreaker("test")
        assert breaker.name == "test"
        assert breaker.state == CircuitState.CLOSED
    
    def test_creation_with_config(self):
        """Test creating with custom config."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        assert breaker.config.failure_threshold == 3
    
    def test_successful_execution(self):
        """Test successful execution through circuit breaker."""
        breaker = CircuitBreaker("test")
        result = breaker.execute(lambda: 42)
        assert result == 42
        assert breaker.state == CircuitState.CLOSED
    
    def test_failure_counts(self):
        """Test that failures are counted."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise ValueError("test error")
        
        for _ in range(2):
            try:
                breaker.execute(failing_func)
            except ValueError:
                pass
        
        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.CLOSED
    
    def test_opens_after_threshold(self):
        """Test that circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise ValueError("test error")
        
        for _ in range(3):
            try:
                breaker.execute(failing_func)
            except ValueError:
                pass
        
        assert breaker.state == CircuitState.OPEN
    
    def test_rejects_when_open(self):
        """Test that requests are rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        with pytest.raises(CircuitBreakerError):
            breaker.execute(lambda: 42)
    
    def test_transitions_to_half_open(self):
        """Test transition from open to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        assert breaker.state == CircuitState.OPEN
        
        time.sleep(0.15)
        
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_closes_from_half_open_on_success(self):
        """Test circuit closes from half-open on successful calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1
        )
        breaker = CircuitBreaker("test", config)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        time.sleep(0.15)
        
        # Two successful calls should close the circuit
        breaker.execute(lambda: 1)
        breaker.execute(lambda: 2)
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_opens_from_half_open_on_failure(self):
        """Test circuit opens from half-open on failure."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        time.sleep(0.15)
        
        # Failure in half-open should reopen
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        assert breaker.state == CircuitState.OPEN
    
    def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker("test")
        
        @breaker
        def my_function(x):
            return x * 2
        
        result = my_function(21)
        assert result == 42
    
    def test_exclude_exceptions(self):
        """Test that excluded exceptions don't trip the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            exclude_exceptions=(ValueError,)
        )
        breaker = CircuitBreaker("test", config)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("ignored")))
        except ValueError:
            pass
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    def test_listener_notification(self):
        """Test state change listener is notified."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)
        
        transitions = []
        
        def listener(old_state, new_state):
            transitions.append((old_state, new_state))
        
        breaker.add_listener(listener)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        assert len(transitions) == 1
        assert transitions[0] == (CircuitState.CLOSED, CircuitState.OPEN)
    
    def test_remove_listener(self):
        """Test removing a listener."""
        breaker = CircuitBreaker("test")
        
        listener = Mock()
        breaker.add_listener(listener)
        breaker.remove_listener(listener)
        
        breaker.reset()  # Would trigger listener if still registered
        assert not listener.called
    
    def test_reset(self):
        """Test resetting the circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)
        
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
        except ValueError:
            pass
        
        assert breaker.state == CircuitState.OPEN
        
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    def test_get_stats(self):
        """Test getting circuit breaker stats."""
        breaker = CircuitBreaker("test")
        stats = breaker.get_stats()
        
        assert stats["name"] == "test"
        assert stats["state"] == "CLOSED"
        assert "failure_count" in stats
        assert "success_count" in stats
    
    def test_thread_safety(self):
        """Test circuit breaker is thread-safe."""
        config = CircuitBreakerConfig(failure_threshold=100)
        breaker = CircuitBreaker("test", config)
        
        def fail_once():
            try:
                breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
            except ValueError:
                pass
        
        threads = [threading.Thread(target=fail_once) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert breaker.failure_count == 50


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    def test_creation(self):
        """Test creating a rate limiter."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=20)
        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 20
    
    def test_creation_with_config(self):
        """Test creating with config."""
        config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        limiter = RateLimiter(config=config)
        assert limiter.requests_per_second == 5.0
        assert limiter.burst_size == 10
    
    def test_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        
        for _ in range(10):
            limiter.acquire()  # Should not raise
    
    def test_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(requests_per_second=1, burst_size=2)
        
        limiter.acquire()
        limiter.acquire()
        
        with pytest.raises(RateLimitExceededError):
            limiter.acquire()
    
    def test_rate_limit_exceeded_error_has_retry_after(self):
        """Test that error includes retry_after time."""
        limiter = RateLimiter(requests_per_second=1, burst_size=1)
        
        limiter.acquire()
        
        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.acquire()
        
        assert exc_info.value.retry_after >= 0
    
    def test_token_bucket_refills(self):
        """Test that token bucket refills over time."""
        limiter = RateLimiter(
            requests_per_second=10.0,
            burst_size=1,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        
        limiter.acquire()
        
        time.sleep(0.15)  # Wait for refill
        
        limiter.acquire()  # Should work now
    
    def test_sliding_window_strategy(self):
        """Test sliding window rate limiting."""
        limiter = RateLimiter(
            requests_per_second=2.0,
            burst_size=2,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        
        limiter.acquire()
        limiter.acquire()
        
        with pytest.raises(RateLimitExceededError):
            limiter.acquire()
    
    def test_fixed_window_strategy(self):
        """Test fixed window rate limiting."""
        limiter = RateLimiter(
            requests_per_second=2.0,
            burst_size=2,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        
        limiter.acquire()
        limiter.acquire()
        
        with pytest.raises(RateLimitExceededError):
            limiter.acquire()
    
    def test_leaky_bucket_strategy(self):
        """Test leaky bucket rate limiting."""
        # Use extremely low leak rate (0.001/sec = 1 per 1000 seconds)
        # This means virtually no leaking during the test
        limiter = RateLimiter(
            requests_per_second=0.001,  # Effectively no leaking
            burst_size=2,
            strategy=RateLimitStrategy.LEAKY_BUCKET
        )
        
        # These calls should fill the bucket to level 2.0
        assert limiter.acquire() == True   # bucket_level = 1.0
        assert limiter.acquire() == True   # bucket_level = 2.0
        
        # Third request should fail since bucket is at capacity (2.0 >= 2)
        with pytest.raises(RateLimitExceededError):
            limiter.acquire()
    
    def test_decorator_usage(self):
        """Test rate limiter as decorator."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        
        @limiter
        def my_function():
            return 42
        
        result = my_function()
        assert result == 42
    
    def test_blocking_acquire(self):
        """Test blocking acquire waits for permission."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=1)
        
        limiter.acquire()
        
        start = time.time()
        result = limiter.acquire(blocking=True, timeout=1.0)
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed > 0.05  # Should have waited some time
    
    def test_blocking_timeout(self):
        """Test blocking acquire respects timeout."""
        limiter = RateLimiter(requests_per_second=0.5, burst_size=1)
        
        limiter.acquire()
        
        start = time.time()
        result = limiter.acquire(blocking=True, timeout=0.1)
        elapsed = time.time() - start
        
        assert result is False
        assert elapsed < 0.5
    
    def test_get_stats_token_bucket(self):
        """Test getting stats for token bucket."""
        limiter = RateLimiter(
            requests_per_second=10.0,
            burst_size=20,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        
        stats = limiter.get_stats()
        
        assert stats["strategy"] == "TOKEN_BUCKET"
        assert "available_tokens" in stats
    
    def test_get_stats_sliding_window(self):
        """Test getting stats for sliding window."""
        limiter = RateLimiter(
            requests_per_second=10.0,
            burst_size=20,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        
        limiter.acquire()
        limiter.acquire()
        
        stats = limiter.get_stats()
        
        assert stats["strategy"] == "SLIDING_WINDOW"
        assert stats["requests_in_window"] == 2


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestCallableHealthCheck:
    """Tests for CallableHealthCheck."""
    
    def test_creation(self):
        """Test creating a callable health check."""
        check = CallableHealthCheck("test", lambda: True)
        assert check.name == "test"
    
    def test_healthy_check(self):
        """Test a healthy check result."""
        check = CallableHealthCheck("test", lambda: True)
        result = check.check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "test"
    
    def test_unhealthy_check(self):
        """Test an unhealthy check result."""
        check = CallableHealthCheck("test", lambda: False)
        result = check.check()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    def test_exception_handling(self):
        """Test that exceptions are caught."""
        def failing_check():
            raise RuntimeError("Check failed")
        
        check = CallableHealthCheck("test", failing_check)
        result = check.check()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
    
    def test_duration_tracking(self):
        """Test that duration is tracked."""
        def slow_check():
            time.sleep(0.1)
            return True
        
        check = CallableHealthCheck("test", slow_check)
        result = check.check()
        
        assert result.duration_ms >= 100


class TestHealthCheckRegistry:
    """Tests for HealthCheckRegistry."""
    
    def test_creation(self):
        """Test creating a registry."""
        registry = HealthCheckRegistry()
        assert registry is not None
    
    def test_register_check(self):
        """Test registering a health check."""
        registry = HealthCheckRegistry()
        check = CallableHealthCheck("db", lambda: True)
        
        registry.register(check)
        
        result = registry.check("db")
        assert result.status == HealthStatus.HEALTHY
    
    def test_unregister_check(self):
        """Test unregistering a health check."""
        registry = HealthCheckRegistry()
        check = CallableHealthCheck("db", lambda: True)
        registry.register(check)
        
        result = registry.unregister("db")
        assert result is True
        
        check_result = registry.check("db")
        assert check_result.status == HealthStatus.UNKNOWN
    
    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent check."""
        registry = HealthCheckRegistry()
        result = registry.unregister("nonexistent")
        assert result is False
    
    def test_check_nonexistent(self):
        """Test checking a non-existent health check."""
        registry = HealthCheckRegistry()
        result = registry.check("nonexistent")
        
        assert result.status == HealthStatus.UNKNOWN
    
    def test_check_all(self):
        """Test checking all health checks."""
        registry = HealthCheckRegistry()
        registry.register(CallableHealthCheck("db", lambda: True))
        registry.register(CallableHealthCheck("cache", lambda: True))
        
        results = registry.check_all()
        
        assert len(results) == 2
        assert results["db"].status == HealthStatus.HEALTHY
        assert results["cache"].status == HealthStatus.HEALTHY
    
    def test_overall_status_healthy(self):
        """Test overall status when all healthy."""
        registry = HealthCheckRegistry()
        registry.register(CallableHealthCheck("db", lambda: True))
        registry.register(CallableHealthCheck("cache", lambda: True))
        
        status = registry.get_overall_status()
        assert status == HealthStatus.HEALTHY
    
    def test_overall_status_unhealthy(self):
        """Test overall status when one unhealthy."""
        registry = HealthCheckRegistry()
        registry.register(CallableHealthCheck("db", lambda: True))
        registry.register(CallableHealthCheck("cache", lambda: False))
        
        status = registry.get_overall_status()
        assert status == HealthStatus.UNHEALTHY
    
    def test_overall_status_empty(self):
        """Test overall status when no checks."""
        registry = HealthCheckRegistry()
        status = registry.get_overall_status()
        assert status == HealthStatus.UNKNOWN
    
    def test_get_last_results(self):
        """Test getting cached results."""
        registry = HealthCheckRegistry()
        registry.register(CallableHealthCheck("db", lambda: True))
        
        registry.check("db")
        
        results = registry.get_last_results()
        assert "db" in results
    
    def test_get_summary(self):
        """Test getting health summary."""
        registry = HealthCheckRegistry()
        registry.register(CallableHealthCheck("db", lambda: True))
        
        summary = registry.get_summary()
        
        assert "overall_status" in summary
        assert "timestamp" in summary
        assert "checks" in summary
        assert "db" in summary["checks"]


# =============================================================================
# GRACEFUL SHUTDOWN TESTS
# =============================================================================


class TestGracefulShutdown:
    """Tests for GracefulShutdown."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        GracefulShutdown.reset()
    
    def teardown_method(self):
        """Clean up after each test."""
        GracefulShutdown.reset()
    
    def test_singleton(self):
        """Test that GracefulShutdown is a singleton."""
        shutdown1 = GracefulShutdown()
        shutdown2 = GracefulShutdown()
        assert shutdown1 is shutdown2
    
    def test_initial_phase(self):
        """Test initial phase is RUNNING."""
        shutdown = GracefulShutdown()
        assert shutdown.phase == ShutdownPhase.RUNNING
    
    def test_is_shutting_down(self):
        """Test is_shutting_down property."""
        shutdown = GracefulShutdown()
        assert shutdown.is_shutting_down is False
    
    def test_register_cleanup(self):
        """Test registering cleanup handlers."""
        shutdown = GracefulShutdown()
        handler = Mock()
        
        shutdown.register_cleanup(handler)
        
        status = shutdown.get_status()
        assert status["handlers_registered"] == 1
    
    def test_unregister_cleanup(self):
        """Test unregistering cleanup handlers."""
        shutdown = GracefulShutdown()
        handler = Mock()
        
        shutdown.register_cleanup(handler)
        result = shutdown.unregister_cleanup(handler)
        
        assert result is True
        assert shutdown.get_status()["handlers_registered"] == 0
    
    def test_unregister_nonexistent(self):
        """Test unregistering non-existent handler."""
        shutdown = GracefulShutdown()
        result = shutdown.unregister_cleanup(Mock())
        assert result is False
    
    def test_track_request(self):
        """Test request tracking."""
        shutdown = GracefulShutdown()
        
        with shutdown.track_request():
            status = shutdown.get_status()
            assert status["active_requests"] == 1
        
        status = shutdown.get_status()
        assert status["active_requests"] == 0
    
    def test_track_request_during_shutdown(self):
        """Test that requests are rejected during shutdown."""
        shutdown = GracefulShutdown()
        shutdown._phase = ShutdownPhase.DRAINING
        
        with pytest.raises(RuntimeError):
            with shutdown.track_request():
                pass
    
    def test_get_status(self):
        """Test getting shutdown status."""
        shutdown = GracefulShutdown()
        status = shutdown.get_status()
        
        assert "phase" in status
        assert "active_requests" in status
        assert "handlers_registered" in status


# =============================================================================
# RETRY POLICY TESTS
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy."""
    
    def test_creation(self):
        """Test creating a retry policy."""
        policy = RetryPolicy(max_retries=3)
        assert policy.max_retries == 3
    
    def test_creation_with_config(self):
        """Test creating with config."""
        config = RetryConfig(max_retries=5, strategy=RetryStrategy.LINEAR)
        policy = RetryPolicy(config=config)
        assert policy.max_retries == 5
        assert policy.strategy == RetryStrategy.LINEAR
    
    def test_successful_execution(self):
        """Test successful execution without retries."""
        policy = RetryPolicy(max_retries=3)
        result = policy.execute(lambda: 42)
        assert result == 42
    
    def test_retries_on_failure(self):
        """Test that failures trigger retries."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay_seconds=0.01,
            jitter=False
        )
        
        call_count = [0]
        
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return 42
        
        result = policy.execute(flaky_func)
        
        assert result == 42
        assert call_count[0] == 3
    
    def test_raises_after_max_retries(self):
        """Test that exception is raised after max retries."""
        policy = RetryPolicy(
            max_retries=2,
            base_delay_seconds=0.01,
            jitter=False
        )
        
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            policy.execute(always_fails)
    
    def test_respects_retry_on(self):
        """Test that only specified exceptions are retried."""
        policy = RetryPolicy(
            max_retries=3,
            retry_on=(ValueError,),
            base_delay_seconds=0.01
        )
        
        def raises_type_error():
            raise TypeError("Not retried")
        
        with pytest.raises(TypeError):
            policy.execute(raises_type_error)
    
    def test_constant_strategy(self):
        """Test constant delay strategy."""
        policy = RetryPolicy(
            max_retries=2,
            base_delay_seconds=0.1,
            strategy=RetryStrategy.CONSTANT,
            jitter=False
        )
        
        # Verify delay calculation
        assert policy._get_delay(0) == 0.1
        assert policy._get_delay(1) == 0.1
        assert policy._get_delay(2) == 0.1
    
    def test_linear_strategy(self):
        """Test linear delay strategy."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay_seconds=1.0,
            strategy=RetryStrategy.LINEAR,
            jitter=False
        )
        
        assert policy._get_delay(0) == 1.0
        assert policy._get_delay(1) == 2.0
        assert policy._get_delay(2) == 3.0
    
    def test_exponential_strategy(self):
        """Test exponential delay strategy."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay_seconds=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False
        )
        
        assert policy._get_delay(0) == 1.0
        assert policy._get_delay(1) == 2.0
        assert policy._get_delay(2) == 4.0
    
    def test_fibonacci_strategy(self):
        """Test fibonacci delay strategy."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay_seconds=1.0,
            strategy=RetryStrategy.FIBONACCI,
            jitter=False
        )
        
        assert policy._get_delay(0) == 1.0  # 1 * 1
        assert policy._get_delay(1) == 1.0  # 1 * 1
        assert policy._get_delay(2) == 2.0  # 1 * 2
        assert policy._get_delay(3) == 3.0  # 1 * 3
        assert policy._get_delay(4) == 5.0  # 1 * 5
    
    def test_max_delay_respected(self):
        """Test that max delay is respected."""
        policy = RetryPolicy(
            max_retries=10,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False
        )
        
        assert policy._get_delay(10) == 5.0  # Would be 1024, but capped
    
    def test_decorator_usage(self):
        """Test retry policy as decorator."""
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.01)
        
        call_count = [0]
        
        @policy
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Retry me")
            return "success"
        
        result = flaky_function()
        assert result == "success"


# =============================================================================
# BULKHEAD TESTS
# =============================================================================


class TestBulkhead:
    """Tests for Bulkhead."""
    
    def test_creation(self):
        """Test creating a bulkhead."""
        bulkhead = Bulkhead("test", max_concurrent=5)
        assert bulkhead.name == "test"
        assert bulkhead.max_concurrent == 5
    
    def test_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        bulkhead = Bulkhead("test", max_concurrent=2)
        
        with bulkhead.acquire():
            assert bulkhead.active_count == 1
        
        assert bulkhead.active_count == 0
    
    def test_rejects_over_limit(self):
        """Test that requests over limit are rejected."""
        bulkhead = Bulkhead("test", max_concurrent=1)
        
        with bulkhead.acquire():
            with pytest.raises(BulkheadError):
                with bulkhead.acquire():
                    pass
    
    def test_waits_with_timeout(self):
        """Test that bulkhead waits with timeout."""
        bulkhead = Bulkhead("test", max_concurrent=1, max_wait_seconds=0.5)
        
        result = []
        
        def acquire_and_wait():
            with bulkhead.acquire():
                time.sleep(0.2)
        
        def try_acquire():
            with bulkhead.acquire():
                result.append("acquired")
        
        t1 = threading.Thread(target=acquire_and_wait)
        t2 = threading.Thread(target=try_acquire)
        
        t1.start()
        time.sleep(0.05)  # Ensure t1 starts first
        t2.start()
        
        t1.join()
        t2.join()
        
        assert "acquired" in result
    
    def test_rejected_count(self):
        """Test rejected count tracking."""
        bulkhead = Bulkhead("test", max_concurrent=1)
        
        with bulkhead.acquire():
            try:
                with bulkhead.acquire():
                    pass
            except BulkheadError:
                pass
        
        assert bulkhead.rejected_count == 1
    
    def test_available_slots(self):
        """Test available slots calculation."""
        bulkhead = Bulkhead("test", max_concurrent=3)
        
        assert bulkhead.available_slots == 3
        
        with bulkhead.acquire():
            assert bulkhead.available_slots == 2
            
            with bulkhead.acquire():
                assert bulkhead.available_slots == 1
    
    def test_decorator_usage(self):
        """Test bulkhead as decorator."""
        bulkhead = Bulkhead("test", max_concurrent=10)
        
        @bulkhead
        def my_function():
            return 42
        
        result = my_function()
        assert result == 42
    
    def test_get_stats(self):
        """Test getting bulkhead stats."""
        bulkhead = Bulkhead("test", max_concurrent=5)
        
        stats = bulkhead.get_stats()
        
        assert stats["name"] == "test"
        assert stats["max_concurrent"] == 5
        assert stats["active_count"] == 0
        assert stats["rejected_count"] == 0
    
    def test_concurrent_access(self):
        """Test concurrent access to bulkhead."""
        # Create bulkhead with wait time so all 10 requests eventually succeed
        bulkhead = Bulkhead("test", max_concurrent=5, max_wait_seconds=5.0)
        
        results = []
        lock = threading.Lock()
        
        def task(i):
            try:
                # acquire() uses the bulkhead's max_wait_seconds
                with bulkhead.acquire():
                    time.sleep(0.01)  # Short sleep
                    with lock:
                        results.append(i)
            except BulkheadError:
                pass
        
        threads = [threading.Thread(target=task, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should eventually succeed because bulkhead has wait time
        assert len(results) == 10


# =============================================================================
# TIMEOUT HANDLER TESTS
# =============================================================================


class TestTimeoutHandler:
    """Tests for TimeoutHandler."""
    
    def test_successful_execution(self):
        """Test successful execution within timeout."""
        handler = TimeoutHandler(5.0)
        result = handler.execute(lambda: 42)
        assert result == 42
    
    def test_decorator_usage(self):
        """Test timeout handler as decorator."""
        handler = TimeoutHandler(5.0)
        
        @handler
        def quick_function():
            return "done"
        
        result = quick_function()
        assert result == "done"


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_circuit_breaker(self):
        """Test create_circuit_breaker function."""
        breaker = create_circuit_breaker("test", failure_threshold=10)
        
        assert breaker.name == "test"
        assert breaker.config.failure_threshold == 10
    
    def test_create_rate_limiter(self):
        """Test create_rate_limiter function."""
        limiter = create_rate_limiter(
            requests_per_second=5.0,
            burst_size=10,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        
        assert limiter.requests_per_second == 5.0
        assert limiter.burst_size == 10
        assert limiter.strategy == RateLimitStrategy.SLIDING_WINDOW
    
    def test_create_health_registry(self):
        """Test create_health_registry function."""
        registry = create_health_registry()
        assert isinstance(registry, HealthCheckRegistry)
    
    def test_create_retry_policy(self):
        """Test create_retry_policy function."""
        policy = create_retry_policy(
            max_retries=5,
            strategy=RetryStrategy.LINEAR
        )
        
        assert policy.max_retries == 5
        assert policy.strategy == RetryStrategy.LINEAR
    
    def test_create_bulkhead(self):
        """Test create_bulkhead function."""
        bulkhead = create_bulkhead("test", max_concurrent=20)
        
        assert bulkhead.name == "test"
        assert bulkhead.max_concurrent == 20


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for production hardening components."""
    
    def test_circuit_breaker_with_retry(self):
        """Test combining circuit breaker with retry policy."""
        breaker = CircuitBreaker("api", CircuitBreakerConfig(failure_threshold=5))
        policy = RetryPolicy(max_retries=2, base_delay_seconds=0.01, jitter=False)
        
        call_count = [0]
        
        def api_call():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        @breaker
        @policy
        def protected_api_call():
            return api_call()
        
        result = protected_api_call()
        assert result == "success"
    
    def test_rate_limiter_with_bulkhead(self):
        """Test combining rate limiter with bulkhead."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        bulkhead = Bulkhead("api", max_concurrent=5)
        
        @limiter
        @bulkhead
        def api_call():
            return "success"
        
        results = [api_call() for _ in range(5)]
        assert all(r == "success" for r in results)
    
    def test_health_check_with_circuit_breaker(self):
        """Test health check monitoring circuit breaker."""
        breaker = CircuitBreaker("api", CircuitBreakerConfig(failure_threshold=3))
        registry = HealthCheckRegistry()
        
        def check_circuit():
            return breaker.state == CircuitState.CLOSED
        
        registry.register(CallableHealthCheck("circuit", check_circuit))
        
        # Initially healthy
        status = registry.get_overall_status()
        assert status == HealthStatus.HEALTHY
        
        # Trip the circuit
        for _ in range(3):
            try:
                breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
            except ValueError:
                pass
        
        # Now unhealthy
        status = registry.get_overall_status()
        assert status == HealthStatus.UNHEALTHY
    
    def test_full_resilience_stack(self):
        """Test complete resilience stack."""
        # Create all components
        breaker = CircuitBreaker(
            "service",
            CircuitBreakerConfig(failure_threshold=5)
        )
        limiter = RateLimiter(requests_per_second=100, burst_size=50)
        bulkhead = Bulkhead("service", max_concurrent=10)
        policy = RetryPolicy(max_retries=2, base_delay_seconds=0.01, jitter=False)
        registry = HealthCheckRegistry()
        
        # Register health checks
        registry.register(CallableHealthCheck(
            "circuit",
            lambda: breaker.state == CircuitState.CLOSED
        ))
        
        # Create protected function
        @breaker
        @limiter
        @bulkhead
        @policy
        def resilient_call():
            return "success"
        
        # Execute multiple calls
        results = [resilient_call() for _ in range(10)]
        
        assert all(r == "success" for r in results)
        assert registry.get_overall_status() == HealthStatus.HEALTHY


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Thread safety tests."""
    
    def test_circuit_breaker_concurrent_state_changes(self):
        """Test circuit breaker handles concurrent state changes."""
        config = CircuitBreakerConfig(failure_threshold=50, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config)
        
        errors = []
        
        def fail():
            try:
                breaker.execute(lambda: (_ for _ in ()).throw(ValueError("error")))
            except (ValueError, CircuitBreakerError):
                pass
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=fail) for _ in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_rate_limiter_concurrent_acquire(self):
        """Test rate limiter handles concurrent acquisitions."""
        limiter = RateLimiter(requests_per_second=100, burst_size=50)
        
        successful = [0]
        lock = threading.Lock()
        
        def acquire():
            try:
                limiter.acquire()
                with lock:
                    successful[0] += 1
            except RateLimitExceededError:
                pass
        
        threads = [threading.Thread(target=acquire) for _ in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # At least burst_size should succeed
        assert successful[0] >= 50
    
    def test_health_registry_concurrent_checks(self):
        """Test health registry handles concurrent checks."""
        registry = HealthCheckRegistry()
        
        for i in range(10):
            registry.register(CallableHealthCheck(f"check_{i}", lambda: True))
        
        results = []
        lock = threading.Lock()
        
        def check_all():
            r = registry.check_all()
            with lock:
                results.append(r)
        
        threads = [threading.Thread(target=check_all) for _ in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 20
        assert all(len(r) == 10 for r in results)
    
    def test_bulkhead_concurrent_access(self):
        """Test bulkhead handles high concurrency."""
        bulkhead = Bulkhead("test", max_concurrent=10, max_wait_seconds=1.0)
        
        active_max = [0]
        lock = threading.Lock()
        
        def task():
            with bulkhead.acquire():
                with lock:
                    active_max[0] = max(active_max[0], bulkhead.active_count)
                time.sleep(0.01)
        
        threads = [threading.Thread(target=task) for _ in range(50)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should never exceed max_concurrent
        assert active_max[0] <= 10
