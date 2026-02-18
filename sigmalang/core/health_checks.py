"""
Enhanced Health Checks - Sprint 4

Comprehensive health checks for ΣLANG services including:
- Dependency checks (Redis, filesystem)
- Codebook load status
- Buffer pool utilization
- Recent error rate
- System uptime
"""

import time
import os
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .monitoring import HealthCheckResult, get_health_checker


# =============================================================================
# Enhanced Health Check Data Classes
# =============================================================================

@dataclass
class DependencyHealth:
    """Health status of a dependency."""
    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    healthy: bool
    uptime_seconds: float
    dependencies: Dict[str, DependencyHealth]
    codebook_loaded: bool
    buffer_pool_utilization: float
    error_rate: float
    memory_usage_percent: float
    cpu_usage_percent: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Dependency Health Checks
# =============================================================================

class RedisHealthCheck:
    """Health check for Redis connectivity."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, timeout: float = 1.0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.timeout = timeout

    def check(self) -> DependencyHealth:
        """Check Redis connectivity."""
        start = time.time()

        try:
            import redis

            client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
                decode_responses=True
            )

            # Ping Redis
            pong = client.ping()
            latency_ms = (time.time() - start) * 1000

            if pong:
                # Get additional info
                info = client.info()
                return DependencyHealth(
                    name="redis",
                    healthy=True,
                    latency_ms=latency_ms,
                    message="Connected",
                    details={
                        "version": info.get("redis_version", "unknown"),
                        "uptime_seconds": info.get("uptime_in_seconds", 0),
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_human": info.get("used_memory_human", "unknown")
                    }
                )
            else:
                return DependencyHealth(
                    name="redis",
                    healthy=False,
                    latency_ms=latency_ms,
                    message="Ping failed"
                )

        except ImportError:
            return DependencyHealth(
                name="redis",
                healthy=False,
                message="Redis client not installed"
            )
        except Exception as e:
            return DependencyHealth(
                name="redis",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Connection failed: {str(e)}"
            )


class FilesystemHealthCheck:
    """Health check for filesystem access."""

    def __init__(self, paths: Optional[list] = None, required_space_mb: int = 100):
        self.paths = paths or ["/app/tmp", "/app/cache", "/app/logs"]
        self.required_space_mb = required_space_mb

    def check(self) -> DependencyHealth:
        """Check filesystem health."""
        start = time.time()
        issues = []

        try:
            for path_str in self.paths:
                path = Path(path_str)

                # Check if path exists
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        issues.append(f"{path_str}: Cannot create ({e})")
                        continue

                # Check write permission
                test_file = path / ".health_check"
                try:
                    test_file.write_text("ok")
                    test_file.unlink()
                except Exception as e:
                    issues.append(f"{path_str}: Not writable ({e})")

                # Check disk space
                try:
                    stat = os.statvfs(str(path))
                    free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
                    if free_mb < self.required_space_mb:
                        issues.append(f"{path_str}: Low disk space ({free_mb:.0f}MB < {self.required_space_mb}MB)")
                except Exception:
                    pass  # statvfs not available on all platforms

            latency_ms = (time.time() - start) * 1000

            if issues:
                return DependencyHealth(
                    name="filesystem",
                    healthy=False,
                    latency_ms=latency_ms,
                    message="; ".join(issues[:3])  # First 3 issues
                )
            else:
                return DependencyHealth(
                    name="filesystem",
                    healthy=True,
                    latency_ms=latency_ms,
                    message="All paths accessible"
                )

        except Exception as e:
            return DependencyHealth(
                name="filesystem",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message=f"Check failed: {str(e)}"
            )


# =============================================================================
# Component Health Checks
# =============================================================================

class CodebookHealthCheck:
    """Health check for codebook load status."""

    def __init__(self):
        self.loaded = False
        self.load_time = None
        self.codebook_size = 0

    def mark_loaded(self, codebook_size: int) -> None:
        """Mark codebook as loaded."""
        self.loaded = True
        self.load_time = datetime.utcnow()
        self.codebook_size = codebook_size

    def check(self) -> DependencyHealth:
        """Check codebook status."""
        start = time.time()

        if self.loaded:
            age_seconds = (datetime.utcnow() - self.load_time).total_seconds() if self.load_time else 0
            return DependencyHealth(
                name="codebook",
                healthy=True,
                latency_ms=(time.time() - start) * 1000,
                message="Loaded",
                details={
                    "size": self.codebook_size,
                    "age_seconds": age_seconds
                }
            )
        else:
            return DependencyHealth(
                name="codebook",
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                message="Not loaded"
            )


class BufferPoolHealthCheck:
    """Health check for buffer pool utilization."""

    def __init__(self):
        self.utilization = 0.0
        self.pool_size = 0
        self.buffers_in_use = 0

    def update(self, buffers_in_use: int, pool_size: int) -> None:
        """Update buffer pool metrics."""
        self.buffers_in_use = buffers_in_use
        self.pool_size = pool_size
        self.utilization = (buffers_in_use / pool_size * 100) if pool_size > 0 else 0.0

    def check(self) -> DependencyHealth:
        """Check buffer pool status."""
        start = time.time()

        healthy = self.utilization < 95  # Warn if >95% full

        return DependencyHealth(
            name="buffer_pool",
            healthy=healthy,
            latency_ms=(time.time() - start) * 1000,
            message=f"{self.utilization:.1f}% utilized",
            details={
                "utilization_percent": self.utilization,
                "buffers_in_use": self.buffers_in_use,
                "pool_size": self.pool_size
            }
        )


# =============================================================================
# Error Rate Tracking
# =============================================================================

class ErrorRateTracker:
    """Tracks recent error rate for health checks."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.errors = []
        self.requests = []

    def record_request(self, is_error: bool = False) -> None:
        """Record a request."""
        now = time.time()
        self.requests.append(now)
        if is_error:
            self.errors.append(now)

        # Clean old entries
        self._cleanup(now)

    def _cleanup(self, now: float) -> None:
        """Remove entries outside the time window."""
        cutoff = now - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]
        self.errors = [t for t in self.errors if t > cutoff]

    def get_error_rate(self) -> float:
        """Get current error rate (percentage)."""
        self._cleanup(time.time())

        if not self.requests:
            return 0.0

        return (len(self.errors) / len(self.requests)) * 100

    def check(self) -> DependencyHealth:
        """Check error rate health."""
        start = time.time()
        error_rate = self.get_error_rate()

        # Healthy if error rate < 5%
        healthy = error_rate < 5.0

        return DependencyHealth(
            name="error_rate",
            healthy=healthy,
            latency_ms=(time.time() - start) * 1000,
            message=f"{error_rate:.2f}% error rate",
            details={
                "error_rate_percent": error_rate,
                "total_requests": len(self.requests),
                "total_errors": len(self.errors),
                "window_seconds": self.window_seconds
            }
        )


# =============================================================================
# System Uptime Tracker
# =============================================================================

class UptimeTracker:
    """Tracks system uptime."""

    def __init__(self):
        self.start_time = time.time()

    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time

    def get_uptime_str(self) -> str:
        """Get uptime as formatted string."""
        uptime = self.get_uptime_seconds()
        return str(timedelta(seconds=int(uptime)))


# =============================================================================
# Enhanced Health Checker
# =============================================================================

class EnhancedHealthChecker:
    """Enhanced health checker with all Sprint 4 improvements."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        filesystem_paths: Optional[list] = None,
        enable_redis: bool = True
    ):
        self.uptime_tracker = UptimeTracker()
        self.error_tracker = ErrorRateTracker()
        self.codebook_check = CodebookHealthCheck()
        self.buffer_pool_check = BufferPoolHealthCheck()

        # Dependency checks
        self.dependency_checks = {}

        if enable_redis:
            self.dependency_checks['redis'] = RedisHealthCheck(redis_host, redis_port)

        self.dependency_checks['filesystem'] = FilesystemHealthCheck(filesystem_paths)

        # Register with global health checker
        health_checker = get_health_checker()
        health_checker.register("enhanced", self.check_all_as_result)

    def check_all(self) -> SystemHealth:
        """Perform all health checks and return comprehensive status."""
        # Check all dependencies
        dep_results = {}
        for name, checker in self.dependency_checks.items():
            dep_results[name] = checker.check()

        # Check codebook
        dep_results['codebook'] = self.codebook_check.check()

        # Check buffer pool
        dep_results['buffer_pool'] = self.buffer_pool_check.check()

        # Check error rate
        dep_results['error_rate'] = self.error_tracker.check()

        # Get system metrics
        try:
            process = psutil.Process()
            memory_usage = process.memory_percent()
            cpu_usage = process.cpu_percent(interval=0.1)
        except Exception:
            memory_usage = 0.0
            cpu_usage = 0.0

        # Overall health (all components must be healthy)
        overall_healthy = all(dep.healthy for dep in dep_results.values())

        return SystemHealth(
            healthy=overall_healthy,
            uptime_seconds=self.uptime_tracker.get_uptime_seconds(),
            dependencies=dep_results,
            codebook_loaded=self.codebook_check.loaded,
            buffer_pool_utilization=self.buffer_pool_check.utilization,
            error_rate=self.error_tracker.get_error_rate(),
            memory_usage_percent=memory_usage,
            cpu_usage_percent=cpu_usage
        )

    def check_all_as_result(self) -> HealthCheckResult:
        """Return health status as HealthCheckResult for compatibility."""
        health = self.check_all()
        message = f"System {'healthy' if health.healthy else 'unhealthy'}"

        if not health.healthy:
            unhealthy = [name for name, dep in health.dependencies.items() if not dep.healthy]
            message += f" ({', '.join(unhealthy)} failing)"

        return HealthCheckResult(
            name="enhanced",
            healthy=health.healthy,
            message=message
        )


# =============================================================================
# Global Enhanced Health Checker
# =============================================================================

_enhanced_health_checker: Optional[EnhancedHealthChecker] = None


def get_enhanced_health_checker() -> EnhancedHealthChecker:
    """Get or create the global enhanced health checker."""
    global _enhanced_health_checker
    if _enhanced_health_checker is None:
        _enhanced_health_checker = EnhancedHealthChecker()
    return _enhanced_health_checker


def initialize_enhanced_health_checks(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    filesystem_paths: Optional[list] = None,
    enable_redis: bool = True
) -> EnhancedHealthChecker:
    """
    Initialize enhanced health checks.

    Usage:
        from sigmalang.core.health_checks import initialize_enhanced_health_checks

        health_checker = initialize_enhanced_health_checks(
            redis_host="redis.example.com",
            redis_port=6379,
            enable_redis=True
        )
    """
    global _enhanced_health_checker
    _enhanced_health_checker = EnhancedHealthChecker(
        redis_host=redis_host,
        redis_port=redis_port,
        filesystem_paths=filesystem_paths,
        enable_redis=enable_redis
    )
    return _enhanced_health_checker
