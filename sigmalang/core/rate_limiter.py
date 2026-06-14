"""Token bucket rate limiter for SigmaLang API requests.

Thread-safe implementation supporting per-IP and per-API-key limiting
with whitelist support, backed by RateLimitConfig.
"""

import threading
import time
from typing import Dict, Optional

from .config import RateLimitConfig


class _Bucket:
    """Token bucket for a single key."""

    __slots__ = ("_lock", "_tokens", "_last_refill", "_capacity", "_rate")

    def __init__(self, capacity: int, rate: float) -> None:
        self._lock = threading.Lock()
        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()
        self._capacity: float = float(capacity)
        self._rate: float = rate  # tokens per second

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class RateLimiter:
    """Rate limiter backed by per-key token buckets.

    Args:
        config: RateLimitConfig controlling limits and feature flags.

    Example::

        from sigmalang.core.config import RateLimitConfig
        from sigmalang.core.rate_limiter import RateLimiter

        cfg = RateLimitConfig(requests_per_minute=60, burst_size=10)
        limiter = RateLimiter(cfg)

        allowed = limiter.check(ip="1.2.3.4")
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        # tokens per second derived from requests_per_minute
        self._rate: float = config.requests_per_minute / 60.0
        self._capacity: int = config.burst_size
        self._buckets: Dict[str, _Bucket] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_allowed(self, key: str) -> bool:
        """Check and consume one token for *key*.

        Returns:
            ``True`` if the request is within limits, ``False`` otherwise.
            Always returns ``True`` when the limiter is disabled.
        """
        if not self._config.enabled:
            return True
        bucket = self._get_or_create(key)
        return bucket.consume()

    def check(
        self,
        ip: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """Check limits for a request identified by optional IP and API key.

        Whitelist IPs are always allowed.  When both *ip* and *api_key* are
        provided the request is allowed only when **both** checks pass.

        Returns:
            ``True`` if the request is permitted.
        """
        if not self._config.enabled:
            return True

        # Whitelist short-circuit
        if ip and ip in self._config.whitelist_ips:
            return True

        ip_allowed = True
        key_allowed = True

        if self._config.by_ip and ip:
            ip_allowed = self.is_allowed(f"ip:{ip}")

        if self._config.by_api_key and api_key:
            key_allowed = self.is_allowed(f"apikey:{api_key}")

        return ip_allowed and key_allowed

    def reset(self, key: Optional[str] = None) -> None:
        """Remove bucket(s), resetting token counts.

        Args:
            key: Specific bucket key to reset.  Omit to reset all buckets.
        """
        with self._global_lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, key: str) -> _Bucket:
        with self._global_lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(self._capacity, self._rate)
                self._buckets[key] = bucket
        return bucket
