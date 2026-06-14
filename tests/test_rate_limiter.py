"""Tests for sigmalang.core.rate_limiter."""

from unittest.mock import patch

import pytest

from sigmalang.core.config import RateLimitConfig
from sigmalang.core.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limiter(
    enabled: bool = True,
    requests_per_minute: int = 60,
    burst_size: int = 3,
    by_ip: bool = True,
    by_api_key: bool = False,
    whitelist_ips: list | None = None,
) -> RateLimiter:
    cfg = RateLimitConfig(
        enabled=enabled,
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        by_ip=by_ip,
        by_api_key=by_api_key,
        whitelist_ips=whitelist_ips or [],
    )
    return RateLimiter(cfg)


# ---------------------------------------------------------------------------
# is_allowed / _Bucket behaviour
# ---------------------------------------------------------------------------


class TestDisabled:
    def test_disabled_always_allows_any_key(self):
        rl = _limiter(enabled=False, burst_size=1)
        # Exhaust what would have been the bucket
        for _ in range(20):
            assert rl.is_allowed("ip:1.2.3.4") is True

    def test_disabled_check_always_true(self):
        rl = _limiter(enabled=False, burst_size=1)
        for _ in range(10):
            assert rl.check("1.2.3.4", "key123") is True


class TestBurst:
    def test_burst_allows_exactly_burst_size_requests(self):
        burst = 3
        rl = _limiter(burst_size=burst, requests_per_minute=600)
        key = "ip:10.0.0.1"
        results = [rl.is_allowed(key) for _ in range(burst + 2)]
        assert results[:burst] == [True] * burst
        assert results[burst:] == [False] * 2

    def test_burst_independent_per_key(self):
        rl = _limiter(burst_size=2, requests_per_minute=600)
        assert rl.is_allowed("ip:a") is True
        assert rl.is_allowed("ip:a") is True
        assert rl.is_allowed("ip:a") is False
        # Different key has its own bucket
        assert rl.is_allowed("ip:b") is True
        assert rl.is_allowed("ip:b") is True
        assert rl.is_allowed("ip:b") is False


class TestReplenishment:
    def test_tokens_replenish_after_time(self):
        burst = 2
        rate_per_minute = 60  # 1 req/sec
        rl = _limiter(burst_size=burst, requests_per_minute=rate_per_minute)
        key = "ip:10.0.0.2"

        start = 1_000.0

        with patch("sigmalang.core.rate_limiter.time.monotonic") as mock_t:
            mock_t.return_value = start
            # Exhaust burst
            for _ in range(burst):
                assert rl.is_allowed(key) is True
            assert rl.is_allowed(key) is False

            # Advance 1 second → 1 new token
            mock_t.return_value = start + 1.0
            assert rl.is_allowed(key) is True
            assert rl.is_allowed(key) is False

    def test_replenishment_caps_at_burst_size(self):
        burst = 2
        rl = _limiter(burst_size=burst, requests_per_minute=60)
        key = "ip:10.0.0.3"

        start = 2_000.0
        with patch("sigmalang.core.rate_limiter.time.monotonic") as mock_t:
            mock_t.return_value = start
            # Exhaust
            for _ in range(burst):
                rl.is_allowed(key)

            # Advance 1000 seconds → tokens should cap at burst_size (not go above)
            mock_t.return_value = start + 1000.0
            results = [rl.is_allowed(key) for _ in range(burst + 1)]
        assert results[:burst] == [True] * burst
        assert results[burst] is False


# ---------------------------------------------------------------------------
# whitelist
# ---------------------------------------------------------------------------


class TestWhitelist:
    def test_whitelisted_ip_always_passes_check(self):
        rl = _limiter(burst_size=1, whitelist_ips=["192.168.1.1"])
        for _ in range(10):
            assert rl.check("192.168.1.1", None) is True

    def test_non_whitelisted_ip_is_rate_limited(self):
        rl = _limiter(burst_size=1, whitelist_ips=["192.168.1.1"])
        assert rl.check("10.0.0.1", None) is True  # first allowed
        assert rl.check("10.0.0.1", None) is False  # burst exhausted


# ---------------------------------------------------------------------------
# check() flag combinations
# ---------------------------------------------------------------------------


class TestCheckFlags:
    def test_by_ip_false_skips_ip_check(self):
        rl = _limiter(by_ip=False, by_api_key=False, burst_size=1)
        # With both flags off, check should always succeed (no enforcement)
        for _ in range(5):
            assert rl.check("1.2.3.4", "k") is True

    def test_by_ip_true_enforces_ip(self):
        rl = _limiter(by_ip=True, by_api_key=False, burst_size=1)
        assert rl.check("5.5.5.5", "k") is True
        assert rl.check("5.5.5.5", "k") is False

    def test_by_api_key_enforces_api_key(self):
        rl = _limiter(by_ip=False, by_api_key=True, burst_size=1)
        assert rl.check("5.5.5.5", "mykey") is True
        assert rl.check("5.5.5.5", "mykey") is False

    def test_api_key_bucket_uses_apikey_prefix(self):
        rl = _limiter(by_ip=False, by_api_key=True, burst_size=2)
        # Same API key from different IPs → same bucket
        assert rl.check("1.1.1.1", "shared") is True
        assert rl.check("2.2.2.2", "shared") is True
        assert rl.check("3.3.3.3", "shared") is False

    def test_both_must_pass(self):
        rl = _limiter(by_ip=True, by_api_key=True, burst_size=1)
        ip = "6.6.6.6"
        key = "testkey"

        # First call passes both
        assert rl.check(ip, key) is True

        # Both ip and api_key buckets are now empty → fails
        assert rl.check(ip, key) is False

    def test_ip_exhausted_api_key_still_fresh(self):
        """If by_ip exhausts, check returns False even if api_key is fresh."""
        rl = _limiter(by_ip=True, by_api_key=True, burst_size=1)
        # Use up ip bucket
        rl.check("7.7.7.7", "newkey")
        # ip exhausted, api_key has never been used
        assert rl.check("7.7.7.7", "newkey") is False


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_specific_key_clears_only_that_bucket(self):
        rl = _limiter(burst_size=1)
        rl.is_allowed("ip:a")  # exhaust
        rl.is_allowed("ip:b")  # exhaust

        assert rl.is_allowed("ip:a") is False
        assert rl.is_allowed("ip:b") is False

        rl.reset("ip:a")

        assert rl.is_allowed("ip:a") is True   # reset → fresh bucket
        assert rl.is_allowed("ip:b") is False  # still exhausted

    def test_reset_all_clears_all_buckets(self):
        rl = _limiter(burst_size=1)
        rl.is_allowed("ip:x")
        rl.is_allowed("ip:y")
        assert rl.is_allowed("ip:x") is False
        assert rl.is_allowed("ip:y") is False

        rl.reset()

        assert rl.is_allowed("ip:x") is True
        assert rl.is_allowed("ip:y") is True

    def test_reset_unknown_key_is_noop(self):
        rl = _limiter(burst_size=2)
        rl.reset("ip:nonexistent")  # should not raise
        assert rl.is_allowed("ip:fresh") is True

    def test_reset_after_check_restores_capacity(self):
        rl = _limiter(by_ip=True, by_api_key=False, burst_size=2)
        ip = "9.9.9.9"
        key = ip

        rl.check(ip, None)
        rl.check(ip, None)
        assert rl.check(ip, None) is False

        rl.reset(f"ip:{ip}")
        assert rl.check(ip, None) is True


# ---------------------------------------------------------------------------
# Thread-safety smoke test (not exhaustive, just ensures no deadlock/crash)
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_access_does_not_raise(self):
        import threading

        rl = _limiter(burst_size=100, requests_per_minute=6000)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    rl.check("1.1.1.1", "k")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in threads: {errors}"
