"""
Health Monitor & Auto-Restart - Phase 6 Task 6.3

Continuous health monitoring for the personal Docker stack.
Checks service health, triggers auto-restart on failures,
and logs all events for post-mortem analysis.

Architecture:
    Monitor Loop (every 30s)
        |
        +-- Check API Health (HTTP /health)
        +-- Check Redis Connectivity (via API /health/redis)
        +-- Check Prometheus (HTTP /-/healthy)
        +-- Check Grafana (HTTP /api/health)
        |
        +-- On Failure:
            +-- Log failure event
            +-- Increment failure counter
            +-- If consecutive failures > threshold:
                +-- Restart failed service
                +-- Wait for recovery
                +-- Log restart event

Usage:
    python scripts/health_monitor.py                 # Run monitor loop
    python scripts/health_monitor.py --once          # Single check
    python scripts/health_monitor.py --status        # Show service status
    python scripts/health_monitor.py --interval 60   # Custom check interval
"""

import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('health_monitor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.personal.yml"
STATE_FILE = PROJECT_ROOT / ".health-monitor-state.json"


# =============================================================================
# Service Health Models
# =============================================================================

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health state of a single service."""

    name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    consecutive_failures: int = 0
    last_check: float = 0.0
    last_healthy: float = 0.0
    total_checks: int = 0
    total_failures: int = 0
    total_restarts: int = 0
    last_restart: float = 0.0
    response_time_ms: float = 0.0
    details: str = ""


@dataclass
class MonitorConfig:
    """Health monitor configuration."""

    check_interval: int = 30  # seconds between checks
    failure_threshold: int = 3  # consecutive failures before restart
    restart_cooldown: int = 300  # seconds between restarts (5 min)
    max_restarts_per_hour: int = 3  # prevent restart loops
    http_timeout: int = 10  # seconds for HTTP checks

    # Service endpoints (relative to Docker network)
    api_url: str = "http://localhost:8000"
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"


# =============================================================================
# Health Checks
# =============================================================================

def http_check(url: str, timeout: int = 10) -> Tuple[bool, float, str]:
    """
    Perform HTTP health check using curl.

    Returns: (success, response_time_ms, details)
    """
    try:
        start = time.time()
        result = subprocess.run(
            [
                "curl", "-sf", "--connect-timeout", str(timeout),
                "--max-time", str(timeout), "-o", "/dev/null",
                "-w", "%{http_code}|%{time_total}", url
            ],
            capture_output=True, text=True, timeout=timeout + 5
        )
        elapsed_ms = (time.time() - start) * 1000

        if result.returncode == 0:
            parts = result.stdout.strip().split("|")
            status_code = parts[0] if parts else "000"
            curl_time = float(parts[1]) * 1000 if len(parts) > 1 else elapsed_ms

            if status_code.startswith("2"):
                return True, curl_time, f"HTTP {status_code}"
            return False, curl_time, f"HTTP {status_code}"

        return False, elapsed_ms, f"curl exit code {result.returncode}"

    except subprocess.TimeoutExpired:
        return False, timeout * 1000, "Connection timed out"
    except FileNotFoundError:
        # curl not available, fallback to Python
        return _python_http_check(url, timeout)
    except Exception as e:
        return False, 0, str(e)


def _python_http_check(url: str, timeout: int = 10) -> Tuple[bool, float, str]:
    """Fallback HTTP check using Python urllib."""
    try:
        import urllib.request
        import urllib.error

        start = time.time()
        req = urllib.request.Request(url, method='GET')
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed_ms = (time.time() - start) * 1000

        if 200 <= resp.status < 300:
            return True, elapsed_ms, f"HTTP {resp.status}"
        return False, elapsed_ms, f"HTTP {resp.status}"

    except urllib.error.HTTPError as e:
        elapsed_ms = (time.time() - start) * 1000
        return False, elapsed_ms, f"HTTP {e.code}"
    except Exception as e:
        return False, 0, str(e)


def check_docker_container(service_name: str) -> Tuple[bool, str]:
    """Check if a Docker container is running and healthy."""
    try:
        result = subprocess.run(
            [
                "docker-compose", "-f", str(COMPOSE_FILE),
                "ps", "--format", "json", service_name
            ],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode != 0:
            # Fallback to docker ps
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={service_name}",
                 "--format", "{{.Status}}"],
                capture_output=True, text=True, timeout=10
            )
            status = result.stdout.strip()
            if "Up" in status:
                if "healthy" in status:
                    return True, f"Container up (healthy): {status}"
                return True, f"Container up: {status}"
            return False, f"Container not running: {status or 'not found'}"

        output = result.stdout.strip()
        if not output:
            return False, "Container not found"

        return "running" in output.lower(), output[:200]

    except Exception as e:
        return False, str(e)


# =============================================================================
# Service Checkers
# =============================================================================

def check_api(config: MonitorConfig) -> ServiceHealth:
    """Check SigmaLang API health."""
    health = ServiceHealth(name="sigmalang-api")

    # Check container
    container_ok, container_detail = check_docker_container("sigmalang-api")
    if not container_ok:
        health.status = ServiceStatus.UNREACHABLE
        health.details = f"Container: {container_detail}"
        return health

    # Check HTTP endpoint
    ok, response_ms, detail = http_check(
        f"{config.api_url}/health", config.http_timeout
    )
    health.response_time_ms = response_ms

    if ok:
        health.status = ServiceStatus.HEALTHY
        health.details = f"{detail} ({response_ms:.0f}ms)"

        # Check if response time is degraded (>2s)
        if response_ms > 2000:
            health.status = ServiceStatus.DEGRADED
            health.details += " [slow response]"
    else:
        health.status = ServiceStatus.UNHEALTHY
        health.details = f"Health check failed: {detail}"

    return health


def check_redis(config: MonitorConfig) -> ServiceHealth:
    """Check Redis health via API metrics endpoint."""
    health = ServiceHealth(name="redis")

    container_ok, container_detail = check_docker_container("redis")
    if not container_ok:
        health.status = ServiceStatus.UNREACHABLE
        health.details = f"Container: {container_detail}"
        return health

    # Check via API's Redis health endpoint
    ok, response_ms, detail = http_check(
        f"{config.api_url}/health/redis", config.http_timeout
    )
    health.response_time_ms = response_ms

    if ok:
        health.status = ServiceStatus.HEALTHY
        health.details = f"Redis OK ({response_ms:.0f}ms)"
    else:
        # Redis container is up but API can't reach it
        health.status = ServiceStatus.DEGRADED
        health.details = f"Container up but API reports: {detail}"

    return health


def check_prometheus(config: MonitorConfig) -> ServiceHealth:
    """Check Prometheus health."""
    health = ServiceHealth(name="prometheus")

    container_ok, container_detail = check_docker_container("prometheus")
    if not container_ok:
        health.status = ServiceStatus.UNREACHABLE
        health.details = f"Container: {container_detail}"
        return health

    ok, response_ms, detail = http_check(
        f"{config.prometheus_url}/-/healthy", config.http_timeout
    )
    health.response_time_ms = response_ms

    if ok:
        health.status = ServiceStatus.HEALTHY
        health.details = f"Prometheus OK ({response_ms:.0f}ms)"
    else:
        health.status = ServiceStatus.UNHEALTHY
        health.details = f"Prometheus unhealthy: {detail}"

    return health


def check_grafana(config: MonitorConfig) -> ServiceHealth:
    """Check Grafana health."""
    health = ServiceHealth(name="grafana")

    container_ok, container_detail = check_docker_container("grafana")
    if not container_ok:
        health.status = ServiceStatus.UNREACHABLE
        health.details = f"Container: {container_detail}"
        return health

    ok, response_ms, detail = http_check(
        f"{config.grafana_url}/api/health", config.http_timeout
    )
    health.response_time_ms = response_ms

    if ok:
        health.status = ServiceStatus.HEALTHY
        health.details = f"Grafana OK ({response_ms:.0f}ms)"
    else:
        health.status = ServiceStatus.UNHEALTHY
        health.details = f"Grafana unhealthy: {detail}"

    return health


# =============================================================================
# Auto-Restart Logic
# =============================================================================

def restart_service(service_name: str) -> bool:
    """Restart a Docker Compose service."""
    logger.warning(f"Restarting service: {service_name}")

    try:
        result = subprocess.run(
            [
                "docker-compose", "-f", str(COMPOSE_FILE),
                "restart", service_name
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode == 0:
            logger.info(f"Service {service_name} restarted successfully")
            return True

        logger.error(f"Restart failed for {service_name}: {result.stderr}")
        return False

    except Exception as e:
        logger.error(f"Restart error for {service_name}: {e}")
        return False


def should_restart(
    health: ServiceHealth,
    state: Dict[str, Any],
    config: MonitorConfig
) -> bool:
    """Determine if a service should be auto-restarted."""
    service_state = state.get(health.name, {})

    consecutive = service_state.get('consecutive_failures', 0)
    last_restart = service_state.get('last_restart', 0)
    restarts_this_hour = service_state.get('restarts_this_hour', 0)

    # Not enough consecutive failures
    if consecutive < config.failure_threshold:
        return False

    # Restart cooldown
    if time.time() - last_restart < config.restart_cooldown:
        logger.info(
            f"  {health.name}: In cooldown period "
            f"({config.restart_cooldown - (time.time() - last_restart):.0f}s remaining)"
        )
        return False

    # Max restarts per hour
    if restarts_this_hour >= config.max_restarts_per_hour:
        logger.warning(
            f"  {health.name}: Max restarts per hour reached ({config.max_restarts_per_hour})"
        )
        return False

    return True


# =============================================================================
# State Management
# =============================================================================

def load_state() -> Dict[str, Any]:
    """Load monitor state from disk."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def save_state(state: Dict[str, Any]) -> None:
    """Save monitor state to disk."""
    STATE_FILE.write_text(
        json.dumps(state, indent=2, default=str),
        encoding='utf-8'
    )


def update_service_state(
    state: Dict[str, Any],
    health: ServiceHealth,
    restarted: bool = False
) -> None:
    """Update state for a service after a health check."""
    now = time.time()

    if health.name not in state:
        state[health.name] = {
            'consecutive_failures': 0,
            'total_checks': 0,
            'total_failures': 0,
            'total_restarts': 0,
            'last_restart': 0,
            'restarts_this_hour': 0,
            'hour_start': now,
        }

    svc = state[health.name]
    svc['total_checks'] = svc.get('total_checks', 0) + 1
    svc['last_check'] = now
    svc['last_status'] = health.status.value
    svc['last_details'] = health.details
    svc['response_time_ms'] = health.response_time_ms

    if health.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED):
        svc['consecutive_failures'] = 0
        svc['last_healthy'] = now
    else:
        svc['consecutive_failures'] = svc.get('consecutive_failures', 0) + 1
        svc['total_failures'] = svc.get('total_failures', 0) + 1

    if restarted:
        svc['total_restarts'] = svc.get('total_restarts', 0) + 1
        svc['last_restart'] = now
        svc['restarts_this_hour'] = svc.get('restarts_this_hour', 0) + 1

    # Reset hourly counter
    hour_start = svc.get('hour_start', now)
    if now - hour_start > 3600:
        svc['restarts_this_hour'] = 0
        svc['hour_start'] = now


# =============================================================================
# Monitor Loop
# =============================================================================

def run_health_check(config: MonitorConfig) -> Dict[str, ServiceHealth]:
    """Run a single round of health checks."""
    results = {}

    checkers = [
        ("sigmalang-api", check_api),
        ("redis", check_redis),
        ("prometheus", check_prometheus),
        ("grafana", check_grafana),
    ]

    for name, checker in checkers:
        try:
            health = checker(config)
            results[name] = health
        except Exception as e:
            health = ServiceHealth(
                name=name,
                status=ServiceStatus.UNKNOWN,
                details=f"Check error: {e}"
            )
            results[name] = health

    return results


def run_monitor_loop(config: MonitorConfig) -> None:
    """Run the continuous monitoring loop."""
    logger.info("=" * 60)
    logger.info("SigmaLang Health Monitor - Phase 6")
    logger.info(f"  Check interval: {config.check_interval}s")
    logger.info(f"  Failure threshold: {config.failure_threshold}")
    logger.info(f"  Restart cooldown: {config.restart_cooldown}s")
    logger.info(f"  Max restarts/hour: {config.max_restarts_per_hour}")
    logger.info("=" * 60)

    state = load_state()
    check_count = 0

    try:
        while True:
            check_count += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")

            results = run_health_check(config)

            # Log results
            all_healthy = True
            for name, health in results.items():
                status_icon = {
                    ServiceStatus.HEALTHY: "[OK]",
                    ServiceStatus.DEGRADED: "[WARN]",
                    ServiceStatus.UNHEALTHY: "[FAIL]",
                    ServiceStatus.UNREACHABLE: "[DOWN]",
                    ServiceStatus.UNKNOWN: "[????]",
                }.get(health.status, "[????]")

                if health.status not in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED):
                    all_healthy = False

                log_fn = logger.info if health.status == ServiceStatus.HEALTHY else logger.warning
                log_fn(f"  {status_icon} {name}: {health.details}")

                # Check if restart needed
                restarted = False
                if health.status in (ServiceStatus.UNHEALTHY, ServiceStatus.UNREACHABLE):
                    if should_restart(health, state, config):
                        logger.warning(f"  -> Auto-restarting {name}...")
                        restarted = restart_service(name)
                        if restarted:
                            logger.info(f"  -> {name} restart initiated")
                        else:
                            logger.error(f"  -> {name} restart FAILED")

                update_service_state(state, health, restarted)

            # Summary line
            if all_healthy:
                logger.info(f"[{now}] Check #{check_count}: All services healthy")
            else:
                unhealthy = [n for n, h in results.items()
                             if h.status not in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)]
                logger.warning(
                    f"[{now}] Check #{check_count}: "
                    f"Unhealthy services: {', '.join(unhealthy)}"
                )

            # Save state
            state['last_check'] = time.time()
            state['check_count'] = check_count
            save_state(state)

            # Wait for next check
            time.sleep(config.check_interval)

    except KeyboardInterrupt:
        logger.info("\nMonitor stopped by user")
        save_state(state)


def show_status() -> None:
    """Show current service status from saved state."""
    state = load_state()

    if not state:
        print("No health monitor state found. Run the monitor first.")
        return

    print("=" * 60)
    print("SigmaLang Health Monitor - Service Status")
    print("=" * 60)

    last_check = state.get('last_check', 0)
    if last_check:
        dt = datetime.fromtimestamp(last_check, tz=timezone.utc)
        print(f"Last check: {dt.isoformat()}")
    print(f"Total checks: {state.get('check_count', 0)}")
    print()

    services = ["sigmalang-api", "redis", "prometheus", "grafana"]
    for svc_name in services:
        svc = state.get(svc_name, {})
        if not svc:
            print(f"  {svc_name}: No data")
            continue

        status = svc.get('last_status', 'unknown')
        failures = svc.get('consecutive_failures', 0)
        total_restarts = svc.get('total_restarts', 0)
        response_ms = svc.get('response_time_ms', 0)
        details = svc.get('last_details', '')

        status_marker = "[OK]" if status == "healthy" else "[!!]"
        print(f"  {status_marker} {svc_name}:")
        print(f"      Status: {status}")
        print(f"      Response: {response_ms:.0f}ms")
        print(f"      Consecutive failures: {failures}")
        print(f"      Total restarts: {total_restarts}")
        if details:
            print(f"      Details: {details}")
        print()

    print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SigmaLang Health Monitor & Auto-Restart"
    )
    parser.add_argument(
        '--once', action='store_true',
        help='Run a single health check and exit'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show current service status'
    )
    parser.add_argument(
        '--interval', type=int, default=30,
        help='Check interval in seconds (default: 30)'
    )
    parser.add_argument(
        '--threshold', type=int, default=3,
        help='Consecutive failures before restart (default: 3)'
    )
    parser.add_argument(
        '--no-restart', action='store_true',
        help='Monitor only, no auto-restart'
    )

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    config = MonitorConfig(
        check_interval=args.interval,
        failure_threshold=args.threshold if not args.no_restart else 999999,
    )

    if args.once:
        results = run_health_check(config)
        all_ok = True
        for name, health in results.items():
            status_icon = "[OK]" if health.status == ServiceStatus.HEALTHY else "[!!]"
            print(f"  {status_icon} {name}: {health.status.value} - {health.details}")
            if health.status not in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED):
                all_ok = False
        sys.exit(0 if all_ok else 1)

    run_monitor_loop(config)


if __name__ == "__main__":
    main()
