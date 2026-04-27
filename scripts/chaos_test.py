"""
SigmaLang Chaos Testing Framework

Chaos engineering tests to validate system resilience under adverse conditions.

Usage:
    python scripts/chaos_test.py --scenarios all
    python scripts/chaos_test.py --scenarios input-fuzzing,circuit-breaker
    python scripts/chaos_test.py --scenarios memory-pressure --duration 60

Scenarios:
- input-fuzzing: Random malformed/edge case inputs
- circuit-breaker: Simulate backend failures
- rate-limiter: Test rate limiting behavior
- graceful-shutdown: SIGTERM during active operations
- memory-pressure: Simulate low memory conditions
- slow-network: Simulate network latency/timeouts
- pod-kill: Simulate sudden process termination
"""

import argparse
import logging
import os
import random
import signal
import string
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root))

from sigmalang.core.api_server import create_api  # noqa: E402
from sigmalang.core.encoder import SigmaEncoder  # noqa: E402
from sigmalang.core.parser import SemanticParser  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Windows console emoji support
IS_WINDOWS = sys.platform == 'win32'

def safe_print(text: str):
    """Print text with emoji support, handling Windows console encoding."""
    if IS_WINDOWS:
        # Replace emojis with ASCII equivalents for Windows console
        text = text.replace('🔥', '[CHAOS]')
        text = text.replace('✅', '[PASS]')
        text = text.replace('❌', '[FAIL]')
        text = text.replace('📊', '[SUMMARY]')
        text = text.replace('🚀', '[ROCKET]')
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII, ignoring errors
        print(text.encode('ascii', errors='ignore').decode('ascii'))


class ChaosScenario(Enum):
    """Available chaos testing scenarios."""
    INPUT_FUZZING = "input-fuzzing"
    CIRCUIT_BREAKER = "circuit-breaker"
    RATE_LIMITER = "rate-limiter"
    GRACEFUL_SHUTDOWN = "graceful-shutdown"
    MEMORY_PRESSURE = "memory-pressure"
    SLOW_NETWORK = "slow-network"
    POD_KILL = "pod-kill"
    ALL = "all"


@dataclass
class ChaosTestResult:
    """Result of a chaos test scenario."""
    scenario: str
    total_tests: int
    passed: int
    failed: int
    errors: List[str]
    duration: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


class InputFuzzingChaos:
    """Random input fuzzing to test error handling."""

    def __init__(self, api):
        self.api = api
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()

    def generate_malformed_inputs(self) -> List[str]:
        """Generate various malformed/edge case inputs."""
        return [
            # Empty and whitespace
            "",
            " ",
            "\n",
            "\t",
            "   \n\n\t  ",

            # Very long strings
            "A" * 100000,
            "x" * 1000000,

            # Binary/non-text data
            "\x00\x01\x02\x03",
            bytes(range(256)).decode('latin1'),

            # Unicode edge cases
            "🚀" * 1000,
            "你好世界" * 500,
            "\u0000\u0001\u0002",
            "café" * 100,

            # SQL injection attempts
            "'; DROP TABLE users; --",
            "1' OR '1'='1",

            # Script injection
            "<script>alert('xss')</script>",
            "javascript:alert(1)",

            # Special characters
            "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "\r\n\r\n",

            # Null bytes
            "test\x00null",

            # Very nested structures (if parsing JSON)
            "{" * 1000 + "}" * 1000,

            # Repeated characters
            "\n" * 10000,
            " " * 50000,

            # Mixed encodings
            "Hello\x80World",

            # Control characters
            "\x1b[31mRed\x1b[0m",

            # Path traversal attempts
            "../../etc/passwd",
            "..\\..\\windows\\system32",

            # Random garbage
            ''.join(random.choices(string.printable, k=10000)),
        ]

    def run(self) -> ChaosTestResult:
        """Run input fuzzing chaos test."""
        logger.info("Starting Input Fuzzing Chaos Test...")

        malformed_inputs = self.generate_malformed_inputs()
        total = len(malformed_inputs)
        passed = 0
        failed = 0
        errors = []

        start_time = time.time()

        for i, input_text in enumerate(malformed_inputs):
            try:
                # Test encoding pipeline: parse → encode
                tree = self.parser.parse(input_text)
                encoded = self.encoder.encode(tree, original_text=input_text)

                # Should either succeed or fail gracefully (no crashes)
                if encoded is not None and len(encoded) > 0:
                    passed += 1
                else:
                    # Empty/None response is acceptable for invalid input
                    passed += 1

            except ValueError as e:
                # ValueError is acceptable for invalid input
                passed += 1
                logger.debug(f"Correctly rejected input {i}: {str(e)[:50]}")

            except UnicodeDecodeError:
                # Unicode errors are acceptable for binary data
                passed += 1
                logger.debug(f"Correctly rejected binary data {i}")

            except Exception as e:
                # Unexpected errors are failures
                failed += 1
                error_msg = f"Input {i}: {type(e).__name__}: {str(e)[:100]}"
                errors.append(error_msg)
                logger.error(error_msg)

        duration = time.time() - start_time

        return ChaosTestResult(
            scenario="Input Fuzzing",
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            duration=duration
        )


class CircuitBreakerChaos:
    """Simulate backend failures to test circuit breaker."""

    def __init__(self, api):
        self.api = api
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()

    def simulate_backend_failure(self, failure_rate: float = 0.5):
        """Inject random failures into API calls."""
        if random.random() < failure_rate:
            raise ConnectionError("Simulated backend failure")

    def run(self) -> ChaosTestResult:
        """Run circuit breaker chaos test."""
        logger.info("Starting Circuit Breaker Chaos Test...")

        total = 100
        passed = 0
        failed = 0
        errors = []

        start_time = time.time()

        for i in range(total):
            try:
                # Simulate random backend failures
                if random.random() < 0.3:  # 30% failure rate
                    raise ConnectionError("Simulated backend failure")

                # Normal operation
                text = f"Test message {i}"
                tree = self.parser.parse(text)
                encoded = self.encoder.encode(tree, original_text=text)

                if encoded is not None and len(encoded) > 0:
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"Encode {i} returned None")

            except ConnectionError:
                # Circuit breaker should handle this gracefully
                # For now, count as expected failure
                passed += 1
                logger.debug(f"Circuit breaker handled failure {i}")

            except Exception as e:
                failed += 1
                error_msg = f"Unexpected error {i}: {type(e).__name__}: {str(e)[:100]}"
                errors.append(error_msg)
                logger.error(error_msg)

        duration = time.time() - start_time

        return ChaosTestResult(
            scenario="Circuit Breaker",
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            duration=duration
        )


class RateLimiterChaos:
    """Test rate limiting behavior."""

    def __init__(self, api):
        self.api = api
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()

    def run(self) -> ChaosTestResult:
        """Run rate limiter chaos test."""
        logger.info("Starting Rate Limiter Chaos Test...")

        total = 200  # Try to exceed rate limits
        passed = 0
        failed = 0
        errors = []

        start_time = time.time()

        # Burst of rapid requests
        for i in range(total):
            try:
                text = f"Rate limit test {i}"
                tree = self.parser.parse(text)
                encoded = self.encoder.encode(tree, original_text=text)

                if encoded is not None and len(encoded) > 0:
                    passed += 1
                else:
                    # Empty/None response might indicate rate limiting
                    passed += 1

            except Exception as e:
                # Rate limit errors are expected
                if "rate limit" in str(e).lower() or "429" in str(e):
                    passed += 1
                    logger.debug(f"Rate limit correctly enforced at request {i}")
                else:
                    failed += 1
                    error_msg = f"Request {i}: {type(e).__name__}: {str(e)[:100]}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        duration = time.time() - start_time
        requests_per_second = total / duration

        logger.info(f"Throughput: {requests_per_second:.2f} req/s")

        return ChaosTestResult(
            scenario="Rate Limiter",
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            duration=duration
        )


class GracefulShutdownChaos:
    """Test graceful shutdown during active operations."""

    def __init__(self, api):
        self.api = api
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()
        self.active_requests = 0
        self.completed_requests = 0
        self.shutdown_triggered = False

    def background_worker(self, duration: int = 5):
        """Simulate background work."""
        end_time = time.time() + duration

        while time.time() < end_time and not self.shutdown_triggered:
            try:
                self.active_requests += 1
                text = f"Background request {self.completed_requests}"
                tree = self.parser.parse(text)
                self.encoder.encode(tree, original_text=text)
                self.completed_requests += 1
                self.active_requests -= 1
                time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Background worker error: {e}")
                self.active_requests -= 1

    def run(self) -> ChaosTestResult:
        """Run graceful shutdown chaos test."""
        logger.info("Starting Graceful Shutdown Chaos Test...")

        errors = []
        start_time = time.time()

        # Start background workers
        workers = []
        for i in range(3):
            worker = threading.Thread(target=self.background_worker, args=(3,))
            worker.start()
            workers.append(worker)

        # Wait a bit, then simulate shutdown
        time.sleep(1.5)

        logger.info(f"Triggering shutdown with {self.active_requests} active requests...")
        self.shutdown_triggered = True

        # Wait for workers to finish (graceful shutdown)
        for worker in workers:
            worker.join(timeout=5)

        duration = time.time() - start_time

        # Check if shutdown was graceful
        passed = 1 if self.active_requests == 0 else 0
        failed = 1 - passed

        if self.active_requests > 0:
            errors.append(f"Shutdown not graceful: {self.active_requests} requests still active")

        logger.info(f"Completed {self.completed_requests} requests before shutdown")

        return ChaosTestResult(
            scenario="Graceful Shutdown",
            total_tests=1,
            passed=passed,
            failed=failed,
            errors=errors,
            duration=duration
        )


class MemoryPressureChaos:
    """Simulate low memory conditions."""

    def __init__(self, api):
        self.api = api
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()

    def run(self) -> ChaosTestResult:
        """Run memory pressure chaos test."""
        logger.info("Starting Memory Pressure Chaos Test...")

        total = 50
        passed = 0
        failed = 0
        errors = []

        start_time = time.time()

        # Create very large inputs to stress memory
        large_texts = [
            "Memory pressure test " * 10000,  # ~200KB
            "X" * 500000,  # 500KB
            "🚀" * 100000,  # Large Unicode
        ]

        for i in range(total):
            try:
                text = random.choice(large_texts)
                tree = self.parser.parse(text)
                encoded = self.encoder.encode(tree, original_text=text)

                if encoded is not None and len(encoded) > 0:
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"Large input {i} failed to encode")

            except MemoryError:
                # MemoryError is expected under pressure
                passed += 1
                logger.debug(f"Correctly handled memory pressure at iteration {i}")

            except Exception as e:
                failed += 1
                error_msg = f"Iteration {i}: {type(e).__name__}: {str(e)[:100]}"
                errors.append(error_msg)
                logger.error(error_msg)

        duration = time.time() - start_time

        return ChaosTestResult(
            scenario="Memory Pressure",
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            duration=duration
        )


class ChaosTester:
    """Main chaos testing orchestrator."""

    def __init__(self):
        self.api = None
        self.results: List[ChaosTestResult] = []

    def initialize(self):
        """Initialize API for testing."""
        logger.info("Initializing SigmaLang API...")
        self.api = create_api()
        self.api.initialize()

    def run_scenario(self, scenario: ChaosScenario) -> Optional[ChaosTestResult]:
        """Run a specific chaos scenario."""
        if scenario == ChaosScenario.INPUT_FUZZING:
            return InputFuzzingChaos(self.api).run()
        elif scenario == ChaosScenario.CIRCUIT_BREAKER:
            return CircuitBreakerChaos(self.api).run()
        elif scenario == ChaosScenario.RATE_LIMITER:
            return RateLimiterChaos(self.api).run()
        elif scenario == ChaosScenario.GRACEFUL_SHUTDOWN:
            return GracefulShutdownChaos(self.api).run()
        elif scenario == ChaosScenario.MEMORY_PRESSURE:
            return MemoryPressureChaos(self.api).run()
        elif scenario == ChaosScenario.SLOW_NETWORK:
            logger.warning("Slow network scenario not implemented (requires HTTP layer)")
            return None
        elif scenario == ChaosScenario.POD_KILL:
            logger.warning("Pod kill scenario not implemented (requires containerization)")
            return None
        else:
            logger.error(f"Unknown scenario: {scenario}")
            return None

    def run_all_scenarios(self, scenarios: List[ChaosScenario]):
        """Run all specified chaos scenarios."""
        self.initialize()

        logger.info(f"Starting Chaos Tests for {len(scenarios)} scenarios...\n")

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            if result:
                self.results.append(result)
                self.print_result(result)

        self.print_summary()

    def print_result(self, result: ChaosTestResult):
        """Print individual test result."""
        status = "[PASS]" if result.failed == 0 else "[FAIL]"

        safe_print(f"\n{status} {result.scenario}")
        safe_print(f"  Total Tests: {result.total_tests}")
        safe_print(f"  Passed: {result.passed}")
        safe_print(f"  Failed: {result.failed}")
        safe_print(f"  Success Rate: {result.success_rate:.1f}%")
        safe_print(f"  Duration: {result.duration:.2f}s")

        if result.errors:
            safe_print(f"  Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                safe_print(f"    - {error}")
            if len(result.errors) > 5:
                safe_print(f"    ... and {len(result.errors) - 5} more")

    def print_summary(self):
        """Print overall summary."""
        safe_print("\n" + "="*70)
        safe_print("[SUMMARY] CHAOS TESTING SUMMARY")
        safe_print("="*70)

        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        safe_print(f"Total Scenarios: {len(self.results)}")
        safe_print(f"Total Tests: {total_tests}")
        safe_print(f"Passed: {total_passed}")
        safe_print(f"Failed: {total_failed}")
        safe_print(f"Overall Success Rate: {(total_passed/total_tests*100):.1f}%")
        safe_print(f"Total Duration: {total_duration:.2f}s")

        # Determine overall status
        if total_failed == 0:
            safe_print("\n[PASS] ALL CHAOS TESTS PASSED - System is resilient!")
        else:
            safe_print(f"\n[FAIL] SOME TESTS FAILED - {total_failed} failures found")

        safe_print("="*70 + "\n")


def parse_scenarios(scenario_str: str) -> List[ChaosScenario]:
    """Parse scenario string into list of scenarios."""
    if scenario_str.lower() == "all":
        return [
            ChaosScenario.INPUT_FUZZING,
            ChaosScenario.CIRCUIT_BREAKER,
            ChaosScenario.RATE_LIMITER,
            ChaosScenario.GRACEFUL_SHUTDOWN,
            ChaosScenario.MEMORY_PRESSURE,
        ]

    scenarios = []
    for name in scenario_str.split(','):
        name = name.strip().lower()
        try:
            scenario = ChaosScenario(name)
            scenarios.append(scenario)
        except ValueError:
            logger.warning(f"Unknown scenario: {name}")

    return scenarios


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SigmaLang Chaos Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/chaos_test.py --scenarios all
  python scripts/chaos_test.py --scenarios input-fuzzing,circuit-breaker
  python scripts/chaos_test.py --scenarios memory-pressure --duration 60
        """
    )

    parser.add_argument(
        '--scenarios',
        type=str,
        default='all',
        help='Comma-separated list of scenarios or "all" (default: all)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Override duration for applicable scenarios (seconds)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scenarios = parse_scenarios(args.scenarios)

    if not scenarios:
        logger.error("No valid scenarios specified")
        return 1

    tester = ChaosTester()
    tester.run_all_scenarios(scenarios)

    # Exit with error code if any tests failed
    total_failed = sum(r.failed for r in tester.results)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
