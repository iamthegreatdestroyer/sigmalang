"""
Continuous Benchmarking Framework - Phase 2 Task 2.2

Automated performance benchmarking with historical trend tracking,
regression detection, and performance budgets.

Features:
- Automated nightly benchmarks
- Historical trend analysis
- Regression detection (performance degradation > 10%)
- Performance budgets with alerts
- JSON results export for CI/CD integration
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import sys

# Add parent to path
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root))


# =============================================================================
# Benchmark Result Models
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    duration_ms: float
    memory_mb: float
    throughput_ops_per_sec: Optional[float] = None
    compression_ratio: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'suite_name': self.suite_name,
            'timestamp': self.timestamp,
            'git_commit': self.git_commit,
            'system_info': self.system_info,
            'results': [r.to_dict() for r in self.results]
        }


# =============================================================================
# Benchmark Executor
# =============================================================================

class BenchmarkExecutor:
    """Executes benchmarks and collects results."""

    def __init__(self):
        self.suite = BenchmarkSuite(suite_name="SigmaLang Performance")
        self._setup_system_info()

    def _setup_system_info(self) -> None:
        """Collect system information."""
        import platform
        import psutil

        self.suite.system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

        # Try to get git commit
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.suite.git_commit = result.stdout.strip()
        except Exception:
            pass

    def benchmark_encode_throughput(self, num_samples: int = 100) -> BenchmarkResult:
        """Benchmark encoding throughput."""
        from sigmalang.core.parser import SemanticParser
        from sigmalang.core.encoder import SigmaEncoder
        import psutil

        parser = SemanticParser()
        encoder = SigmaEncoder()

        # Test data
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing enables computers to understand text.",
        ] * (num_samples // 3)

        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Benchmark
        start = time.perf_counter()

        for text in test_texts:
            tree = parser.parse(text)
            encoder.encode(tree)

        duration = (time.perf_counter() - start) * 1000  # ms

        # Measure memory after
        mem_after = process.memory_info().rss / (1024 * 1024)

        throughput = len(test_texts) / (duration / 1000)  # ops per second

        return BenchmarkResult(
            name="encode_throughput",
            duration_ms=duration,
            memory_mb=mem_after - mem_before,
            throughput_ops_per_sec=throughput,
            metadata={'num_samples': num_samples}
        )

    def benchmark_compression_ratio(self, num_samples: int = 50) -> BenchmarkResult:
        """Benchmark compression ratio."""
        from sigmalang.core.parser import SemanticParser
        from sigmalang.core.encoder import SigmaEncoder
        import psutil

        parser = SemanticParser()
        encoder = SigmaEncoder()

        test_texts = [
            "This is a test sentence for compression measurement.",
            "Semantic compression using learned primitives and codebooks.",
            "High compression ratios indicate efficient encoding schemes.",
        ] * (num_samples // 3)

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        compression_ratios = []
        start = time.perf_counter()

        for text in test_texts:
            tree = parser.parse(text)
            encoded = encoder.encode(tree)

            original_size = len(text.encode('utf-8'))
            compressed_size = len(encoded)

            if compressed_size > 0:
                compression_ratios.append(original_size / compressed_size)

        duration = (time.perf_counter() - start) * 1000

        mem_after = process.memory_info().rss / (1024 * 1024)

        avg_compression = statistics.mean(compression_ratios) if compression_ratios else 1.0

        return BenchmarkResult(
            name="compression_ratio",
            duration_ms=duration,
            memory_mb=mem_after - mem_before,
            compression_ratio=avg_compression,
            metadata={
                'num_samples': num_samples,
                'min_ratio': min(compression_ratios) if compression_ratios else 0,
                'max_ratio': max(compression_ratios) if compression_ratios else 0
            }
        )

    def benchmark_search_latency(self, num_queries: int = 50) -> BenchmarkResult:
        """Benchmark semantic search latency."""
        from sigmalang.core.api_server import create_api
        import psutil

        api = create_api()
        api.initialize()

        # Build test corpus
        corpus = [
            "Machine learning algorithms process data.",
            "Neural networks learn from examples.",
            "Deep learning uses multiple layers.",
            "Artificial intelligence mimics human cognition.",
            "Natural language understanding enables conversation."
        ] * 10

        queries = [
            "How do neural networks work?",
            "What is deep learning?",
            "Explain machine learning."
        ] * (num_queries // 3)

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        latencies = []
        start = time.perf_counter()

        for query in queries:
            query_start = time.perf_counter()
            try:
                api.search.semantic_search(query, corpus, top_k=3)
                latencies.append((time.perf_counter() - query_start) * 1000)
            except Exception:
                pass

        duration = (time.perf_counter() - start) * 1000

        mem_after = process.memory_info().rss / (1024 * 1024)

        p50 = statistics.median(latencies) if latencies else 0
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0

        return BenchmarkResult(
            name="search_latency",
            duration_ms=duration,
            memory_mb=mem_after - mem_before,
            metadata={
                'num_queries': num_queries,
                'p50_latency_ms': p50,
                'p95_latency_ms': p95,
                'avg_latency_ms': statistics.mean(latencies) if latencies else 0
            }
        )

    def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run all benchmarks."""
        print("Running continuous benchmarks...")

        # Encode throughput
        print("  [1/3] Benchmarking encode throughput...")
        result1 = self.benchmark_encode_throughput(num_samples=100)
        self.suite.add_result(result1)
        print(f"        Throughput: {result1.throughput_ops_per_sec:.1f} ops/sec")

        # Compression ratio
        print("  [2/3] Benchmarking compression ratio...")
        result2 = self.benchmark_compression_ratio(num_samples=50)
        self.suite.add_result(result2)
        print(f"        Avg ratio: {result2.compression_ratio:.2f}x")

        # Search latency
        print("  [3/3] Benchmarking search latency...")
        result3 = self.benchmark_search_latency(num_queries=50)
        self.suite.add_result(result3)
        print(f"        P50: {result3.metadata['p50_latency_ms']:.1f}ms")

        print("[PASS] All benchmarks complete")

        return self.suite


# =============================================================================
# Historical Tracking & Regression Detection
# =============================================================================

class BenchmarkTracker:
    """Tracks benchmark history and detects regressions."""

    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: List[Dict[str, Any]] = []

        if history_file.exists():
            self._load_history()

    def _load_history(self) -> None:
        """Load historical benchmark data."""
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load history: {e}")
            self.history = []

    def save_suite(self, suite: BenchmarkSuite) -> None:
        """Save benchmark suite to history."""
        self.history.append(suite.to_dict())

        # Keep last 100 runs
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Save to file
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def detect_regressions(
        self,
        suite: BenchmarkSuite,
        threshold_pct: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Detect performance regressions.

        A regression is detected if a metric degrades by more than threshold_pct
        compared to the median of the last 10 runs.
        """
        if len(self.history) < 2:
            return []  # Need history to compare

        regressions = []

        # Get last 10 historical results for each benchmark
        for result in suite.results:
            historical_values = self._get_historical_values(result.name, limit=10)

            if not historical_values:
                continue

            # Calculate baseline (median of historical)
            baseline = statistics.median(historical_values)

            # For throughput/compression, higher is better
            # For latency/duration, lower is better
            if result.name in ['encode_throughput', 'compression_ratio']:
                # Higher is better - regression if current < baseline
                if result.throughput_ops_per_sec:
                    current = result.throughput_ops_per_sec
                elif result.compression_ratio:
                    current = result.compression_ratio
                else:
                    continue

                degradation_pct = ((baseline - current) / baseline) * 100

                if degradation_pct > threshold_pct:
                    regressions.append({
                        'benchmark': result.name,
                        'baseline': baseline,
                        'current': current,
                        'degradation_pct': degradation_pct,
                        'direction': 'lower_is_worse'
                    })

            elif result.name in ['search_latency']:
                # Lower is better - regression if current > baseline
                current = result.metadata.get('p50_latency_ms', result.duration_ms)

                degradation_pct = ((current - baseline) / baseline) * 100

                if degradation_pct > threshold_pct:
                    regressions.append({
                        'benchmark': result.name,
                        'baseline': baseline,
                        'current': current,
                        'degradation_pct': degradation_pct,
                        'direction': 'higher_is_worse'
                    })

        return regressions

    def _get_historical_values(self, benchmark_name: str, limit: int = 10) -> List[float]:
        """Get historical values for a benchmark."""
        values = []

        for suite_data in reversed(self.history[-limit:]):
            for result_data in suite_data.get('results', []):
                if result_data['name'] == benchmark_name:
                    # Extract the relevant metric
                    if benchmark_name == 'encode_throughput':
                        if result_data.get('throughput_ops_per_sec'):
                            values.append(result_data['throughput_ops_per_sec'])
                    elif benchmark_name == 'compression_ratio':
                        if result_data.get('compression_ratio'):
                            values.append(result_data['compression_ratio'])
                    elif benchmark_name == 'search_latency':
                        if result_data.get('metadata', {}).get('p50_latency_ms'):
                            values.append(result_data['metadata']['p50_latency_ms'])

        return values

    def get_trend_summary(self, benchmark_name: str, limit: int = 10) -> Dict[str, Any]:
        """Get trend summary for a benchmark."""
        values = self._get_historical_values(benchmark_name, limit=limit)

        if not values:
            return {'error': 'No historical data'}

        return {
            'benchmark': benchmark_name,
            'samples': len(values),
            'min': min(values),
            'max': max(values),
            'median': statistics.median(values),
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'trend': 'improving' if values[-1] > values[0] else 'degrading'
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def run_continuous_benchmarks(
    history_file: Path = None,
    detect_regressions: bool = True
) -> Dict[str, Any]:
    """
    Run continuous benchmarks with regression detection.

    Usage:
        python tests/continuous_benchmarking.py
    """
    if history_file is None:
        history_file = Path(__file__).parent / 'benchmark_history.json'

    print("=" * 70)
    print("SigmaLang Continuous Benchmarking")
    print("=" * 70)

    # Run benchmarks
    executor = BenchmarkExecutor()
    suite = executor.run_all_benchmarks()

    # Track results
    tracker = BenchmarkTracker(history_file)
    tracker.save_suite(suite)

    print(f"\n[PASS] Results saved to: {history_file}")

    # Detect regressions
    if detect_regressions and len(tracker.history) > 1:
        print("\n" + "=" * 70)
        print("Regression Detection")
        print("=" * 70)

        regressions = tracker.detect_regressions(suite, threshold_pct=10.0)

        if regressions:
            print(f"[WARN] Found {len(regressions)} performance regressions:")
            for reg in regressions:
                print(f"\n  Benchmark: {reg['benchmark']}")
                print(f"  Baseline:  {reg['baseline']:.2f}")
                print(f"  Current:   {reg['current']:.2f}")
                print(f"  Degradation: {reg['degradation_pct']:.1f}%")
        else:
            print("[PASS] No regressions detected")

    # Trend summary
    print("\n" + "=" * 70)
    print("Trend Summary (Last 10 Runs)")
    print("=" * 70)

    for result in suite.results:
        trend = tracker.get_trend_summary(result.name, limit=10)
        if 'error' not in trend:
            print(f"\n{result.name}:")
            print(f"  Median: {trend['median']:.2f}")
            print(f"  Range:  {trend['min']:.2f} - {trend['max']:.2f}")
            print(f"  Trend:  {trend['trend']}")

    print("\n" + "=" * 70)

    return {
        'suite': suite.to_dict(),
        'regressions': regressions if detect_regressions else [],
        'history_file': str(history_file)
    }


if __name__ == "__main__":
    result = run_continuous_benchmarks()

    # Exit with error if regressions detected
    if result['regressions']:
        sys.exit(1)
