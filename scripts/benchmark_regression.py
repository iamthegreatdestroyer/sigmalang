"""
Continuous Benchmark Regression Testing

Runs benchmarks and compares against baseline to detect performance regressions.

Usage:
    python scripts/benchmark_regression.py --run
    python scripts/benchmark_regression.py --compare benchmark_results/20260217.json
    python scripts/benchmark_regression.py --run --alert-threshold 10

Features:
- Runs all benchmark tests
- Saves results to JSON with timestamp
- Compares against baseline (latest or specified)
- Alerts on regressions > threshold (default 10%)
- CI/CD ready exit codes
"""

import argparse
import json
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    name: str
    mean_time: float
    min_time: float
    max_time: float
    stddev: float
    iterations: int

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean_time * 1000


@dataclass
class RegressionAlert:
    """Benchmark regression alert."""
    name: str
    baseline_time: float
    current_time: float
    regression_percent: float

    @property
    def is_regression(self) -> bool:
        """Check if this is a regression (positive %)."""
        return self.regression_percent > 0


class BenchmarkRunner:
    """Run benchmarks and save results."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmarks(self, pytest_args: Optional[List[str]] = None) -> Optional[Path]:
        """
        Run pytest benchmarks and save to JSON.

        Returns:
            Path to output JSON file, or None if failed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"

        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-k", "benchmark",
            "--benchmark-only",
            f"--benchmark-json={output_file}",
            "--benchmark-warmup=on",
            "--benchmark-disable-gc",
        ]

        if pytest_args:
            cmd.extend(pytest_args)

        print(f"Running benchmarks...")
        print(f"Command: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True
            )

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            if result.returncode != 0:
                print(f"[WARN]  Benchmarks completed with warnings (exit code {result.returncode})")
                # Check if JSON was still created
                if not output_file.exists():
                    print(f"[FAIL] Benchmark results file not created: {output_file}")
                    return None
            else:
                print("[PASS] Benchmarks completed successfully")

            if output_file.exists():
                print(f"[REPORT] Results saved to: {output_file}")
                return output_file
            else:
                print(f"[FAIL] Results file not found: {output_file}")
                return None

        except Exception as e:
            print(f"[FAIL] Error running benchmarks: {e}")
            return None


class BenchmarkComparator:
    """Compare benchmark results against baseline."""

    def __init__(self, alert_threshold: float = 10.0):
        self.alert_threshold = alert_threshold

    def load_results(self, json_file: Path) -> Dict[str, BenchmarkResult]:
        """Load benchmark results from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        results = {}

        # pytest-benchmark JSON format
        if "benchmarks" in data:
            for bench in data["benchmarks"]:
                name = bench.get("name") or bench.get("fullname", "unknown")
                stats = bench.get("stats", {})

                result = BenchmarkResult(
                    name=name,
                    mean_time=stats.get("mean", 0.0),
                    min_time=stats.get("min", 0.0),
                    max_time=stats.get("max", 0.0),
                    stddev=stats.get("stddev", 0.0),
                    iterations=stats.get("iterations", 1)
                )

                results[name] = result

        return results

    def compare(
        self,
        baseline_file: Path,
        current_file: Path
    ) -> Tuple[List[RegressionAlert], Dict[str, BenchmarkResult]]:
        """
        Compare current results against baseline.

        Returns:
            (regressions, all_comparisons)
        """
        baseline = self.load_results(baseline_file)
        current = self.load_results(current_file)

        regressions = []
        comparisons = {}

        for name, current_result in current.items():
            if name not in baseline:
                print(f"[INFO]  New benchmark: {name}")
                continue

            baseline_result = baseline[name]

            # Calculate regression percentage
            if baseline_result.mean_time > 0:
                regression_percent = (
                    (current_result.mean_time - baseline_result.mean_time) /
                    baseline_result.mean_time * 100
                )
            else:
                regression_percent = 0.0

            comparisons[name] = current_result

            if abs(regression_percent) > self.alert_threshold:
                alert = RegressionAlert(
                    name=name,
                    baseline_time=baseline_result.mean_time,
                    current_time=current_result.mean_time,
                    regression_percent=regression_percent
                )
                regressions.append(alert)

        return regressions, comparisons

    def find_latest_baseline(self, output_dir: Path) -> Optional[Path]:
        """Find the most recent baseline JSON file."""
        json_files = sorted(output_dir.glob("benchmark_*.json"))

        if len(json_files) >= 2:
            # Return second-to-last (latest is the one we just created)
            return json_files[-2]
        elif len(json_files) == 1:
            return json_files[0]
        else:
            return None

    def print_report(
        self,
        regressions: List[RegressionAlert],
        comparisons: Dict[str, BenchmarkResult]
    ):
        """Print regression report."""
        print("\n" + "=" * 70)
        print("[REPORT] BENCHMARK REGRESSION REPORT")
        print("=" * 70)

        if not comparisons:
            print("[WARN]  No benchmarks to compare")
            return

        print(f"\nTotal Benchmarks: {len(comparisons)}")
        print(f"Regressions Found: {len(regressions)}")
        print(f"Alert Threshold: ±{self.alert_threshold}%")

        if regressions:
            print(f"\n{'Benchmark':<50} {'Baseline':<12} {'Current':<12} {'Change':<10}")
            print("-" * 94)

            for alert in sorted(regressions, key=lambda x: abs(x.regression_percent), reverse=True):
                status = "[REG]" if alert.is_regression else "[IMP]"
                sign = "+" if alert.regression_percent > 0 else ""

                print(
                    f"{alert.name:<50} "
                    f"{alert.baseline_time*1000:>10.2f}ms "
                    f"{alert.current_time*1000:>10.2f}ms "
                    f"{status} {sign}{alert.regression_percent:>6.1f}%"
                )

        else:
            print("\n[PASS] No significant regressions detected")

        # Summary statistics
        if comparisons:
            print(f"\n{'Benchmark':<50} {'Mean':<12} {'Min':<12} {'Max':<12}")
            print("-" * 86)

            for name, result in sorted(comparisons.items())[:10]:  # Show first 10
                print(
                    f"{name:<50} "
                    f"{result.mean_ms:>10.2f}ms "
                    f"{result.min_time*1000:>10.2f}ms "
                    f"{result.max_time*1000:>10.2f}ms"
                )

            if len(comparisons) > 10:
                print(f"... and {len(comparisons) - 10} more benchmarks")

        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Continuous Benchmark Regression Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--run',
        action='store_true',
        help='Run benchmarks and save results'
    )

    parser.add_argument(
        '--compare',
        type=str,
        metavar='BASELINE',
        help='Compare current results against baseline JSON file'
    )

    parser.add_argument(
        '--alert-threshold',
        type=float,
        default=10.0,
        help='Regression alert threshold in percent (default: 10.0)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for benchmark results (default: benchmark_results)'
    )

    parser.add_argument(
        '--pytest-args',
        type=str,
        default='',
        help='Additional pytest arguments (e.g., "--verbose --timeout=600")'
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output_dir)
    comparator = BenchmarkComparator(alert_threshold=args.alert_threshold)

    # Run benchmarks
    if args.run:
        pytest_args = args.pytest_args.split() if args.pytest_args else None
        current_file = runner.run_benchmarks(pytest_args=pytest_args)

        if not current_file:
            print("[FAIL] Failed to run benchmarks")
            return 1

        # Auto-compare against latest baseline
        baseline_file = comparator.find_latest_baseline(runner.output_dir)

        if baseline_file and baseline_file != current_file:
            print(f"\n[REPORT] Comparing against baseline: {baseline_file}")
            regressions, comparisons = comparator.compare(baseline_file, current_file)
            comparator.print_report(regressions, comparisons)

            # Exit with error if regressions found
            if any(r.is_regression for r in regressions):
                print(f"\n[FAIL] Performance regressions detected!")
                return 1
            else:
                print(f"\n[PASS] No performance regressions")
                return 0
        else:
            print(f"\n[INFO]  No baseline found for comparison")
            print(f"   This will be used as the new baseline")
            return 0

    # Compare against specific baseline
    elif args.compare:
        baseline_file = Path(args.compare)

        if not baseline_file.exists():
            print(f"[FAIL] Baseline file not found: {baseline_file}")
            return 1

        # Find latest current results
        current_file = comparator.find_latest_baseline(runner.output_dir)

        if not current_file:
            print(f"[FAIL] No current results found in {args.output_dir}")
            return 1

        print(f"[REPORT] Comparing:")
        print(f"  Baseline: {baseline_file}")
        print(f"  Current:  {current_file}")

        regressions, comparisons = comparator.compare(baseline_file, current_file)
        comparator.print_report(regressions, comparisons)

        # Exit with error if regressions found
        if any(r.is_regression for r in regressions):
            return 1
        else:
            return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
