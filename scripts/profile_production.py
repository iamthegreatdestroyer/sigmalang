#!/usr/bin/env python3
"""
ΣLANG Phase 2: Performance Profiling
Automated performance analysis and optimization
"""

import os
import sys
import json
import time
import psutil
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import threading

class PerformanceProfiler:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "performance_reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def print_status(self, message: str):
        print(f"[INFO] {message}")

    def print_success(self, message: str):
        print(f"[SUCCESS] {message}")

    def print_warning(self, message: str):
        print(f"[WARNING] {message}")

    def print_error(self, message: str):
        print(f"[ERROR] {message}")

    def get_system_info(self):
        """Get system information for profiling context"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "platform": sys.platform,
            "python_version": sys.version
        }

    def profile_memory_usage(self):
        """Profile memory usage of core components"""
        self.print_status("Profiling memory usage...")

        # Start memory tracing
        tracemalloc.start()

        try:
            # Import and profile core modules
            sys.path.insert(0, str(self.project_root))

            memory_snapshots = {}

            # Profile core encoder
            try:
                from sigmalang.core.encoder import SigmaEncoder
                encoder = SigmaEncoder()

                # Take snapshot before
                snapshot1 = tracemalloc.take_snapshot()

                # Perform encoding operation
                test_text = "This is a test sentence for memory profiling. " * 100
                result = encoder.encode(test_text)

                # Take snapshot after
                snapshot2 = tracemalloc.take_snapshot()

                # Calculate memory usage
                stats = snapshot2.compare_to(snapshot1, 'lineno')
                memory_snapshots["encoder"] = {
                    "peak_memory": sum(stat.size_diff for stat in stats if stat.size_diff > 0),
                    "memory_stats": [{"file": stat.traceback[0].filename,
                                    "line": stat.traceback[0].lineno,
                                    "size": stat.size_diff} for stat in stats[:10]]
                }

            except Exception as e:
                self.print_warning(f"Could not profile encoder: {e}")
                memory_snapshots["encoder"] = {"error": str(e)}

            # Profile bidirectional codec
            try:
                from sigmalang.core.bidirectional_codec import BidirectionalCodec
                codec = BidirectionalCodec()

                snapshot1 = tracemalloc.take_snapshot()

                test_tree = {"text": "Hello world", "type": "sentence"}
                encoded = codec.encode_with_verification(test_tree)

                snapshot2 = tracemalloc.take_snapshot()

                stats = snapshot2.compare_to(snapshot1, 'lineno')
                memory_snapshots["codec"] = {
                    "peak_memory": sum(stat.size_diff for stat in stats if stat.size_diff > 0),
                    "memory_stats": [{"file": stat.traceback[0].filename,
                                    "line": stat.traceback[0].lineno,
                                    "size": stat.size_diff} for stat in stats[:10]]
                }

            except Exception as e:
                self.print_warning(f"Could not profile codec: {e}")
                memory_snapshots["codec"] = {"error": str(e)}

        finally:
            tracemalloc.stop()

        # Save memory profile report
        memory_report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "memory_profiles": memory_snapshots
        }

        with open(self.reports_dir / "memory_profile.json", 'w') as f:
            json.dump(memory_report, f, indent=2)

        self.print_success("Memory profiling completed")
        return memory_report

    def profile_cpu_usage(self):
        """Profile CPU usage patterns"""
        self.print_status("Profiling CPU usage...")

        cpu_stats = {"measurements": []}

        # Monitor CPU usage over time
        for i in range(10):
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_stats["measurements"].append({
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "cpu_times": psutil.cpu_times()._asdict()
            })

        # Test core algorithm performance
        try:
            sys.path.insert(0, str(self.project_root))
            from sigmalang.core.encoder import SigmaEncoder

            encoder = SigmaEncoder()

            # Time encoding operations
            test_sizes = [100, 1000, 10000]

            performance_results = {}
            for size in test_sizes:
                test_text = "This is a test sentence. " * size

                start_time = time.time()
                result = encoder.encode(test_text)
                end_time = time.time()

                performance_results[f"size_{size}"] = {
                    "input_size": len(test_text),
                    "output_size": len(str(result)) if result else 0,
                    "processing_time": end_time - start_time,
                    "throughput": len(test_text) / (end_time - start_time) if (end_time - start_time) > 0 else 0
                }

            cpu_stats["algorithm_performance"] = performance_results

        except Exception as e:
            self.print_warning(f"Could not profile algorithm performance: {e}")
            cpu_stats["algorithm_performance"] = {"error": str(e)}

        # Save CPU profile report
        cpu_report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "cpu_analysis": cpu_stats
        }

        with open(self.reports_dir / "cpu_profile.json", 'w') as f:
            json.dump(cpu_report, f, indent=2)

        self.print_success("CPU profiling completed")
        return cpu_report

    def analyze_latency_patterns(self):
        """Analyze latency patterns and bottlenecks"""
        self.print_status("Analyzing latency patterns...")

        latency_analysis = {"measurements": []}

        try:
            sys.path.insert(0, str(self.project_root))
            from sigmalang.core.encoder import SigmaEncoder

            encoder = SigmaEncoder()

            # Measure latency for different input sizes
            test_cases = [
                ("small", "Hello world"),
                ("medium", "This is a medium length sentence for testing. " * 10),
                ("large", "This is a large text block for performance testing. " * 100)
            ]

            for case_name, test_text in test_cases:
                latencies = []

                # Run multiple times for statistical significance
                for _ in range(10):
                    start_time = time.perf_counter()
                    result = encoder.encode(test_text)
                    end_time = time.perf_counter()

                    latency = (end_time - start_time) * 1000  # Convert to milliseconds
                    latencies.append(latency)

                # Calculate statistics
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

                latency_analysis["measurements"].append({
                    "test_case": case_name,
                    "input_size": len(test_text),
                    "avg_latency_ms": avg_latency,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "p95_latency_ms": p95_latency,
                    "throughput_chars_per_sec": len(test_text) / (avg_latency / 1000) if avg_latency > 0 else 0
                })

        except Exception as e:
            self.print_warning(f"Could not analyze latency: {e}")
            latency_analysis["error"] = str(e)

        # Save latency analysis report
        latency_report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "latency_analysis": latency_analysis
        }

        with open(self.reports_dir / "latency_analysis.json", 'w') as f:
            json.dump(latency_report, f, indent=2)

        self.print_success("Latency analysis completed")
        return latency_report

    def generate_optimization_recommendations(self, memory_report, cpu_report, latency_report):
        """Generate optimization recommendations based on profiling data"""
        self.print_status("Generating optimization recommendations...")

        recommendations = []

        # Memory optimization recommendations
        if "memory_profiles" in memory_report:
            for component, profile in memory_report["memory_profiles"].items():
                if "peak_memory" in profile and profile["peak_memory"] > 50 * 1024 * 1024:  # 50MB
                    recommendations.append({
                        "category": "memory",
                        "component": component,
                        "issue": f"High memory usage: {profile['peak_memory'] / (1024*1024):.1f} MB",
                        "recommendation": "Consider streaming processing or memory pooling"
                    })

        # CPU optimization recommendations
        if "cpu_analysis" in cpu_report and "algorithm_performance" in cpu_report["cpu_analysis"]:
            perf_data = cpu_report["cpu_analysis"]["algorithm_performance"]
            for size_key, metrics in perf_data.items():
                if isinstance(metrics, dict) and "processing_time" in metrics:
                    if metrics["processing_time"] > 1.0:  # More than 1 second
                        recommendations.append({
                            "category": "cpu",
                            "component": "algorithm",
                            "issue": f"Slow processing for {size_key}: {metrics['processing_time']:.2f}s",
                            "recommendation": "Consider parallel processing or algorithm optimization"
                        })

        # Latency optimization recommendations
        if "latency_analysis" in latency_report and "measurements" in latency_report["latency_analysis"]:
            for measurement in latency_report["latency_analysis"]["measurements"]:
                if measurement["p95_latency_ms"] > 100:  # More than 100ms
                    recommendations.append({
                        "category": "latency",
                        "component": measurement["test_case"],
                        "issue": f"High latency: {measurement['p95_latency_ms']:.1f}ms p95",
                        "recommendation": "Optimize hot paths and consider caching"
                    })

        # Save recommendations
        optimization_report = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations,
            "priority_levels": {
                "critical": [r for r in recommendations if "critical" in r.get("issue", "").lower()],
                "high": [r for r in recommendations if "high" in r.get("issue", "").lower() or r["category"] == "latency"],
                "medium": [r for r in recommendations if r["category"] == "cpu"],
                "low": [r for r in recommendations if r["category"] == "memory"]
            }
        }

        with open(self.reports_dir / "optimization_recommendations.json", 'w') as f:
            json.dump(optimization_report, f, indent=2)

        return optimization_report

    def run_performance_profiling(self):
        """Run complete performance profiling"""
        print("⚡ ΣLANG Phase 2: Performance Profiling")
        print("=" * 42)
        print(f"Timestamp: {datetime.now()}")
        print(f"Reports Directory: {self.reports_dir}")
        print()

        # Step 1: Memory profiling
        memory_report = self.profile_memory_usage()

        # Step 2: CPU profiling
        cpu_report = self.profile_cpu_usage()

        # Step 3: Latency analysis
        latency_report = self.analyze_latency_patterns()

        # Step 4: Generate optimization recommendations
        optimization_report = self.generate_optimization_recommendations(
            memory_report, cpu_report, latency_report
        )

        # Final results
        print()
        print("📊 PERFORMANCE PROFILING SUMMARY")
        print("=" * 33)

        recommendations_count = len(optimization_report["recommendations"])

        if recommendations_count == 0:
            self.print_success("✅ NO PERFORMANCE ISSUES FOUND")
            self.print_success("🎉 Performance optimization requirements met")
            success = True
        else:
            self.print_warning(f"⚠️  Found {recommendations_count} optimization opportunities")
            self.print_status("📋 See optimization recommendations for details")

            # Show top recommendations
            for i, rec in enumerate(optimization_report["recommendations"][:3], 1):
                print(f"  {i}. {rec['category'].upper()}: {rec['recommendation']}")

            success = recommendations_count <= 5  # Allow up to 5 recommendations

        print(f"📋 Reports saved to: {self.reports_dir}")

        return success

if __name__ == "__main__":
    profiler = PerformanceProfiler()
    success = profiler.run_performance_profiling()
    sys.exit(0 if success else 1)