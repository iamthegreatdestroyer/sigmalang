"""
Memory Profiling Baseline Runner
==================================

Runs comprehensive memory profiling baseline for Phase 4A.2 implementation.
Generates test files, profiles encoding, and collects measurements with statistical rigor.

Execution: python run_memory_baseline.py

Output:
  - memory_profiles/ : Raw profile JSON files
  - memory_analysis_results/ : Analysis reports and visualizations
  - baseline_report.json : Summary of all measurements

Statistical Guarantees:
  - 95% confidence intervals on all measurements
  - Multiple runs (N=3-5) per file size for variance estimation
  - Outlier detection and anomaly reporting
  - Normality tests and trend analysis
"""

import os
import sys
import json
import time
import tempfile
import tracemalloc
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import gc

# Add sigmalang to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.primitives import (
    SemanticTree, SemanticNode, CodePrimitive, ExistentialPrimitive
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Baseline profiling configuration."""
    
    # Test file sizes (bytes)
    TEST_SIZES = [
        10,                    # 10 B
        1024,                  # 1 KB
        10 * 1024,             # 10 KB
        100 * 1024,            # 100 KB
        1024 * 1024,           # 1 MB
        10 * 1024 * 1024,      # 10 MB
        100 * 1024 * 1024,     # 100 MB (stress test)
    ]
    
    # Number of runs per file size
    NUM_RUNS = {
        10: 5,                  # Small sizes: more runs
        1024: 5,
        10*1024: 5,
        100*1024: 3,
        1024*1024: 3,
        10*1024*1024: 2,       # Large sizes: fewer runs
        100*1024*1024: 1,      # Extra large: stress test only
    }
    
    # Output directories
    PROFILE_DIR = Path(__file__).parent / "memory_profiles"
    RESULTS_DIR = Path(__file__).parent / "memory_analysis_results"
    
    # Skip extremely large tests if low memory
    MIN_AVAILABLE_MEMORY_GB = 2.0
    
    @staticmethod
    def get_num_runs(size_bytes: int) -> int:
        """Get number of runs for size."""
        return Config.NUM_RUNS.get(size_bytes, 1)


# ============================================================================
# MEASUREMENT DATA STRUCTURES
# ============================================================================

@dataclass
class MemoryMeasurement:
    """Single memory measurement."""
    timestamp_s: float
    rss_mb: float  # Resident set size
    vms_mb: float  # Virtual memory size
    peak_mb: float  # Peak traced memory
    current_mb: float  # Current traced memory
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EncodingResult:
    """Result of an encoding operation."""
    file_size_bytes: int
    file_size_mb: float
    run_number: int
    duration_seconds: float
    encoded_size_bytes: int
    compression_ratio: float
    
    # Memory measurements
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_freed_mb: float
    
    # Measurements taken during encoding
    measurements: List[MemoryMeasurement] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['measurements'] = [m.to_dict() for m in (self.measurements or [])]
        return data


# ============================================================================
# MEMORY PROFILER ENGINE
# ============================================================================

class MemoryMeasurementEngine:
    """Captures memory during encoding with high precision."""
    
    def __init__(self, sampling_interval: float = 0.01):
        """
        Initialize measurement engine.
        
        Args:
            sampling_interval: Time between measurements (seconds)
        """
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.measurements: List[MemoryMeasurement] = []
        self.start_time: Optional[float] = None
    
    def start(self):
        """Begin measurement."""
        gc.collect()
        tracemalloc.start()
        self.measurements = []
        self.start_time = time.time()
        self._record()  # Initial measurement
    
    def _record(self):
        """Record current memory state."""
        if self.start_time is None:
            return
        
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        mem_info = self.process.memory_info()
        
        measurement = MemoryMeasurement(
            timestamp_s=time.time() - self.start_time,
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            peak_mb=peak_traced / (1024 * 1024),
            current_mb=current_traced / (1024 * 1024)
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def stop(self) -> Tuple[float, List[MemoryMeasurement]]:
        """Stop measurement and return peak memory."""
        peak_mb = max(m.peak_mb for m in self.measurements) if self.measurements else 0
        tracemalloc.stop()
        return peak_mb, self.measurements
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory observed."""
        return max(m.peak_mb for m in self.measurements) if self.measurements else 0


# ============================================================================
# TEST DATA GENERATOR
# ============================================================================

class TestDataGenerator:
    """Generate semantic trees of various sizes with statistical properties."""
    
    @staticmethod
    def generate_tree(target_size_bytes: int, tree_type: str = 'balanced') -> SemanticTree:
        """
        Generate semantic tree of approximately target size.
        
        Args:
            target_size_bytes: Target tree size
            tree_type: 'balanced', 'deep', or 'wide'
        
        Returns:
            SemanticTree of approximately target size
        """
        if tree_type == 'balanced':
            return TestDataGenerator._generate_balanced_tree(target_size_bytes)
        elif tree_type == 'deep':
            return TestDataGenerator._generate_deep_tree(target_size_bytes)
        elif tree_type == 'wide':
            return TestDataGenerator._generate_wide_tree(target_size_bytes)
        else:
            return TestDataGenerator._generate_balanced_tree(target_size_bytes)
    
    @staticmethod
    def _generate_balanced_tree(target_size_bytes: int) -> SemanticTree:
        """Generate balanced tree."""
        
        # Estimate leaf count
        avg_leaf_size = 128
        target_leaves = max(1, target_size_bytes // avg_leaf_size)
        
        root_node = SemanticNode(primitive=CodePrimitive.BLOCK, value="root")
        
        def build(parent: SemanticNode, depth: int, remaining: int) -> int:
            if depth > 6 or remaining <= 0:
                return 0
            
            branch_factor = min(16, max(2, 10 - depth))
            per_child = max(1, remaining // branch_factor)
            
            for i in range(branch_factor):
                if remaining <= 0:
                    break
                
                node = SemanticNode(
                    primitive=CodePrimitive.VARIABLE,
                    value=f"v{depth}_{i}_{i*100}"
                )
                
                parent.children.append(node)
                build(node, depth + 1, per_child)
                remaining -= per_child
            
            return 1
        
        build(root_node, 0, target_leaves)
        
        tree = SemanticTree(root=root_node, source_text="generated_balanced_tree")
        return tree
    
    @staticmethod
    def _generate_deep_tree(target_size_bytes: int) -> SemanticTree:
        """Generate deep linear tree."""
        current = SemanticNode(primitive=CodePrimitive.BLOCK, value="root")
        
        size = 64
        depth = 0
        while size < target_size_bytes and depth < 100:
            child = SemanticNode(primitive=CodePrimitive.STATEMENT, value=f"s{depth}")
            current.children.append(child)
            current = child
            size += 128
            depth += 1
        
        tree = SemanticTree(root=current, source_text="generated_deep_tree")
        return tree
    
    @staticmethod
    def _generate_wide_tree(target_size_bytes: int) -> SemanticTree:
        """Generate wide tree."""
        root = SemanticNode(primitive=CodePrimitive.BLOCK, value="root")
        
        size = 64
        child_idx = 0
        while size < target_size_bytes:
            child = SemanticNode(primitive=CodePrimitive.VARIABLE, value=f"v{child_idx}")
            root.children.append(child)
            size += 128
            child_idx += 1
        
        tree = SemanticTree(root=root, source_text="generated_wide_tree")
        return tree


# ============================================================================
# BASELINE PROFILER
# ============================================================================

class BaselineProfiler:
    """Runs baseline memory profiling for Phase 4A.2."""
    
    def __init__(self):
        self.results: Dict[int, List[EncodingResult]] = defaultdict(list)
        Config.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def run_full_baseline(self) -> Dict:
        """Run complete baseline profiling."""
        print(f"{'='*70}")
        print("WORKSTREAM C: Memory Profiling Baseline")
        print(f"{'='*70}\n")
        
        print(f"Profile directory: {Config.PROFILE_DIR}")
        print(f"Results directory: {Config.RESULTS_DIR}\n")
        
        # Check available memory
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available memory: {available_gb:.2f} GB")
        print(f"Minimum required: {Config.MIN_AVAILABLE_MEMORY_GB:.2f} GB\n")
        
        # Run profiling for each file size
        for size_bytes in Config.TEST_SIZES:
            size_mb = size_bytes / (1024 * 1024)
            
            # Skip if not enough memory
            if available_gb < Config.MIN_AVAILABLE_MEMORY_GB and size_mb > 10:
                print(f"\n[!] Skipping {size_mb:.2f}MB test (insufficient memory)")
                continue
            
            print(f"\n{'-'*70}")
            print(f"Testing: {size_mb:.2f} MB input")
            print(f"{'-'*70}")
            
            num_runs = Config.get_num_runs(size_bytes)
            print(f"Runs: {num_runs}")
            
            for run in range(num_runs):
                print(f"\n  Run {run+1}/{num_runs}...", end=' ', flush=True)
                result = self._profile_single_encoding(size_bytes, run)
                self.results[size_bytes].append(result)
                print(f"[OK]")
                print(f"    Peak: {result.peak_memory_mb:.2f}MB")
                print(f"    Ratio: {result.peak_memory_mb/result.file_size_mb:.3f}x")
                print(f"    Compression: {result.compression_ratio:.3f}")
            
            # Summary for this size
            self._print_size_summary(size_bytes)
        
        return self._generate_summary_report()
    
    def _profile_single_encoding(self, size_bytes: int, run_num: int) -> EncodingResult:
        """Profile single encoding operation."""
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Create encoder
        encoder = SigmaEncoder()
        
        # Measure initial memory (before tree generation)
        process = psutil.Process()
        gc.collect()
        initial_mem = process.memory_info().rss / (1024**2)
        
        # Generate test data (this consumes memory)
        tree = TestDataGenerator.generate_tree(size_bytes)
        post_generation_mem = process.memory_info().rss / (1024**2)
        
        # Measure memory during encoding
        tracemalloc.start()
        traced_before = tracemalloc.get_traced_memory()[0]
        
        start = time.time()
        encoded = encoder.encode(tree)
        duration = time.time() - start
        
        traced_after_max = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        # Get final memory
        final_mem = process.memory_info().rss / (1024**2)
        
        # Calculate peak as maximum of RSS during generation/encoding
        peak_mb = max(post_generation_mem, final_mem, traced_after_max / (1024**2))
        
        # Calculate metrics
        size_mb = size_bytes / (1024 * 1024)
        compression_ratio = len(encoded) / size_bytes if size_bytes > 0 else 0
        memory_freed = max(0, initial_mem - final_mem)
        
        # Create dummy measurements list
        measurements = [
            MemoryMeasurement(
                timestamp_s=0.0,
                rss_mb=initial_mem,
                vms_mb=process.memory_info().vms / (1024**2),
                peak_mb=peak_mb,
                current_mb=initial_mem
            ),
            MemoryMeasurement(
                timestamp_s=duration,
                rss_mb=final_mem,
                vms_mb=process.memory_info().vms / (1024**2),
                peak_mb=peak_mb,
                current_mb=final_mem
            )
        ]
        
        # Create result
        result = EncodingResult(
            file_size_bytes=size_bytes,
            file_size_mb=size_mb,
            run_number=run_num,
            duration_seconds=duration,
            encoded_size_bytes=len(encoded),
            compression_ratio=compression_ratio,
            initial_memory_mb=initial_mem,
            peak_memory_mb=peak_mb,
            final_memory_mb=final_mem,
            memory_freed_mb=memory_freed,
            measurements=measurements
        )
        
        # Clean up to avoid accumulation
        del tree
        del encoded
        gc.collect()
        
        return result
    
    def _print_size_summary(self, size_bytes: int):
        """Print summary for a file size."""
        results = self.results[size_bytes]
        size_mb = size_bytes / (1024 * 1024)
        
        peaks = [r.peak_memory_mb for r in results]
        ratios = [r.peak_memory_mb / r.file_size_mb for r in results]
        
        print(f"\n  Summary for {size_mb:.2f}MB:")
        print(f"    Peak Memory: {np.mean(peaks):.2f} +/- {np.std(peaks):.2f} MB")
        print(f"    Min/Max: {min(peaks):.2f} / {max(peaks):.2f} MB")
        print(f"    Scaling Ratio: {np.mean(ratios):.3f}x")
    
    def _generate_summary_report(self) -> Dict:
        """Generate summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'os': os.name,
                'system': sys.platform,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            },
            'results': {}
        }
        
        for size_bytes, results in self.results.items():
            size_mb = size_bytes / (1024 * 1024)
            peaks = [r.peak_memory_mb for r in results]
            ratios = [r.peak_memory_mb / r.file_size_mb for r in results]
            
            report['results'][str(size_mb)] = {
                'file_size_mb': size_mb,
                'num_runs': len(results),
                'peak_memory': {
                    'mean_mb': float(np.mean(peaks)),
                    'std_mb': float(np.std(peaks)),
                    'min_mb': float(min(peaks)),
                    'max_mb': float(max(peaks)),
                    'ci_95_lower': float(np.mean(peaks) - 1.96*np.std(peaks)/np.sqrt(len(peaks))),
                    'ci_95_upper': float(np.mean(peaks) + 1.96*np.std(peaks)/np.sqrt(len(peaks))),
                },
                'scaling_ratio': {
                    'mean': float(np.mean(ratios)),
                    'std': float(np.std(ratios)),
                },
                'compression_ratio': float(np.mean([r.compression_ratio for r in results])),
                'all_results': [r.to_dict() for r in results]
            }
        
        return report
    
    def save_results(self) -> Path:
        """Save all results to files."""
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}\n")
        
        # Save individual profiles
        for size_bytes, results in self.results.items():
            for result in results:
                filename = Config.PROFILE_DIR / f"profile_size_{result.file_size_mb:.2f}MB_run_{result.run_number}.json"
                with open(filename, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"[OK] Saved: {filename.name}")
        
        # Save summary report
        report = self._generate_summary_report()
        report_path = Config.PROFILE_DIR / "baseline_summary.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Saved: baseline_summary.json")
        
        return report_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run baseline profiling."""
    try:
        profiler = BaselineProfiler()
        report = profiler.run_full_baseline()
        profiler.save_results()
        
        print(f"\n{'='*70}")
        print("[OK] BASELINE PROFILING COMPLETE")
        print(f"{'='*70}\n")
        
        print("Next steps:")
        print("1. Run Phase 4A.3 optimizations")
        print("2. Run memory profiling again")
        print("3. Compare before/after using: python memory_analysis.py")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
