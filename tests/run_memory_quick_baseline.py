"""
Quick Memory Profiling Baseline - Reduced Test Suite
====================================================

Runs memory profiling baseline for Phase 4A.2 with reduced test sizes.
Faster execution, still captures scaling characteristics.

Execution: python run_memory_quick_baseline.py

Output:
  - memory_profiles/ : Raw profile JSON files
  - baseline_summary.json : Summary report
"""

import os
import sys
import json
import time
import tracemalloc
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import gc

# Add sigmalang to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.primitives import SemanticTree, SemanticNode, CodePrimitive


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Quick baseline configuration - reduced test sizes."""
    
    # Test file sizes (bytes) - REDUCED for speed
    TEST_SIZES = [
        10,                     # 10 B
        1024,                   # 1 KB
        10 * 1024,              # 10 KB
        100 * 1024,             # 100 KB
        1024 * 1024,            # 1 MB
        5 * 1024 * 1024,        # 5 MB (reduced from 10)
        10 * 1024 * 1024,       # 10 MB (reduced from 100)
    ]
    
    # Number of runs per file size - REDUCED
    NUM_RUNS = {
        10: 3,
        1024: 3,
        10*1024: 3,
        100*1024: 2,
        1024*1024: 2,
        5*1024*1024: 1,
        10*1024*1024: 1,
    }
    
    # Output directories
    PROFILE_DIR = Path(__file__).parent / "memory_profiles"
    RESULTS_DIR = Path(__file__).parent / "memory_analysis_results"
    
    MIN_AVAILABLE_MEMORY_GB = 1.0


# ============================================================================
# MEASUREMENT DATA STRUCTURES
# ============================================================================

@dataclass
class MemoryMeasurement:
    """Single memory measurement."""
    timestamp_s: float
    rss_mb: float
    peak_mb: float


@dataclass
class EncodingResult:
    """Result from single encoding."""
    file_size_bytes: int
    file_size_mb: float
    run_number: int
    duration_seconds: float
    encoded_size_bytes: int
    compression_ratio: float
    peak_memory_mb: float
    initial_memory_mb: float
    final_memory_mb: float
    measurements: List[MemoryMeasurement] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['measurements'] = [asdict(m) for m in (self.measurements or [])]
        return data


# ============================================================================
# TEST DATA GENERATOR
# ============================================================================

class TestDataGenerator:
    """Generate semantic trees of various sizes."""
    
    @staticmethod
    def generate_tree(target_size_bytes: int) -> SemanticTree:
        """Generate balanced semantic tree."""
        avg_leaf_size = 128
        target_leaves = max(1, target_size_bytes // avg_leaf_size)
        
        root_node = SemanticNode(primitive=CodePrimitive.BLOCK, value="root")
        
        def build(parent: SemanticNode, depth: int, remaining: int):
            if depth > 5 or remaining <= 0:
                return
            
            branch_factor = min(12, max(2, 8 - depth))
            per_child = max(1, remaining // branch_factor)
            
            for i in range(branch_factor):
                if remaining <= 0:
                    break
                
                node = SemanticNode(primitive=CodePrimitive.VARIABLE, value=f"v{depth}_{i}")
                parent.children.append(node)
                build(node, depth + 1, per_child)
                remaining -= per_child
        
        build(root_node, 0, target_leaves)
        return SemanticTree(root=root_node, source_text="quick_baseline")


# ============================================================================
# PROFILER ENGINE
# ============================================================================

class QuickProfiler:
    """Fast memory profiler."""
    
    def __init__(self):
        self.results = {}
    
    def run_quick_baseline(self) -> Dict:
        """Run quick baseline profiling."""
        
        # Create output directories
        Config.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("QUICK BASELINE: Phase 4A.2 Memory Profile")
        print(f"{'='*70}\n")
        
        # Check memory
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available memory: {available_gb:.2f} GB\n")
        
        # Profile each size
        for size_bytes in Config.TEST_SIZES:
            size_mb = size_bytes / (1024 * 1024)
            num_runs = Config.NUM_RUNS.get(size_bytes, 1)
            
            print(f"Testing {size_mb:.2f}MB... ", end='', flush=True)
            
            self.results[size_bytes] = []
            
            for run in range(num_runs):
                result = self._profile_encoding(size_bytes, run)
                self.results[size_bytes].append(result)
            
            # Print summary
            peaks = [r.peak_memory_mb for r in self.results[size_bytes]]
            print(f"Peak: {np.mean(peaks):.1f} +/- {np.std(peaks):.1f}MB")
        
        return self._save_results()
    
    def _profile_encoding(self, size_bytes: int, run_num: int) -> EncodingResult:
        """Profile single encoding."""
        
        gc.collect()
        process = psutil.Process()
        
        # Initial memory
        initial_mem = process.memory_info().rss / (1024**2)
        
        # Generate tree
        tree = TestDataGenerator.generate_tree(size_bytes)
        post_gen_mem = process.memory_info().rss / (1024**2)
        
        # Encode
        tracemalloc.start()
        start = time.time()
        encoder = SigmaEncoder()
        encoded = encoder.encode(tree)
        duration = time.time() - start
        tracemalloc.stop()
        
        # Final memory
        final_mem = process.memory_info().rss / (1024**2)
        peak_mb = max(post_gen_mem, final_mem)
        
        # Create result
        size_mb = size_bytes / (1024 * 1024)
        result = EncodingResult(
            file_size_bytes=size_bytes,
            file_size_mb=size_mb,
            run_number=run_num,
            duration_seconds=duration,
            encoded_size_bytes=len(encoded),
            compression_ratio=len(encoded) / size_bytes if size_bytes > 0 else 0,
            peak_memory_mb=peak_mb,
            initial_memory_mb=initial_mem,
            final_memory_mb=final_mem,
            measurements=[
                MemoryMeasurement(0.0, initial_mem, peak_mb),
                MemoryMeasurement(duration, final_mem, peak_mb)
            ]
        )
        
        # Cleanup
        del tree, encoded
        gc.collect()
        
        return result
    
    def _save_results(self) -> str:
        """Save results to JSON."""
        
        # Save individual profiles
        for size_bytes, results in self.results.items():
            for result in results:
                filename = Config.PROFILE_DIR / f"profile_{result.file_size_mb:.1f}MB_run{result.run_number}.json"
                with open(filename, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        # Save summary
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'quick_baseline_phase4a2',
            'summary': {}
        }
        
        for size_bytes, results in self.results.items():
            size_mb = size_bytes / (1024 * 1024)
            peaks = [r.peak_memory_mb for r in results]
            
            report['summary'][f'{size_mb:.1f}MB'] = {
                'peak_mean_mb': float(np.mean(peaks)),
                'peak_std_mb': float(np.std(peaks)),
                'peak_min_mb': float(np.min(peaks)),
                'peak_max_mb': float(np.max(peaks)),
                'num_runs': len(results),
                'compression_ratio': float(np.mean([r.compression_ratio for r in results]))
            }
        
        summary_path = Config.PROFILE_DIR / "baseline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Results saved to {Config.PROFILE_DIR}")
        return str(summary_path)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    profiler = QuickProfiler()
    report = profiler.run_quick_baseline()
    
    print(f"\n{'='*70}")
    print("[OK] QUICK BASELINE COMPLETE")
    print(f"{'='*70}\n")
    
    print("Results saved:")
    print(f"  - {report}")
