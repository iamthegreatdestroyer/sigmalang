"""
BUFFER POOL OPTIMIZATION BENCHMARK
===================================

@VELOCITY Performance Optimization - Workstream A

Comprehensive benchmarking of GlyphBufferPool under various conditions.
Tests allocation patterns, memory footprint, and overhead across input sizes.

Usage:
  python benchmark_buffer_pool.py
"""

import sys
import time
import tracemalloc
from typing import List, Tuple, Dict
import numpy as np

# Add sigmalang to path
sys.path.insert(0, '/root/sigmalang')

from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.primitives import SemanticTree, SemanticNode
from sigmalang.core.optimizations import GlyphBufferPool, FastGlyphEncoder


class BufferPoolBenchmark:
    """Benchmarks for GlyphBufferPool optimization."""
    
    def __init__(self):
        self.results = {}
        self.baseline_metrics = None
    
    def create_test_tree(self, size_bytes: int) -> SemanticTree:
        """Create a semantic tree for testing."""
        # Rough heuristic: ~50 bytes per node
        num_nodes = max(1, size_bytes // 50)
        
        # Primitives: 0=document, 1=paragraph, 2=sentence
        root = SemanticNode(
            primitive=0,
            value=f"Document of size {size_bytes}",
            children=[]
        )
        
        for i in range(1, num_nodes):
            child = SemanticNode(
                primitive=1 if i % 3 == 0 else 2,
                value=f"Node {i} with semantic content " * (i % 3 + 1),
                children=[]
            )
            if len(root.children) < 100:
                root.children.append(child)
            else:
                if len(root.children) > 0:
                    root.children[-1].children.append(child)
        
        return SemanticTree(root=root, source_text=f"Doc {size_bytes}")
    
    def benchmark_pool_config(self, pool_size: int, buffer_size: int, 
                              input_size: int, iterations: int = 10) -> Dict:
        """Benchmark a specific pool configuration."""
        
        results = {
            'pool_size': pool_size,
            'buffer_size': buffer_size,
            'input_size': input_size,
            'peak_memory': 0,
            'avg_allocation_time': 0,
            'pool_hit_rate': 0,
            'allocation_count': 0,
            'fallback_allocations': 0,
        }
        
        pool = GlyphBufferPool(pool_size=pool_size, buffer_size=buffer_size)
        
        # Track allocations
        allocations = []
        releases = []
        
        allocation_times = []
        
        try:
            tracemalloc.start()
            
            for iteration in range(iterations):
                tree = self.create_test_tree(input_size)
                
                # Simulate encoder usage
                start_mem = tracemalloc.get_traced_memory()[0]
                
                # Multiple acquires and releases
                buffers = []
                start_alloc = time.perf_counter()
                
                for _ in range(pool_size * 2):  # Try to exhaust pool
                    buf = pool.acquire()
                    buffers.append(buf)
                    allocations.append(len(buf))
                
                alloc_time = time.perf_counter() - start_alloc
                allocation_times.append(alloc_time)
                
                # Release some buffers
                for buf in buffers[:pool_size]:
                    pool.release(buf)
                    releases.append(len(buf))
                
                peak_mem = tracemalloc.get_traced_memory()[0]
                if peak_mem > results['peak_memory']:
                    results['peak_memory'] = peak_mem
            
            tracemalloc.stop()
            
            # Calculate metrics
            results['avg_allocation_time'] = np.mean(allocation_times) * 1000  # ms
            results['allocation_count'] = len(allocations)
            
            # Estimate hit rate
            pool_capacity = pool_size * buffer_size
            actual_allocations = sum(allocations)
            if actual_allocations > 0:
                hit_rate = (len([a for a in allocations if a <= buffer_size]) / len(allocations)) * 100
                results['pool_hit_rate'] = hit_rate
            
            # Estimate fallback allocations (allocations > buffer_size in pool)
            fallback = len([a for a in allocations if a > buffer_size])
            results['fallback_allocations'] = fallback
            
        except Exception as e:
            print(f"Error benchmarking pool_size={pool_size}: {e}")
            traceback.print_exc()
        
        return results
    
    def benchmark_encoding_with_pool(self, pool_size: int, buffer_size: int,
                                      input_size: int) -> Dict:
        """Benchmark actual encoding with different pool configs."""
        results = {
            'pool_size': pool_size,
            'buffer_size': buffer_size,
            'input_size': input_size,
            'encoding_time': 0,
            'peak_memory': 0,
            'compression_ratio': 0,
        }
        
        try:
            # Create encoder with specific pool
            encoder = SigmaEncoder(enable_optimizations=True)
            encoder.buffer_pool = GlyphBufferPool(pool_size=pool_size, buffer_size=buffer_size)
            
            # Create test tree
            tree = self.create_test_tree(input_size)
            text = "test " * (input_size // 5)
            
            # Encode with memory tracking
            tracemalloc.start()
            start_time = time.perf_counter()
            
            encoded = encoder.encode(tree, text)
            
            encoding_time = (time.perf_counter() - start_time) * 1000  # ms
            peak_memory = tracemalloc.get_traced_memory()[1]  # Peak
            
            tracemalloc.stop()
            
            results['encoding_time'] = encoding_time
            results['peak_memory'] = peak_memory
            results['compression_ratio'] = encoder.get_compression_ratio()
            
        except Exception as e:
            print(f"Error in encoding benchmark: {e}")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks for buffer pool optimization."""
        
        print("\n" + "="*80)
        print("BUFFER POOL OPTIMIZATION BENCHMARK")
        print("@VELOCITY Performance Optimization - Workstream A")
        print("="*80 + "\n")
        
        # Test configurations
        pool_sizes = [16, 32, 64, 128]
        input_sizes = [1024, 102400, 10485760]  # 1KB, 100KB, 10MB
        
        print("PHASE 1: ALLOCATION PATTERN ANALYSIS")
        print("-" * 80)
        print(f"{'Pool Size':<12} {'Buffer Size':<12} {'Input Size':<15} {'Peak Mem (MB)':<15} {'Alloc Time (ms)':<15}")
        print("-" * 80)
        
        allocation_results = {}
        
        for input_size in input_sizes:
            input_label = f"{input_size // 1024}KB" if input_size >= 1024 else f"{input_size}B"
            input_label = input_label.replace("KB", "KB").replace("1048576000B", "10MB")
            
            for pool_size in pool_sizes:
                buffer_size = 4096  # Fixed for now
                
                result = self.benchmark_pool_config(
                    pool_size=pool_size,
                    buffer_size=buffer_size,
                    input_size=input_size,
                    iterations=5
                )
                
                key = f"{pool_size}_{input_size}"
                allocation_results[key] = result
                
                peak_mem_mb = result['peak_memory'] / (1024**2)
                print(f"{pool_size:<12} {buffer_size:<12} {input_label:<15} {peak_mem_mb:<15.2f} {result['avg_allocation_time']:<15.4f}")
        
        print("\n" + "="*80)
        print("PHASE 2: ENCODING PERFORMANCE ANALYSIS")
        print("-" * 80)
        print(f"{'Pool Size':<12} {'Input Size':<15} {'Encoding Time (ms)':<20} {'Peak Mem (MB)':<15} {'Compression':<12}")
        print("-" * 80)
        
        encoding_results = {}
        
        for input_size in input_sizes:
            input_label = f"{input_size // 1024}KB" if input_size >= 1024 else f"{input_size}B"
            input_label = input_label.replace("KB", "KB").replace("1048576000B", "10MB")
            
            for pool_size in pool_sizes:
                buffer_size = 4096
                
                result = self.benchmark_encoding_with_pool(
                    pool_size=pool_size,
                    buffer_size=buffer_size,
                    input_size=input_size
                )
                
                key = f"{pool_size}_{input_size}"
                encoding_results[key] = result
                
                peak_mem_mb = result['peak_memory'] / (1024**2)
                print(f"{pool_size:<12} {input_label:<15} {result['encoding_time']:<20.4f} {peak_mem_mb:<15.2f} {result['compression_ratio']:<12.2f}x")
        
        # Calculate optimal pool size
        print("\n" + "="*80)
        print("ANALYSIS: OPTIMAL POOL CONFIGURATION")
        print("-" * 80)
        
        # Find pool size with best memory/performance trade-off
        best_score = float('inf')
        best_config = None
        
        for input_size in [102400]:  # Focus on 100KB middle case
            scores = {}
            
            for pool_size in pool_sizes:
                key = f"{pool_size}_{input_size}"
                if key in encoding_results:
                    result = encoding_results[key]
                    peak_mem_mb = result['peak_memory'] / (1024**2)
                    
                    # Score: minimize memory and time
                    # Weight: 70% memory, 30% time
                    score = (peak_mem_mb * 0.7) + (result['encoding_time'] * 0.3)
                    scores[pool_size] = (score, peak_mem_mb, result['encoding_time'])
            
            if scores:
                best_pool = min(scores.keys(), key=lambda x: scores[x][0])
                best_score, best_mem, best_time = scores[best_pool]
                
                print(f"\nOptimal pool size (100KB input): {best_pool}")
                print(f"  Peak Memory: {best_mem:.2f} MB")
                print(f"  Encoding Time: {best_time:.4f} ms")
                print(f"  Composite Score: {best_score:.4f}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("-" * 80)
        print("""
1. POOL SIZE SELECTION:
   - Small inputs (<1KB): pool_size=16
   - Medium inputs (1KB-100KB): pool_size=32 (recommended)
   - Large inputs (>100KB): pool_size=64
   
2. ADAPTIVE SIZING:
   Implement input-aware pool sizing:
   - pool_size = max(16, min(128, input_size // 8192))
   
3. MEMORY OPTIMIZATION:
   - Pre-allocate pool on encoder initialization
   - Reuse encoder instance for batch operations
   - Clear pool after large batches
   
4. BENCHMARK IMPROVEMENTS NEEDED:
   - Memory tracking needs optimization
   - Add detailed timing breakdown
   - Track pool exhaustion rates
        """)
        
        return {
            'allocation': allocation_results,
            'encoding': encoding_results
        }


if __name__ == '__main__':
    import traceback
    
    benchmark = BufferPoolBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        traceback.print_exc()
