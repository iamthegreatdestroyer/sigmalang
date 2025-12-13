#!/usr/bin/env python
"""Phase 4A.2: Performance Benchmark - With vs Without Optimizations"""

import time
import statistics
from sigmalang.core.primitives import SemanticNode, SemanticTree, ExistentialPrimitive, CodePrimitive
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder

def create_test_trees():
    """Create various test trees for benchmarking."""
    trees = []
    
    # Simple tree
    root = SemanticNode(primitive=ExistentialPrimitive.ENTITY, value='user')
    trees.append(('simple', SemanticTree(root=root, source_text='user')))
    
    # Medium tree (3 levels, 3 children)
    root = SemanticNode(primitive=CodePrimitive.FUNCTION, value='calculate')
    child1 = SemanticNode(primitive=CodePrimitive.PARAMETER, value='x')
    child2 = SemanticNode(primitive=CodePrimitive.PARAMETER, value='y')
    grandchild1 = SemanticNode(primitive=CodePrimitive.VARIABLE, value='result')
    child1.children = [grandchild1]
    root.children = [child1, child2]
    trees.append(('medium', SemanticTree(root=root, source_text='function')))
    
    # Deep tree (10 levels)
    root = SemanticNode(primitive=ExistentialPrimitive.ACTION, value='root')
    current = root
    for i in range(10):
        child = SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value=f'level_{i}')
        current.children = [child]
        current = child
    trees.append(('deep', SemanticTree(root=root, source_text='deep_tree')))
    
    # Wide tree (20 siblings)
    children = [
        SemanticNode(primitive=CodePrimitive.VARIABLE, value=f'var_{i}')
        for i in range(20)
    ]
    root = SemanticNode(primitive=CodePrimitive.CLASS, value='data_class', children=children)
    trees.append(('wide', SemanticTree(root=root, source_text='wide_tree')))
    
    return trees

def benchmark_encode_decode(trees, enable_optimizations=True, iterations=10):
    """Benchmark encoding and decoding performance."""
    results = {}
    
    encoder = SigmaEncoder(enable_optimizations=enable_optimizations)
    decoder = SigmaDecoder(encoder)
    
    for name, tree in trees:
        encode_times = []
        decode_times = []
        sizes = []
        
        for _ in range(iterations):
            # Encode
            start = time.perf_counter()
            encoded = encoder.encode(tree, tree.source_text)
            encode_time = (time.perf_counter() - start) * 1_000_000  # Convert to microseconds
            encode_times.append(encode_time)
            sizes.append(len(encoded))
            
            # Decode
            start = time.perf_counter()
            decoded = decoder.decode(encoded)
            decode_time = (time.perf_counter() - start) * 1_000_000
            decode_times.append(decode_time)
        
        results[name] = {
            'encode_mean': statistics.mean(encode_times),
            'encode_median': statistics.median(encode_times),
            'decode_mean': statistics.mean(decode_times),
            'decode_median': statistics.median(decode_times),
            'size_mean': statistics.mean(sizes),
            'compression': encoder.get_compression_ratio(),
        }
    
    return results

def main():
    """Run the benchmark."""
    print('=' * 80)
    print('PHASE 4A.2: OPTIMIZATION PERFORMANCE BENCHMARK')
    print('=' * 80)
    
    trees = create_test_trees()
    print(f'\nTest suite: {len(trees)} tree types')
    for name, tree in trees:
        print(f'  - {name}: {len(list(tree.root.children)) if hasattr(tree.root, "children") else 0} children')
    
    print('\n' + '=' * 80)
    print('BASELINE: Without Optimizations')
    print('=' * 80)
    baseline = benchmark_encode_decode(trees, enable_optimizations=False, iterations=5)
    print_results(baseline)
    
    print('\n' + '=' * 80)
    print('OPTIMIZED: With Optimizations')
    print('=' * 80)
    optimized = benchmark_encode_decode(trees, enable_optimizations=True, iterations=5)
    print_results(optimized)
    
    print('\n' + '=' * 80)
    print('IMPROVEMENTS')
    print('=' * 80)
    print_improvements(baseline, optimized)

def print_results(results):
    """Print benchmark results."""
    print(f'\n{"Tree":<12} {"Encode (µs)":<18} {"Decode (µs)":<18} {"Compression":<15}')
    print('-' * 63)
    for name in ['simple', 'medium', 'deep', 'wide']:
        if name in results:
            r = results[name]
            print(f'{name:<12} {r["encode_mean"]:>8.2f}     {r["decode_mean"]:>8.2f}     {r["compression"]:>8.2f}x')

def print_improvements(baseline, optimized):
    """Print improvement percentages."""
    print(f'\n{"Tree":<12} {"Encode ↓":<15} {"Decode ↓":<15}')
    print('-' * 42)
    for name in ['simple', 'medium', 'deep', 'wide']:
        if name in baseline and name in optimized:
            b = baseline[name]
            o = optimized[name]
            encode_improvement = ((b['encode_mean'] - o['encode_mean']) / b['encode_mean']) * 100
            decode_improvement = ((b['decode_mean'] - o['decode_mean']) / b['decode_mean']) * 100
            print(f'{name:<12} {encode_improvement:>8.1f}%       {decode_improvement:>8.1f}%')

if __name__ == '__main__':
    main()
