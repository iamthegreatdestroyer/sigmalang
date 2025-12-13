#!/usr/bin/env python
"""Phase 4A.2 Task 3-4: Iterative vs Recursive Traversal Benchmark"""

import time
import statistics
from sigmalang.core.primitives import SemanticNode, SemanticTree, ExistentialPrimitive, CodePrimitive
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder

def create_deep_tree(depth: int) -> SemanticTree:
    """Create a deep tree for testing traversal performance."""
    root = SemanticNode(primitive=ExistentialPrimitive.ACTION, value='root')
    current = root
    for i in range(depth):
        child = SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value=f'level_{i}')
        current.children = [child]
        current = child
    return SemanticTree(root=root, source_text='deep_tree')

def create_wide_tree(width: int) -> SemanticTree:
    """Create a wide tree for testing traversal performance."""
    children = [
        SemanticNode(primitive=CodePrimitive.VARIABLE, value=f'var_{i}')
        for i in range(width)
    ]
    root = SemanticNode(primitive=CodePrimitive.CLASS, value='data_class', children=children)
    return SemanticTree(root=root, source_text='wide_tree')

def benchmark_traversal(trees, enable_optimizations=True, iterations=5):
    """Benchmark iterative vs recursive traversal."""
    results = {}
    
    for name, tree in trees:
        times = []
        
        for _ in range(iterations):
            # Create encoder with specified optimization setting
            encoder = SigmaEncoder(enable_optimizations=enable_optimizations)
            
            # Encode (which triggers traversal)
            start = time.perf_counter()
            encoded = encoder.encode(tree, tree.source_text)
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
        
        results[name] = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        }
    
    return results

def main():
    """Run traversal benchmark."""
    print('=' * 80)
    print('PHASE 4A.2 TASK 3-4: ITERATIVE VS RECURSIVE TRAVERSAL BENCHMARK')
    print('=' * 80)
    
    # Create test trees with various shapes
    trees = [
        ('shallow', create_deep_tree(2)),
        ('medium_deep', create_deep_tree(10)),
        ('very_deep', create_deep_tree(50)),
        ('wide_10', create_wide_tree(10)),
        ('wide_50', create_wide_tree(50)),
    ]
    
    print('\nTest suite:')
    for name, tree in trees:
        depth = _get_tree_depth(tree.root)
        width = len(tree.root.children) if hasattr(tree.root, 'children') else 0
        print(f'  - {name}: depth={depth}, width={width}')
    
    # Benchmark with optimizations disabled (recursive)
    print('\n' + '=' * 80)
    print('BASELINE: Recursive Traversal (optimizations disabled)')
    print('=' * 80)
    recursive = benchmark_traversal(trees, enable_optimizations=False, iterations=3)
    print_results(recursive)
    
    # Benchmark with optimizations enabled (iterative)
    print('\n' + '=' * 80)
    print('OPTIMIZED: Iterative Traversal (optimizations enabled)')
    print('=' * 80)
    iterative = benchmark_traversal(trees, enable_optimizations=True, iterations=3)
    print_results(iterative)
    
    # Print comparison
    print('\n' + '=' * 80)
    print('IMPROVEMENTS')
    print('=' * 80)
    print_improvements(recursive, iterative)

def _get_tree_depth(node, current_depth=0):
    """Helper to calculate tree depth."""
    if not hasattr(node, 'children') or not node.children:
        return current_depth
    return max(_get_tree_depth(child, current_depth + 1) for child in node.children)

def print_results(results):
    """Print benchmark results."""
    print(f'\n{"Tree":<20} {"Mean (µs)":<15} {"Median (µs)":<15} {"Stdev (µs)":<15}')
    print('-' * 65)
    for name in sorted(results.keys()):
        r = results[name]
        print(f'{name:<20} {r["mean"]:>10.2f}    {r["median"]:>10.2f}    {r["stdev"]:>10.2f}')

def print_improvements(baseline, iterative):
    """Print improvement percentages."""
    print(f'\n{"Tree":<20} {"Improvement":<15} {"Status":<20}')
    print('-' * 55)
    for name in sorted(baseline.keys()):
        if name in iterative:
            b = baseline[name]
            i = iterative[name]
            improvement = ((b['mean'] - i['mean']) / b['mean']) * 100
            status = "✅ Faster" if improvement > 5 else "⚠️  Similar" if improvement > -5 else "❌ Slower"
            print(f'{name:<20} {improvement:>10.1f}%      {status:<20}')

if __name__ == '__main__':
    main()
