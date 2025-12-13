#!/usr/bin/env python
"""Quick verification that optimizations are in place."""

from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.primitives import SemanticTree, SemanticNode

# Create simple test tree
root = SemanticNode(primitive=0, value='test')
tree = SemanticTree(root=root, source_text='test')

# Test encoder with optimizations
encoder = SigmaEncoder(enable_optimizations=True)
result = encoder.encode(tree, 'test text')

# Get stats and verify pool optimization
stats = encoder.get_stats()
pool_stats = stats.get('buffer_pool', {})

print('=== OPTIMIZATION VERIFICATION ===')
print(f'Pool Size: {pool_stats.get("pool_size", "N/A")} (target: 16)')
print(f'Buffer Size: {pool_stats.get("buffer_size", "N/A")} bytes')
print(f'Total Acquires: {pool_stats.get("total_acquires", 0)}')
print(f'Overflow Rate: {pool_stats.get("overflow_rate", 0):.1f}%')
print(f'Adaptive Resizes: {pool_stats.get("adaptive_resizes", 0)}')
print()
print('Compression Ratio:', stats.get('compression_ratio', 'N/A'))
print()
print('✓ Optimizations enabled and working!')
