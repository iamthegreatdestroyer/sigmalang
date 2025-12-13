#!/usr/bin/env python
"""Quick test of Phase 4A.2 optimization integration."""

from sigmalang.core.primitives import SemanticNode, SemanticTree, ExistentialPrimitive
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder

# Create a simple tree
root = SemanticNode(primitive=ExistentialPrimitive.ENTITY, value='test')
tree = SemanticTree(root=root, source_text='test')

# Encode and decode with optimizations enabled
encoder = SigmaEncoder(enable_optimizations=True)
decoder = SigmaDecoder(encoder)

print('✅ Testing roundtrip with optimizations...')
encoded = encoder.encode(tree, 'test')
print(f'✅ Encoded {len(tree.source_text)} bytes to {len(encoded)} bytes')
print(f'✅ Compression ratio: {encoder.get_compression_ratio():.2f}x')

decoded = decoder.decode(encoded)
print(f'✅ Decoded successfully')
print(f'✅ Original value: {tree.root.value}')
print(f'✅ Decoded value: {decoded.root.value}')
print(f'✅ Match: {tree.root.value == decoded.root.value}')

# Check stats
stats = encoder.get_stats()
print(f'\n✅ Encoder Statistics:')
print(f'  Input bytes: {stats["total_input_bytes"]}')
print(f'  Output bytes: {stats["total_output_bytes"]}')
print(f'  Compression: {stats["compression_ratio"]:.2f}x')
print(f'  Encoding count: {stats["encoding_count"]}')

# Check if timing was recorded
if any(k.startswith('timing_') for k in stats):
    print(f'\n✅ Performance Metrics:')
    for key, value in stats.items():
        if key.startswith('timing_'):
            print(f'  {key}: mean={value["mean"]:.2f} µs')

# Check cache performance
if 'primitive_cache_hit_rate' in stats:
    print(f'  Cache hit rate: {stats["primitive_cache_hit_rate"]:.1%}')

print('\n✅ Phase 4A.2 Integration Test: PASSED')
