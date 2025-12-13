# Phase 4 Option A: Performance Optimization Plan

## Objective

Optimize encoder/decoder to achieve sub-linear compression with minimal latency overhead.

## Performance Goals

- ✅ **Compression**: 10-50x ratio (baseline: working)
- ⏱️ **Encoding**: < 100µs for typical input
- ⏱️ **Decoding**: < 300µs for typical output
- 💾 **Memory**: < 50MB for full pipeline
- 📊 **Throughput**: > 10K items/sec

## Current Performance Baseline

From test results:

- **Encode**: 109.5 µs (mean) ✅ Already good
- **Decode**: 375.4 µs (mean) ⚠️ Needs optimization
- **Roundtrip**: 191.3 µs (mean) ✅ Good

## Identified Bottlenecks

### 1. **Glyph Encoding/Decoding**

- Variable-length encoding overhead
- Repeated lookups in PRIMITIVE_REGISTRY
- String encoding/decoding in payload

### 2. **Tree Traversal**

- Recursive descent in \_encode_node/\_decode_node
- No memoization of tree walks
- O(n) traversal for each operation

### 3. **Delta Compression**

- Full context stack comparison
- O(m²) primitive matching
- No incremental computation

### 4. **Memory Allocation**

- List appending in hot loops
- GlyphStream construction overhead
- Temporary array allocations

### 5. **Hashing Operations**

- Semantic hash computation in LSHIndex
- Repeated projection matrices
- O(k) candidate evaluation

## Optimization Strategy

### Phase 4A.1: Fast Paths

- [ ] Implement cached primitive lookups
- [ ] Add path compression for common tree shapes
- [ ] Pre-allocate buffers for hot paths

### Phase 4A.2: Algorithm Optimization

- [ ] Replace recursive with iterative traversal
- [ ] Implement incremental delta computation
- [ ] Use bit-packing for glyph headers

### Phase 4A.3: Memory Efficiency

- [ ] Pool allocations for GlyphStreams
- [ ] Use numpy arrays for bulk operations
- [ ] Lazy materialization of trees

### Phase 4A.4: Profiling & Validation

- [ ] Profile with cProfile
- [ ] Benchmark against baselines
- [ ] Memory profiling with tracemalloc
- [ ] Validate correctness after each optimization

## Expected Improvements

- 30% latency reduction (Decode: 375 → 260 µs)
- 25% memory reduction
- 2x throughput improvement

## Success Criteria

- ✅ All tests still passing
- ✅ Compression ratio unchanged
- ✅ Latency reduced by 20%+
- ✅ Memory usage reduced by 15%+
- ✅ No correctness regressions

## Timeline

- Phase 4A.1: 2 hours
- Phase 4A.2: 3 hours
- Phase 4A.3: 2 hours
- Phase 4A.4: 1 hour
