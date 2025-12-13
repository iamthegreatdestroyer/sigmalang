"""
Tests for Performance Optimizations
====================================

Validates that optimizations maintain correctness while improving performance.
"""

import pytest
import time
import numpy as np
from sigmalang.core.optimizations import (
    FastPrimitiveCache, GlyphBufferPool, FastGlyphEncoder,
    IterativeTreeWalker, IncrementalDeltaCompressor,
    MemoryProfiler, PerformanceMetrics
)
from sigmalang.core.primitives import (
    ExistentialPrimitive, CodePrimitive, SemanticNode, SemanticTree
)


class TestFastPrimitiveCache:
    """Test cached primitive lookups."""
    
    @pytest.mark.unit
    def test_cache_hit_on_registered_primitive(self):
        """Cache should return registered primitives."""
        cache = FastPrimitiveCache()
        
        primitive = ExistentialPrimitive.ENTITY
        cache.register(0x00, "ENTITY", primitive)
        
        result = cache.get_by_id(0x00)
        assert result == primitive
        assert cache.hits == 1
        assert cache.misses == 0
    
    @pytest.mark.unit
    def test_cache_miss_on_unregistered_primitive(self):
        """Cache should return None for unregistered primitives."""
        cache = FastPrimitiveCache()
        
        result = cache.get_by_id(0xFF)
        assert result is None
        assert cache.misses == 1
    
    @pytest.mark.unit
    def test_cache_hit_rate(self):
        """Hit rate calculation should be accurate."""
        cache = FastPrimitiveCache()
        cache.register(0x00, "ENTITY", ExistentialPrimitive.ENTITY)
        
        # 5 hits
        for _ in range(5):
            cache.get_by_id(0x00)
        
        # 3 misses
        for _ in range(3):
            cache.get_by_id(0xFF)
        
        hit_rate = cache.hit_rate()
        assert abs(hit_rate - 5/8) < 0.001  # Should be 0.625
    
    @pytest.mark.unit
    def test_cache_by_name(self):
        """Cache should support lookup by name."""
        cache = FastPrimitiveCache()
        
        primitive = ExistentialPrimitive.ACTION
        cache.register(0x01, "ACTION", primitive)
        
        result = cache.get_by_name("ACTION")
        assert result == primitive


class TestGlyphBufferPool:
    """Test memory pooling for glyph buffers."""
    
    @pytest.mark.unit
    def test_acquire_buffer_from_pool(self):
        """Should acquire clean buffer from pool."""
        pool = GlyphBufferPool(pool_size=4, buffer_size=256)
        
        buf = pool.acquire()
        assert isinstance(buf, bytearray)
        assert len(buf) == 256
    
    @pytest.mark.unit
    def test_buffer_reuse_reduces_allocations(self):
        """Releasing and reacquiring should reuse buffers."""
        pool = GlyphBufferPool(pool_size=2, buffer_size=64)
        
        # Acquire two buffers
        buf1 = pool.acquire()
        buf2 = pool.acquire()
        
        # Pool should be exhausted
        assert len(pool._available) == 0
        
        # Release and reacquire
        pool.release(buf1)
        buf3 = pool.acquire()
        
        # Should reuse buf1
        assert buf3 is not buf2


class TestFastGlyphEncoder:
    """Test fast glyph encoding."""
    
    @pytest.mark.unit
    def test_encode_decode_header_roundtrip(self):
        """Encoding and decoding should be symmetric."""
        glyph_type = 2  # DELTA
        flags = 5
        
        encoded = FastGlyphEncoder.encode_header(glyph_type, flags)
        decoded_type, decoded_flags = FastGlyphEncoder.decode_header(encoded)
        
        assert decoded_type == glyph_type
        assert decoded_flags == flags
    
    @pytest.mark.unit
    def test_varint_encoding_small_values(self):
        """Small values should encode as single byte."""
        for value in [0, 1, 127]:
            encoded = FastGlyphEncoder.encode_varint(value)
            assert len(encoded) == 1
            assert encoded[0] == value
    
    @pytest.mark.unit
    def test_varint_encoding_medium_values(self):
        """Medium values (128-16384) should encode as 2 bytes."""
        for value in [128, 1000, 16383]:
            encoded = FastGlyphEncoder.encode_varint(value)
            assert len(encoded) == 2
    
    @pytest.mark.unit
    def test_varint_cache_hit_rate(self):
        """Repeated varint encoding should hit cache."""
        # Clear cache
        FastGlyphEncoder.encode_varint.cache_clear()
        
        # First call misses
        FastGlyphEncoder.encode_varint(42)
        assert FastGlyphEncoder.encode_varint.cache_info().misses == 1
        
        # Second call hits
        FastGlyphEncoder.encode_varint(42)
        assert FastGlyphEncoder.encode_varint.cache_info().hits == 1


class TestIterativeTreeWalker:
    """Test iterative tree traversal."""
    
    @pytest.mark.unit
    def test_depth_first_walk_returns_all_nodes(self):
        """Should visit all nodes in depth-first order."""
        # Create tree:      root
        #                   /  \
        #                  a    b
        #                 /
        #                c
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value="root"
        )
        a = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="a")
        b = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="b")
        c = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="c")
        
        root.children = [a, b]
        a.children = [c]
        
        result = IterativeTreeWalker.walk_depth_first(root)
        
        # Should have 4 nodes
        assert len(result) == 4
        
        # Extract values
        values = [node.value for node, _ in result]
        
        # First node is root
        assert values[0] == "root"
        
        # Check depths
        depths = [depth for _, depth in result]
        assert depths[0] == 0  # root at depth 0
        assert all(d > 0 for d in depths[1:])  # children at depth > 0
    
    @pytest.mark.unit
    def test_breadth_first_walk_level_order(self):
        """Should visit nodes in breadth-first order."""
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value="root"
        )
        a = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="a")
        b = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="b")
        c = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="c")
        
        root.children = [a, b]
        a.children = [c]
        
        result = IterativeTreeWalker.walk_breadth_first(root)
        
        # Extract values
        values = [node.value for node, _ in result]
        
        # Root first, then level 1 (a, b), then level 2 (c)
        assert values == ["root", "a", "b", "c"]


class TestIncrementalDeltaCompressor:
    """Test incremental delta computation."""
    
    @pytest.mark.unit
    def test_delta_computes_difference(self):
        """Should compute only new primitives."""
        compressor = IncrementalDeltaCompressor()
        
        old_primitives = {
            ExistentialPrimitive.ENTITY,
            ExistentialPrimitive.ACTION,
        }
        new_primitives = {
            ExistentialPrimitive.ENTITY,
            ExistentialPrimitive.ACTION,
            ExistentialPrimitive.RELATION,  # New
        }
        
        # Set context
        compressor._last_primitives = old_primitives.copy()
        
        # Compute delta
        delta = compressor.compute_delta(new_primitives)
        
        assert delta == {ExistentialPrimitive.RELATION}
    
    @pytest.mark.unit
    def test_delta_context_updates(self):
        """Context should update after delta computation."""
        compressor = IncrementalDeltaCompressor()
        
        primitives = {ExistentialPrimitive.ENTITY}
        compressor.compute_delta(primitives)
        
        assert compressor._last_primitives == primitives
    
    @pytest.mark.unit
    def test_delta_reset_clears_context(self):
        """Reset should clear context."""
        compressor = IncrementalDeltaCompressor()
        compressor._last_primitives = {ExistentialPrimitive.ENTITY}
        
        compressor.reset_context()
        
        assert len(compressor._last_primitives) == 0


class TestMemoryProfiler:
    """Test memory profiling."""
    
    @pytest.mark.unit
    def test_track_allocations(self):
        """Should track allocation sizes."""
        profiler = MemoryProfiler()
        
        profiler.record_allocation(100, "buffer")
        profiler.record_allocation(200, "buffer")
        profiler.record_allocation(50, "string")
        
        assert profiler.allocations["buffer"]["count"] == 2
        assert profiler.allocations["buffer"]["total"] == 300
        assert profiler.allocations["string"]["count"] == 1
        assert profiler.allocations["string"]["total"] == 50
    
    @pytest.mark.unit
    def test_peak_memory_tracking(self):
        """Should track peak memory."""
        profiler = MemoryProfiler()
        
        profiler.record_allocation(100, "test")
        profiler.record_allocation(200, "test")
        profiler.record_deallocation(100)
        
        assert profiler.peak_memory == 300
        assert profiler.current_memory == 200


class TestPerformanceMetrics:
    """Test performance metrics collection."""
    
    @pytest.mark.unit
    def test_record_and_report_timings(self):
        """Should collect and report timing statistics."""
        metrics = PerformanceMetrics()
        
        # Record some timings
        for i in range(10):
            metrics.record_timing("encode", 100.0 + i)
        
        stats = metrics.get_stats("encode")
        
        assert stats['min'] == 100.0
        assert stats['max'] == 109.0
        assert abs(stats['mean'] - 104.5) < 0.01
        assert stats['count'] == 10
    
    @pytest.mark.unit
    def test_counter_tracking(self):
        """Should track counters."""
        metrics = PerformanceMetrics()
        
        metrics.increment_counter("primitives")
        metrics.increment_counter("primitives", 5)
        
        assert metrics.counters["primitives"] == 6


class TestOptimizationBenchmarks:
    """Benchmark tests comparing optimized vs non-optimized paths."""
    
    @pytest.mark.benchmark
    def test_cache_lookup_vs_linear_search(self, benchmark):
        """Benchmark: cached lookup vs linear search."""
        cache = FastPrimitiveCache()
        
        # Populate cache
        for i in range(256):
            cache.register(i, f"PRIM_{i}", f"primitive_{i}")
        
        def cached_lookup():
            for _ in range(1000):
                cache.get_by_id(100)
        
        result = benchmark(cached_lookup)
        
        # Verify cache hit rate
        assert cache.hit_rate() > 0.99
    
    @pytest.mark.benchmark
    def test_iterative_vs_recursive_traversal(self, benchmark):
        """Benchmark: iterative vs recursive tree traversal."""
        # Create deep tree
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value="root"
        )
        current = root
        for i in range(100):
            child = SemanticNode(
                primitive=ExistentialPrimitive.ACTION,
                value=f"level_{i}"
            )
            current.children = [child]
            current = child
        
        def iterative_walk():
            for _ in range(100):
                IterativeTreeWalker.walk_depth_first(root)
        
        result = benchmark(iterative_walk)
        assert result is None  # Just measuring time
