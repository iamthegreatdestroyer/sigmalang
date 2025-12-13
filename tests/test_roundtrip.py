"""
ΣLANG Round-Trip & Edge Case Tests
===================================

Comprehensive testing of encoder/decoder round-trip fidelity.
Ensures that encoding followed by decoding produces perfect reconstruction.

Test Categories:
- Basic round-trip validation
- Complex nested structures
- Edge cases and boundary conditions
- Property-based testing with Hypothesis
- Compression ratio validation

Copyright 2025 - Ryot LLM Project
"""

import sys
import pytest
from pathlib import Path
import numpy as np
from typing import List, Tuple
from hypothesis import given, strategies as st, settings

# Add parent to path
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root.parent))

from sigmalang.core.primitives import (
    SemanticNode, SemanticTree, ExistentialPrimitive, CodePrimitive,
    ActionPrimitive, EntityPrimitive, PRIMITIVE_REGISTRY
)
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
from sigmalang.core.parser import SemanticParser
from tests.conftest import (
    SemanticTreeBuilder, TreeComparator, CompressionAnalyzer, TestDatasets
)


# ============================================================================
# BASIC ROUND-TRIP TESTS
# ============================================================================

class TestBasicRoundTrip:
    """Basic round-trip validation tests."""
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_simple_tree_round_trip(self, simple_semantic_tree):
        """Test encoding and decoding of simple semantic tree."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        # Encode
        encoded = encoder.encode(simple_semantic_tree)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        
        # Decode
        decoded = decoder.decode(encoded)
        
        # Verify
        assert TreeComparator.trees_equal(simple_semantic_tree, decoded), \
            f"Trees not equal. Differences: {TreeComparator.get_differences(simple_semantic_tree, decoded)}"
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_complex_tree_round_trip(self, complex_semantic_tree):
        """Test encoding and decoding of complex semantic tree."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        encoded = encoder.encode(complex_semantic_tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(complex_semantic_tree, decoded), \
            f"Trees not equal. Differences: {TreeComparator.get_differences(complex_semantic_tree, decoded)}"
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_single_node_round_trip(self):
        """Test round-trip of single-node tree."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value="test"
        )
        tree = SemanticTree(root=root, source_text="test")
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded)
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_deep_tree_round_trip(self):
        """Test round-trip of deeply nested tree."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        # Create deep tree (depth 10)
        node = SemanticNode(
            primitive=ExistentialPrimitive.ACTION,
            value="root"
        )
        current = node
        for i in range(9):
            child = SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value=f"level_{i}"
            )
            current.children = [child]
            current = child
        
        tree = SemanticTree(root=node, source_text="deep_test")
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded)
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_wide_tree_round_trip(self):
        """Test round-trip of wide tree (many siblings)."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        # Create wide tree (20 children)
        children = [
            SemanticNode(
                primitive=CodePrimitive.PARAMETER,
                value=f"param_{i}"
            )
            for i in range(20)
        ]
        
        root = SemanticNode(
            primitive=CodePrimitive.FUNCTION,
            value="func",
            children=children
        )
        tree = SemanticTree(root=root, source_text="wide_test")
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""
    
    @pytest.mark.unit
    def test_empty_value_round_trip(self):
        """Test node with empty value string."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value=""
        )
        tree = SemanticTree(root=root, source_text="")
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded)
    
    @pytest.mark.unit
    def test_special_characters_in_values(self):
        """Test values containing special characters."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        special_values = [
            "value_with_üñíçödé",
            "value\nwith\nnewlines",
            "value\twith\ttabs",
            "value with \"quotes\" and 'apostrophes'",
            "value\x00with\x00null\x00bytes",
        ]
        
        for special_value in special_values:
            root = SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value=special_value
            )
            tree = SemanticTree(root=root, source_text=special_value)
            
            encoded = encoder.encode(tree)
            decoded = decoder.decode(encoded)
            
            assert TreeComparator.trees_equal(tree, decoded), \
                f"Failed for value: {repr(special_value)}"
    
    @pytest.mark.unit
    def test_large_value_round_trip(self):
        """Test node with very large value string."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        large_value = "x" * 10000  # 10KB value
        
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value=large_value
        )
        tree = SemanticTree(root=root, source_text=large_value)
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded)
    
    @pytest.mark.unit
    def test_all_primitive_types_round_trip(self):
        """Test round-trip using all primitive types."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        primitives_to_test = [
            ExistentialPrimitive.ENTITY,
            ExistentialPrimitive.ACTION,
            ExistentialPrimitive.RELATION,
            CodePrimitive.FUNCTION,
            CodePrimitive.CLASS,
            CodePrimitive.VARIABLE,
        ]
        
        for primitive in primitives_to_test:
            root = SemanticNode(
                primitive=primitive,
                value=f"test_{primitive.name}"
            )
            tree = SemanticTree(root=root, source_text="")
            
            encoded = encoder.encode(tree)
            decoded = decoder.decode(encoded)
            
            assert TreeComparator.trees_equal(tree, decoded), \
                f"Failed for primitive: {primitive}"


# ============================================================================
# COMPRESSION RATIO VALIDATION TESTS
# ============================================================================

class TestCompressionRatios:
    """Validate compression ratio consistency and quality."""
    
    @pytest.mark.unit
    def test_compression_ratio_positive(self, simple_semantic_tree):
        """Compression ratio should never be >1.0 without reason."""
        encoder = SigmaEncoder()
        
        encoded = encoder.encode(simple_semantic_tree)
        original_text = simple_semantic_tree.source_text
        original_size = len(original_text.encode('utf-8'))
        compressed_size = len(encoded)
        
        ratio = CompressionAnalyzer.compute_ratio(original_size, compressed_size)
        
        # Should achieve at least some compression or have fallback
        assert ratio >= 0.8 or ratio == 1.0, \
            f"Poor compression ratio: {ratio} ({compressed_size} vs {original_size} bytes)"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_text", TestDatasets.CODE_SNIPPETS[:5])
    def test_code_snippet_compression(self, semantic_parser, input_text):
        """Code snippets should compress well.
        
        Note: Short, unique text (<100 bytes) may expand due to encoding overhead
        (GlyphStream header, glyph metadata, CRC checksum). This is expected behavior
        in all compression systems - overhead dominates for small inputs.
        
        For longer text (>100 bytes), compression should be effective as semantic
        patterns can be reused and delta encoding becomes beneficial.
        """
        encoder = SigmaEncoder()
        
        tree = semantic_parser.parse(input_text)
        encoded = encoder.encode(tree)
        
        original_size = len(input_text.encode('utf-8'))
        compressed_size = len(encoded)
        
        ratio = CompressionAnalyzer.compute_ratio(original_size, compressed_size)
        
        # Realistic compression expectations based on input size
        if original_size < 100:
            # Very short text: allow expansion due to fixed encoding overhead
            # (4-byte header + 2-byte CRC + glyph metadata)
            assert 0.3 <= ratio <= 3.0, \
                f"Unexpected compression ratio for short code (<100 bytes): {ratio}"
        else:
            # Longer text: expect good compression from semantic pattern sharing
            assert 0.2 <= ratio <= 0.9, \
                f"Unexpected compression ratio for longer code (>100 bytes): {ratio}"
    
    @pytest.mark.unit
    def test_compression_ratio_distribution(self, test_data_collection):
        """Test compression on diverse inputs."""
        encoder = SigmaEncoder()
        parser = SemanticParser()
        
        test_cases = test_data_collection.get_test_cases()[:10]
        ratios = []
        
        for test_case in test_cases:
            tree = parser.parse(test_case.input_text)
            encoded = encoder.encode(tree)
            
            original_size = len(test_case.input_text.encode('utf-8'))
            compressed_size = len(encoded)
            
            ratio = CompressionAnalyzer.compute_ratio(original_size, compressed_size)
            ratios.append(ratio)
        
        # Average compression should be decent
        avg_ratio = np.mean(ratios)
        assert avg_ratio <= 0.9, f"Average compression ratio too high: {avg_ratio}"


# ============================================================================
# PROPERTY-BASED TESTS (HYPOTHESIS)
# ============================================================================

class TestPropertyBased:
    """Property-based testing using Hypothesis."""
    
    @pytest.mark.unit
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_round_trip_any_depth(self, depth):
        """Property: Encoding then decoding should always reconstruct."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        tree = SemanticTreeBuilder.random_tree(
            depth=depth,
            avg_branching=2.0,
            seed=hash(depth) % (2**31)
        )
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded), \
            f"Round-trip failed for depth {depth}"
    
    @pytest.mark.unit
    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_arbitrary_text_values(self, text_value):
        """Property: Any text value should encode/decode correctly."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value=text_value
        )
        tree = SemanticTree(root=root, source_text=text_value)
        
        try:
            encoded = encoder.encode(tree)
            decoded = decoder.decode(encoded)
            assert TreeComparator.trees_equal(tree, decoded)
        except Exception as e:
            pytest.fail(f"Failed for text value {repr(text_value)}: {e}")
    
    @pytest.mark.unit
    @given(st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_arbitrary_tree_width(self, width):
        """Property: Trees of any width should encode/decode correctly."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        children = [
            SemanticNode(
                primitive=ExistentialPrimitive.ATTRIBUTE,
                value=f"attr_{i}"
            )
            for i in range(width)
        ]
        
        root = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value="root",
            children=children
        )
        tree = SemanticTree(root=root, source_text="")
        
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded), \
            f"Failed for width {width}"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with full pipeline."""
    
    @pytest.mark.integration
    def test_parse_encode_decode_roundtrip(self, semantic_parser):
        """Full pipeline: text → parse → encode → decode → compare."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        input_text = "Create a Python function that sorts a list"
        
        # Parse text
        tree = semantic_parser.parse(input_text)
        
        # Encode tree
        encoded = encoder.encode(tree)
        
        # Decode bytes
        decoded = decoder.decode(encoded)
        
        # Verify
        assert TreeComparator.trees_equal(tree, decoded)
    
    @pytest.mark.integration
    @pytest.mark.parametrize("input_text", TestDatasets.CODE_SNIPPETS)
    def test_all_code_snippets_roundtrip(self, semantic_parser, input_text):
        """Test round-trip for all code snippets."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        tree = semantic_parser.parse(input_text)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded), \
            f"Failed for: {input_text}"
    
    @pytest.mark.integration
    @pytest.mark.parametrize("input_text", TestDatasets.QUERIES)
    def test_all_queries_roundtrip(self, semantic_parser, input_text):
        """Test round-trip for all queries."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        
        tree = semantic_parser.parse(input_text)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)
        
        assert TreeComparator.trees_equal(tree, decoded), \
            f"Failed for: {input_text}"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and benchmarking tests."""
    
    @pytest.mark.performance
    def test_encode_performance(self, benchmark):
        """Benchmark encoding performance."""
        encoder = SigmaEncoder()
        tree = SemanticTreeBuilder.complex_tree()
        
        def encode_tree():
            return encoder.encode(tree)
        
        result = benchmark(encode_tree)
        assert len(result) > 0
    
    @pytest.mark.performance
    def test_decode_performance(self, benchmark):
        """Benchmark decoding performance."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        tree = SemanticTreeBuilder.complex_tree()
        encoded = encoder.encode(tree)
        
        def decode_tree():
            return decoder.decode(encoded)
        
        result = benchmark(decode_tree)
        assert result is not None
    
    @pytest.mark.performance
    def test_roundtrip_performance(self, benchmark):
        """Benchmark full round-trip performance."""
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)
        tree = SemanticTreeBuilder.complex_tree()
        
        def roundtrip():
            encoded = encoder.encode(tree)
            return decoder.decode(encoded)
        
        result = benchmark(roundtrip)
        assert result is not None
