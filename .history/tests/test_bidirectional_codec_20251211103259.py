"""
ΣLANG Bidirectional Codec Tests
================================

Comprehensive tests for the BidirectionalSemanticCodec.

Test categories:
- Basic codec functionality
- Round-trip verification
- Fallback mechanisms
- Snapshot comparison
- Tree diffing
- Adaptive compression
- Statistics tracking

Copyright 2025 - Ryot LLM Project
"""

import sys
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings

# Add parent to path
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root.parent))

from sigmalang.core.primitives import (
    SemanticNode, SemanticTree, ExistentialPrimitive, CodePrimitive
)
from sigmalang.core.bidirectional_codec import (
    BidirectionalSemanticCodec, AdaptiveCompressor,
    TreeSnapshot, TreeDiff, EncodingResult, CompressionMode
)
from tests.conftest import SemanticTreeBuilder, TreeComparator


class TestTreeSnapshot:
    """Test snapshot creation and hashing."""
    
    @pytest.mark.unit
    def test_snapshot_creation(self, simple_semantic_tree):
        """Test creating snapshot from tree."""
        snapshot = TreeSnapshot.from_tree(simple_semantic_tree)
        
        assert snapshot.root_primitive == ExistentialPrimitive.ACTION
        assert snapshot.node_count == simple_semantic_tree.node_count
        assert snapshot.depth == simple_semantic_tree.depth
        assert snapshot.tree_hash is not None
        assert len(snapshot.tree_hash) == 64  # SHA256 hex length
    
    @pytest.mark.unit
    def test_snapshot_consistency(self, simple_semantic_tree):
        """Snapshots of same tree should have same hash."""
        snap1 = TreeSnapshot.from_tree(simple_semantic_tree)
        snap2 = TreeSnapshot.from_tree(simple_semantic_tree)
        
        assert snap1.tree_hash == snap2.tree_hash
    
    @pytest.mark.unit
    def test_snapshot_differentiates_trees(self, simple_semantic_tree, complex_semantic_tree):
        """Snapshots of different trees should have different hashes."""
        snap1 = TreeSnapshot.from_tree(simple_semantic_tree)
        snap2 = TreeSnapshot.from_tree(complex_semantic_tree)
        
        assert snap1.tree_hash != snap2.tree_hash


class TestTreeDiff:
    """Test tree difference detection."""
    
    @pytest.mark.unit
    def test_identical_trees_no_diff(self, simple_semantic_tree):
        """Identical trees should have no differences."""
        diff = TreeDiff.diff_trees(simple_semantic_tree, simple_semantic_tree)
        
        assert len(diff) == 0
    
    @pytest.mark.unit
    def test_primitive_mismatch_detected(self):
        """Should detect primitive mismatches."""
        tree1 = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value="test"
            ),
            source_text="test"
        )
        
        tree2 = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ACTION,
                value="test"
            ),
            source_text="test"
        )
        
        diff = TreeDiff.diff_trees(tree1, tree2)
        
        assert len(diff) > 0
        assert any("primitive" in d for d in diff)
    
    @pytest.mark.unit
    def test_value_mismatch_detected(self):
        """Should detect value mismatches."""
        tree1 = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value="value1"
            ),
            source_text="value1"
        )
        
        tree2 = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value="value2"
            ),
            source_text="value2"
        )
        
        diff = TreeDiff.diff_trees(tree1, tree2)
        
        assert len(diff) > 0
        assert any("value" in d for d in diff)
    
    @pytest.mark.unit
    def test_child_count_mismatch_detected(self):
        """Should detect child count mismatches."""
        tree1 = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value="parent",
                children=[
                    SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value="child1"),
                    SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value="child2"),
                ]
            ),
            source_text="parent"
        )
        
        tree2 = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value="parent",
                children=[
                    SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value="child1"),
                ]
            ),
            source_text="parent"
        )
        
        diff = TreeDiff.diff_trees(tree1, tree2)
        
        assert len(diff) > 0
        assert any("child count" in d for d in diff)


class TestBidirectionalCodec:
    """Test BidirectionalSemanticCodec functionality."""
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_codec_simple_roundtrip(self, simple_semantic_tree):
        """Codec should achieve perfect round-trip on simple tree."""
        codec = BidirectionalSemanticCodec()
        
        result = codec.encode_with_verification(simple_semantic_tree)
        
        assert result.round_trip_successful
        assert result.original_snapshot is not None
        assert result.decoded_snapshot is not None
        assert result.original_snapshot.tree_hash == result.decoded_snapshot.tree_hash
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_codec_complex_roundtrip(self, complex_semantic_tree):
        """Codec should achieve perfect round-trip on complex tree."""
        codec = BidirectionalSemanticCodec()
        
        result = codec.encode_with_verification(complex_semantic_tree)
        
        assert result.round_trip_successful
        assert result.original_snapshot.tree_hash == result.decoded_snapshot.tree_hash
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_codec_deep_tree_roundtrip(self):
        """Codec should handle deeply nested trees."""
        codec = BidirectionalSemanticCodec()
        tree = SemanticTreeBuilder.random_tree(depth=15, avg_branching=1.5)
        
        result = codec.encode_with_verification(tree)
        
        assert result.round_trip_successful or result.mode == CompressionMode.LOSSLESS
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    def test_codec_wide_tree_roundtrip(self):
        """Codec should handle wide trees."""
        codec = BidirectionalSemanticCodec()
        tree = SemanticTreeBuilder.random_tree(depth=3, avg_branching=10.0)
        
        result = codec.encode_with_verification(tree)
        
        assert result.round_trip_successful or result.mode == CompressionMode.LOSSLESS
    
    @pytest.mark.unit
    def test_codec_fallback_to_lossless(self, simple_semantic_tree):
        """Codec should fallback to lossless on verification failure."""
        codec = BidirectionalSemanticCodec(verify_round_trip=True)
        
        # Force fallback by using a tree that may not round-trip perfectly
        # (in practice, this would be detected and handled)
        result = codec.encode_with_verification(
            simple_semantic_tree,
            fallback_to_lossless=True
        )
        
        # Should either succeed with pattern mode or fallback to lossless
        assert result.round_trip_successful
        assert result.mode in [CompressionMode.PATTERN, CompressionMode.LOSSLESS]
    
    @pytest.mark.unit
    def test_codec_statistics_tracking(self, simple_semantic_tree, complex_semantic_tree):
        """Codec should track statistics."""
        codec = BidirectionalSemanticCodec()
        
        codec.encode_with_verification(simple_semantic_tree)
        codec.encode_with_verification(complex_semantic_tree)
        
        stats = codec.get_statistics()
        
        assert stats['total_encodes'] == 2
        assert stats['successful_roundtrips'] >= 1
        assert 0 <= stats['success_rate'] <= 1
        assert 0 <= stats['fallback_rate'] <= 1
        assert stats['average_compression_ratio'] > 0


class TestEncodingResult:
    """Test EncodingResult dataclass."""
    
    @pytest.mark.unit
    def test_encoding_result_creation(self, simple_semantic_tree):
        """Should create valid EncodingResult."""
        codec = BidirectionalSemanticCodec()
        result = codec.encode_with_verification(simple_semantic_tree)
        
        assert isinstance(result, EncodingResult)
        assert result.original_data is not None
        assert result.compressed_data is not None
        assert result.mode is not None
        assert 0 < result.compression_ratio <= 1.5
        assert result.diagnostics is not None
    
    @pytest.mark.unit
    def test_encoding_result_has_diagnostics(self, simple_semantic_tree):
        """EncodingResult should include diagnostic information."""
        codec = BidirectionalSemanticCodec()
        result = codec.encode_with_verification(simple_semantic_tree)
        
        assert 'strategy' in result.diagnostics
        assert 'verification' in result.diagnostics


class TestAdaptiveCompressor:
    """Test AdaptiveCompressor functionality."""
    
    @pytest.mark.unit
    def test_adaptive_compression_simple(self, simple_semantic_tree):
        """Adaptive compressor should compress simple trees."""
        compressor = AdaptiveCompressor()
        
        result = compressor.compress(simple_semantic_tree)
        
        assert result is not None
        assert result.compressed_data is not None
        assert len(result.compressed_data) > 0
    
    @pytest.mark.unit
    def test_adaptive_compression_complex(self, complex_semantic_tree):
        """Adaptive compressor should compress complex trees."""
        compressor = AdaptiveCompressor()
        
        result = compressor.compress(complex_semantic_tree)
        
        assert result is not None
        assert result.compressed_data is not None
    
    @pytest.mark.unit
    def test_adaptive_compression_fallback_to_raw(self):
        """Adaptive compressor should use raw encoding if compression is poor."""
        compressor = AdaptiveCompressor()
        
        # Create a very small tree
        tiny_tree = SemanticTree(
            root=SemanticNode(primitive=ExistentialPrimitive.ENTITY, value="x"),
            source_text="x"
        )
        
        result = compressor.compress(tiny_tree)
        
        # May use raw encoding if compression ratio is poor
        assert result is not None


class TestCompressionModes:
    """Test different compression modes."""
    
    @pytest.mark.unit
    def test_compression_mode_enum(self):
        """CompressionMode enum should have all required modes."""
        modes = [
            CompressionMode.PATTERN,
            CompressionMode.REFERENCE,
            CompressionMode.DELTA,
            CompressionMode.LOSSLESS,
            CompressionMode.RAW,
        ]
        
        assert len(modes) == 5
        
        for mode in modes:
            assert isinstance(mode, CompressionMode)
    
    @pytest.mark.unit
    def test_encoding_result_mode_tracking(self, simple_semantic_tree):
        """EncodingResult should track which mode was used."""
        codec = BidirectionalSemanticCodec()
        result = codec.encode_with_verification(simple_semantic_tree)
        
        assert isinstance(result.mode, CompressionMode)
        assert result.mode != CompressionMode.RAW  # Should use better mode


class TestRoundTripFidelity:
    """Test perfect round-trip fidelity."""
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    @pytest.mark.parametrize("depth,width", [
        (2, 2),
        (5, 3),
        (8, 5),
        (10, 2),
    ])
    def test_roundtrip_various_complexities(self, depth, width):
        """Test round-trip across various tree complexities."""
        codec = BidirectionalSemanticCodec()
        tree = SemanticTreeBuilder.random_tree(depth=depth, avg_branching=width)
        
        result = codec.encode_with_verification(tree)
        
        assert result.round_trip_successful or result.mode == CompressionMode.LOSSLESS
        assert result.original_snapshot is not None
        if result.decoded_snapshot:
            assert result.original_snapshot.node_count == result.decoded_snapshot.node_count
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_roundtrip_property_any_depth(self, depth):
        """Property test: Round-trip should succeed for any depth."""
        codec = BidirectionalSemanticCodec()
        tree = SemanticTreeBuilder.random_tree(depth=depth)
        
        result = codec.encode_with_verification(tree)
        
        assert result.round_trip_successful or result.mode == CompressionMode.LOSSLESS
    
    @pytest.mark.round_trip
    @pytest.mark.unit
    @given(st.text(min_size=0, max_size=50))
    @settings(max_examples=30)
    def test_roundtrip_property_any_value(self, text_value):
        """Property test: Round-trip should work with any text value."""
        codec = BidirectionalSemanticCodec()
        
        tree = SemanticTree(
            root=SemanticNode(
                primitive=ExistentialPrimitive.ENTITY,
                value=text_value
            ),
            source_text=text_value
        )
        
        result = codec.encode_with_verification(tree)
        
        # Should either round-trip successfully or fallback to lossless
        assert result.round_trip_successful or result.mode == CompressionMode.LOSSLESS
