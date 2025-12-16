"""
ΣLANG <-> Ryot LLM Integration Tests
====================================

Test suite for Phase 2A: Ryot adapter integration
"""

import pytest
from sigmalang.adapters import (
    SigmaCompressionAdapter,
    create_ryot_compression_adapter,
    RyotTokenSequence,
)


class TestRyotAdapter:
    """Test Ryot LLM adapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return create_ryot_compression_adapter(mode="balanced")
    
    def test_adapter_creation(self):
        """Test adapter can be created."""
        adapter = SigmaCompressionAdapter()
        assert adapter is not None
        assert adapter.is_available()
    
    def test_encode_decode_roundtrip(self, adapter):
        """Test encode->decode roundtrip."""
        original = RyotTokenSequence.from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        encoded = adapter.encode(original)
        assert encoded.original_token_count == 10
        assert encoded.compressed_glyph_count > 0
        
        decoded = adapter.decode(encoded)
        assert len(decoded) == 10
    
    def test_compression_ratio(self, adapter):
        """Test compression achieves meaningful ratio."""
        tokens = RyotTokenSequence.from_list(list(range(100)))
        
        encoded = adapter.encode(tokens)
        
        assert encoded.compression_ratio > 1.0
        assert adapter.get_compression_ratio() > 1.0
    
    def test_semantic_hash(self, adapter):
        """Test semantic hash is computed."""
        tokens = RyotTokenSequence.from_list([1, 2, 3, 4, 5])
        
        encoded = adapter.encode(tokens)
        
        assert encoded.semantic_hash != 0
        assert isinstance(encoded.semantic_hash, int)
    
    def test_is_available(self, adapter):
        """Test availability check."""
        assert adapter.is_available()
    
    def test_statistics(self, adapter):
        """Test statistics retrieval."""
        tokens = RyotTokenSequence.from_list([1, 2, 3])
        adapter.encode(tokens)
        
        stats = adapter.get_statistics()
        
        assert "compression_ratio" in stats
        assert "total_tokens_processed" in stats
        assert "average_compression_ratio" in stats
    
    def test_conversation_tracking(self, adapter):
        """Test conversation context tracking."""
        tokens = RyotTokenSequence.from_list([1, 2, 3])
        
        encoded = adapter.encode(tokens, conversation_id="conv_123")
        
        assert "conv_123" in adapter._conversation_contexts
    
    def test_mode_selection(self):
        """Test different encoding modes."""
        for mode in ["fast", "balanced", "deep", "streaming"]:
            adapter = create_ryot_compression_adapter(mode=mode)
            assert adapter is not None
            assert adapter.is_available()
    
    def test_token_sequence_conversions(self):
        """Test RyotTokenSequence conversions."""
        # From list
        tokens = RyotTokenSequence.from_list([1, 2, 3, 4, 5])
        assert len(tokens) == 5
        assert list(tokens.tokens) == [1, 2, 3, 4, 5]
        
        # To list
        as_list = tokens.to_list()
        assert as_list == [1, 2, 3, 4, 5]


def test_adapter_standalone():
    """Quick standalone test."""
    adapter = create_ryot_compression_adapter()
    
    tokens = RyotTokenSequence.from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoded = adapter.encode(tokens)
    
    print(f"\nOriginal tokens: {len(tokens)}")
    print(f"Compressed glyphs: {encoded.compressed_glyph_count}")
    print(f"Compression ratio: {encoded.compression_ratio:.1f}x")
    print(f"Semantic hash: {encoded.semantic_hash:016x}")
    
    # Test decode
    decoded = adapter.decode(encoded)
    print(f"Decoded tokens: {len(decoded)}")
    
    # Test statistics
    stats = adapter.get_statistics()
    print(f"Average compression: {stats['average_compression_ratio']:.1f}x")
    
    print("\n✓ ΣLANG Ryot adapter test passed")


if __name__ == "__main__":
    test_adapter_standalone()
