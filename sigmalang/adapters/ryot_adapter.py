"""
ΣLANG Adapter for Ryot LLM Integration
======================================

Implements Ryot's CompressionEngineProtocol using ΣLANG compression.
Bridges ΣLANG's type system with Ryot's expected interface.
"""

from typing import Optional, List, Iterator
import hashlib
from dataclasses import dataclass
from typing import Tuple

# Import from ΣLANG
from ..api.interfaces import CompressionEngine as SigmaCompressionEngine
from ..api.types import (
    EncodingMode,
    EncodingResult,
    DecodingResult,
    SigmaEncodedContext,
)


# ============================================================================
# RYOT-COMPATIBLE TYPE WRAPPERS
# ============================================================================

@dataclass
class RyotTokenSequence:
    """Compatible with Ryot's TokenSequence."""
    tokens: Tuple[int, ...]
    sigma_encoded: bool = False
    compression_ratio: Optional[float] = None
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    @classmethod
    def from_list(cls, tokens: List[int]) -> 'RyotTokenSequence':
        """Create from list of tokens."""
        return cls(tokens=tuple(tokens))
    
    def to_list(self) -> List[int]:
        """Convert to list of tokens."""
        return list(self.tokens)


@dataclass
class RyotSigmaEncodedContext:
    """Compatible with Ryot's SigmaEncodedContext."""
    glyph_sequence: bytes
    original_token_count: int
    compressed_glyph_count: int
    decompressed_tokens: Optional[RyotTokenSequence] = None
    compression_ratio: float = 1.0
    semantic_hash: int = 0
    is_delta_encoded: bool = False
    parent_rsu_reference: Optional[str] = None


# ============================================================================
# ADAPTER IMPLEMENTATION
# ============================================================================

class SigmaCompressionAdapter:
    """
    Adapter implementing Ryot's CompressionEngineProtocol.
    
    Bridges ΣLANG's compression system to Ryot LLM's expected interface.
    Handles type conversions and protocol mapping.
    """
    
    def __init__(
        self,
        sigma_engine: Optional[SigmaCompressionEngine] = None,
        default_mode: EncodingMode = EncodingMode.BALANCED,
    ):
        """
        Initialize adapter.
        
        Args:
            sigma_engine: Optional ΣLANG engine (creates mock if None)
            default_mode: Default encoding mode for operations
        """
        if sigma_engine is None:
            # Use mock for testing
            from ..stubs import MockCompressionEngine
            sigma_engine = MockCompressionEngine()
        
        self._sigma = sigma_engine
        self._default_mode = default_mode
        self._compression_ratio = 1.0
        self._conversation_contexts: dict = {}
    
    def encode(
        self,
        tokens: RyotTokenSequence,
        conversation_id: Optional[str] = None,
    ) -> RyotSigmaEncodedContext:
        """
        Encode tokens using ΣLANG compression.
        
        Args:
            tokens: Ryot TokenSequence to compress
            conversation_id: Optional conversation ID for RSU chaining
        
        Returns:
            RyotSigmaEncodedContext with compressed data
        """
        # Convert to list for ΣLANG
        token_list = list(tokens.tokens)
        
        # Encode with ΣLANG
        result: EncodingResult = self._sigma.encode(
            token_list,
            mode=self._default_mode
        )
        
        # Update compression ratio
        self._compression_ratio = result.encoded_context.compression_ratio
        
        # Compute semantic hash
        semantic_hash = self._compute_semantic_hash(result.encoded_context)
        
        # Convert to Ryot format
        ryot_context = RyotSigmaEncodedContext(
            glyph_sequence=self._serialize_glyphs(result.encoded_context),
            original_token_count=len(token_list),
            compressed_glyph_count=result.encoded_context.glyph_count,
            compression_ratio=result.encoded_context.compression_ratio,
            semantic_hash=semantic_hash,
            is_delta_encoded=False,
        )
        
        # Track conversation context
        if conversation_id:
            self._conversation_contexts[conversation_id] = ryot_context
        
        return ryot_context
    
    def decode(
        self,
        encoded: RyotSigmaEncodedContext,
    ) -> RyotTokenSequence:
        """
        Decode ΣLANG-compressed context back to tokens.
        
        Args:
            encoded: RyotSigmaEncodedContext to decompress
        
        Returns:
            RyotTokenSequence with original tokens
        """
        # Deserialize glyphs back to SigmaEncodedContext
        sigma_context = self._deserialize_glyphs(
            encoded.glyph_sequence,
            encoded.original_token_count,
            encoded.compression_ratio
        )
        
        # Decode with ΣLANG
        result: DecodingResult = self._sigma.decode(sigma_context)
        
        # Convert to Ryot format
        return RyotTokenSequence(
            tokens=tuple(result.decoded_context.tokens),
            sigma_encoded=False,
            compression_ratio=1.0,
        )
    
    def get_compression_ratio(self) -> float:
        """
        Get current compression ratio.
        
        Returns:
            Average compression ratio from recent operations
        """
        return self._compression_ratio
    
    def is_available(self) -> bool:
        """
        Check if compression engine is available.
        
        Returns:
            True if ready for encoding/decoding
        """
        return self._sigma.is_available()
    
    def encode_streaming(
        self,
        tokens: RyotTokenSequence,
        chunk_size: int = 512,
    ) -> Iterator[RyotSigmaEncodedContext]:
        """
        Encode tokens in streaming mode.
        
        Yields compressed chunks for long sequences.
        
        Args:
            tokens: Tokens to encode in chunks
            chunk_size: Size of each chunk to process
        
        Yields:
            RyotSigmaEncodedContext chunks
        """
        token_list = list(tokens.tokens)
        
        for chunk_result in self._sigma.encode_streaming(
            iter(token_list),
            mode=self._default_mode
        ):
            yield RyotSigmaEncodedContext(
                glyph_sequence=self._serialize_glyphs(chunk_result.base_glyph),
                original_token_count=chunk_size,
                compressed_glyph_count=1,
                compression_ratio=chunk_result.encoded_context.compression_ratio if hasattr(chunk_result, 'encoded_context') else 1.0,
                semantic_hash=self._compute_semantic_hash(chunk_result.base_glyph) if hasattr(chunk_result, 'base_glyph') else 0,
            )
    
    def get_statistics(self) -> dict:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression performance metrics
        """
        sigma_stats = self._sigma.get_statistics()
        return {
            "compression_ratio": self._compression_ratio,
            "total_tokens_processed": sigma_stats.total_tokens_processed,
            "total_glyphs_produced": sigma_stats.total_glyphs_produced,
            "total_bytes_saved": sigma_stats.total_bytes_saved,
            "average_compression_ratio": sigma_stats.average_compression_ratio,
            "average_encoding_time_ms": sigma_stats.average_encoding_time_ms,
            "average_decoding_time_ms": sigma_stats.average_decoding_time_ms,
            "cache_hit_rate": sigma_stats.rsu_hit_rate,
            "average_fidelity": sigma_stats.average_semantic_fidelity,
        }
    
    def _compute_semantic_hash(self, context) -> int:
        """
        Compute semantic hash for RSU matching.
        
        Args:
            context: SigmaEncodedContext or similar
        
        Returns:
            Hash value for semantic matching
        """
        if hasattr(context, 'semantic_hash') and context.semantic_hash:
            return context.semantic_hash
        
        # Fallback: hash first 16 bytes of encoded data
        if hasattr(context, 'encoded_bytes'):
            hash_input = context.encoded_bytes[:16] if len(context.encoded_bytes) >= 16 else context.encoded_bytes
        else:
            hash_input = b'default'
        
        return int(hashlib.sha256(hash_input).hexdigest()[:16], 16)
    
    def _serialize_glyphs(self, context: SigmaEncodedContext) -> bytes:
        """
        Serialize glyph sequence to bytes.
        
        Args:
            context: SigmaEncodedContext with glyphs
        
        Returns:
            Serialized bytes representation
        """
        # Use the encoded_bytes directly if available
        if hasattr(context, 'encoded_bytes') and context.encoded_bytes:
            return context.encoded_bytes
        
        # Otherwise serialize glyph sequence
        if hasattr(context, 'glyph_sequence') and context.glyph_sequence:
            return context.glyph_sequence.to_bytes()
        
        # Fallback
        return b''
    
    def _deserialize_glyphs(
        self,
        data: bytes,
        original_token_count: int,
        compression_ratio: float
    ) -> SigmaEncodedContext:
        """
        Deserialize bytes back to SigmaEncodedContext.
        
        Args:
            data: Serialized glyph data
            original_token_count: Original token count
            compression_ratio: Compression ratio value
        
        Returns:
            Reconstructed SigmaEncodedContext
        """
        # Create minimal context for decoding
        # In a full implementation, we'd deserialize the actual glyph sequence
        from ..api.types import GlyphSequence, EncodedGlyph, SemanticGlyph, GlyphTier, CompressionQuality
        
        # Create a minimal glyph sequence from serialized data
        # For now, create empty glyphs that can be decoded
        empty_glyph = SemanticGlyph(
            glyph_id=0,
            tier=GlyphTier.DOMAIN,
            semantic_category="placeholder",
            base_meaning="deserialized",
            binary_code=b'\x00'
        )
        
        glyph_sequence = GlyphSequence(
            glyphs=tuple(),
            original_token_count=original_token_count,
            compressed_size_bytes=len(data),
            semantic_hash=self._compute_semantic_hash(None),
            codebook_version="1.0",
        )
        
        return SigmaEncodedContext(
            glyph_sequence=glyph_sequence,
            encoded_bytes=data,
            original_token_count=original_token_count,
            original_text_preview=f"[{original_token_count} tokens]",
            compression_ratio=compression_ratio,
            encoding_mode=self._default_mode,
            quality=CompressionQuality.BALANCED,
            semantic_hash=self._compute_semantic_hash(None),
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_ryot_compression_adapter(
    mode: str = "balanced",
) -> SigmaCompressionAdapter:
    """
    Factory function to create adapter with specified mode.
    
    Args:
        mode: Encoding mode ("fast", "balanced", "deep", "streaming")
    
    Returns:
        Configured SigmaCompressionAdapter
    """
    mode_map = {
        "fast": EncodingMode.FAST,
        "balanced": EncodingMode.BALANCED,
        "deep": EncodingMode.DEEP,
        "streaming": EncodingMode.STREAMING,
    }
    
    return SigmaCompressionAdapter(
        default_mode=mode_map.get(mode, EncodingMode.BALANCED)
    )
