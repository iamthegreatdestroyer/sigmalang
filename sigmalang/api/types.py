"""
ΣLANG Core Type Definitions
===========================

Shared types used across all ΣLANG interfaces.
These types define the data structures for integration with:
- Ryot LLM (inference engine)
- ΣVAULT (secure storage)
- Neurectomy (IDE orchestration)
- Elite Agent Collective (multi-agent system)

STABILITY: These types are part of the public API contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# ENUMERATIONS
# =============================================================================

class GlyphTier(Enum):
    """Semantic primitive tiers in the ΣLANG hierarchy."""
    EXISTENTIAL = auto()   # Tier 0: Universal concepts (being, causation, relation)
    DOMAIN = auto()        # Tier 1: Domain-specific (code, math, language)
    LEARNED = auto()       # Tier 2: User-specific patterns from training


class EncodingMode(Enum):
    """ΣLANG encoding strategies."""
    FAST = auto()          # Quick encoding, lower compression
    BALANCED = auto()      # Default balance of speed/compression
    DEEP = auto()          # Maximum compression, slower
    STREAMING = auto()     # Token-by-token for real-time


class ProcessingMode(Enum):
    """RSU processing outcomes."""
    FAST_PATH = auto()       # Exact hash match, O(1) lookup
    EXACT_HIT = auto()       # Content-identical RSU found
    APPROXIMATE_HIT = auto() # Similar RSU found, delta encoding
    DELTA_CHAIN = auto()     # Part of conversation chain
    FRESH_ENCODE = auto()    # No match, full encoding required


class StorageTier(Enum):
    """RSU storage tiers for lifecycle management."""
    HOT = auto()    # In-memory, instant access
    WARM = auto()   # On-disk SSD, fast access
    COLD = auto()   # External/ΣVAULT, slower access


class CompressionQuality(Enum):
    """Quality levels for compression/decompression."""
    LOSSLESS = auto()      # Perfect reconstruction
    HIGH = auto()          # 99%+ semantic fidelity
    BALANCED = auto()      # 95%+ semantic fidelity
    AGGRESSIVE = auto()    # 90%+ semantic fidelity, max compression


# =============================================================================
# CORE GLYPH STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class SemanticGlyph:
    """
    A single semantic primitive in ΣLANG.
    
    Glyphs are the atomic units of meaning in ΣLANG encoding.
    Each glyph represents a semantic concept that can be combined
    with modifiers to express complex meanings.
    """
    glyph_id: int              # 0-255 for base glyphs
    tier: GlyphTier
    
    # Semantic properties
    semantic_category: str     # e.g., "entity", "action", "relation"
    base_meaning: str          # Human-readable meaning
    
    # Encoding properties
    binary_code: bytes         # Binary representation
    embedding: Optional[NDArray[np.float32]] = None  # Semantic embedding
    
    def __hash__(self) -> int:
        return hash((self.glyph_id, self.tier))


@dataclass(frozen=True)
class GlyphModifier:
    """
    Modifier that adjusts glyph meaning.
    
    Modifiers allow a small glyph vocabulary to express
    nuanced meanings through composition.
    """
    modifier_id: int           # 0-63 for modifiers
    modifier_type: str         # e.g., "intensity", "negation", "temporal"
    
    # Effect on meaning
    semantic_shift: str        # Description of modification
    weight: float = 1.0        # Strength of modification


@dataclass
class EncodedGlyph:
    """
    A glyph with applied modifiers, ready for serialization.
    """
    base_glyph: SemanticGlyph
    modifiers: Tuple[GlyphModifier, ...] = ()
    
    # Position in sequence
    position: int = 0
    
    # Link to original tokens
    source_token_span: Tuple[int, int] = (0, 0)  # (start, end) in original
    
    def to_bytes(self) -> bytes:
        """Serialize to binary representation."""
        result = bytes([self.base_glyph.glyph_id, len(self.modifiers)])
        for mod in self.modifiers:
            result += bytes([mod.modifier_id])
        return result


# =============================================================================
# SEQUENCE AND CONTEXT STRUCTURES
# =============================================================================

@dataclass
class GlyphSequence:
    """
    A sequence of encoded glyphs representing compressed content.
    """
    glyphs: Tuple[EncodedGlyph, ...]
    
    # Compression metadata
    original_token_count: int
    compressed_size_bytes: int
    
    # For reconstruction
    semantic_hash: int
    codebook_version: str
    
    @property
    def compression_ratio(self) -> float:
        if self.compressed_size_bytes == 0:
            return 0.0
        original_bytes = self.original_token_count * 4
        return original_bytes / self.compressed_size_bytes
    
    def to_bytes(self) -> bytes:
        result = b''
        for glyph in self.glyphs:
            result += glyph.to_bytes()
        return result


@dataclass
class SigmaEncodedContext:
    """
    Complete ΣLANG-encoded context ready for Ryot LLM.
    
    This is the bridge structure between ΣLANG and Ryot LLM.
    """
    glyph_sequence: GlyphSequence
    encoded_bytes: bytes
    original_token_count: int
    original_text_preview: str
    compression_ratio: float
    encoding_mode: EncodingMode
    quality: CompressionQuality
    semantic_hash: int
    is_delta_encoded: bool = False
    parent_rsu_id: Optional[str] = None
    _cached_tokens: Optional[List[int]] = field(default=None, repr=False)
    
    @property
    def glyph_count(self) -> int:
        return len(self.glyph_sequence.glyphs)


@dataclass
class DecodedContext:
    """Result of decoding ΣLANG back to tokens."""
    tokens: List[int]
    semantic_fidelity: float
    reconstruction_exact: bool
    source_hash: int
    source_rsu_id: Optional[str] = None


# =============================================================================
# RSU STRUCTURES
# =============================================================================

@dataclass
class RSUMetadata:
    """Metadata for a Recyclable Semantic Unit."""
    rsu_id: str
    semantic_hash: int
    token_count: int
    glyph_count: int
    compression_ratio: float
    storage_tier: StorageTier
    storage_path: Optional[str] = None
    created_timestamp: float = 0.0
    last_access_timestamp: float = 0.0
    access_count: int = 0
    conversation_id: Optional[str] = None
    chain_position: int = 0
    parent_rsu_id: Optional[str] = None


@dataclass
class RSUEntry:
    """Complete RSU entry with encoded content."""
    metadata: RSUMetadata
    encoded_context: SigmaEncodedContext
    has_kv_cache: bool = False
    kv_cache_id: Optional[str] = None
    delta_from_parent: Optional[bytes] = None


@dataclass
class RSUReference:
    """Lightweight reference to an RSU for lookups."""
    rsu_id: str
    semantic_hash: int
    storage_tier: StorageTier
    compression_ratio: float
    has_kv_cache: bool
    
    def __hash__(self) -> int:
        return hash(self.rsu_id)


@dataclass
class RSUChain:
    """A chain of RSUs representing conversation history."""
    conversation_id: str
    entries: List[RSUReference]
    total_tokens: int
    total_compressed_bytes: int
    
    @property
    def chain_compression_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 0.0
        return (self.total_tokens * 4) / self.total_compressed_bytes


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class EncodingResult:
    """Result of encoding tokens through ΣLANG."""
    encoded_context: SigmaEncodedContext
    processing_mode: ProcessingMode
    encoding_time_ms: float
    rsu_created: bool
    rsu_reference: Optional[RSUReference] = None
    delta_size_bytes: Optional[int] = None
    parent_rsu_reference: Optional[RSUReference] = None


@dataclass
class DecodingResult:
    """Result of decoding ΣLANG back to tokens."""
    decoded_context: DecodedContext
    decoding_time_ms: float
    cache_hit: bool
    semantic_fidelity: float


@dataclass
class CompressionStatistics:
    """Statistics about compression performance."""
    total_tokens_processed: int
    total_glyphs_produced: int
    total_bytes_saved: int
    average_compression_ratio: float
    average_encoding_time_ms: float
    average_decoding_time_ms: float
    rsu_cache_hits: int
    rsu_cache_misses: int
    rsu_hit_rate: float
    average_semantic_fidelity: float
    lossless_ratio: float


# =============================================================================
# CODEBOOK STRUCTURES
# =============================================================================

@dataclass
class CodebookEntry:
    """Single entry in the ΣLANG codebook."""
    glyph: SemanticGlyph
    frequency: int
    confidence: float
    common_token_patterns: List[Tuple[int, ...]]


@dataclass
class CodebookMetadata:
    """Metadata about the loaded codebook."""
    version: str
    created_timestamp: float
    num_base_glyphs: int
    num_modifiers: int
    num_learned_patterns: int
    training_corpus_size: int
    training_domain: str
    expected_compression_ratio: float
    semantic_coverage: float


@dataclass
class CodebookState:
    """Complete codebook state for serialization."""
    metadata: CodebookMetadata
    base_glyphs: Dict[int, SemanticGlyph]
    modifiers: Dict[int, GlyphModifier]
    learned_patterns: Dict[int, CodebookEntry]
    token_glyph_map: Dict[Tuple[int, ...], int]
    glyph_embeddings: Optional[NDArray[np.float32]] = None
