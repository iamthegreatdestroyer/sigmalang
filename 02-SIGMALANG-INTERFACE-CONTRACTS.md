# COPILOT DIRECTIVE: ΣLANG Interface Contracts

## Mission Objective

You are implementing **Phase 0: Interface Contracts** for the ΣLANG (Sub-Linear Algorithmic Neural Glyph Language) project. This establishes the APIs that Ryot LLM, Neurectomy, and Elite Agents will use to access compression capabilities.

**CRITICAL:** ΣLANG is the compression layer. All context passing through Ryot LLM can optionally be compressed here first. These interfaces must be rock-solid.

---

## Project Context

**ΣLANG** is a semantic compression system that encodes meaning directly rather than compressing text, achieving 10-50x compression ratios through:
- 256 semantic primitives (glyphs)
- Learned codebook with pattern recognition
- RSU (Recyclable Semantic Unit) caching

**Current Status:** Core primitives and encoder exist, integration layer needed

**Integration Role:** ΣLANG sits between user input and Ryot LLM inference, compressing context for efficiency.

---

## Task Specification

### Task 1: Create Directory Structure

Create the following directory structure:

```
sigmalang/
├── api/
│   ├── __init__.py
│   ├── interfaces.py          # Protocol definitions
│   ├── types.py               # Shared type definitions
│   └── exceptions.py          # Custom exceptions
├── contracts/
│   ├── __init__.py
│   ├── compression_contract.py  # CompressionEngine protocol
│   ├── rsu_contract.py         # RSU management protocol
│   └── codebook_contract.py    # Codebook protocol
└── stubs/
    ├── __init__.py
    └── mock_sigma.py           # Mock implementation for testing
```

### Task 2: Create Core Type Definitions

**File: `sigmalang/api/types.py`**

```python
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
```

### Task 3: Create Core Interface Protocols

**File: `sigmalang/api/interfaces.py`**

```python
"""
ΣLANG Core Interface Protocols
==============================

These protocols define the contracts that Ryot LLM, Neurectomy,
and Elite Agents code against.

STABILITY GUARANTEE: These interfaces are versioned.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from .types import (
    CodebookMetadata,
    CodebookState,
    CompressionQuality,
    CompressionStatistics,
    DecodedContext,
    DecodingResult,
    EncodedGlyph,
    EncodingMode,
    EncodingResult,
    RSUChain,
    RSUEntry,
    RSUReference,
    SemanticGlyph,
    SigmaEncodedContext,
    StorageTier,
)


@runtime_checkable
class CompressionEngine(Protocol):
    """
    Core ΣLANG compression engine protocol.
    
    This is the PRIMARY integration point for Ryot LLM.
    """
    
    @abstractmethod
    def encode(
        self,
        tokens: List[int],
        mode: EncodingMode = EncodingMode.BALANCED,
        quality: CompressionQuality = CompressionQuality.HIGH,
        conversation_id: Optional[str] = None,
    ) -> EncodingResult:
        """Encode token sequence to ΣLANG representation."""
        ...
    
    @abstractmethod
    def decode(
        self,
        encoded: SigmaEncodedContext,
    ) -> DecodingResult:
        """Decode ΣLANG representation back to tokens."""
        ...
    
    @abstractmethod
    def encode_streaming(
        self,
        token_stream: Iterator[int],
        mode: EncodingMode = EncodingMode.STREAMING,
    ) -> Iterator[EncodedGlyph]:
        """Encode tokens in streaming mode for real-time processing."""
        ...
    
    @abstractmethod
    def get_compression_ratio(self) -> float:
        """Get average compression ratio from recent operations."""
        ...
    
    @abstractmethod
    def get_statistics(self) -> CompressionStatistics:
        """Get comprehensive compression statistics."""
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if ΣLANG is ready for encoding."""
        ...


@runtime_checkable
class RSUManager(Protocol):
    """RSU (Recyclable Semantic Unit) management protocol."""
    
    @abstractmethod
    def store(
        self,
        encoded: SigmaEncodedContext,
        kv_cache_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> RSUReference:
        """Store encoded context as a reusable RSU."""
        ...
    
    @abstractmethod
    def retrieve(
        self,
        reference: RSUReference,
    ) -> RSUEntry:
        """Retrieve RSU by reference."""
        ...
    
    @abstractmethod
    def find_by_hash(
        self,
        semantic_hash: int,
        similarity_threshold: float = 0.85,
    ) -> Optional[RSUReference]:
        """Find RSU by semantic hash (O(1) lookup)."""
        ...
    
    @abstractmethod
    def find_similar(
        self,
        encoded: SigmaEncodedContext,
        max_results: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Tuple[RSUReference, float]]:
        """Find similar RSUs by semantic similarity."""
        ...
    
    @abstractmethod
    def get_chain(
        self,
        conversation_id: str,
    ) -> RSUChain:
        """Get all RSUs in a conversation chain."""
        ...
    
    @abstractmethod
    def create_delta(
        self,
        new_encoded: SigmaEncodedContext,
        parent_reference: RSUReference,
    ) -> RSUReference:
        """Create delta-encoded RSU from parent."""
        ...
    
    @abstractmethod
    def promote_tier(
        self,
        reference: RSUReference,
        target_tier: StorageTier,
    ) -> bool:
        """Move RSU to different storage tier."""
        ...
    
    @abstractmethod
    def evict(
        self,
        reference: RSUReference,
    ) -> bool:
        """Remove RSU from storage."""
        ...
    
    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get RSU storage statistics."""
        ...


@runtime_checkable
class CodebookProtocol(Protocol):
    """ΣLANG codebook management protocol."""
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load codebook from file."""
        ...
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """Save codebook to file."""
        ...
    
    @abstractmethod
    def lookup_glyph(
        self,
        tokens: List[int],
    ) -> Optional[SemanticGlyph]:
        """Find glyph that represents given token sequence."""
        ...
    
    @abstractmethod
    def lookup_tokens(
        self,
        glyph_id: int,
    ) -> List[List[int]]:
        """Find token patterns associated with glyph."""
        ...
    
    @abstractmethod
    def add_learned_pattern(
        self,
        tokens: List[int],
        frequency: int = 1,
    ) -> Optional[int]:
        """Add a learned pattern to Tier 2."""
        ...
    
    @abstractmethod
    def get_metadata(self) -> CodebookMetadata:
        """Get codebook metadata."""
        ...
    
    @abstractmethod
    def export_state(self) -> CodebookState:
        """Export complete codebook state."""
        ...
    
    @abstractmethod
    def import_state(self, state: CodebookState) -> bool:
        """Import codebook state."""
        ...


@runtime_checkable
class StorageBackend(Protocol):
    """Storage backend protocol for RSU persistence. ΣVAULT implements this."""
    
    @abstractmethod
    def store(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store binary data with key."""
        ...
    
    @abstractmethod
    def retrieve(
        self,
        key: str,
    ) -> Optional[bytes]:
        """Retrieve data by key."""
        ...
    
    @abstractmethod
    def delete(
        self,
        key: str,
    ) -> bool:
        """Delete data by key."""
        ...
    
    @abstractmethod
    def exists(
        self,
        key: str,
    ) -> bool:
        """Check if key exists."""
        ...
    
    @abstractmethod
    def list_keys(
        self,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """List all keys with optional prefix filter."""
        ...


@runtime_checkable
class SigmaFactory(Protocol):
    """Factory for creating configured ΣLANG components."""
    
    @abstractmethod
    def create_engine(
        self,
        codebook_path: Optional[str] = None,
        enable_rsu: bool = True,
        storage_backend: Optional[StorageBackend] = None,
    ) -> CompressionEngine:
        """Create configured compression engine."""
        ...
    
    @abstractmethod
    def create_rsu_manager(
        self,
        storage_backend: Optional[StorageBackend] = None,
        max_hot_entries: int = 1000,
        max_warm_entries: int = 10000,
    ) -> RSUManager:
        """Create configured RSU manager."""
        ...
```

### Task 4: Create Custom Exceptions

**File: `sigmalang/api/exceptions.py`**

```python
"""
ΣLANG Custom Exceptions
=======================
"""

from typing import Optional


class SigmaError(Exception):
    """Base exception for all ΣLANG errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SIGMA_ERROR",
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.is_retryable = is_retryable


class CodebookNotLoadedError(SigmaError):
    """Raised when encoding attempted without loaded codebook."""
    
    def __init__(self, codebook_path: Optional[str] = None):
        message = "No codebook loaded"
        if codebook_path:
            message = f"Failed to load codebook from: {codebook_path}"
        super().__init__(message, "CODEBOOK_NOT_LOADED", is_retryable=True)


class EncodingError(SigmaError):
    """Raised when encoding fails."""
    
    def __init__(self, message: str, token_position: Optional[int] = None):
        super().__init__(message, "ENCODING_ERROR", is_retryable=True)
        self.token_position = token_position


class DecodingError(SigmaError):
    """Raised when decoding fails."""
    
    def __init__(self, message: str, glyph_position: Optional[int] = None):
        super().__init__(message, "DECODING_ERROR", is_retryable=False)
        self.glyph_position = glyph_position


class RSUNotFoundError(SigmaError):
    """Raised when RSU lookup fails."""
    
    def __init__(self, rsu_id: Optional[str] = None, semantic_hash: Optional[int] = None):
        if rsu_id:
            message = f"RSU not found: {rsu_id}"
        elif semantic_hash:
            message = f"No RSU for hash: {semantic_hash:016x}"
        else:
            message = "RSU not found"
        super().__init__(message, "RSU_NOT_FOUND", is_retryable=False)
        self.rsu_id = rsu_id
        self.semantic_hash = semantic_hash


class RSUStorageError(SigmaError):
    """Raised when RSU storage operations fail."""
    
    def __init__(self, message: str, operation: str = "unknown"):
        super().__init__(message, "RSU_STORAGE_ERROR", is_retryable=True)
        self.operation = operation


class SemanticHashCollisionError(SigmaError):
    """Raised when semantic hash collision detected."""
    
    def __init__(self, hash_value: int):
        message = f"Semantic hash collision: {hash_value:016x}"
        super().__init__(message, "HASH_COLLISION", is_retryable=False)
        self.hash_value = hash_value


class CompressionQualityError(SigmaError):
    """Raised when compression quality below threshold."""
    
    def __init__(self, achieved: float, required: float):
        message = f"Compression quality {achieved:.2%} below required {required:.2%}"
        super().__init__(message, "QUALITY_ERROR", is_retryable=True)
        self.achieved = achieved
        self.required = required
```

### Task 5: Create Mock Implementation

**File: `sigmalang/stubs/mock_sigma.py`**

```python
"""
Mock ΣLANG Implementation for Integration Testing
==================================================
"""

import hashlib
import time
from typing import Dict, Iterator, List, Optional, Tuple

from ..api.types import (
    CompressionQuality,
    CompressionStatistics,
    DecodedContext,
    DecodingResult,
    EncodedGlyph,
    EncodingMode,
    EncodingResult,
    GlyphSequence,
    GlyphTier,
    ProcessingMode,
    RSUChain,
    RSUEntry,
    RSUMetadata,
    RSUReference,
    SemanticGlyph,
    SigmaEncodedContext,
    StorageTier,
)
from ..api.interfaces import CompressionEngine, RSUManager


class MockCompressionEngine(CompressionEngine):
    """Mock compression engine for integration testing."""
    
    def __init__(
        self,
        mock_compression_ratio: float = 15.0,
        simulate_latency: bool = False,
    ):
        self._mock_ratio = mock_compression_ratio
        self._simulate_latency = simulate_latency
        self._total_tokens = 0
        self._total_glyphs = 0
        self._encode_count = 0
    
    def encode(
        self,
        tokens: List[int],
        mode: EncodingMode = EncodingMode.BALANCED,
        quality: CompressionQuality = CompressionQuality.HIGH,
        conversation_id: Optional[str] = None,
    ) -> EncodingResult:
        if self._simulate_latency:
            time.sleep(len(tokens) * 0.001)
        
        start_time = time.time()
        num_glyphs = max(1, len(tokens) // int(self._mock_ratio))
        
        mock_glyphs = tuple(
            EncodedGlyph(
                base_glyph=SemanticGlyph(
                    glyph_id=i % 256,
                    tier=GlyphTier.DOMAIN,
                    semantic_category="mock",
                    base_meaning=f"mock_glyph_{i}",
                    binary_code=bytes([i % 256]),
                ),
                position=i,
                source_token_span=(i * int(self._mock_ratio), (i + 1) * int(self._mock_ratio)),
            )
            for i in range(num_glyphs)
        )
        
        glyph_sequence = GlyphSequence(
            glyphs=mock_glyphs,
            original_token_count=len(tokens),
            compressed_size_bytes=num_glyphs * 2,
            semantic_hash=self._compute_hash(tokens),
            codebook_version="mock-1.0",
        )
        
        encoded_context = SigmaEncodedContext(
            glyph_sequence=glyph_sequence,
            encoded_bytes=glyph_sequence.to_bytes(),
            original_token_count=len(tokens),
            original_text_preview=f"[{len(tokens)} tokens]",
            compression_ratio=len(tokens) / max(1, num_glyphs),
            encoding_mode=mode,
            quality=quality,
            semantic_hash=glyph_sequence.semantic_hash,
        )
        
        self._total_tokens += len(tokens)
        self._total_glyphs += num_glyphs
        self._encode_count += 1
        
        return EncodingResult(
            encoded_context=encoded_context,
            processing_mode=ProcessingMode.FRESH_ENCODE,
            encoding_time_ms=(time.time() - start_time) * 1000,
            rsu_created=False,
        )
    
    def decode(
        self,
        encoded: SigmaEncodedContext,
    ) -> DecodingResult:
        start_time = time.time()
        
        if self._simulate_latency:
            time.sleep(encoded.glyph_count * 0.0005)
        
        decoded_tokens = list(range(encoded.original_token_count))
        
        decoded_context = DecodedContext(
            tokens=decoded_tokens,
            semantic_fidelity=0.99,
            reconstruction_exact=True,
            source_hash=encoded.semantic_hash,
        )
        
        return DecodingResult(
            decoded_context=decoded_context,
            decoding_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            semantic_fidelity=0.99,
        )
    
    def encode_streaming(
        self,
        token_stream: Iterator[int],
        mode: EncodingMode = EncodingMode.STREAMING,
    ) -> Iterator[EncodedGlyph]:
        buffer = []
        glyph_count = 0
        
        for token in token_stream:
            buffer.append(token)
            
            if len(buffer) >= int(self._mock_ratio):
                yield EncodedGlyph(
                    base_glyph=SemanticGlyph(
                        glyph_id=glyph_count % 256,
                        tier=GlyphTier.DOMAIN,
                        semantic_category="mock",
                        base_meaning=f"streaming_glyph_{glyph_count}",
                        binary_code=bytes([glyph_count % 256]),
                    ),
                    position=glyph_count,
                )
                glyph_count += 1
                buffer = []
        
        if buffer:
            yield EncodedGlyph(
                base_glyph=SemanticGlyph(
                    glyph_id=glyph_count % 256,
                    tier=GlyphTier.DOMAIN,
                    semantic_category="mock",
                    base_meaning=f"streaming_glyph_{glyph_count}",
                    binary_code=bytes([glyph_count % 256]),
                ),
                position=glyph_count,
            )
    
    def get_compression_ratio(self) -> float:
        if self._total_glyphs == 0:
            return self._mock_ratio
        return self._total_tokens / self._total_glyphs
    
    def get_statistics(self) -> CompressionStatistics:
        return CompressionStatistics(
            total_tokens_processed=self._total_tokens,
            total_glyphs_produced=self._total_glyphs,
            total_bytes_saved=self._total_tokens * 4 - self._total_glyphs * 2,
            average_compression_ratio=self.get_compression_ratio(),
            average_encoding_time_ms=1.0,
            average_decoding_time_ms=0.5,
            rsu_cache_hits=0,
            rsu_cache_misses=self._encode_count,
            rsu_hit_rate=0.0,
            average_semantic_fidelity=0.99,
            lossless_ratio=1.0,
        )
    
    def is_available(self) -> bool:
        return True
    
    def _compute_hash(self, tokens: List[int]) -> int:
        data = bytes(t % 256 for t in tokens[:100])
        return int(hashlib.sha256(data).hexdigest()[:16], 16)


class MockRSUManager(RSUManager):
    """Mock RSU manager for integration testing."""
    
    def __init__(self):
        self._storage: Dict[str, RSUEntry] = {}
        self._hash_index: Dict[int, str] = {}
        self._chains: Dict[str, List[str]] = {}
        self._access_count = 0
        self._hit_count = 0
    
    def store(
        self,
        encoded: SigmaEncodedContext,
        kv_cache_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> RSUReference:
        rsu_id = f"rsu_{encoded.semantic_hash:016x}_{int(time.time())}"
        
        metadata = RSUMetadata(
            rsu_id=rsu_id,
            semantic_hash=encoded.semantic_hash,
            token_count=encoded.original_token_count,
            glyph_count=encoded.glyph_count,
            compression_ratio=encoded.compression_ratio,
            storage_tier=StorageTier.HOT,
            created_timestamp=time.time(),
            last_access_timestamp=time.time(),
            conversation_id=conversation_id,
        )
        
        entry = RSUEntry(
            metadata=metadata,
            encoded_context=encoded,
            has_kv_cache=kv_cache_id is not None,
            kv_cache_id=kv_cache_id,
        )
        
        self._storage[rsu_id] = entry
        self._hash_index[encoded.semantic_hash] = rsu_id
        
        if conversation_id:
            if conversation_id not in self._chains:
                self._chains[conversation_id] = []
            self._chains[conversation_id].append(rsu_id)
        
        return RSUReference(
            rsu_id=rsu_id,
            semantic_hash=encoded.semantic_hash,
            storage_tier=StorageTier.HOT,
            compression_ratio=encoded.compression_ratio,
            has_kv_cache=entry.has_kv_cache,
        )
    
    def retrieve(self, reference: RSUReference) -> RSUEntry:
        self._access_count += 1
        if reference.rsu_id not in self._storage:
            from ..api.exceptions import RSUNotFoundError
            raise RSUNotFoundError(rsu_id=reference.rsu_id)
        
        entry = self._storage[reference.rsu_id]
        entry.metadata.last_access_timestamp = time.time()
        entry.metadata.access_count += 1
        self._hit_count += 1
        return entry
    
    def find_by_hash(
        self,
        semantic_hash: int,
        similarity_threshold: float = 0.85,
    ) -> Optional[RSUReference]:
        self._access_count += 1
        if semantic_hash in self._hash_index:
            rsu_id = self._hash_index[semantic_hash]
            entry = self._storage[rsu_id]
            self._hit_count += 1
            return RSUReference(
                rsu_id=rsu_id,
                semantic_hash=semantic_hash,
                storage_tier=entry.metadata.storage_tier,
                compression_ratio=entry.metadata.compression_ratio,
                has_kv_cache=entry.has_kv_cache,
            )
        return None
    
    def find_similar(
        self,
        encoded: SigmaEncodedContext,
        max_results: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Tuple[RSUReference, float]]:
        exact = self.find_by_hash(encoded.semantic_hash)
        if exact:
            return [(exact, 1.0)]
        return []
    
    def get_chain(self, conversation_id: str) -> RSUChain:
        rsu_ids = self._chains.get(conversation_id, [])
        entries = []
        total_tokens = 0
        total_bytes = 0
        
        for rsu_id in rsu_ids:
            if rsu_id in self._storage:
                entry = self._storage[rsu_id]
                entries.append(RSUReference(
                    rsu_id=rsu_id,
                    semantic_hash=entry.metadata.semantic_hash,
                    storage_tier=entry.metadata.storage_tier,
                    compression_ratio=entry.metadata.compression_ratio,
                    has_kv_cache=entry.has_kv_cache,
                ))
                total_tokens += entry.metadata.token_count
                total_bytes += entry.encoded_context.glyph_count * 2
        
        return RSUChain(
            conversation_id=conversation_id,
            entries=entries,
            total_tokens=total_tokens,
            total_compressed_bytes=total_bytes,
        )
    
    def create_delta(
        self,
        new_encoded: SigmaEncodedContext,
        parent_reference: RSUReference,
    ) -> RSUReference:
        ref = self.store(new_encoded)
        entry = self._storage[ref.rsu_id]
        entry.metadata.parent_rsu_id = parent_reference.rsu_id
        return ref
    
    def promote_tier(
        self,
        reference: RSUReference,
        target_tier: StorageTier,
    ) -> bool:
        if reference.rsu_id in self._storage:
            self._storage[reference.rsu_id].metadata.storage_tier = target_tier
            return True
        return False
    
    def evict(self, reference: RSUReference) -> bool:
        if reference.rsu_id in self._storage:
            entry = self._storage.pop(reference.rsu_id)
            self._hash_index.pop(entry.metadata.semantic_hash, None)
            return True
        return False
    
    def get_statistics(self) -> Dict:
        return {
            "total_rsus": len(self._storage),
            "total_chains": len(self._chains),
            "access_count": self._access_count,
            "hit_count": self._hit_count,
            "hit_rate": self._hit_count / max(1, self._access_count),
        }
```

### Task 6: Create Package Init Files

**File: `sigmalang/api/__init__.py`**

```python
"""
ΣLANG Public API
================
"""

from .interfaces import (
    CompressionEngine,
    RSUManager,
    CodebookProtocol,
    StorageBackend,
    SigmaFactory,
)

from .types import (
    GlyphTier, EncodingMode, ProcessingMode, StorageTier, CompressionQuality,
    SemanticGlyph, GlyphModifier, EncodedGlyph, GlyphSequence,
    SigmaEncodedContext, DecodedContext,
    RSUMetadata, RSUEntry, RSUReference, RSUChain,
    EncodingResult, DecodingResult, CompressionStatistics,
    CodebookEntry, CodebookMetadata, CodebookState,
)

from .exceptions import (
    SigmaError, CodebookNotLoadedError, EncodingError, DecodingError,
    RSUNotFoundError, RSUStorageError, SemanticHashCollisionError, CompressionQualityError,
)

__all__ = [
    "CompressionEngine", "RSUManager", "CodebookProtocol", "StorageBackend", "SigmaFactory",
    "GlyphTier", "EncodingMode", "ProcessingMode", "StorageTier", "CompressionQuality",
    "SemanticGlyph", "GlyphModifier", "EncodedGlyph", "GlyphSequence",
    "SigmaEncodedContext", "DecodedContext",
    "RSUMetadata", "RSUEntry", "RSUReference", "RSUChain",
    "EncodingResult", "DecodingResult", "CompressionStatistics",
    "CodebookEntry", "CodebookMetadata", "CodebookState",
    "SigmaError", "CodebookNotLoadedError", "EncodingError", "DecodingError",
    "RSUNotFoundError", "RSUStorageError", "SemanticHashCollisionError", "CompressionQualityError",
]

__version__ = "0.1.0"
```

**File: `sigmalang/stubs/__init__.py`**

```python
"""ΣLANG Stubs for Integration Testing"""

from .mock_sigma import MockCompressionEngine, MockRSUManager

__all__ = ["MockCompressionEngine", "MockRSUManager"]
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] All files created in correct locations
- [ ] Type hints are correct (run `mypy sigmalang/api/`)
- [ ] Mock engine works:

```python
from sigmalang.api import CompressionEngine, EncodingMode
from sigmalang.stubs import MockCompressionEngine, MockRSUManager

sigma: CompressionEngine = MockCompressionEngine()
assert sigma.is_available()

result = sigma.encode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mode=EncodingMode.BALANCED)
assert result.encoded_context.compression_ratio > 1.0

rsu_mgr = MockRSUManager()
ref = rsu_mgr.store(result.encoded_context)
assert rsu_mgr.find_by_hash(result.encoded_context.semantic_hash) is not None

print("✓ ΣLANG interface contracts verified")
```

---

**END OF DIRECTIVE**
