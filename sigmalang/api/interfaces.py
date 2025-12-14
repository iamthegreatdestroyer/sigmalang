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
