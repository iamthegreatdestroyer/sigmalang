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
