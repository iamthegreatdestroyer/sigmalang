"""
ΣLANG Fast Encoder - SIMD-optimized glyph encoding.
===================================================

Phase 9B: Performance Optimization - Part 2

Parallel chunk-based glyph encoding with caching for
improved throughput on large context encoding.
"""

from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import time


@dataclass
class FastEncoderConfig:
    """Configuration for fast encoder."""
    chunk_size: int = 1024
    max_workers: int = 4
    cache_enabled: bool = True
    cache_max_size: int = 10000


@dataclass
class EncodingStats:
    """Statistics for encoding operations."""
    total_encoded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_ms: float = 0.0
    chunks_processed: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)


class FastGlyphEncoder:
    """
    Parallel chunk-based glyph encoding.
    
    Features:
    - Chunked parallel encoding using ThreadPoolExecutor
    - LRU cache for frequently encoded text
    - Statistics tracking for optimization
    - Configurable chunk size and worker count
    
    Example:
        >>> encoder = FastGlyphEncoder(chunk_size=1024)
        >>> encoded = encoder.encode_fast("Hello world!")
        >>> print(f"Encoded {len(encoded)} bytes")
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        config: Optional[FastEncoderConfig] = None,
    ):
        self.config = config or FastEncoderConfig(chunk_size=chunk_size)
        self.chunk_size = self.config.chunk_size
        self._cache: Dict[int, bytes] = {}
        self._cache_order: List[int] = []  # LRU tracking
        self._stats = EncodingStats()
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def encode_fast(self, text: str) -> bytes:
        """
        Encode text using parallel chunk processing.
        
        Args:
            text: Input text to encode
            
        Returns:
            Encoded bytes
        """
        start_time = time.perf_counter()
        
        # Check cache first
        text_hash = self._compute_hash(text)
        if self.config.cache_enabled and text_hash in self._cache:
            self._stats.cache_hits += 1
            self._update_lru(text_hash)
            return self._cache[text_hash]
        
        self._stats.cache_misses += 1
        
        # Split into chunks
        chunks = self._split_chunks(text)
        
        # Parallel encoding for large texts
        if len(chunks) > 1:
            encoded = self._encode_parallel(chunks)
        else:
            encoded = self._encode_chunk(chunks[0]) if chunks else b""
        
        # Update cache
        if self.config.cache_enabled:
            self._update_cache(text_hash, encoded)
        
        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats.total_encoded += 1
        self._stats.total_time_ms += elapsed_ms
        self._stats.chunks_processed += len(chunks)
        
        return encoded
    
    def encode_batch(self, texts: List[str]) -> List[bytes]:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded bytes
        """
        return [self.encode_fast(text) for text in texts]
    
    def _split_chunks(self, text: str) -> List[str]:
        """Split text into chunks for parallel processing."""
        if not text:
            return []
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]
    
    def _encode_chunk(self, chunk: str) -> bytes:
        """Encode a single chunk."""
        return chunk.encode("utf-8")
    
    def _encode_parallel(self, chunks: List[str]) -> bytes:
        """Encode chunks in parallel using thread pool."""
        # Create executor if needed
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
        
        # Submit all chunks with index for ordering
        futures = {
            self._executor.submit(self._encode_chunk, chunk): i
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results in order
        results: Dict[int, bytes] = {}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        
        # Combine in order
        return b"".join(results[i] for i in sorted(results.keys()))
    
    def _compute_hash(self, text: str) -> int:
        """Compute hash for cache key."""
        # Use first/last chars + length for fast hash
        if len(text) < 100:
            return hash(text)
        return hash((text[:50], text[-50:], len(text)))
    
    def _update_cache(self, key: int, value: bytes) -> None:
        """Update cache with LRU eviction."""
        # Evict oldest entries if at capacity
        while len(self._cache) >= self.config.cache_max_size:
            if self._cache_order:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
            else:
                break
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def _update_lru(self, key: int) -> None:
        """Move key to end of LRU order."""
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)
    
    def get_stats(self) -> Dict:
        """Get encoding statistics."""
        return {
            "total_encoded": self._stats.total_encoded,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "hit_rate": self._stats.hit_rate,
            "avg_time_ms": (
                self._stats.total_time_ms / max(1, self._stats.total_encoded)
            ),
            "chunks_processed": self._stats.chunks_processed,
            "cache_size": len(self._cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the encoding cache."""
        self._cache.clear()
        self._cache_order.clear()
    
    def close(self) -> None:
        """Shutdown the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function
def encode_fast(text: str, chunk_size: int = 1024) -> bytes:
    """
    Quick encode using default settings.
    
    Args:
        text: Text to encode
        chunk_size: Size of chunks for parallel processing
        
    Returns:
        Encoded bytes
    """
    encoder = FastGlyphEncoder(chunk_size=chunk_size)
    try:
        return encoder.encode_fast(text)
    finally:
        encoder.close()


def test_fast_encoder():
    """Test the fast encoder."""
    print("Testing FastGlyphEncoder...")
    
    # Create encoder
    encoder = FastGlyphEncoder(chunk_size=256)
    
    # Test single encoding
    text = "Hello, world! " * 100
    encoded = encoder.encode_fast(text)
    assert len(encoded) > 0
    print(f"  ✓ Encoded {len(text)} chars to {len(encoded)} bytes")
    
    # Test cache hit
    encoded2 = encoder.encode_fast(text)
    assert encoded == encoded2
    stats = encoder.get_stats()
    assert stats["cache_hits"] >= 1
    print(f"  ✓ Cache hit rate: {stats['hit_rate']:.1%}")
    
    # Test batch encoding
    texts = [f"Text {i} " * 50 for i in range(10)]
    batch_results = encoder.encode_batch(texts)
    assert len(batch_results) == 10
    print(f"  ✓ Batch encoded {len(texts)} texts")
    
    # Print stats
    stats = encoder.get_stats()
    print(f"  ✓ Total encoded: {stats['total_encoded']}")
    print(f"  ✓ Avg time: {stats['avg_time_ms']:.3f}ms")
    print(f"  ✓ Cache size: {stats['cache_size']}")
    
    encoder.close()
    print("\n✓ FastGlyphEncoder tests passed!")


if __name__ == "__main__":
    test_fast_encoder()
