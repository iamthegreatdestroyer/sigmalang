"""
Context Window Extension - Phase 4 Task 4.3

Extends effective context window from 200K to 2M+ tokens using SigmaLang
compressed context injection. Compresses older context segments while
preserving recent context in full fidelity.

Architecture:
    Full Context (200K limit)
    |
    +-- Recent Context (uncompressed, high fidelity)
    |   |-- Last N turns of conversation
    |   |-- Current working files
    |
    +-- Compressed Context (SigmaLang encoded)
    |   |-- Older conversation turns
    |   |-- Reference documents
    |   |-- Background knowledge
    |
    +-- Index Layer
        |-- Semantic search over compressed segments
        |-- Priority-based retrieval
        |-- Dynamic expansion on demand

Expected Impact: 200K -> 2M+ effective tokens (10x extension)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
import logging

sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ContextExtensionConfig:
    """Configuration for context window extension."""

    # Window sizes (in tokens, ~4 chars per token)
    max_native_tokens: int = 200_000  # Native context limit
    recent_context_tokens: int = 50_000  # Reserved for recent/uncompressed
    compressed_context_tokens: int = 150_000  # Available for compressed
    target_effective_tokens: int = 2_000_000  # Target effective capacity

    # Compression settings
    compression_ratio_estimate: float = 10.0  # Expected compression ratio
    chunk_size_tokens: int = 2000  # Tokens per chunk for compression

    # Retrieval
    max_retrieval_chunks: int = 20  # Max chunks to retrieve per query
    relevance_threshold: float = 0.1  # Min relevance score for retrieval


class ContextPriority(Enum):
    """Priority levels for context segments."""

    CRITICAL = 4  # Always keep uncompressed
    HIGH = 3  # Compress last
    MEDIUM = 2  # Standard compression
    LOW = 1  # Compress aggressively
    ARCHIVE = 0  # Deep archive, retrieve on demand


# =============================================================================
# Context Segment Models
# =============================================================================

@dataclass
class ContextSegment:
    """A segment of context (compressed or uncompressed)."""

    segment_id: str
    content: str  # Original content (empty if purged)
    compressed: Optional[bytes] = None  # SigmaLang encoded
    priority: ContextPriority = ContextPriority.MEDIUM
    token_count: int = 0
    compressed_token_count: int = 0  # Tokens when compressed
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source: str = ""  # e.g., "conversation", "document", "search_result"
    tags: List[str] = field(default_factory=list)

    @property
    def is_compressed(self) -> bool:
        return self.compressed is not None

    @property
    def effective_tokens(self) -> int:
        """Get effective token count (original size)."""
        return self.token_count

    @property
    def actual_tokens(self) -> int:
        """Get actual token usage in context window."""
        if self.is_compressed:
            return self.compressed_token_count
        return self.token_count

    @property
    def compression_ratio(self) -> float:
        if self.compressed_token_count == 0:
            return 1.0
        return self.token_count / max(1, self.compressed_token_count)


# =============================================================================
# Context Manager
# =============================================================================

class ContextWindowExtender:
    """
    Manages extended context window using SigmaLang compression.

    Automatically compresses older context segments to fit more information
    into the native context window while preserving semantic content.
    """

    def __init__(self, config: Optional[ContextExtensionConfig] = None):
        self.config = config or ContextExtensionConfig()

        # Segment storage
        self.segments: Dict[str, ContextSegment] = {}
        self.segment_order: deque = deque()  # Ordered by creation time

        # Token accounting
        self.uncompressed_tokens = 0
        self.compressed_tokens = 0

        # Search index (term -> segment_ids)
        self._term_index: Dict[str, List[str]] = {}

        # Encoder (lazy init)
        self._encoder_initialized = False
        self._parser = None
        self._encoder = None

        self.stats = {
            'segments_added': 0,
            'segments_compressed': 0,
            'segments_retrieved': 0,
            'total_effective_tokens': 0,
            'compressions_performed': 0
        }

    def _ensure_encoder(self) -> None:
        """Lazy initialize the SigmaLang encoder."""
        if self._encoder_initialized:
            return

        try:
            from sigmalang.core.parser import SemanticParser
            from sigmalang.core.encoder import SigmaEncoder

            self._parser = SemanticParser()
            self._encoder = SigmaEncoder()
            self._encoder_initialized = True
        except Exception as e:
            logger.warning(f"SigmaLang encoder not available: {e}")

    def add_context(
        self,
        content: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
        source: str = "conversation",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a new context segment.

        Args:
            content: Text content to add
            priority: Priority level
            source: Source identifier
            tags: Optional tags for search

        Returns:
            Segment ID
        """
        # Estimate token count (~4 chars per token)
        token_count = len(content) // 4

        segment_id = hashlib.sha256(
            (content[:200] + str(time.time())).encode()
        ).hexdigest()[:12]

        segment = ContextSegment(
            segment_id=segment_id,
            content=content,
            priority=priority,
            token_count=token_count,
            source=source,
            tags=tags or []
        )

        # Add to storage
        self.segments[segment_id] = segment
        self.segment_order.append(segment_id)
        self.uncompressed_tokens += token_count

        # Update index
        self._index_segment(segment)

        # Update stats
        self.stats['segments_added'] += 1
        self.stats['total_effective_tokens'] += token_count

        # Check if we need to compress
        self._maybe_compress()

        return segment_id

    def get_context_window(self) -> List[Dict[str, Any]]:
        """
        Get the current context window contents.

        Returns segments ordered by priority and recency,
        with compressed segments providing summaries.
        """
        window = []

        # Sort segments: CRITICAL first, then by recency
        sorted_segments = sorted(
            self.segments.values(),
            key=lambda s: (s.priority.value, s.last_accessed),
            reverse=True
        )

        for segment in sorted_segments:
            entry = {
                'segment_id': segment.segment_id,
                'source': segment.source,
                'priority': segment.priority.name,
                'is_compressed': segment.is_compressed,
                'effective_tokens': segment.effective_tokens,
                'actual_tokens': segment.actual_tokens,
            }

            if not segment.is_compressed:
                entry['content'] = segment.content
            else:
                entry['content'] = f"[Compressed: {segment.token_count} tokens, "
                entry['content'] += f"{segment.compression_ratio:.1f}x ratio]"

            window.append(entry)

        return window

    def retrieve(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context segments for a query.

        Searches both compressed and uncompressed segments,
        decompressing on demand.
        """
        results = []
        query_terms = set(query.lower().split())

        # Score each segment by term overlap
        scored = []
        for seg_id, segment in self.segments.items():
            score = self._score_segment(segment, query_terms)
            if score > self.config.relevance_threshold:
                scored.append((segment, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        for segment, score in scored[:max_results]:
            # Decompress if needed
            content = segment.content
            if segment.is_compressed and not content:
                content = self._decompress_segment(segment)

            segment.last_accessed = time.time()
            segment.access_count += 1
            self.stats['segments_retrieved'] += 1

            results.append({
                'segment_id': segment.segment_id,
                'content': content,
                'score': round(score, 4),
                'source': segment.source,
                'priority': segment.priority.name,
                'was_compressed': segment.is_compressed
            })

        return results

    def _score_segment(self, segment: ContextSegment, query_terms: set) -> float:
        """Score a segment against query terms."""
        # Use content for uncompressed, tags/source for compressed
        if not segment.is_compressed and segment.content:
            content_terms = set(segment.content.lower().split())
            overlap = len(query_terms & content_terms)
            return overlap / max(1, len(query_terms))

        # For compressed segments, use tags and source
        tag_terms = set(t.lower() for t in segment.tags)
        overlap = len(query_terms & tag_terms)
        return overlap / max(1, len(query_terms)) * 0.5  # Lower weight

    def _maybe_compress(self) -> None:
        """Compress segments if context window is over budget."""
        total_actual = self.uncompressed_tokens + self.compressed_tokens

        if total_actual <= self.config.max_native_tokens:
            return  # Within budget

        # Need to compress. Find candidates (oldest, lowest priority first)
        candidates = sorted(
            [s for s in self.segments.values() if not s.is_compressed],
            key=lambda s: (s.priority.value, -s.created_at)
        )

        for segment in candidates:
            if total_actual <= self.config.max_native_tokens:
                break

            if segment.priority == ContextPriority.CRITICAL:
                continue  # Never compress critical segments

            self._compress_segment(segment)
            total_actual = self.uncompressed_tokens + self.compressed_tokens

    def _compress_segment(self, segment: ContextSegment) -> bool:
        """Compress a single segment using SigmaLang."""
        self._ensure_encoder()

        if not self._encoder_initialized:
            # Fallback: just truncate
            segment.compressed = segment.content[:100].encode('utf-8')
            segment.compressed_token_count = 25
            self.uncompressed_tokens -= segment.token_count
            self.compressed_tokens += segment.compressed_token_count
            self.stats['segments_compressed'] += 1
            return True

        try:
            tree = self._parser.parse(segment.content)
            encoded = self._encoder.encode(tree)

            segment.compressed = encoded
            segment.compressed_token_count = len(encoded) // 4  # Estimate

            # Update token accounting
            self.uncompressed_tokens -= segment.token_count
            self.compressed_tokens += segment.compressed_token_count

            self.stats['segments_compressed'] += 1
            self.stats['compressions_performed'] += 1

            logger.debug(
                f"Compressed segment {segment.segment_id}: "
                f"{segment.token_count} -> {segment.compressed_token_count} tokens "
                f"({segment.compression_ratio:.1f}x)"
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to compress segment: {e}")
            return False

    def _decompress_segment(self, segment: ContextSegment) -> str:
        """Decompress a segment on demand."""
        if segment.content:
            return segment.content

        if segment.compressed is None:
            return "[Content unavailable]"

        self._ensure_encoder()

        if not self._encoder_initialized:
            return segment.compressed.decode('utf-8', errors='replace')

        try:
            from sigmalang.core.encoder import SigmaDecoder

            decoder = SigmaDecoder(self._encoder)
            decoded_tree = decoder.decode(segment.compressed)
            return str(decoded_tree)
        except Exception as e:
            logger.warning(f"Failed to decompress: {e}")
            return "[Decompression failed]"

    def _index_segment(self, segment: ContextSegment) -> None:
        """Add segment to search index."""
        terms = set()

        if segment.content:
            terms.update(segment.content.lower().split()[:200])  # First 200 words
        terms.update(t.lower() for t in segment.tags)
        terms.add(segment.source.lower())

        for term in terms:
            if term not in self._term_index:
                self._term_index[term] = []
            self._term_index[term].append(segment.segment_id)

    def get_capacity(self) -> Dict[str, Any]:
        """Get current context window capacity."""
        total_actual = self.uncompressed_tokens + self.compressed_tokens
        total_effective = self.stats['total_effective_tokens']

        extension_ratio = total_effective / max(1, total_actual)

        return {
            'native_limit': self.config.max_native_tokens,
            'actual_usage': total_actual,
            'effective_tokens': total_effective,
            'extension_ratio': round(extension_ratio, 1),
            'uncompressed_tokens': self.uncompressed_tokens,
            'compressed_tokens': self.compressed_tokens,
            'available_tokens': self.config.max_native_tokens - total_actual,
            'segments_total': len(self.segments),
            'segments_compressed': sum(1 for s in self.segments.values() if s.is_compressed),
            'utilization_pct': round(total_actual / self.config.max_native_tokens * 100, 1)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get full statistics."""
        return {
            **self.stats,
            'capacity': self.get_capacity()
        }

    def export_state(self, path: Path) -> None:
        """Export context state to JSON."""
        state = {
            'segments': {},
            'stats': self.stats,
            'capacity': self.get_capacity()
        }

        for seg_id, segment in self.segments.items():
            state['segments'][seg_id] = {
                'segment_id': segment.segment_id,
                'source': segment.source,
                'priority': segment.priority.value,
                'token_count': segment.token_count,
                'compressed_token_count': segment.compressed_token_count,
                'is_compressed': segment.is_compressed,
                'tags': segment.tags,
                'access_count': segment.access_count
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)


# =============================================================================
# Global Context Extender
# =============================================================================

_global_extender: Optional[ContextWindowExtender] = None


def get_context_extender() -> ContextWindowExtender:
    """Get or create the global context extender."""
    global _global_extender
    if _global_extender is None:
        _global_extender = ContextWindowExtender()
    return _global_extender


def initialize_context_extension(
    max_native_tokens: int = 200_000,
    target_effective_tokens: int = 2_000_000
) -> ContextWindowExtender:
    """
    Initialize the context window extender.

    Usage:
        from tools.context_extender import initialize_context_extension, ContextPriority

        extender = initialize_context_extension(
            max_native_tokens=200_000,
            target_effective_tokens=2_000_000
        )

        # Add context
        extender.add_context("Important document...", priority=ContextPriority.HIGH)
        extender.add_context("Old conversation...", priority=ContextPriority.LOW)

        # Check capacity
        capacity = extender.get_capacity()
        print(f"Extension: {capacity['extension_ratio']}x")
        print(f"Effective: {capacity['effective_tokens']:,} tokens")

        # Retrieve relevant context
        results = extender.retrieve("machine learning")
        for r in results:
            print(f"  [{r['score']}] {r['content'][:100]}...")
    """
    global _global_extender
    config = ContextExtensionConfig(
        max_native_tokens=max_native_tokens,
        target_effective_tokens=target_effective_tokens
    )
    _global_extender = ContextWindowExtender(config)
    return _global_extender
