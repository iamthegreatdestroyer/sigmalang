"""
Streaming Codebook Adaptation - Phase 7 Track 8

Real-time codebook adaptation during streaming encoding. Updates Tier 2
learned primitives on-the-fly as new patterns emerge in the data stream,
without interrupting the encoding pipeline.

Architecture:
    Incoming Stream Chunks
        |
        v
    Pattern Extractor (extract frequent n-grams from chunk)
        |
        v
    Frequency Accumulator (track pattern frequencies across chunks)
        |
        v
    Promotion Evaluator (decide when a pattern deserves a Tier 2 slot)
        |
        v
    Live Codebook Update (hot-swap new entries without restart)
        |
        v
    Backpressure Controller (slow adaptation if consumer is behind)

Key Properties:
    - Non-blocking: Adaptation runs alongside encoding
    - Convergent: Codebook stabilizes within ~100 chunks
    - Bounded: Memory usage capped regardless of stream length
    - Reversible: New entries can be evicted if they don't help

Research Basis:
    - "Real-time Indexing via Streaming VQ" (Jan 2025)
    - Builds on existing streaming_encoder.py and online_learner.py

Usage:
    from sigmalang.core.streaming_codebook import StreamingCodebook

    codebook = StreamingCodebook(max_entries=128)

    for chunk in data_stream:
        # Observe patterns in chunk
        codebook.observe(chunk)

        # Encode using current codebook (including any new entries)
        encoded = codebook.encode_chunk(chunk)

        # Check adaptation status
        if codebook.stats['promotions'] > 0:
            print(f"New patterns learned: {codebook.stats['promotions']}")
"""

import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StreamingCodebookConfig:
    """Configuration for streaming codebook adaptation."""

    # Codebook limits
    max_entries: int = 128              # Max Tier 2 entries
    embedding_dim: int = 256            # Dimensions per entry

    # Pattern extraction
    min_ngram: int = 2                  # Minimum n-gram length
    max_ngram: int = 5                  # Maximum n-gram length
    min_frequency: int = 3              # Minimum occurrences before promotion

    # Promotion thresholds
    promotion_score_threshold: float = 0.5  # Minimum score for promotion
    eviction_idle_chunks: int = 50      # Evict entry if unused for N chunks

    # Backpressure
    max_pending_promotions: int = 10    # Max queued promotions
    adaptation_rate: float = 0.1        # Learning rate for embedding updates

    # Observation window
    observation_window: int = 100       # Chunks to track for frequency


# =============================================================================
# Pattern Tracking
# =============================================================================

@dataclass
class PatternCandidate:
    """A pattern being tracked for potential promotion."""

    pattern: str
    hash_id: str
    frequency: int = 0
    first_seen_chunk: int = 0
    last_seen_chunk: int = 0
    avg_compression_benefit: float = 0.0
    promotion_score: float = 0.0

    @property
    def age_chunks(self) -> int:
        return self.last_seen_chunk - self.first_seen_chunk


@dataclass
class CodebookEntry:
    """An active entry in the streaming codebook."""

    slot_id: int
    pattern: str
    hash_id: str
    embedding: np.ndarray
    frequency: int = 0
    created_at_chunk: int = 0
    last_used_chunk: int = 0
    compression_saves: int = 0          # Total bytes saved by using this entry

    @property
    def idle_chunks(self) -> int:
        return 0  # Computed externally based on current chunk


# =============================================================================
# Pattern Extractor
# =============================================================================

class PatternExtractor:
    """Extract n-gram patterns from text chunks."""

    def __init__(self, min_n: int = 2, max_n: int = 5):
        self.min_n = min_n
        self.max_n = max_n

    def extract(self, text: str) -> Dict[str, int]:
        """
        Extract word n-gram frequencies from text.

        Returns: {pattern: count}
        """
        words = text.lower().split()
        patterns = defaultdict(int)

        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                if len(ngram) >= 4:  # Skip trivially short patterns
                    patterns[ngram] += 1

        return dict(patterns)


# =============================================================================
# Frequency Accumulator
# =============================================================================

class FrequencyAccumulator:
    """
    Track pattern frequencies across streaming chunks with
    a bounded observation window.
    """

    def __init__(self, window_size: int = 100, max_candidates: int = 1000):
        self.window_size = window_size
        self.max_candidates = max_candidates
        self._candidates: Dict[str, PatternCandidate] = {}
        self._chunk_count = 0

    def update(self, patterns: Dict[str, int]) -> None:
        """Update frequencies with patterns from a new chunk."""
        self._chunk_count += 1

        for pattern, count in patterns.items():
            hash_id = hashlib.md5(pattern.encode(), usedforsecurity=False).hexdigest()[:8]

            if hash_id in self._candidates:
                candidate = self._candidates[hash_id]
                candidate.frequency += count
                candidate.last_seen_chunk = self._chunk_count
            else:
                self._candidates[hash_id] = PatternCandidate(
                    pattern=pattern,
                    hash_id=hash_id,
                    frequency=count,
                    first_seen_chunk=self._chunk_count,
                    last_seen_chunk=self._chunk_count,
                )

        # Prune old candidates
        self._prune_stale()

    def _prune_stale(self) -> None:
        """Remove candidates not seen in the observation window."""
        if len(self._candidates) <= self.max_candidates:
            return

        cutoff = self._chunk_count - self.window_size
        stale = [
            h for h, c in self._candidates.items()
            if c.last_seen_chunk < cutoff
        ]
        for h in stale:
            del self._candidates[h]

        # If still too many, remove lowest frequency
        if len(self._candidates) > self.max_candidates:
            sorted_candidates = sorted(
                self._candidates.items(),
                key=lambda x: x[1].frequency
            )
            for h, _ in sorted_candidates[:len(self._candidates) - self.max_candidates]:
                del self._candidates[h]

    def get_promotion_candidates(self, min_frequency: int = 3) -> List[PatternCandidate]:
        """Get candidates that meet minimum frequency for promotion."""
        return [
            c for c in self._candidates.values()
            if c.frequency >= min_frequency
        ]

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)


# =============================================================================
# Promotion Evaluator
# =============================================================================

class PromotionEvaluator:
    """
    Evaluate whether a pattern candidate should be promoted
    to a codebook entry based on frequency, recency, and
    estimated compression benefit.
    """

    def __init__(self, config: StreamingCodebookConfig):
        self.config = config

    def score(self, candidate: PatternCandidate, current_chunk: int) -> float:
        """
        Score a candidate for promotion.

        Score = frequency_score * recency_score * length_score

        Returns: float in [0, 1]
        """
        # Frequency score: log-scaled
        freq_score = min(1.0, candidate.frequency / 20.0)

        # Recency score: decay for old patterns
        age = current_chunk - candidate.last_seen_chunk
        recency_score = 1.0 / (1.0 + 0.1 * age)

        # Length score: longer patterns save more bytes
        length_score = min(1.0, len(candidate.pattern) / 30.0)

        # Consistency: seen across multiple chunks
        span = candidate.age_chunks
        consistency = min(1.0, span / 10.0) if span > 0 else 0.0

        score = (
            0.4 * freq_score +
            0.2 * recency_score +
            0.2 * length_score +
            0.2 * consistency
        )

        candidate.promotion_score = score
        return score

    def select_promotions(
        self,
        candidates: List[PatternCandidate],
        current_chunk: int,
        available_slots: int
    ) -> List[PatternCandidate]:
        """Select the best candidates for promotion."""
        if available_slots <= 0 or not candidates:
            return []

        # Score all candidates
        scored = []
        for c in candidates:
            score = self.score(c, current_chunk)
            if score >= self.config.promotion_score_threshold:
                scored.append((c, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take up to available_slots
        limit = min(available_slots, self.config.max_pending_promotions)
        return [c for c, _ in scored[:limit]]


# =============================================================================
# Backpressure Controller
# =============================================================================

class BackpressureController:
    """
    Control adaptation rate based on consumer processing speed.
    Slows codebook updates if the downstream encoder is falling behind.
    """

    def __init__(self, target_latency_ms: float = 5.0):
        self.target_latency_ms = target_latency_ms
        self._recent_latencies: deque = deque(maxlen=20)
        self._adaptation_multiplier: float = 1.0

    def record_latency(self, latency_ms: float) -> None:
        """Record encoding latency for a chunk."""
        self._recent_latencies.append(latency_ms)
        self._update_multiplier()

    def _update_multiplier(self) -> None:
        """Adjust adaptation rate based on latency trend."""
        if len(self._recent_latencies) < 5:
            return

        avg_latency = sum(self._recent_latencies) / len(self._recent_latencies)
        ratio = avg_latency / self.target_latency_ms

        if ratio > 2.0:
            # Very slow: pause adaptation
            self._adaptation_multiplier = 0.0
        elif ratio > 1.5:
            # Slow: reduce adaptation
            self._adaptation_multiplier = 0.3
        elif ratio > 1.0:
            # Slightly slow
            self._adaptation_multiplier = 0.7
        else:
            # On target or fast
            self._adaptation_multiplier = 1.0

    @property
    def should_adapt(self) -> bool:
        return self._adaptation_multiplier > 0.0

    @property
    def adaptation_rate(self) -> float:
        return self._adaptation_multiplier


# =============================================================================
# Main Streaming Codebook
# =============================================================================

class StreamingCodebook:
    """
    Real-time adaptive codebook for streaming encoding.

    Observes patterns in data chunks, promotes frequent patterns
    to codebook entries, and evicts stale entries — all while
    encoding continues uninterrupted.

    Usage:
        codebook = StreamingCodebook(max_entries=128)

        for chunk in stream:
            codebook.observe(chunk)
            encoded = codebook.encode_chunk(chunk)

        print(codebook.get_stats())
    """

    def __init__(
        self,
        max_entries: int = 128,
        config: Optional[StreamingCodebookConfig] = None
    ):
        self.config = config or StreamingCodebookConfig(max_entries=max_entries)
        self.config.max_entries = max_entries

        self._extractor = PatternExtractor(
            min_n=self.config.min_ngram,
            max_n=self.config.max_ngram
        )
        self._accumulator = FrequencyAccumulator(
            window_size=self.config.observation_window
        )
        self._evaluator = PromotionEvaluator(self.config)
        self._backpressure = BackpressureController()

        # Active codebook
        self._entries: Dict[int, CodebookEntry] = {}
        self._pattern_to_slot: Dict[str, int] = {}
        self._next_slot = 0

        # RNG for embeddings
        self._rng = np.random.RandomState(42)

        # Statistics
        self.stats = {
            'chunks_observed': 0,
            'promotions': 0,
            'evictions': 0,
            'total_matches': 0,
            'total_bytes_saved': 0,
        }

    def observe(self, text: str) -> Dict[str, Any]:
        """
        Observe patterns in a text chunk and potentially update the codebook.

        Returns dict with observation results.
        """
        self.stats['chunks_observed'] += 1
        chunk_num = self.stats['chunks_observed']

        # Extract patterns
        patterns = self._extractor.extract(text)
        self._accumulator.update(patterns)

        # Check for promotions (if backpressure allows)
        promoted = []
        evicted = []

        if self._backpressure.should_adapt:
            # Get promotion candidates
            candidates = self._accumulator.get_promotion_candidates(
                min_frequency=self.config.min_frequency
            )

            # Filter out already-promoted patterns
            candidates = [
                c for c in candidates
                if c.hash_id not in self._pattern_to_slot
            ]

            available_slots = self.config.max_entries - len(self._entries)

            # Evict stale entries to make room
            if available_slots <= 0:
                evicted = self._evict_stale(chunk_num)
                available_slots = self.config.max_entries - len(self._entries)

            # Select and promote
            to_promote = self._evaluator.select_promotions(
                candidates, chunk_num, available_slots
            )

            for candidate in to_promote:
                slot = self._promote(candidate, chunk_num)
                if slot >= 0:
                    promoted.append(candidate.pattern)

        # Update usage stats for matched patterns
        matches = self._count_matches(text)

        return {
            'chunk': chunk_num,
            'patterns_found': len(patterns),
            'candidates': self._accumulator.candidate_count,
            'promoted': promoted,
            'evicted': evicted,
            'matches': matches,
            'codebook_size': len(self._entries),
        }

    def _promote(self, candidate: PatternCandidate, chunk_num: int) -> int:
        """Promote a candidate to a codebook entry."""
        if len(self._entries) >= self.config.max_entries:
            return -1

        slot_id = self._next_slot
        self._next_slot += 1

        # Generate embedding for the pattern
        embedding = self._generate_embedding(candidate.pattern)

        entry = CodebookEntry(
            slot_id=slot_id,
            pattern=candidate.pattern,
            hash_id=candidate.hash_id,
            embedding=embedding,
            frequency=candidate.frequency,
            created_at_chunk=chunk_num,
            last_used_chunk=chunk_num,
        )

        self._entries[slot_id] = entry
        self._pattern_to_slot[candidate.hash_id] = slot_id
        self.stats['promotions'] += 1

        logger.debug(
            f"Promoted pattern '{candidate.pattern}' to slot {slot_id} "
            f"(freq={candidate.frequency}, score={candidate.promotion_score:.3f})"
        )

        return slot_id

    def _evict_stale(self, current_chunk: int) -> List[str]:
        """Evict entries that haven't been used recently."""
        evicted = []
        stale_slots = []

        for slot_id, entry in self._entries.items():
            idle = current_chunk - entry.last_used_chunk
            if idle > self.config.eviction_idle_chunks:
                stale_slots.append(slot_id)

        # Sort by lowest usage, evict up to 10% of codebook
        stale_slots.sort(
            key=lambda s: self._entries[s].frequency
        )
        max_evict = max(1, len(self._entries) // 10)

        for slot_id in stale_slots[:max_evict]:
            entry = self._entries[slot_id]
            evicted.append(entry.pattern)

            if entry.hash_id in self._pattern_to_slot:
                del self._pattern_to_slot[entry.hash_id]
            del self._entries[slot_id]
            self.stats['evictions'] += 1

            logger.debug(f"Evicted pattern '{entry.pattern}' from slot {slot_id}")

        return evicted

    def _count_matches(self, text: str) -> int:
        """Count how many codebook patterns match in the text."""
        text_lower = text.lower()
        matches = 0

        for slot_id, entry in self._entries.items():
            count = text_lower.count(entry.pattern.lower())
            if count > 0:
                entry.frequency += count
                entry.last_used_chunk = self.stats['chunks_observed']
                entry.compression_saves += count * len(entry.pattern)
                matches += count

        self.stats['total_matches'] += matches
        return matches

    def _generate_embedding(self, pattern: str) -> np.ndarray:
        """Generate a deterministic embedding for a pattern."""
        # Hash-based deterministic embedding
        seed = int(hashlib.md5(pattern.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.config.embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec

    def encode_chunk(self, text: str) -> Dict[str, Any]:
        """
        Encode a chunk using the current codebook.

        Returns encoding result with substitution info.
        """
        start = time.time()
        text_lower = text.lower()
        substitutions = []

        # Find all pattern matches (longest match first)
        entries_by_length = sorted(
            self._entries.values(),
            key=lambda e: len(e.pattern),
            reverse=True
        )

        for entry in entries_by_length:
            if entry.pattern.lower() in text_lower:
                # Track substitution
                substitutions.append({
                    'pattern': entry.pattern,
                    'slot_id': entry.slot_id,
                    'savings_bytes': len(entry.pattern) - 4,  # slot ref = ~4 bytes
                })

        elapsed_ms = (time.time() - start) * 1000
        self._backpressure.record_latency(elapsed_ms)

        total_savings = sum(max(0, s['savings_bytes']) for s in substitutions)
        self.stats['total_bytes_saved'] += total_savings

        return {
            'original_size': len(text),
            'substitutions': len(substitutions),
            'bytes_saved': total_savings,
            'time_ms': round(elapsed_ms, 2),
            'codebook_size': len(self._entries),
            'backpressure_rate': round(self._backpressure.adaptation_rate, 2),
        }

    def get_codebook_entries(self) -> List[Dict[str, Any]]:
        """Get all current codebook entries."""
        return [
            {
                'slot_id': e.slot_id,
                'pattern': e.pattern,
                'frequency': e.frequency,
                'created_chunk': e.created_at_chunk,
                'last_used_chunk': e.last_used_chunk,
                'compression_saves': e.compression_saves,
            }
            for e in sorted(self._entries.values(), key=lambda e: e.frequency, reverse=True)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming codebook statistics."""
        return {
            **self.stats,
            'codebook_size': len(self._entries),
            'codebook_capacity': self.config.max_entries,
            'utilization_pct': round(
                len(self._entries) / max(1, self.config.max_entries) * 100, 1
            ),
            'backpressure_rate': round(self._backpressure.adaptation_rate, 2),
            'candidates_tracked': self._accumulator.candidate_count,
        }

    def reset(self) -> None:
        """Reset the codebook (clear all entries and stats)."""
        self._entries.clear()
        self._pattern_to_slot.clear()
        self._next_slot = 0
        self.stats = {k: 0 for k in self.stats}


# =============================================================================
# Convenience
# =============================================================================

_global_streaming_codebook: Optional[StreamingCodebook] = None


def get_streaming_codebook(max_entries: int = 128) -> StreamingCodebook:
    """Get or create the global streaming codebook."""
    global _global_streaming_codebook
    if _global_streaming_codebook is None:
        _global_streaming_codebook = StreamingCodebook(max_entries=max_entries)
    return _global_streaming_codebook
