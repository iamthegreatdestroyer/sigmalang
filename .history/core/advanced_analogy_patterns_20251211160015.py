"""Advanced Analogy Patterns and Multi-Type Reasoning Module.

This module extends the foundational SemanticAnalogyEngine with advanced
pattern recognition and multi-type reasoning capabilities.

Classes:
    AnalogyCachingLayer: LRU cache for analogy results
    FuzzyAnalogyMatcher: Handle approximate analogies with confidence
    InverseAnalogyResolver: Find missing elements in analogies
    AnalogyChainingEngine: Solve chained analogies (A:B::B:C::C:D)
    AnalogyCacheManager: Manage cache statistics and performance

Example:
    >>> from core.semantic_analogy_engine import SemanticAnalogyEngine
    >>> from core.advanced_analogy_patterns import (
    ...     AnalogyCachingLayer,
    ...     FuzzyAnalogyMatcher,
    ...     InverseAnalogyResolver
    ... )
    >>> engine = SemanticAnalogyEngine()
    >>> cache = AnalogyCachingLayer(capacity=1000)
    >>> fuzzy = FuzzyAnalogyMatcher(similarity_threshold=0.6)
    >>> result = cache.get("king", "man", "queen")
    >>> if not result:
    ...     result = engine.solve_analogy("king", "man", "queen")
    ...     cache.put("king", "man", "queen", result)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import time
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES FOR ADVANCED ANALOGY RESULTS
# ============================================================================

@dataclass
class CacheStatistics:
    """Statistics for the analogy caching layer."""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    avg_latency_saved_ms: float = 0.0
    evictions: int = 0
    expired_entries: int = 0
    current_size: int = 0
    capacity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(self.hit_rate, 4),
            'avg_latency_saved_ms': round(self.avg_latency_saved_ms, 4),
            'evictions': self.evictions,
            'expired_entries': self.expired_entries,
            'current_size': self.current_size,
            'capacity': self.capacity,
        }


@dataclass
class ChainedAnalogyResult:
    """Result of solving a chained analogy."""
    chain: List[str]
    chain_length: int
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    total_confidence: float = 0.0
    validity_score: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'chain': self.chain,
            'chain_length': self.chain_length,
            'total_confidence': round(self.total_confidence, 4),
            'validity_score': round(self.validity_score, 4),
            'latency_ms': round(self.latency_ms, 4),
            'intermediate_steps_count': len(self.intermediate_steps),
        }


@dataclass
class InverseAnalogyResult:
    """Result of solving an inverse analogy."""
    found_element: str
    position: int  # 0=A, 1=B, 2=C, 3=D
    confidence: float = 0.0
    candidates_explored: int = 0
    reasoning: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        position_names = {0: "A", 1: "B", 2: "C", 3: "D"}
        return {
            'found_element': self.found_element,
            'position': position_names.get(self.position, "?"),
            'confidence': round(self.confidence, 4),
            'candidates_explored': self.candidates_explored,
            'reasoning': self.reasoning,
            'latency_ms': round(self.latency_ms, 4),
        }


@dataclass
class FuzzyAnalogyResult:
    """Result of fuzzy analogy matching."""
    answer: str
    confidence: float = 0.0
    similarity_to_ideal: float = 0.0
    fuzzy_candidates: List[Tuple[str, float]] = field(default_factory=list)
    threshold_used: float = 0.6
    matched_fuzzily: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'answer': self.answer,
            'confidence': round(self.confidence, 4),
            'similarity_to_ideal': round(self.similarity_to_ideal, 4),
            'fuzzy_candidates_count': len(self.fuzzy_candidates),
            'threshold_used': round(self.threshold_used, 4),
            'matched_fuzzily': self.matched_fuzzily,
            'latency_ms': round(self.latency_ms, 4),
        }


@dataclass
class CompositeAnalogyResult:
    """Result of composite analogy solving."""
    answer: str
    component_count: int = 1
    pattern_confidence: float = 0.0
    component_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'answer': self.answer,
            'component_count': self.component_count,
            'pattern_confidence': round(self.pattern_confidence, 4),
            'reasoning': self.reasoning,
            'latency_ms': round(self.latency_ms, 4),
        }


# ============================================================================
# ANALOGY CACHING LAYER
# ============================================================================

class AnalogyCachingLayer:
    """LRU cache for analogy results with TTL support and statistics.

    Features:
        - LRU eviction when capacity exceeded
        - TTL (time-to-live) for automatic expiration
        - Hit/miss statistics tracking
        - Performance metrics (latency saved)

    Example:
        >>> cache = AnalogyCachingLayer(capacity=1000, ttl_seconds=3600)
        >>> result = cache.get("king", "man", "queen")
        >>> if not result:
        ...     result = solve_analogy("king", "man", "queen")
        ...     cache.put("king", "man", "queen", result)
        >>> stats = cache.get_statistics()
        >>> print(f"Hit rate: {stats.hit_rate}")
    """

    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        """Initialize caching layer.

        Args:
            capacity: Maximum number of cached analogies
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_entries = 0
        self.latencies_saved: List[float] = []

    def _make_key(self, a: str, b: str, c: str) -> str:
        """Create cache key from analogy components.

        Args:
            a: First concept
            b: Second concept
            c: Third concept

        Returns:
            Cache key string
        """
        return f"{a}:{b}:{c}"

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cached entry has expired.

        Args:
            timestamp: Timestamp when entry was cached

        Returns:
            True if entry expired, False otherwise
        """
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, a: str, b: str, c: str) -> Optional[Any]:
        """Retrieve cached analogy result.

        Args:
            a: First concept
            b: Second concept
            c: Third concept

        Returns:
            Cached result if found and not expired, None otherwise
        """
        key = self._make_key(a, b, c)

        if key not in self.cache:
            self.misses += 1
            return None

        result, timestamp = self.cache[key]

        if self._is_expired(timestamp):
            del self.cache[key]
            self.expired_entries += 1
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return result

    def put(self, a: str, b: str, c: str, result: Any, latency_ms: float = 0.0):
        """Cache an analogy result.

        Args:
            a: First concept
            b: Second concept
            c: Third concept
            result: Analogy result to cache
            latency_ms: Latency of original computation (optional)
        """
        key = self._make_key(a, b, c)
        timestamp = time.time()

        self.cache[key] = (result, timestamp)
        self.cache.move_to_end(key)

        if latency_ms > 0:
            self.latencies_saved.append(latency_ms)

        # Evict if over capacity
        if len(self.cache) > self.capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.evictions += 1

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Analogy cache cleared")

    def get_statistics(self) -> CacheStatistics:
        """Get caching layer statistics.

        Returns:
            CacheStatistics object with performance metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        avg_latency_saved = (
            sum(self.latencies_saved) / len(self.latencies_saved)
            if self.latencies_saved else 0.0
        )

        return CacheStatistics(
            hits=self.hits,
            misses=self.misses,
            hit_rate=hit_rate,
            avg_latency_saved_ms=avg_latency_saved,
            evictions=self.evictions,
            expired_entries=self.expired_entries,
            current_size=len(self.cache),
            capacity=self.capacity,
        )

    def export_statistics(self, filepath: Path):
        """Export statistics to JSON file.

        Args:
            filepath: Path to export statistics to
        """
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        logger.info(f"Cache statistics exported to {filepath}")


# ============================================================================
# FUZZY ANALOGY MATCHER
# ============================================================================

class FuzzyAnalogyMatcher:
    """Match analogies with fuzzy thresholds and confidence scoring.

    Features:
        - Configurable similarity thresholds
        - Confidence scoring for approximate matches
        - Quality grading for analogy solutions
        - Fuzzy vs exact comparison

    Example:
        >>> matcher = FuzzyAnalogyMatcher(similarity_threshold=0.6)
        >>> result = matcher.find_approximate_answer(
        ...     "king", "man", "queen"
        ... )
        >>> print(f"Answer: {result.answer}, Confidence: {result.confidence}")
    """

    def __init__(self, similarity_threshold: float = 0.6):
        """Initialize fuzzy matcher.

        Args:
            similarity_threshold: Minimum similarity score for matches [0, 1]
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.similarity_threshold = similarity_threshold

    def match_similar_analogies(
        self,
        a: str,
        b: str,
        c: str,
        tolerance: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """Find analogies similar to query within tolerance.

        Args:
            a: First concept
            b: Second concept
            c: Third concept
            tolerance: Tolerance for similarity matching [0, 1]

        Returns:
            List of (concept, similarity_score) tuples
        """
        # Placeholder for implementation
        return []

    def find_approximate_answer(
        self,
        a: str,
        b: str,
        c: str,
    ) -> FuzzyAnalogyResult:
        """Find approximate answer to analogy with fuzzy matching.

        Args:
            a: First concept
            b: Second concept
            c: Third concept

        Returns:
            FuzzyAnalogyResult with approximate answer and confidence
        """
        # Placeholder for implementation
        return FuzzyAnalogyResult(answer="", confidence=0.0)

    def grade_analogy_quality(
        self,
        a: str,
        b: str,
        c: str,
        d: str,
    ) -> float:
        """Grade the quality of an analogy (A:B::C:D).

        Args:
            a: First concept
            b: Second concept
            c: Third concept
            d: Proposed fourth concept

        Returns:
            Quality score between 0 and 1
        """
        # Placeholder for implementation
        return 0.0

    def set_threshold(self, threshold: float):
        """Update similarity threshold.

        Args:
            threshold: New similarity threshold [0, 1]
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.similarity_threshold = threshold
        logger.debug(f"Fuzzy matcher threshold updated to {threshold}")


# ============================================================================
# INVERSE ANALOGY RESOLVER
# ============================================================================

class InverseAnalogyResolver:
    """Solve inverse analogies to find missing elements.

    Features:
        - Find A given B, C, D (A:B::C:D)
        - Find B given A, C, D
        - Find C given A, B, D
        - Find D given A, B, C
        - Multiple solution discovery

    Example:
        >>> resolver = InverseAnalogyResolver(base_engine)
        >>> result = resolver.find_first_element("man", "queen", "woman")
        >>> print(f"Answer: {result.found_element}")
    """

    def __init__(self, base_engine):
        """Initialize inverse resolver.

        Args:
            base_engine: SemanticAnalogyEngine instance
        """
        self.base_engine = base_engine

    def find_first_element(
        self,
        b: str,
        c: str,
        d: str,
    ) -> InverseAnalogyResult:
        """Find first element A given A:B::C:D.

        Args:
            b: Second element
            c: Third element
            d: Fourth element

        Returns:
            InverseAnalogyResult with found first element
        """
        # Placeholder for implementation
        return InverseAnalogyResult(found_element="", position=0)

    def find_second_element(
        self,
        a: str,
        c: str,
        d: str,
    ) -> InverseAnalogyResult:
        """Find second element B given A:B::C:D.

        Args:
            a: First element
            c: Third element
            d: Fourth element

        Returns:
            InverseAnalogyResult with found second element
        """
        # Placeholder for implementation
        return InverseAnalogyResult(found_element="", position=1)

    def find_third_element(
        self,
        a: str,
        b: str,
        d: str,
    ) -> InverseAnalogyResult:
        """Find third element C given A:B::C:D.

        Args:
            a: First element
            b: Second element
            d: Fourth element

        Returns:
            InverseAnalogyResult with found third element
        """
        # Placeholder for implementation
        return InverseAnalogyResult(found_element="", position=2)

    def find_any_element(
        self,
        a: Optional[str] = None,
        b: Optional[str] = None,
        c: Optional[str] = None,
        d: Optional[str] = None,
    ) -> InverseAnalogyResult:
        """Find any missing element in analogy.

        Args:
            a: First element (None if unknown)
            b: Second element (None if unknown)
            c: Third element (None if unknown)
            d: Fourth element (None if unknown)

        Returns:
            InverseAnalogyResult with found element

        Raises:
            ValueError: If less than 3 elements provided
        """
        elements = [a, b, c, d]
        missing_count = sum(1 for e in elements if e is None)

        if missing_count != 1:
            raise ValueError("Exactly one element must be missing")

        # Delegate to specific finder
        if a is None:
            return self.find_first_element(b, c, d)
        elif b is None:
            return self.find_second_element(a, c, d)
        elif c is None:
            return self.find_third_element(a, b, d)
        else:
            # d is None - use base engine
            result = self.base_engine.solve_analogy(a, b, c)
            return InverseAnalogyResult(
                found_element=result.answer if hasattr(result, 'answer') else "",
                position=3,
                confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
            )


# ============================================================================
# ANALOGY CHAINING ENGINE
# ============================================================================

class AnalogyChainingEngine:
    """Solve chained analogies (A:B::B:C::C:D).

    Features:
        - Multi-step analogy chains
        - Consistency validation
        - Intermediate step tracking
        - Path exploration

    Example:
        >>> chainer = AnalogyChainingEngine(base_engine)
        >>> result = chainer.solve_chain("king", "man", "queen", length=3)
        >>> print(f"Chain: {result.chain}")
    """

    def __init__(self, base_engine):
        """Initialize chaining engine.

        Args:
            base_engine: SemanticAnalogyEngine instance
        """
        self.base_engine = base_engine

    def chain_analogies(
        self,
        concepts: List[str],
        chain_length: int,
    ) -> ChainedAnalogyResult:
        """Chain multiple analogies together.

        Args:
            concepts: List of starting concepts
            chain_length: Desired length of chain

        Returns:
            ChainedAnalogyResult with full chain
        """
        import time
        start_time = time.time()
        
        if len(concepts) < 2:
            return ChainedAnalogyResult(chain=[], chain_length=0)
        
        chain = concepts[:chain_length] if chain_length <= len(concepts) else concepts
        intermediate_steps = []
        total_confidence = 0.0
        
        # Build intermediate steps by solving analogies sequentially
        if chain_length > 2 and len(chain) >= 3:
            # For longer chains, solve intermediate analogies
            for i in range(1, len(chain) - 2):
                try:
                    result = self.base_engine.solve_analogy(chain[i-1], chain[i], chain[i+1])
                    if hasattr(result, 'answer'):
                        intermediate_steps.append(result.answer)
                        if hasattr(result, 'confidence'):
                            total_confidence += result.confidence
                except:
                    pass
        
        validity_score = self.validate_chain_consistency(chain)
        latency_ms = (time.time() - start_time) * 1000
        
        return ChainedAnalogyResult(
            chain=chain,
            chain_length=len(chain),
            intermediate_steps=intermediate_steps,
            total_confidence=total_confidence,
            validity_score=validity_score,
            latency_ms=latency_ms,
        )

    def solve_chain(
        self,
        a: str,
        b: str,
        c: str,
        length: int = 2,
    ) -> ChainedAnalogyResult:
        """Solve chained analogy starting with A:B.

        Args:
            a: First concept
            b: Second concept
            c: Third concept (start of next analogy)
            length: How many links in the chain

        Returns:
            ChainedAnalogyResult
        """
        import time
        start_time = time.time()
        
        # Build chain based on requested length
        chain = [a, b, c]
        intermediate_steps = []
        total_confidence = 0.0
        
        # For longer chains, extend with additional solved concepts
        if length > 3:
            # Use base engine to predict next elements
            for i in range(3, length):
                try:
                    # Try to solve: chain[-2]:chain[-1]::chain[-1]:?
                    result = self.base_engine.solve_analogy(chain[-2], chain[-1], chain[-1])
                    if hasattr(result, 'answer'):
                        next_concept = result.answer
                        chain.append(next_concept)
                        intermediate_steps.append(next_concept)
                        if hasattr(result, 'confidence'):
                            total_confidence += result.confidence
                    else:
                        break
                except:
                    break
        
        validity_score = self.validate_chain_consistency(chain)
        latency_ms = (time.time() - start_time) * 1000
        
        return ChainedAnalogyResult(
            chain=chain,
            chain_length=length,
            intermediate_steps=intermediate_steps,
            total_confidence=total_confidence,
            validity_score=validity_score,
            latency_ms=latency_ms,
        )

    def validate_chain_consistency(
        self,
        chain: List[str],
    ) -> float:
        """Validate consistency of analogy chain.

        Args:
            chain: List of concepts in chain order

        Returns:
            Consistency score [0, 1]
        """
        if len(chain) < 2:
            return 0.0
        
        if len(chain) == 2:
            return 1.0  # Perfect consistency for 2-element chains
        
        # For longer chains, check consistency by validating relationships
        consistency_scores = []
        for i in range(len(chain) - 2):
            try:
                # Check if relationship is consistent
                # For simplicity, return high consistency if base_engine can solve it
                result = self.base_engine.solve_analogy(chain[i], chain[i+1], chain[i+1])
                if hasattr(result, 'confidence'):
                    consistency_scores.append(result.confidence)
                else:
                    consistency_scores.append(0.5)  # Default moderate consistency
            except:
                consistency_scores.append(0.3)  # Lower consistency for failed relationships
        
        if not consistency_scores:
            return 0.5
        
        # Average the consistency scores
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        return min(1.0, max(0.0, avg_consistency))


# ============================================================================
# ANALOGY CACHE MANAGER
# ============================================================================

class AnalogyCacheManager:
    """Manage cache statistics and performance analysis.

    Features:
        - Cache statistics aggregation
        - Access pattern analysis
        - Capacity recommendations
        - Metrics export

    Example:
        >>> manager = AnalogyCacheManager(cache_layer)
        >>> stats = manager.get_cache_stats()
        >>> manager.export_cache_metrics(Path("metrics.json"))
    """

    def __init__(self, cache_layer: AnalogyCachingLayer):
        """Initialize cache manager.

        Args:
            cache_layer: AnalogyCachingLayer instance to manage
        """
        self.cache_layer = cache_layer

    def get_cache_stats(self) -> CacheStatistics:
        """Get comprehensive cache statistics.

        Returns:
            CacheStatistics object
        """
        return self.cache_layer.get_statistics()

    def analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze cache access patterns.

        Returns:
            Dictionary with pattern analysis
        """
        # Placeholder for implementation
        return {}

    def recommend_capacity(self) -> int:
        """Recommend optimal cache capacity.

        Returns:
            Recommended capacity size
        """
        # Placeholder for implementation
        return self.cache_layer.capacity

    def export_cache_metrics(self, filepath: Path):
        """Export cache metrics to file.

        Args:
            filepath: Path to export metrics to
        """
        self.cache_layer.export_statistics(filepath)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'AnalogyCachingLayer',
    'FuzzyAnalogyMatcher',
    'InverseAnalogyResolver',
    'AnalogyChainingEngine',
    'AnalogyCacheManager',
    'CacheStatistics',
    'ChainedAnalogyResult',
    'InverseAnalogyResult',
    'FuzzyAnalogyResult',
    'CompositeAnalogyResult',
]
