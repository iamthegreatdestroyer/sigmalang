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
from typing import List, Optional, Tuple, Dict, Any, Callable
import logging
from pathlib import Path
import time
from collections import OrderedDict, deque
import json
from enum import Enum
from statistics import mean, median

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
    confidence: float = 0.0  # Overall confidence of composite answer

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'answer': self.answer,
            'component_count': self.component_count,
            'pattern_confidence': round(self.pattern_confidence, 4),
            'confidence': round(self.confidence, 4),
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
# ANALYTICS AND SYSTEM INTELLIGENCE (TASK 4 INTEGRATION LAYER)
# ============================================================================

class QueryMethod(Enum):
    """Enumeration of available solving methods."""
    CACHING = "caching"
    FUZZY = "fuzzy"
    INVERSE = "inverse"
    CHAINING = "chaining"
    COMPOSITION = "composition"


@dataclass
class QueryAnalytic:
    """Record of a single query execution."""
    timestamp: float
    query_key: str  # e.g., "king:man::queen:?"
    method_used: QueryMethod
    result: str
    confidence: float
    latency_ms: float
    cache_hit: bool
    success: bool = True
    accuracy: Optional[float] = None  # User-provided feedback

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'query_key': self.query_key,
            'method_used': self.method_used.value,
            'result': self.result,
            'confidence': round(self.confidence, 4),
            'latency_ms': round(self.latency_ms, 4),
            'cache_hit': self.cache_hit,
            'success': self.success,
            'accuracy': round(self.accuracy, 4) if self.accuracy is not None else None,
        }


@dataclass
class AnalyticsSnapshot:
    """Summary of analytics over a time window."""
    window_start: float
    window_end: float
    total_queries: int
    cache_hit_rate: float
    avg_latency_ms: float
    success_rate: float
    method_distribution: Dict[str, int]
    avg_confidence: float
    improvement_rate: float = 0.0  # Tracks if queries improving over time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'window_start': self.window_start,
            'window_end': self.window_end,
            'total_queries': self.total_queries,
            'cache_hit_rate': round(self.cache_hit_rate, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 4),
            'success_rate': round(self.success_rate, 4),
            'method_distribution': self.method_distribution,
            'avg_confidence': round(self.avg_confidence, 4),
            'improvement_rate': round(self.improvement_rate, 4),
        }


class AnalyticsCollector:
    """Collects and analyzes query execution analytics."""

    def __init__(self, max_history: int = 10000):
        """Initialize analytics collector.

        Args:
            max_history: Maximum number of queries to keep in memory
        """
        self.max_history = max_history
        self.analytics: deque = deque(maxlen=max_history)
        self.method_stats: Dict[QueryMethod, Dict[str, Any]] = {
            method: {
                'count': 0,
                'total_latency': 0.0,
                'success_count': 0,
                'avg_confidence': 0.0,
            }
            for method in QueryMethod
        }
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.last_snapshot: Optional[AnalyticsSnapshot] = None

    def record_query(
        self,
        query_key: str,
        method_used: QueryMethod,
        result: str,
        confidence: float,
        latency_ms: float,
        cache_hit: bool,
        success: bool = True
    ) -> QueryAnalytic:
        """Record a query execution.

        Args:
            query_key: The analogy query string
            method_used: Which method was used
            result: The result returned
            confidence: Confidence score (0-1)
            latency_ms: Execution latency
            cache_hit: Whether result came from cache
            success: Whether the query succeeded

        Returns:
            QueryAnalytic record
        """
        analytic = QueryAnalytic(
            timestamp=time.time(),
            query_key=query_key,
            method_used=method_used,
            result=result,
            confidence=confidence,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            success=success,
        )
        self.analytics.append(analytic)

        # Update method stats
        stats = self.method_stats[method_used]
        stats['count'] += 1
        stats['total_latency'] += latency_ms
        if success:
            stats['success_count'] += 1

        # Update cache stats
        if cache_hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1

        return analytic

    def get_snapshot(self) -> AnalyticsSnapshot:
        """Get current analytics snapshot.

        Returns:
            AnalyticsSnapshot with current metrics
        """
        if not self.analytics:
            return AnalyticsSnapshot(
                window_start=time.time(),
                window_end=time.time(),
                total_queries=0,
                cache_hit_rate=0.0,
                avg_latency_ms=0.0,
                success_rate=0.0,
                method_distribution={},
                avg_confidence=0.0,
            )

        total = len(self.analytics)
        cache_total = self.cache_stats['hits'] + self.cache_stats['misses']
        cache_hit_rate = (
            self.cache_stats['hits'] / cache_total if cache_total > 0 else 0.0
        )

        latencies = [a.latency_ms for a in self.analytics]
        confidences = [a.confidence for a in self.analytics]
        successful = sum(1 for a in self.analytics if a.success)

        method_dist = {}
        for method in QueryMethod:
            count = self.method_stats[method]['count']
            if count > 0:
                method_dist[method.value] = count

        snapshot = AnalyticsSnapshot(
            window_start=self.analytics[0].timestamp,
            window_end=self.analytics[-1].timestamp,
            total_queries=total,
            cache_hit_rate=cache_hit_rate,
            avg_latency_ms=mean(latencies) if latencies else 0.0,
            success_rate=successful / total if total > 0 else 0.0,
            method_distribution=method_dist,
            avg_confidence=mean(confidences) if confidences else 0.0,
        )
        self.last_snapshot = snapshot
        return snapshot

    def get_method_stats(self, method: QueryMethod) -> Dict[str, Any]:
        """Get statistics for a specific method.

        Args:
            method: The method to get stats for

        Returns:
            Dictionary with method statistics
        """
        stats = self.method_stats[method]
        if stats['count'] == 0:
            return stats

        return {
            'count': stats['count'],
            'avg_latency_ms': round(stats['total_latency'] / stats['count'], 4),
            'success_rate': round(stats['success_count'] / stats['count'], 4)
            if stats['count'] > 0
            else 0.0,
        }

    def export_analytics(self, filepath: Path) -> None:
        """Export analytics to JSON file.

        Args:
            filepath: Path to export to
        """
        data = {
            'snapshot': self.get_snapshot().to_dict(),
            'method_stats': {
                method.value: self.get_method_stats(method)
                for method in QueryMethod
            },
            'sample_queries': [
                a.to_dict() for a in list(self.analytics)[-100:]
            ],
        }
        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Analytics exported to {filepath}")

    def clear(self) -> None:
        """Clear all analytics."""
        self.analytics.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.last_snapshot = None


class FeedbackLoop:
    """Manages feedback-driven optimization of the analogy system."""

    def __init__(
        self,
        pattern_weight_learner: Optional[Callable] = None,
        threshold_learner: Optional[Callable] = None,
    ):
        """Initialize feedback loop.

        Args:
            pattern_weight_learner: Callback to update pattern weights
            threshold_learner: Callback to retrain thresholds
        """
        self.pattern_weight_learner = pattern_weight_learner
        self.threshold_learner = threshold_learner
        self.feedback_buffer: deque = deque(maxlen=500)
        self.performance_history: deque = deque(maxlen=1000)

    def record_feedback(
        self,
        query_key: str,
        result: str,
        user_rating: float,  # 0-1, 1=excellent, 0=poor
        confidence: float,
    ) -> None:
        """Record user feedback on a result.

        Args:
            query_key: The original query
            result: The provided result
            user_rating: User's rating (0-1)
            confidence: Model's confidence in result
        """
        feedback = {
            'timestamp': time.time(),
            'query_key': query_key,
            'result': result,
            'user_rating': user_rating,
            'confidence': confidence,
            'matches_expected': user_rating > 0.7,  # Threshold for "correct"
        }
        self.feedback_buffer.append(feedback)
        self.performance_history.append(user_rating)

        # Trigger relearning if buffer reaches threshold
        if len(self.feedback_buffer) >= 100:
            self._trigger_optimization()

    def _trigger_optimization(self) -> None:
        """Trigger optimization based on accumulated feedback."""
        if self.pattern_weight_learner:
            # Extract high-feedback queries for weight updating
            high_feedback = [
                f for f in self.feedback_buffer if f['user_rating'] >= 0.7
            ]
            if high_feedback:
                self.pattern_weight_learner(high_feedback)

        if self.threshold_learner:
            # Use recent feedback to retrain thresholds
            recent = list(self.feedback_buffer)[-100:]
            if len(recent) > 20:
                self.threshold_learner(recent)

    def get_effectiveness(self) -> Dict[str, Any]:
        """Get feedback loop effectiveness metrics.

        Returns:
            Dictionary with metrics
        """
        if not self.performance_history:
            return {'effectiveness': 0.0, 'feedback_count': 0}

        ratings = list(self.performance_history)
        return {
            'avg_user_rating': round(mean(ratings), 4),
            'median_user_rating': round(median(ratings), 4),
            'feedback_count': len(self.feedback_buffer),
            'good_feedback_rate': round(
                sum(1 for r in ratings if r >= 0.7) / len(ratings), 4
            ),
        }


class SystemIntelligence:
    """Orchestrates analytics, feedback, and optimization of the analogy system."""

    def __init__(
        self,
        analytics_collector: Optional[AnalyticsCollector] = None,
        feedback_loop: Optional[FeedbackLoop] = None,
    ):
        """Initialize system intelligence.

        Args:
            analytics_collector: Analytics collector instance
            feedback_loop: Feedback loop instance
        """
        self.analytics = analytics_collector or AnalyticsCollector()
        self.feedback = feedback_loop or FeedbackLoop()
        self.optimization_history: deque = deque(maxlen=100)
        self.recommendations: List[Dict[str, Any]] = []

    def record_query_execution(
        self,
        query_key: str,
        method: QueryMethod,
        result: str,
        confidence: float,
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        """Record a query execution for analytics.

        Args:
            query_key: The query key
            method: The method used
            result: The result
            confidence: Confidence score
            latency_ms: Execution latency
            cache_hit: Whether cache was hit
        """
        self.analytics.record_query(
            query_key=query_key,
            method_used=method,
            result=result,
            confidence=confidence,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

    def submit_user_feedback(
        self,
        query_key: str,
        result: str,
        rating: float,
        confidence: float,
    ) -> None:
        """Submit user feedback to the system.

        Args:
            query_key: The query
            result: The result
            rating: User's rating (0-1)
            confidence: Model's confidence
        """
        self.feedback.record_feedback(
            query_key=query_key,
            result=result,
            user_rating=rating,
            confidence=confidence,
        )

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analytics.

        Returns:
            List of recommendations
        """
        self.recommendations = []
        snapshot = self.analytics.get_snapshot()

        # Recommendation 1: Cache efficiency
        if snapshot.cache_hit_rate < 0.3 and snapshot.total_queries > 100:
            self.recommendations.append({
                'type': 'cache_efficiency',
                'severity': 'medium',
                'description': 'Cache hit rate is low',
                'action': 'Consider increasing cache capacity or improving pattern discovery',
            })

        # Recommendation 2: Method performance
        for method, count in snapshot.method_distribution.items():
            if count > 0:
                method_stats = self.analytics.get_method_stats(
                    QueryMethod(method)
                )
                if method_stats['success_rate'] < 0.75:
                    self.recommendations.append({
                        'type': 'method_performance',
                        'severity': 'medium',
                        'method': method,
                        'success_rate': method_stats['success_rate'],
                        'action': f'Method {method} has low success rate',
                    })

        # Recommendation 3: Latency optimization
        if snapshot.avg_latency_ms > 100:
            self.recommendations.append({
                'type': 'latency',
                'severity': 'low',
                'avg_latency_ms': snapshot.avg_latency_ms,
                'action': 'Consider optimizing query processing',
            })

        # Recommendation 4: Feedback effectiveness
        feedback_effectiveness = self.feedback.get_effectiveness()
        if (feedback_effectiveness.get('feedback_count', 0) > 50 and
                feedback_effectiveness.get('good_feedback_rate', 0) < 0.6):
            self.recommendations.append({
                'type': 'feedback_quality',
                'severity': 'high',
                'good_feedback_rate': feedback_effectiveness['good_feedback_rate'],
                'action': 'User feedback indicates room for accuracy improvement',
            })

        return self.recommendations

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics.

        Returns:
            Dictionary with health metrics
        """
        snapshot = self.analytics.get_snapshot()
        feedback_effectiveness = self.feedback.get_effectiveness()

        health_score = 0.0
        factors = []

        # Cache efficiency factor (0-25 points)
        cache_score = min(25, snapshot.cache_hit_rate * 25)
        factors.append(('cache_efficiency', cache_score))

        # Success rate factor (0-25 points)
        success_score = snapshot.success_rate * 25
        factors.append(('success_rate', success_score))

        # Latency factor (0-25 points, lower latency is better)
        latency_score = max(0, 25 - (snapshot.avg_latency_ms / 2))
        factors.append(('latency', latency_score))

        # Feedback quality factor (0-25 points)
        feedback_score = feedback_effectiveness.get('good_feedback_rate', 0) * 25
        factors.append(('feedback_quality', feedback_score))

        health_score = sum(score for _, score in factors) / 4

        return {
            'overall_score': round(health_score, 2),
            'factors': [{'name': name, 'score': round(score, 2)}
                       for name, score in factors],
            'snapshot': snapshot.to_dict(),
            'feedback_effectiveness': feedback_effectiveness,
        }

    def export_system_report(self, filepath: Path) -> None:
        """Export comprehensive system report.

        Args:
            filepath: Path to export to
        """
        report = {
            'timestamp': time.time(),
            'system_health': self.get_system_health(),
            'recommendations': self.generate_recommendations(),
            'analytics': self.analytics.get_snapshot().to_dict(),
            'feedback_effectiveness': self.feedback.get_effectiveness(),
        }
        filepath.write_text(json.dumps(report, indent=2))
        logger.info(f"System report exported to {filepath}")


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
    'QueryMethod',
    'QueryAnalytic',
    'AnalyticsSnapshot',
    'AnalyticsCollector',
    'FeedbackLoop',
    'SystemIntelligence',]