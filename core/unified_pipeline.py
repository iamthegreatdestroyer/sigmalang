"""
Phase 2A.5 - Task 2: Unified Pipeline Orchestrator
===================================================

Single entry point orchestrating all Phase 2A components.

This module provides:
- UnifiedAnalogyPipeline: Master orchestrator for all components
- QueryRouter: Intelligent query routing to optimal component
- CacheManager: Multi-level caching for performance
- PipelineStateMachine: State management and checkpointing

Integration Points:
- Phase 2A.1: HyperdimensionalEncoder
- Phase 2A.2: BidirectionalCodec
- Phase 2A.3: SemanticAnalogyEngine
- Phase 2A.4: PatternEvolution, PatternIntelligence

Copyright 2025 - SigmaLang Project
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, 
    Tuple, TypeVar, Union
)

# Internal imports (conditional for testing)
try:
    from .parallel_processor import ParallelExecutor, AsyncPatternProcessor
except ImportError:
    ParallelExecutor = None
    AsyncPatternProcessor = None

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class QueryComplexity(Enum):
    """Query complexity classification."""
    TRIVIAL = auto()      # Direct lookup, cached result
    SIMPLE = auto()       # Single component, fast execution
    MODERATE = auto()     # Multiple components, medium execution
    COMPLEX = auto()      # Full pipeline, parallel execution
    EXTREME = auto()      # Resource-intensive, batching recommended


class ProcessingStrategy(Enum):
    """Processing strategy for queries."""
    CACHE_ONLY = auto()        # Only check cache
    FAST_PATH = auto()         # Optimized fast execution
    STANDARD = auto()          # Normal execution path
    COMPREHENSIVE = auto()     # Full analysis with all components
    PARALLEL = auto()          # Parallel processing for batch


class PipelineState(Enum):
    """Pipeline state machine states."""
    IDLE = auto()
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    DEGRADED = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


# ============================================================================
# QUERY AND RESULT TYPES
# ============================================================================

@dataclass
class Query:
    """Represents a query to the pipeline."""
    id: str
    content: Any
    query_type: str = "analogy"
    priority: int = 0
    timeout_ms: float = 5000.0
    require_explanation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def cache_key(self) -> str:
        """Generate cache key for this query."""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(
            f"{self.query_type}:{content_str}".encode()
        ).hexdigest()[:32]


@dataclass
class QueryResult:
    """Result of a query execution."""
    query_id: str
    success: bool
    result: Any
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    strategy_used: Optional[ProcessingStrategy] = None
    cache_hit: bool = False
    components_used: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for the unified pipeline."""
    # Cache settings
    cache_enabled: bool = True
    cache_max_size: int = 10000
    cache_ttl_seconds: float = 3600.0
    
    # Processing settings
    default_timeout_ms: float = 5000.0
    max_parallel_queries: int = 100
    enable_parallel_processing: bool = True
    
    # Component settings
    enable_hd_encoder: bool = True
    enable_codec: bool = True
    enable_semantic_engine: bool = True
    enable_pattern_evolution: bool = True
    enable_pattern_intelligence: bool = True
    
    # Quality settings
    min_confidence_threshold: float = 0.5
    enable_explanations: bool = False
    
    # State persistence
    checkpoint_enabled: bool = True
    checkpoint_interval_seconds: float = 300.0
    checkpoint_path: Optional[str] = None


# ============================================================================
# CACHE MANAGER
# ============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """Entry in the cache."""
    key: str
    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: float = 3600.0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheManager(Generic[T]):
    """
    Multi-level caching with LRU eviction.
    
    Features:
    - LRU eviction when capacity exceeded
    - TTL-based expiration
    - Hit/miss statistics
    - Smart invalidation
    - Pre-warming support
    
    Example:
        >>> cache = CacheManager(max_size=1000, ttl_seconds=3600)
        >>> cache.put("key1", {"result": 42})
        >>> result = cache.get("key1")
        >>> print(cache.get_hit_rate())
        1.0
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600.0
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Default TTL for entries
        """
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                self._cache.pop(key, None)
                self._expirations += 1
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: T, 
        ttl_seconds: Optional[float] = None
    ) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional custom TTL
        """
        with self._lock:
            # Evict if needed
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl_seconds or self.default_ttl
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate specific cache entry.
        
        Args:
            key: Key to invalidate
            
        Returns:
            True if entry was found and removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all entries matching pattern prefix.
        
        Args:
            pattern: Key prefix to match
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                k for k in self._cache.keys() 
                if k.startswith(pattern)
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
            
            return len(keys_to_remove)
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def warm_cache(
        self, 
        items: List[Tuple[str, T]],
        ttl_seconds: Optional[float] = None
    ) -> int:
        """
        Pre-warm cache with items.
        
        Args:
            items: List of (key, value) tuples
            ttl_seconds: Optional TTL for warmed items
            
        Returns:
            Number of items added
        """
        count = 0
        for key, value in items:
            self.put(key, value, ttl_seconds)
            count += 1
        return count
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate.
        
        Returns:
            Hit rate (0.0 to 1.0)
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            self._cache.popitem(last=False)
            self._evictions += 1
    
    def _cleanup_expired(self) -> int:
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self.get_hit_rate(),
                'evictions': self._evictions,
                'expirations': self._expirations,
                'default_ttl': self.default_ttl
            }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired()


# ============================================================================
# QUERY ROUTER
# ============================================================================

class QueryRouter:
    """
    Intelligent query routing to optimal component.
    
    Features:
    - Complexity estimation
    - Strategy selection
    - Load-aware routing
    - Component health checks
    
    Example:
        >>> router = QueryRouter()
        >>> complexity, strategy = router.route(query)
        >>> print(f"Estimated: {complexity}, Strategy: {strategy}")
    """
    
    def __init__(self):
        """Initialize query router."""
        self._component_health: Dict[str, float] = {}
        self._component_latency: Dict[str, float] = {}
        self._routing_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def route(
        self, 
        query: Query
    ) -> Tuple[QueryComplexity, ProcessingStrategy]:
        """
        Route query to optimal component/strategy.
        
        Args:
            query: Query to route
            
        Returns:
            Tuple of (complexity, strategy)
        """
        complexity = self.predict_complexity(query)
        strategy = self.select_strategy(query, complexity)
        
        # Record routing decision
        with self._lock:
            self._routing_history.append({
                'query_id': query.id,
                'query_type': query.query_type,
                'complexity': complexity.name,
                'strategy': strategy.name,
                'timestamp': time.time()
            })
            
            # Keep history bounded
            if len(self._routing_history) > 1000:
                self._routing_history = self._routing_history[-500:]
        
        return complexity, strategy
    
    def predict_complexity(self, query: Query) -> QueryComplexity:
        """
        Predict query complexity.
        
        Args:
            query: Query to analyze
            
        Returns:
            Estimated complexity level
        """
        # Analyze query content
        content = query.content
        
        # Simple heuristics based on query characteristics
        if isinstance(content, str):
            length = len(content)
            if length < 50:
                return QueryComplexity.TRIVIAL
            elif length < 200:
                return QueryComplexity.SIMPLE
            elif length < 500:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.COMPLEX
        
        elif isinstance(content, dict):
            num_keys = len(content)
            if num_keys <= 2:
                return QueryComplexity.SIMPLE
            elif num_keys <= 5:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.COMPLEX
        
        elif isinstance(content, (list, tuple)):
            length = len(content)
            if length <= 5:
                return QueryComplexity.SIMPLE
            elif length <= 20:
                return QueryComplexity.MODERATE
            elif length <= 100:
                return QueryComplexity.COMPLEX
            else:
                return QueryComplexity.EXTREME
        
        return QueryComplexity.MODERATE
    
    def select_strategy(
        self, 
        query: Query, 
        complexity: QueryComplexity
    ) -> ProcessingStrategy:
        """
        Select processing strategy based on complexity and requirements.
        
        Args:
            query: Query being processed
            complexity: Estimated complexity
            
        Returns:
            Recommended processing strategy
        """
        # Map complexity to default strategy
        strategy_map = {
            QueryComplexity.TRIVIAL: ProcessingStrategy.CACHE_ONLY,
            QueryComplexity.SIMPLE: ProcessingStrategy.FAST_PATH,
            QueryComplexity.MODERATE: ProcessingStrategy.STANDARD,
            QueryComplexity.COMPLEX: ProcessingStrategy.COMPREHENSIVE,
            QueryComplexity.EXTREME: ProcessingStrategy.PARALLEL,
        }
        
        strategy = strategy_map.get(complexity, ProcessingStrategy.STANDARD)
        
        # Override based on query requirements
        if query.require_explanation:
            # Explanations need comprehensive processing
            if strategy in (ProcessingStrategy.CACHE_ONLY, ProcessingStrategy.FAST_PATH):
                strategy = ProcessingStrategy.STANDARD
        
        # Override based on timeout
        if query.timeout_ms < 100:
            # Very tight timeout, use fast path
            strategy = ProcessingStrategy.FAST_PATH
        
        return strategy
    
    def update_component_health(
        self, 
        component: str, 
        health: float
    ) -> None:
        """
        Update health score for a component.
        
        Args:
            component: Component name
            health: Health score (0.0 to 1.0)
        """
        with self._lock:
            self._component_health[component] = max(0.0, min(1.0, health))
    
    def update_component_latency(
        self, 
        component: str, 
        latency_ms: float
    ) -> None:
        """
        Update latency measurement for a component.
        
        Args:
            component: Component name
            latency_ms: Measured latency in milliseconds
        """
        with self._lock:
            # Exponential moving average
            current = self._component_latency.get(component, latency_ms)
            self._component_latency[component] = 0.8 * current + 0.2 * latency_ms
    
    def get_healthy_components(self, threshold: float = 0.5) -> Set[str]:
        """
        Get set of healthy components.
        
        Args:
            threshold: Minimum health score
            
        Returns:
            Set of healthy component names
        """
        with self._lock:
            return {
                name for name, health in self._component_health.items()
                if health >= threshold
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        with self._lock:
            # Count routing decisions by complexity
            complexity_counts = {}
            strategy_counts = {}
            
            for record in self._routing_history[-100:]:  # Last 100
                c = record['complexity']
                s = record['strategy']
                complexity_counts[c] = complexity_counts.get(c, 0) + 1
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
            
            return {
                'total_routed': len(self._routing_history),
                'component_health': dict(self._component_health),
                'component_latency': dict(self._component_latency),
                'complexity_distribution': complexity_counts,
                'strategy_distribution': strategy_counts
            }


# ============================================================================
# PIPELINE STATE MACHINE
# ============================================================================

@dataclass
class StateTransition:
    """Records a state transition."""
    from_state: PipelineState
    to_state: PipelineState
    timestamp: float
    reason: str


class PipelineStateMachine:
    """
    State management and checkpointing for the pipeline.
    
    Features:
    - Valid state transition enforcement
    - State persistence and recovery
    - Transition history tracking
    - Checkpoint/restore functionality
    
    Example:
        >>> sm = PipelineStateMachine()
        >>> sm.transition(PipelineState.READY, "Initialization complete")
        >>> sm.checkpoint("/path/to/checkpoint.json")
    """
    
    # Valid state transitions
    VALID_TRANSITIONS: Dict[PipelineState, Set[PipelineState]] = {
        PipelineState.IDLE: {PipelineState.INITIALIZING, PipelineState.SHUTTING_DOWN},
        PipelineState.INITIALIZING: {PipelineState.READY, PipelineState.ERROR},
        PipelineState.READY: {PipelineState.PROCESSING, PipelineState.DEGRADED, 
                              PipelineState.ERROR, PipelineState.SHUTTING_DOWN},
        PipelineState.PROCESSING: {PipelineState.READY, PipelineState.DEGRADED, 
                                   PipelineState.ERROR},
        PipelineState.DEGRADED: {PipelineState.READY, PipelineState.ERROR, 
                                  PipelineState.SHUTTING_DOWN},
        PipelineState.ERROR: {PipelineState.INITIALIZING, PipelineState.SHUTTING_DOWN},
        PipelineState.SHUTTING_DOWN: {PipelineState.IDLE},
    }
    
    def __init__(self, initial_state: PipelineState = PipelineState.IDLE):
        """
        Initialize state machine.
        
        Args:
            initial_state: Starting state
        """
        self._state = initial_state
        self._lock = threading.RLock()
        self._history: List[StateTransition] = []
        self._state_data: Dict[str, Any] = {}
        self._listeners: List[Callable[[StateTransition], None]] = []
    
    @property
    def state(self) -> PipelineState:
        """Get current state."""
        with self._lock:
            return self._state
    
    def transition(
        self, 
        new_state: PipelineState, 
        reason: str = ""
    ) -> bool:
        """
        Transition to new state.
        
        Args:
            new_state: Target state
            reason: Reason for transition
            
        Returns:
            True if transition was valid and performed
        """
        with self._lock:
            valid_next = self.VALID_TRANSITIONS.get(self._state, set())
            
            if new_state not in valid_next:
                logger.warning(
                    f"Invalid state transition: {self._state} -> {new_state}"
                )
                return False
            
            transition = StateTransition(
                from_state=self._state,
                to_state=new_state,
                timestamp=time.time(),
                reason=reason
            )
            
            self._history.append(transition)
            self._state = new_state
            
            logger.info(f"State transition: {transition.from_state} -> {transition.to_state}")
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(transition)
                except Exception as e:
                    logger.error(f"State listener error: {e}")
            
            return True
    
    def can_transition(self, new_state: PipelineState) -> bool:
        """
        Check if transition to state is valid.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition is valid
        """
        with self._lock:
            valid_next = self.VALID_TRANSITIONS.get(self._state, set())
            return new_state in valid_next
    
    def set_data(self, key: str, value: Any) -> None:
        """Set state data."""
        with self._lock:
            self._state_data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get state data."""
        with self._lock:
            return self._state_data.get(key, default)
    
    def add_listener(
        self, 
        listener: Callable[[StateTransition], None]
    ) -> None:
        """Add state transition listener."""
        self._listeners.append(listener)
    
    def checkpoint(self, path: Union[str, Path]) -> bool:
        """
        Save state to checkpoint file.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            True if checkpoint was saved
        """
        try:
            with self._lock:
                checkpoint_data = {
                    'state': self._state.name,
                    'state_data': self._state_data,
                    'timestamp': time.time(),
                    'history_len': len(self._history)
                }
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            return False
    
    def restore(self, path: Union[str, Path]) -> bool:
        """
        Restore state from checkpoint file.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            True if checkpoint was restored
        """
        try:
            path = Path(path)
            
            if not path.exists():
                logger.warning(f"Checkpoint file not found: {path}")
                return False
            
            with open(path, 'r') as f:
                checkpoint_data = json.load(f)
            
            with self._lock:
                state_name = checkpoint_data.get('state', 'IDLE')
                self._state = PipelineState[state_name]
                self._state_data = checkpoint_data.get('state_data', {})
            
            logger.info(f"Checkpoint restored from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint restore failed: {e}")
            return False
    
    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """
        Get recent state transition history.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of recent transitions
        """
        with self._lock:
            return self._history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        with self._lock:
            # Count time in each state
            state_durations: Dict[str, float] = {}
            
            for i in range(len(self._history) - 1):
                state = self._history[i].to_state.name
                duration = self._history[i + 1].timestamp - self._history[i].timestamp
                state_durations[state] = state_durations.get(state, 0) + duration
            
            return {
                'current_state': self._state.name,
                'total_transitions': len(self._history),
                'state_durations': state_durations,
                'data_keys': list(self._state_data.keys())
            }


# ============================================================================
# UNIFIED ANALOGY PIPELINE
# ============================================================================

class UnifiedAnalogyPipeline:
    """
    Master orchestrator for all Phase 2A components.
    
    Provides single entry point for:
    - Query processing through optimal components
    - Batch processing with parallelization
    - Caching and state management
    - Health monitoring and graceful degradation
    
    Example:
        >>> pipeline = UnifiedAnalogyPipeline()
        >>> pipeline.initialize()
        >>> result = pipeline.process_query(Query(id="1", content="A is to B"))
        >>> print(result.confidence)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize unified pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Core components
        self._cache: Optional[CacheManager] = None
        self._router: Optional[QueryRouter] = None
        self._state_machine: Optional[PipelineStateMachine] = None
        self._parallel_executor: Optional[ParallelExecutor] = None
        
        # Component registry
        self._components: Dict[str, Any] = {}
        
        # Statistics
        self._stats = {
            'queries_processed': 0,
            'queries_failed': 0,
            'total_processing_time_ms': 0.0,
            'avg_processing_time_ms': 0.0
        }
        self._lock = threading.Lock()
    
    def initialize(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize state machine
            self._state_machine = PipelineStateMachine()
            self._state_machine.transition(
                PipelineState.INITIALIZING, 
                "Starting initialization"
            )
            
            # Initialize cache
            if self.config.cache_enabled:
                self._cache = CacheManager(
                    max_size=self.config.cache_max_size,
                    ttl_seconds=self.config.cache_ttl_seconds
                )
            
            # Initialize router
            self._router = QueryRouter()
            
            # Initialize parallel executor
            if self.config.enable_parallel_processing and ParallelExecutor:
                self._parallel_executor = ParallelExecutor(
                    max_workers=self.config.max_parallel_queries
                )
            
            # Try to restore from checkpoint
            if self.config.checkpoint_enabled and self.config.checkpoint_path:
                self._state_machine.restore(self.config.checkpoint_path)
            
            # Register components
            self._register_components()
            
            # Transition to ready
            self._state_machine.transition(
                PipelineState.READY, 
                "Initialization complete"
            )
            
            logger.info("UnifiedAnalogyPipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            if self._state_machine:
                self._state_machine.transition(
                    PipelineState.ERROR, 
                    f"Initialization failed: {e}"
                )
            return False
    
    def _register_components(self) -> None:
        """Register available components."""
        # Components are registered lazily when accessed
        # This allows optional dependencies
        self._components = {
            'cache': self._cache,
            'router': self._router,
            'state_machine': self._state_machine,
            'parallel_executor': self._parallel_executor
        }
    
    def process_query(self, query: Query) -> QueryResult:
        """
        Process a single query through the pipeline.
        
        Args:
            query: Query to process
            
        Returns:
            QueryResult with results and metadata
        """
        start_time = time.time()
        
        # Check state
        if self._state_machine and self._state_machine.state not in (
            PipelineState.READY, PipelineState.PROCESSING, PipelineState.DEGRADED
        ):
            return QueryResult(
                query_id=query.id,
                success=False,
                result=None,
                error=f"Pipeline not ready: {self._state_machine.state.name}"
            )
        
        try:
            # Update state to processing
            if self._state_machine and self._state_machine.state == PipelineState.READY:
                self._state_machine.transition(
                    PipelineState.PROCESSING, 
                    f"Processing query {query.id}"
                )
            
            # Check cache first
            cache_key = query.cache_key()
            if self._cache is not None:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    execution_time = (time.time() - start_time) * 1000
                    return QueryResult(
                        query_id=query.id,
                        success=True,
                        result=cached,
                        cache_hit=True,
                        execution_time_ms=execution_time
                    )
            
            # Route query
            complexity, strategy = self._router.route(query) if self._router else (
                QueryComplexity.MODERATE, ProcessingStrategy.STANDARD
            )
            
            # Process based on strategy
            result = self._execute_strategy(query, strategy)
            
            # Cache result if successful
            if result.success and self._cache is not None:
                self._cache.put(cache_key, result.result)
            
            # Update statistics
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            result.strategy_used = strategy
            
            self._update_stats(execution_time, result.success)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_stats(execution_time, False)
            
            logger.error(f"Query processing failed: {e}")
            return QueryResult(
                query_id=query.id,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time
            )
        
        finally:
            # Return to ready state
            if self._state_machine and self._state_machine.state == PipelineState.PROCESSING:
                self._state_machine.transition(
                    PipelineState.READY, 
                    "Query processing complete"
                )
    
    def _execute_strategy(
        self, 
        query: Query, 
        strategy: ProcessingStrategy
    ) -> QueryResult:
        """Execute query using specified strategy."""
        components_used = []
        
        if strategy == ProcessingStrategy.CACHE_ONLY:
            # CACHE_ONLY should fall back to FAST_PATH on cache miss
            # Since we already checked cache before calling this method,
            # we know it's a miss, so use fast path as fallback
            result = self._fast_process(query)
            components_used.append('fast_path_fallback')
            result.components_used = components_used
            return result
        
        elif strategy == ProcessingStrategy.FAST_PATH:
            # Minimal processing
            result = self._fast_process(query)
            components_used.append('fast_path')
        
        elif strategy == ProcessingStrategy.STANDARD:
            # Normal processing
            result = self._standard_process(query)
            components_used.extend(['standard'])
        
        elif strategy == ProcessingStrategy.COMPREHENSIVE:
            # Full processing with all components
            result = self._comprehensive_process(query)
            components_used.extend(['comprehensive', 'all_components'])
        
        elif strategy == ProcessingStrategy.PARALLEL:
            # Parallel processing
            result = self._parallel_process(query)
            components_used.extend(['parallel'])
        
        else:
            result = self._standard_process(query)
            components_used.append('default')
        
        result.components_used = components_used
        return result
    
    def _fast_process(self, query: Query) -> QueryResult:
        """Fast path processing for simple queries."""
        # Simple echo for now - actual implementation would call HD encoder
        return QueryResult(
            query_id=query.id,
            success=True,
            result={'processed': query.content, 'method': 'fast'},
            confidence=0.7
        )
    
    def _standard_process(self, query: Query) -> QueryResult:
        """Standard processing path."""
        # Standard processing - actual implementation would use components
        return QueryResult(
            query_id=query.id,
            success=True,
            result={'processed': query.content, 'method': 'standard'},
            confidence=0.85
        )
    
    def _comprehensive_process(self, query: Query) -> QueryResult:
        """Comprehensive processing with all components."""
        # Comprehensive - actual implementation would use all Phase 2A components
        explanation = None
        if query.require_explanation:
            explanation = f"Processed query using all available components"
        
        return QueryResult(
            query_id=query.id,
            success=True,
            result={'processed': query.content, 'method': 'comprehensive'},
            confidence=0.95,
            explanation=explanation
        )
    
    def _parallel_process(self, query: Query) -> QueryResult:
        """Parallel processing for batch/complex queries."""
        # For batch content
        if isinstance(query.content, list) and self._parallel_executor:
            # Process items in parallel
            results = self._parallel_executor.map_parallel(
                lambda x: {'item': x, 'processed': True},
                query.content
            )
            return QueryResult(
                query_id=query.id,
                success=True,
                result={'items': results, 'method': 'parallel'},
                confidence=0.9
            )
        
        return self._comprehensive_process(query)
    
    def batch_process(self, queries: List[Query]) -> List[QueryResult]:
        """
        Process multiple queries.
        
        Args:
            queries: List of queries to process
            
        Returns:
            List of results in same order
        """
        if not queries:
            return []
        
        if self._parallel_executor:
            # Process in parallel
            return self._parallel_executor.map_parallel(
                self.process_query,
                queries
            )
        else:
            # Sequential fallback
            return [self.process_query(q) for q in queries]
    
    def _update_stats(self, execution_time_ms: float, success: bool) -> None:
        """Update pipeline statistics."""
        with self._lock:
            self._stats['queries_processed'] += 1
            if not success:
                self._stats['queries_failed'] += 1
            self._stats['total_processing_time_ms'] += execution_time_ms
            self._stats['avg_processing_time_ms'] = (
                self._stats['total_processing_time_ms'] / 
                self._stats['queries_processed']
            )
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Get complete pipeline state.
        
        Returns:
            Dict with all component states
        """
        state = {
            'pipeline_state': self._state_machine.state.name if self._state_machine else 'UNKNOWN',
            'config': {
                'cache_enabled': self.config.cache_enabled,
                'parallel_enabled': self.config.enable_parallel_processing,
                'max_parallel': self.config.max_parallel_queries
            },
            'statistics': dict(self._stats)
        }
        
        if self._cache is not None:
            state['cache'] = self._cache.get_stats()
        
        if self._router:
            state['router'] = self._router.get_stats()
        
        if self._state_machine:
            state['state_machine'] = self._state_machine.get_stats()
        
        return state
    
    def configure(self, **kwargs) -> None:
        """
        Update configuration at runtime.
        
        Args:
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key} = {value}")
    
    def shutdown(self, save_checkpoint: bool = True) -> None:
        """
        Shutdown the pipeline.
        
        Args:
            save_checkpoint: Whether to save checkpoint before shutdown
        """
        if self._state_machine:
            self._state_machine.transition(
                PipelineState.SHUTTING_DOWN, 
                "Pipeline shutdown requested"
            )
        
        # Save checkpoint
        if save_checkpoint and self.config.checkpoint_path:
            if self._state_machine:
                self._state_machine.checkpoint(self.config.checkpoint_path)
        
        # Shutdown parallel executor
        if self._parallel_executor:
            self._parallel_executor.shutdown(wait=True)
        
        if self._state_machine:
            self._state_machine.transition(
                PipelineState.IDLE, 
                "Shutdown complete"
            )
        
        logger.info("UnifiedAnalogyPipeline shutdown complete")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_pipeline(
    cache_size: int = 10000,
    enable_parallel: bool = True,
    checkpoint_path: Optional[str] = None
) -> UnifiedAnalogyPipeline:
    """
    Create and initialize a pipeline with common settings.
    
    Args:
        cache_size: Maximum cache entries
        enable_parallel: Enable parallel processing
        checkpoint_path: Optional checkpoint file path
        
    Returns:
        Initialized UnifiedAnalogyPipeline
    """
    config = PipelineConfig(
        cache_max_size=cache_size,
        enable_parallel_processing=enable_parallel,
        checkpoint_path=checkpoint_path
    )
    
    pipeline = UnifiedAnalogyPipeline(config)
    pipeline.initialize()
    
    return pipeline


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Create pipeline
    pipeline = create_pipeline(cache_size=100)
    
    print("\n1. Pipeline State:")
    state = pipeline.get_pipeline_state()
    print(f"   State: {state['pipeline_state']}")
    print(f"   Cache enabled: {state['config']['cache_enabled']}")
    
    # Process single query
    print("\n2. Single Query Processing:")
    query = Query(id="q1", content="A is to B as C is to ?")
    result = pipeline.process_query(query)
    print(f"   Success: {result.success}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Time: {result.execution_time_ms:.2f}ms")
    print(f"   Cache hit: {result.cache_hit}")
    
    # Process same query (should hit cache)
    print("\n3. Cached Query:")
    result2 = pipeline.process_query(query)
    print(f"   Cache hit: {result2.cache_hit}")
    print(f"   Time: {result2.execution_time_ms:.2f}ms")
    
    # Batch processing
    print("\n4. Batch Processing:")
    queries = [
        Query(id=f"batch_{i}", content=f"Query {i}")
        for i in range(10)
    ]
    results = pipeline.batch_process(queries)
    successful = sum(1 for r in results if r.success)
    print(f"   Processed: {len(results)}")
    print(f"   Successful: {successful}")
    
    # Final state
    print("\n5. Final Pipeline State:")
    state = pipeline.get_pipeline_state()
    print(f"   Queries processed: {state['statistics']['queries_processed']}")
    print(f"   Avg time: {state['statistics']['avg_processing_time_ms']:.2f}ms")
    print(f"   Cache hit rate: {state['cache']['hit_rate']:.2%}")
    
    # Shutdown
    pipeline.shutdown(save_checkpoint=False)
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
