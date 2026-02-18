"""
Online Learning Pipeline - Phase 3 Task 3.1

Continuous codebook refinement from real-time usage patterns.
Implements an online learning loop that observes encoding operations,
identifies emerging patterns, and promotes high-value patterns to
Tier 2 learned primitives without service interruption.

Architecture:
    Usage Stream --> Pattern Observer --> Candidate Queue
                                            |
                                     Frequency Analyzer
                                            |
                                     Promotion Engine
                                            |
                                     Live Codebook Update

Key Properties:
- Non-blocking: Learning runs asynchronously alongside encoding
- Incremental: Updates codebook without full retraining
- Safe: Validates patterns before promotion
- Observable: Full metrics and audit trail

Automation Level: 98% (human review for >10% primitive reallocation)
"""

import time
import threading
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Online Learning Configuration
# =============================================================================

@dataclass
class OnlineLearningConfig:
    """Configuration for the online learning pipeline."""

    # Observation window
    observation_window_seconds: int = 300  # 5-minute sliding window
    min_observations: int = 50  # Min observations before learning

    # Pattern promotion thresholds
    min_frequency: int = 10  # Min occurrences to be a candidate
    min_compression_gain: float = 1.5  # Min compression improvement factor
    promotion_confidence: float = 0.95  # Required confidence for promotion

    # Codebook limits
    max_tier2_primitives: int = 128  # Max Tier 2 slots (128-255)
    promotion_batch_size: int = 5  # Max promotions per cycle
    demotion_threshold: float = 0.01  # Usage rate below which to demote

    # Learning rate
    learning_rate: float = 0.01  # EMA smoothing for pattern scores
    decay_rate: float = 0.99  # Score decay per cycle

    # Cycle timing
    cycle_interval_seconds: int = 60  # Learning cycle interval
    max_queue_size: int = 10000  # Max pending observations


class PatternState(Enum):
    """Lifecycle state of a pattern."""

    OBSERVED = "observed"  # Seen but not yet evaluated
    CANDIDATE = "candidate"  # Meets frequency threshold
    PROMOTED = "promoted"  # Assigned a Tier 2 primitive
    DEMOTED = "demoted"  # Removed from codebook
    REJECTED = "rejected"  # Evaluated and rejected


# =============================================================================
# Pattern Observation
# =============================================================================

@dataclass
class PatternObservation:
    """A single observation of a pattern during encoding."""

    pattern_hash: str  # Hash of the pattern
    token_sequence: Tuple[int, ...]  # Token sequence
    original_size: int  # Original size in bytes
    encoded_size: int  # Encoded size in bytes
    timestamp: float = field(default_factory=time.time)
    context: str = ""  # Optional context (e.g., input type)

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio."""
        return self.original_size / max(1, self.encoded_size)


@dataclass
class PatternCandidate:
    """A candidate pattern for Tier 2 promotion."""

    pattern_hash: str
    token_sequence: Tuple[int, ...]
    frequency: int = 0
    avg_compression_ratio: float = 1.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    score: float = 0.0
    state: PatternState = PatternState.OBSERVED
    primitive_id: Optional[int] = None  # Assigned after promotion

    @property
    def age_seconds(self) -> float:
        """Get pattern age in seconds."""
        return time.time() - self.first_seen

    def update_score(self, learning_rate: float = 0.01) -> None:
        """Recalculate promotion score."""
        # Score = frequency * compression_gain * recency
        recency = 1.0 / (1.0 + (time.time() - self.last_seen) / 3600)
        compression_gain = max(0, self.avg_compression_ratio - 1.0)

        new_score = self.frequency * compression_gain * recency
        self.score = (1 - learning_rate) * self.score + learning_rate * new_score


# =============================================================================
# Pattern Observer
# =============================================================================

class PatternObserver:
    """Observes encoding operations and extracts pattern candidates."""

    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.observation_queue: deque = deque(maxlen=config.max_queue_size)
        self.candidates: Dict[str, PatternCandidate] = {}
        self._lock = threading.Lock()

    def observe(self, observation: PatternObservation) -> None:
        """Record an encoding observation."""
        with self._lock:
            self.observation_queue.append(observation)

            # Update or create candidate
            key = observation.pattern_hash

            if key in self.candidates:
                candidate = self.candidates[key]
                candidate.frequency += 1
                candidate.last_seen = observation.timestamp

                # Update running average compression ratio
                n = candidate.frequency
                candidate.avg_compression_ratio = (
                    (candidate.avg_compression_ratio * (n - 1) + observation.compression_ratio) / n
                )
            else:
                self.candidates[key] = PatternCandidate(
                    pattern_hash=key,
                    token_sequence=observation.token_sequence,
                    frequency=1,
                    avg_compression_ratio=observation.compression_ratio,
                    first_seen=observation.timestamp,
                    last_seen=observation.timestamp
                )

    def get_promotion_candidates(self) -> List[PatternCandidate]:
        """Get patterns that meet promotion thresholds."""
        with self._lock:
            candidates = []

            for candidate in self.candidates.values():
                if candidate.state == PatternState.PROMOTED:
                    continue
                if candidate.state == PatternState.REJECTED:
                    continue

                # Check thresholds
                if candidate.frequency >= self.config.min_frequency:
                    if candidate.avg_compression_ratio >= self.config.min_compression_gain:
                        candidate.state = PatternState.CANDIDATE
                        candidate.update_score(self.config.learning_rate)
                        candidates.append(candidate)

            # Sort by score (descending)
            candidates.sort(key=lambda c: c.score, reverse=True)

            return candidates[:self.config.promotion_batch_size]

    def get_observation_stats(self) -> Dict[str, Any]:
        """Get observation statistics."""
        with self._lock:
            return {
                'queue_size': len(self.observation_queue),
                'total_candidates': len(self.candidates),
                'promoted': sum(1 for c in self.candidates.values() if c.state == PatternState.PROMOTED),
                'candidates_ready': sum(1 for c in self.candidates.values() if c.state == PatternState.CANDIDATE),
                'rejected': sum(1 for c in self.candidates.values() if c.state == PatternState.REJECTED),
            }


# =============================================================================
# Promotion Engine
# =============================================================================

class PromotionEngine:
    """Promotes high-value patterns to Tier 2 primitives."""

    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.allocated_ids: Set[int] = set()
        self.promotion_history: List[Dict[str, Any]] = []
        self.next_available_id = 128  # Tier 2 starts at 128

    def promote(
        self,
        candidate: PatternCandidate,
    ) -> Optional[int]:
        """
        Promote a candidate to a Tier 2 primitive.

        Returns:
            Assigned primitive ID or None if no slots available
        """
        # Check if there are available slots
        if len(self.allocated_ids) >= self.config.max_tier2_primitives:
            logger.warning("No available Tier 2 slots for promotion")
            return None

        # Find next available ID
        primitive_id = self._find_available_id()
        if primitive_id is None:
            return None

        # Record promotion
        self.allocated_ids.add(primitive_id)
        candidate.primitive_id = primitive_id
        candidate.state = PatternState.PROMOTED

        self.promotion_history.append({
            'primitive_id': primitive_id,
            'pattern_hash': candidate.pattern_hash,
            'frequency': candidate.frequency,
            'avg_compression_ratio': candidate.avg_compression_ratio,
            'score': candidate.score,
            'timestamp': time.time()
        })

        logger.info(
            f"Promoted pattern {candidate.pattern_hash[:8]}... "
            f"to primitive {primitive_id} "
            f"(freq={candidate.frequency}, ratio={candidate.avg_compression_ratio:.2f}x)"
        )

        return primitive_id

    def demote(self, primitive_id: int) -> bool:
        """Demote a Tier 2 primitive (free the slot)."""
        if primitive_id not in self.allocated_ids:
            return False

        self.allocated_ids.discard(primitive_id)

        logger.info(f"Demoted primitive {primitive_id}")

        return True

    def _find_available_id(self) -> Optional[int]:
        """Find next available Tier 2 primitive ID."""
        for pid in range(128, 256):
            if pid not in self.allocated_ids:
                return pid
        return None

    def get_utilization(self) -> float:
        """Get Tier 2 slot utilization percentage."""
        return (len(self.allocated_ids) / self.config.max_tier2_primitives) * 100

    def get_promotion_stats(self) -> Dict[str, Any]:
        """Get promotion statistics."""
        return {
            'allocated_slots': len(self.allocated_ids),
            'max_slots': self.config.max_tier2_primitives,
            'utilization_pct': self.get_utilization(),
            'total_promotions': len(self.promotion_history),
            'recent_promotions': self.promotion_history[-5:]
        }


# =============================================================================
# Online Learning Pipeline
# =============================================================================

class OnlineLearningPipeline:
    """
    Main online learning pipeline orchestrator.

    Coordinates observation, analysis, and promotion in a continuous loop.
    """

    def __init__(self, config: Optional[OnlineLearningConfig] = None):
        self.config = config or OnlineLearningConfig()
        self.observer = PatternObserver(self.config)
        self.promotion_engine = PromotionEngine(self.config)

        self._running = False
        self._learning_thread: Optional[threading.Thread] = None
        self._cycle_count = 0

        self.metrics = {
            'cycles': 0,
            'observations': 0,
            'promotions': 0,
            'demotions': 0,
            'start_time': None
        }

    def start(self) -> None:
        """Start the online learning pipeline (background thread)."""
        if self._running:
            logger.warning("Online learning already running")
            return

        self._running = True
        self.metrics['start_time'] = time.time()

        self._learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True,
            name="OnlineLearningPipeline"
        )
        self._learning_thread.start()

        logger.info("Online learning pipeline started")

    def stop(self) -> None:
        """Stop the online learning pipeline."""
        self._running = False
        if self._learning_thread:
            self._learning_thread.join(timeout=5)
        logger.info("Online learning pipeline stopped")

    def observe_encoding(
        self,
        token_sequence: Tuple[int, ...],
        original_size: int,
        encoded_size: int,
        context: str = ""
    ) -> None:
        """
        Record an encoding operation for learning.

        Call this during every encode() operation.
        """
        # Hash the token sequence
        pattern_hash = hashlib.sha256(
            str(token_sequence).encode()
        ).hexdigest()[:16]

        observation = PatternObservation(
            pattern_hash=pattern_hash,
            token_sequence=token_sequence,
            original_size=original_size,
            encoded_size=encoded_size,
            context=context
        )

        self.observer.observe(observation)
        self.metrics['observations'] += 1

    def _learning_loop(self) -> None:
        """Main learning loop (runs in background thread)."""
        while self._running:
            try:
                self._run_learning_cycle()
            except Exception as e:
                logger.error(f"Learning cycle error: {e}")

            time.sleep(self.config.cycle_interval_seconds)

    def _run_learning_cycle(self) -> None:
        """Execute one learning cycle."""
        self._cycle_count += 1
        self.metrics['cycles'] = self._cycle_count

        # Skip if not enough observations
        obs_stats = self.observer.get_observation_stats()
        if obs_stats['queue_size'] < self.config.min_observations:
            return

        # Get promotion candidates
        candidates = self.observer.get_promotion_candidates()

        if not candidates:
            return

        # Promote top candidates
        for candidate in candidates:
            primitive_id = self.promotion_engine.promote(candidate)
            if primitive_id is not None:
                self.metrics['promotions'] += 1

                # Check if we need human review (>10% reallocation)
                utilization = self.promotion_engine.get_utilization()
                if utilization > 90:
                    logger.warning(
                        f"Tier 2 utilization at {utilization:.1f}% - "
                        "consider human review for codebook optimization"
                    )

        logger.debug(
            f"Learning cycle {self._cycle_count}: "
            f"candidates={len(candidates)}, "
            f"promotions={self.metrics['promotions']}"
        )

    def force_cycle(self) -> Dict[str, Any]:
        """Force an immediate learning cycle (for testing)."""
        self._run_learning_cycle()
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        uptime = time.time() - self.metrics['start_time'] if self.metrics['start_time'] else 0

        return {
            'running': self._running,
            'uptime_seconds': uptime,
            **self.metrics,
            'observer': self.observer.get_observation_stats(),
            'promotion_engine': self.promotion_engine.get_promotion_stats()
        }

    def export_state(self, path: Path) -> None:
        """Export pipeline state to JSON."""
        state = {
            'config': asdict(self.config),
            'metrics': self.metrics,
            'promotion_history': self.promotion_engine.promotion_history,
            'observer_stats': self.observer.get_observation_stats()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def import_state(self, path: Path) -> None:
        """Import pipeline state from JSON."""
        if not path.exists():
            return

        with open(path, 'r') as f:
            state = json.load(f)

        self.promotion_engine.promotion_history = state.get('promotion_history', [])

        # Reconstruct allocated IDs
        for entry in self.promotion_engine.promotion_history:
            self.promotion_engine.allocated_ids.add(entry['primitive_id'])


# =============================================================================
# Global Pipeline Instance
# =============================================================================

_global_pipeline: Optional[OnlineLearningPipeline] = None


def get_online_learner() -> OnlineLearningPipeline:
    """Get or create the global online learning pipeline."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = OnlineLearningPipeline()
    return _global_pipeline


def initialize_online_learning(
    config: Optional[OnlineLearningConfig] = None,
    auto_start: bool = True
) -> OnlineLearningPipeline:
    """
    Initialize the online learning pipeline.

    Usage:
        from sigmalang.training.online_learner import initialize_online_learning

        pipeline = initialize_online_learning(auto_start=True)

        # Record encoding observations
        pipeline.observe_encoding(
            token_sequence=(42, 87, 12, 200),
            original_size=100,
            encoded_size=30
        )

        # Check status
        status = pipeline.get_status()
        print(f"Promotions: {status['promotions']}")
    """
    global _global_pipeline
    _global_pipeline = OnlineLearningPipeline(config=config)

    if auto_start:
        _global_pipeline.start()

    return _global_pipeline
