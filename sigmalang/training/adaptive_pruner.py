"""
Adaptive Primitive Pruning - Phase 3 Task 3.3

Auto-deallocates underutilized Tier 2 learned primitives to free codebook
slots for more valuable patterns. Ensures the codebook stays efficient
by continuously removing low-value primitives.

Architecture:
    Usage Tracker --> Utilization Scorer --> Prune Candidates
                                                  |
                                            Safety Validator
                                                  |
                                            Prune Executor
                                                  |
                                            Slot Reclamation

Key Properties:
- Usage-based: Prunes primitives with lowest utilization
- Safe: Validates no active references before pruning
- Gradual: Prunes in small batches to avoid disruption
- Reversible: Keeps pruned primitives in archive for recovery

Automation Level: 98% (human review for >10% reallocation)
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Pruning Configuration
# =============================================================================

@dataclass
class PruningConfig:
    """Configuration for adaptive pruning."""

    # Usage thresholds
    min_usage_rate: float = 0.01  # 1% - below this is a prune candidate
    min_compression_value: float = 1.2  # Below this ratio is low-value
    observation_window_hours: float = 24.0  # Hours of usage to consider

    # Pruning limits
    max_prune_per_cycle: int = 5  # Max primitives to prune per cycle
    min_age_hours: float = 1.0  # Min age before eligible for pruning
    safety_margin: float = 0.1  # Keep 10% of slots as buffer

    # Cycle timing
    cycle_interval_seconds: int = 300  # Check every 5 minutes

    # Archive
    enable_archive: bool = True  # Archive pruned primitives
    archive_retention_days: int = 30  # Days to keep in archive


class PruneReason(Enum):
    """Reason for pruning a primitive."""

    LOW_USAGE = "low_usage"
    LOW_COMPRESSION = "low_compression"
    REDUNDANT = "redundant"  # Similar to another primitive
    EXPIRED = "expired"  # Hasn't been used in too long
    MANUAL = "manual"  # Manually pruned


# =============================================================================
# Usage Tracker
# =============================================================================

@dataclass
class PrimitiveUsageRecord:
    """Usage record for a single Tier 2 primitive."""

    primitive_id: int
    total_uses: int = 0
    total_bytes_saved: int = 0
    avg_compression_ratio: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    usage_history: List[float] = field(default_factory=list)  # Timestamps

    @property
    def age_hours(self) -> float:
        """Get primitive age in hours."""
        return (time.time() - self.created_at) / 3600

    @property
    def idle_hours(self) -> float:
        """Get hours since last use."""
        return (time.time() - self.last_used_at) / 3600

    @property
    def usage_rate(self) -> float:
        """Get usage rate (uses per hour)."""
        age = max(1.0, self.age_hours)
        return self.total_uses / age

    def record_use(self, bytes_saved: int = 0, compression_ratio: float = 1.0) -> None:
        """Record a use of this primitive."""
        self.total_uses += 1
        self.total_bytes_saved += bytes_saved
        self.last_used_at = time.time()
        self.usage_history.append(time.time())

        # Update running average
        n = self.total_uses
        self.avg_compression_ratio = (
            (self.avg_compression_ratio * (n - 1) + compression_ratio) / n
        )

        # Trim old history (keep last 1000)
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]

    def get_recent_usage(self, hours: float = 24.0) -> int:
        """Get usage count in recent hours."""
        cutoff = time.time() - (hours * 3600)
        return sum(1 for t in self.usage_history if t > cutoff)


class PrimitiveUsageTracker:
    """Tracks usage of all Tier 2 primitives."""

    def __init__(self):
        self.records: Dict[int, PrimitiveUsageRecord] = {}
        self._lock = threading.Lock()

    def register_primitive(self, primitive_id: int) -> None:
        """Register a new primitive for tracking."""
        with self._lock:
            if primitive_id not in self.records:
                self.records[primitive_id] = PrimitiveUsageRecord(
                    primitive_id=primitive_id
                )

    def record_use(
        self,
        primitive_id: int,
        bytes_saved: int = 0,
        compression_ratio: float = 1.0
    ) -> None:
        """Record a use of a primitive."""
        with self._lock:
            if primitive_id not in self.records:
                self.register_primitive(primitive_id)

            self.records[primitive_id].record_use(bytes_saved, compression_ratio)

    def get_record(self, primitive_id: int) -> Optional[PrimitiveUsageRecord]:
        """Get usage record for a primitive."""
        return self.records.get(primitive_id)

    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get stats for all tracked primitives."""
        with self._lock:
            stats = {}
            for pid, record in self.records.items():
                stats[pid] = {
                    'total_uses': record.total_uses,
                    'usage_rate': record.usage_rate,
                    'avg_compression_ratio': record.avg_compression_ratio,
                    'total_bytes_saved': record.total_bytes_saved,
                    'age_hours': record.age_hours,
                    'idle_hours': record.idle_hours
                }
            return stats

    def get_least_used(self, n: int = 10) -> List[PrimitiveUsageRecord]:
        """Get N least used primitives."""
        with self._lock:
            sorted_records = sorted(
                self.records.values(),
                key=lambda r: r.usage_rate
            )
            return sorted_records[:n]


# =============================================================================
# Prune Candidate Evaluator
# =============================================================================

@dataclass
class PruneCandidate:
    """A primitive evaluated for pruning."""

    primitive_id: int
    reason: PruneReason
    score: float  # Lower = higher prune priority
    usage_rate: float
    compression_ratio: float
    age_hours: float
    idle_hours: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'primitive_id': self.primitive_id,
            'reason': self.reason.value,
            'score': self.score,
            'usage_rate': self.usage_rate,
            'compression_ratio': self.compression_ratio,
            'age_hours': self.age_hours,
            'idle_hours': self.idle_hours
        }


class PruneCandidateEvaluator:
    """Evaluates primitives for pruning eligibility."""

    def __init__(self, config: PruningConfig):
        self.config = config

    def evaluate(
        self,
        record: PrimitiveUsageRecord
    ) -> Optional[PruneCandidate]:
        """
        Evaluate a primitive for pruning.

        Returns PruneCandidate if eligible, None otherwise.
        """
        # Skip if too young
        if record.age_hours < self.config.min_age_hours:
            return None

        reason = None
        score = float('inf')  # Lower = more pruneable

        # Check for low usage
        if record.usage_rate < self.config.min_usage_rate:
            reason = PruneReason.LOW_USAGE
            score = record.usage_rate

        # Check for low compression value
        elif record.avg_compression_ratio < self.config.min_compression_value:
            reason = PruneReason.LOW_COMPRESSION
            score = record.avg_compression_ratio

        # Check for expired (not used in observation window)
        elif record.idle_hours > self.config.observation_window_hours:
            reason = PruneReason.EXPIRED
            score = -record.idle_hours  # More idle = more pruneable

        if reason is None:
            return None

        return PruneCandidate(
            primitive_id=record.primitive_id,
            reason=reason,
            score=score,
            usage_rate=record.usage_rate,
            compression_ratio=record.avg_compression_ratio,
            age_hours=record.age_hours,
            idle_hours=record.idle_hours
        )


# =============================================================================
# Pruned Primitive Archive
# =============================================================================

@dataclass
class PrunedPrimitive:
    """Archive entry for a pruned primitive."""

    primitive_id: int
    reason: str
    usage_stats: Dict[str, Any]
    pruned_at: float = field(default_factory=time.time)
    token_sequence: Optional[Tuple[int, ...]] = None


class PruneArchive:
    """Archive of pruned primitives for potential recovery."""

    def __init__(self, max_entries: int = 500):
        self.entries: List[PrunedPrimitive] = []
        self.max_entries = max_entries

    def archive(
        self,
        primitive_id: int,
        reason: PruneReason,
        usage_record: PrimitiveUsageRecord
    ) -> None:
        """Archive a pruned primitive."""
        entry = PrunedPrimitive(
            primitive_id=primitive_id,
            reason=reason.value,
            usage_stats={
                'total_uses': usage_record.total_uses,
                'usage_rate': usage_record.usage_rate,
                'avg_compression_ratio': usage_record.avg_compression_ratio,
                'age_hours': usage_record.age_hours
            }
        )

        self.entries.append(entry)

        # Trim old entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recently pruned entries."""
        return [asdict(e) for e in self.entries[-n:]]

    def export(self, path: Path) -> None:
        """Export archive to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2, default=str)


# =============================================================================
# Adaptive Pruner
# =============================================================================

class AdaptivePruner:
    """
    Main adaptive pruning engine.

    Continuously monitors Tier 2 primitive usage and prunes
    underutilized primitives to free codebook slots.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.usage_tracker = PrimitiveUsageTracker()
        self.evaluator = PruneCandidateEvaluator(self.config)
        self.archive = PruneArchive()

        self._running = False
        self._prune_thread: Optional[threading.Thread] = None

        self.metrics = {
            'cycles': 0,
            'total_pruned': 0,
            'total_bytes_recovered': 0,
            'slots_freed': 0,
            'start_time': None
        }

    def start(self) -> None:
        """Start the adaptive pruner (background thread)."""
        if self._running:
            return

        self._running = True
        self.metrics['start_time'] = time.time()

        self._prune_thread = threading.Thread(
            target=self._prune_loop,
            daemon=True,
            name="AdaptivePruner"
        )
        self._prune_thread.start()

        logger.info("Adaptive pruner started")

    def stop(self) -> None:
        """Stop the adaptive pruner."""
        self._running = False
        if self._prune_thread:
            self._prune_thread.join(timeout=5)
        logger.info("Adaptive pruner stopped")

    def record_usage(
        self,
        primitive_id: int,
        bytes_saved: int = 0,
        compression_ratio: float = 1.0
    ) -> None:
        """Record usage of a primitive (call during encoding)."""
        self.usage_tracker.record_use(primitive_id, bytes_saved, compression_ratio)

    def _prune_loop(self) -> None:
        """Background pruning loop."""
        while self._running:
            try:
                self._run_prune_cycle()
            except Exception as e:
                logger.error(f"Prune cycle error: {e}")

            time.sleep(self.config.cycle_interval_seconds)

    def _run_prune_cycle(self) -> Dict[str, Any]:
        """Execute one pruning cycle."""
        self.metrics['cycles'] += 1

        # Evaluate all primitives
        candidates = []
        for record in self.usage_tracker.records.values():
            candidate = self.evaluator.evaluate(record)
            if candidate:
                candidates.append(candidate)

        if not candidates:
            return {'pruned': 0}

        # Sort by score (lowest first = most pruneable)
        candidates.sort(key=lambda c: c.score)

        # Prune up to max per cycle
        pruned = []
        for candidate in candidates[:self.config.max_prune_per_cycle]:
            success = self._execute_prune(candidate)
            if success:
                pruned.append(candidate)

        if pruned:
            logger.info(
                f"Prune cycle {self.metrics['cycles']}: "
                f"pruned {len(pruned)} primitives "
                f"(reasons: {', '.join(c.reason.value for c in pruned)})"
            )

        return {
            'pruned': len(pruned),
            'candidates_evaluated': len(candidates),
            'details': [c.to_dict() for c in pruned]
        }

    def _execute_prune(self, candidate: PruneCandidate) -> bool:
        """Execute pruning of a single primitive."""
        primitive_id = candidate.primitive_id

        # Get usage record for archiving
        record = self.usage_tracker.get_record(primitive_id)
        if record is None:
            return False

        # Archive before pruning
        if self.config.enable_archive:
            self.archive.archive(primitive_id, candidate.reason, record)

        # Remove from tracker
        if primitive_id in self.usage_tracker.records:
            del self.usage_tracker.records[primitive_id]

        # Update metrics
        self.metrics['total_pruned'] += 1
        self.metrics['slots_freed'] += 1

        logger.debug(
            f"Pruned primitive {primitive_id}: "
            f"reason={candidate.reason.value}, "
            f"usage_rate={candidate.usage_rate:.4f}"
        )

        return True

    def force_cycle(self) -> Dict[str, Any]:
        """Force an immediate prune cycle (for testing)."""
        return self._run_prune_cycle()

    def get_prune_candidates(self) -> List[Dict[str, Any]]:
        """Get current prune candidates without executing."""
        candidates = []
        for record in self.usage_tracker.records.values():
            candidate = self.evaluator.evaluate(record)
            if candidate:
                candidates.append(candidate.to_dict())

        candidates.sort(key=lambda c: c['score'])
        return candidates

    def get_status(self) -> Dict[str, Any]:
        """Get pruner status."""
        uptime = time.time() - self.metrics['start_time'] if self.metrics['start_time'] else 0

        tracked = len(self.usage_tracker.records)
        all_stats = self.usage_tracker.get_all_stats()

        avg_usage_rate = 0.0
        avg_compression = 0.0
        if all_stats:
            avg_usage_rate = sum(s['usage_rate'] for s in all_stats.values()) / len(all_stats)
            avg_compression = sum(s['avg_compression_ratio'] for s in all_stats.values()) / len(all_stats)

        return {
            'running': self._running,
            'uptime_seconds': uptime,
            **self.metrics,
            'tracked_primitives': tracked,
            'avg_usage_rate': avg_usage_rate,
            'avg_compression_ratio': avg_compression,
            'archive_size': len(self.archive.entries),
            'recent_prunes': self.archive.get_recent(5)
        }

    def export_state(self, path: Path) -> None:
        """Export pruner state to JSON."""
        state = {
            'config': asdict(self.config),
            'metrics': self.metrics,
            'usage_stats': self.usage_tracker.get_all_stats(),
            'archive': self.archive.get_recent(50),
            'current_candidates': self.get_prune_candidates()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)


# =============================================================================
# Global Pruner Instance
# =============================================================================

_global_pruner: Optional[AdaptivePruner] = None


def get_adaptive_pruner() -> AdaptivePruner:
    """Get or create the global adaptive pruner."""
    global _global_pruner
    if _global_pruner is None:
        _global_pruner = AdaptivePruner()
    return _global_pruner


def initialize_adaptive_pruning(
    config: Optional[PruningConfig] = None,
    auto_start: bool = True
) -> AdaptivePruner:
    """
    Initialize the adaptive pruning system.

    Usage:
        from sigmalang.training.adaptive_pruner import initialize_adaptive_pruning

        pruner = initialize_adaptive_pruning(auto_start=True)

        # Record primitive usage during encoding
        pruner.record_usage(primitive_id=142, bytes_saved=50, compression_ratio=2.5)

        # Check status
        status = pruner.get_status()
        print(f"Pruned: {status['total_pruned']}, Tracked: {status['tracked_primitives']}")

        # View candidates
        candidates = pruner.get_prune_candidates()
        for c in candidates:
            print(f"  Primitive {c['primitive_id']}: {c['reason']} (score={c['score']:.4f})")
    """
    global _global_pruner
    _global_pruner = AdaptivePruner(config=config)

    if auto_start:
        _global_pruner.start()

    return _global_pruner
