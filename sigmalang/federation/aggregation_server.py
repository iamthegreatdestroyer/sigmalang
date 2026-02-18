"""
Aggregation Server - Phase 7 Track 2

Central coordinator for federated codebook learning.
Receives privatized updates from clients, aggregates them,
and distributes the global update.

Supports:
    - Weighted federated averaging (FedAvg)
    - Secure aggregation (simulated)
    - Staleness-aware weighting
    - Round management and convergence tracking

Usage:
    server = AggregationServer(min_clients=3)

    # Clients submit updates
    server.receive_update("node-1", update_1, weight=100)
    server.receive_update("node-2", update_2, weight=200)
    server.receive_update("node-3", update_3, weight=150)

    # Aggregate and broadcast
    global_update = server.aggregate_round()
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np

from sigmalang.federation.privacy import SecureAggregator

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Update submitted by a federation client."""
    node_id: str
    update: np.ndarray
    weight: float  # Typically number of local samples
    timestamp: float = field(default_factory=time.time)
    round_number: int = 0


@dataclass
class RoundResult:
    """Result of one aggregation round."""
    round_number: int
    global_update: np.ndarray
    num_participants: int
    total_weight: float
    timestamp: float = field(default_factory=time.time)
    convergence_delta: float = 0.0


class AggregationServer:
    """
    Central aggregation server for federated codebook learning.

    Implements Federated Averaging (FedAvg) with optional
    secure aggregation and staleness weighting.
    """

    def __init__(
        self,
        min_clients: int = 3,
        max_staleness: int = 3,
        use_secure_aggregation: bool = True,
    ):
        self.min_clients = min_clients
        self.max_staleness = max_staleness

        self._current_round = 0
        self._pending_updates: Dict[str, ClientUpdate] = {}
        self._round_history: List[RoundResult] = []
        self._global_codebook: Optional[np.ndarray] = None

        # Secure aggregation
        self._secure_agg = SecureAggregator(min_clients) if use_secure_aggregation else None

        # Client tracking
        self._client_rounds: Dict[str, int] = {}

    def receive_update(
        self,
        node_id: str,
        update: np.ndarray,
        weight: float = 1.0,
        client_round: Optional[int] = None,
    ) -> bool:
        """
        Receive a privatized update from a client.

        Args:
            node_id: unique client identifier
            update: privatized codebook delta
            weight: aggregation weight (e.g., num local samples)
            client_round: client's current round number

        Returns:
            True if accepted, False if rejected (staleness, duplicate)
        """
        # Check staleness
        if client_round is not None:
            staleness = self._current_round - client_round
            if staleness > self.max_staleness:
                logger.warning(
                    f"Rejecting stale update from {node_id}: "
                    f"round {client_round} vs current {self._current_round}"
                )
                return False

        # Check for duplicate
        if node_id in self._pending_updates:
            logger.warning(f"Duplicate update from {node_id}, replacing")

        self._pending_updates[node_id] = ClientUpdate(
            node_id=node_id,
            update=update,
            weight=weight,
            round_number=client_round or self._current_round,
        )

        # Also submit to secure aggregation if enabled
        if self._secure_agg is not None:
            self._secure_agg.submit_update(node_id, update * weight)

        self._client_rounds[node_id] = self._current_round

        logger.info(
            f"Received update from {node_id} "
            f"(weight={weight}, pending={len(self._pending_updates)})"
        )
        return True

    def can_aggregate(self) -> bool:
        """Check if enough clients have submitted for this round."""
        return len(self._pending_updates) >= self.min_clients

    def aggregate_round(self) -> Optional[RoundResult]:
        """
        Aggregate all pending updates into a global update.

        Uses weighted averaging (FedAvg):
            global = sum(weight_i * update_i) / sum(weight_i)

        Returns:
            RoundResult, or None if insufficient participants
        """
        if not self.can_aggregate():
            logger.warning(
                f"Cannot aggregate: {len(self._pending_updates)}/{self.min_clients} clients"
            )
            return None

        updates = list(self._pending_updates.values())

        # Weighted average
        total_weight = sum(u.weight for u in updates)
        if total_weight < 1e-8:
            total_weight = len(updates)  # Fallback to equal weights

        # Apply staleness discount
        weighted_sum = np.zeros_like(updates[0].update)
        for u in updates:
            staleness = self._current_round - u.round_number
            staleness_factor = 1.0 / (1.0 + 0.5 * staleness)
            weighted_sum += (u.weight * staleness_factor / total_weight) * u.update

        # Track convergence
        convergence_delta = float(np.linalg.norm(weighted_sum))

        # Update global codebook
        if self._global_codebook is not None:
            self._global_codebook = self._global_codebook + weighted_sum
        else:
            self._global_codebook = weighted_sum

        result = RoundResult(
            round_number=self._current_round,
            global_update=weighted_sum,
            num_participants=len(updates),
            total_weight=total_weight,
            convergence_delta=convergence_delta,
        )

        self._round_history.append(result)
        self._current_round += 1
        self._pending_updates.clear()

        logger.info(
            f"Round {result.round_number}: aggregated {result.num_participants} updates, "
            f"convergence_delta={convergence_delta:.6f}"
        )

        return result

    def get_global_update(self) -> Optional[np.ndarray]:
        """Get the latest global update to distribute to clients."""
        if not self._round_history:
            return None
        return self._round_history[-1].global_update

    def is_converged(self, threshold: float = 1e-4, lookback: int = 5) -> bool:
        """
        Check if federation has converged.

        Converged when last `lookback` rounds all have
        convergence_delta below threshold.
        """
        if len(self._round_history) < lookback:
            return False

        recent = self._round_history[-lookback:]
        return all(r.convergence_delta < threshold for r in recent)

    def get_status(self) -> Dict[str, Any]:
        """Get server status summary."""
        return {
            'current_round': self._current_round,
            'pending_updates': len(self._pending_updates),
            'min_clients': self.min_clients,
            'total_rounds_completed': len(self._round_history),
            'registered_clients': list(self._client_rounds.keys()),
            'converged': self.is_converged(),
            'recent_deltas': [
                round(r.convergence_delta, 6)
                for r in self._round_history[-10:]
            ],
        }

    def get_round_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent round history."""
        return [
            {
                'round': r.round_number,
                'participants': r.num_participants,
                'total_weight': r.total_weight,
                'convergence_delta': round(r.convergence_delta, 6),
                'timestamp': r.timestamp,
            }
            for r in self._round_history[-last_n:]
        ]
