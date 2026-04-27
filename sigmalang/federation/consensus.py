"""
Consensus Protocol - Phase 7 Track 2

Decentralized consensus for codebook primitive promotion/demotion
decisions across federation nodes. When sufficient nodes agree that
a pattern should become a shared primitive, it gets promoted.

Protocols:
    1. Voting: Simple majority voting on primitive proposals
    2. Weighted: Vote weight proportional to data volume
    3. Byzantine-tolerant: Tolerate up to f faulty nodes (f < n/3)

Usage:
    protocol = ConsensusProtocol(num_nodes=5, threshold=0.6)

    # Nodes propose primitives
    protocol.propose("node-1", primitive_vector, {"pattern": "the"})
    protocol.propose("node-2", primitive_vector, {"pattern": "the"})
    protocol.propose("node-3", primitive_vector, {"pattern": "the"})

    # Check consensus
    approved = protocol.check_consensus()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrimitiveProposal:
    """A proposed primitive from a federation node."""
    node_id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    timestamp: float = 0.0


@dataclass
class ConsensusResult:
    """Result of consensus check for a primitive candidate."""
    proposal_key: str
    approved: bool
    vote_count: int
    total_weight: float
    threshold_met: float  # Actual vote fraction
    merged_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusProtocol:
    """
    Voting-based consensus for primitive promotion.

    Nodes propose primitive vectors. Similar proposals (cosine similarity
    above merge_threshold) are grouped. When a group's vote weight
    exceeds the approval threshold, the primitive is approved.
    """

    def __init__(
        self,
        num_nodes: int = 5,
        approval_threshold: float = 0.6,
        similarity_threshold: float = 0.85,
        byzantine_tolerance: bool = False,
    ):
        self.num_nodes = num_nodes
        self.approval_threshold = approval_threshold
        self.similarity_threshold = similarity_threshold
        self.byzantine_tolerance = byzantine_tolerance

        # Proposals grouped by similarity
        self._proposal_groups: Dict[str, List[PrimitiveProposal]] = {}
        self._group_centroids: Dict[str, np.ndarray] = {}
        self._next_group_id = 0

        # History of approved primitives
        self._approved: List[ConsensusResult] = []

    def propose(
        self,
        node_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> str:
        """
        Submit a primitive proposal from a node.

        Args:
            node_id: proposing node's ID
            vector: proposed primitive vector
            metadata: optional metadata (pattern text, frequency, etc.)
            weight: vote weight (e.g., proportional to local data size)

        Returns:
            Group key the proposal was assigned to
        """
        import time

        if vector.ndim > 1:
            vector = vector.flatten()

        # Normalize
        norm = np.linalg.norm(vector) + 1e-8
        vector = vector / norm

        proposal = PrimitiveProposal(
            node_id=node_id,
            vector=vector.copy(),
            metadata=metadata or {},
            weight=weight,
            timestamp=time.time(),
        )

        # Find matching group by cosine similarity
        group_key = self._find_matching_group(vector)

        if group_key is None:
            # Create new group
            group_key = f"group_{self._next_group_id}"
            self._next_group_id += 1
            self._proposal_groups[group_key] = []
            self._group_centroids[group_key] = vector.copy()

        # Check for duplicate votes from same node
        existing_nodes = {p.node_id for p in self._proposal_groups[group_key]}
        if node_id in existing_nodes:
            logger.debug(f"Node {node_id} already voted in {group_key}, updating")
            self._proposal_groups[group_key] = [
                p for p in self._proposal_groups[group_key] if p.node_id != node_id
            ]

        self._proposal_groups[group_key].append(proposal)

        # Update centroid
        vectors = np.stack([p.vector for p in self._proposal_groups[group_key]])
        self._group_centroids[group_key] = vectors.mean(axis=0)

        return group_key

    def _find_matching_group(self, vector: np.ndarray) -> Optional[str]:
        """Find existing proposal group matching this vector."""
        best_key = None
        best_sim = -1.0

        for key, centroid in self._group_centroids.items():
            sim = float(np.dot(vector, centroid) / (
                np.linalg.norm(centroid) + 1e-8
            ))
            if sim > self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_key = key

        return best_key

    def check_consensus(self) -> List[ConsensusResult]:
        """
        Check all proposal groups for consensus.

        Returns list of ConsensusResult for groups that reached approval.
        """
        newly_approved = []

        for group_key, proposals in list(self._proposal_groups.items()):
            total_weight = sum(p.weight for p in proposals)
            vote_count = len(proposals)

            # Compute vote fraction
            if self.num_nodes > 0:
                vote_fraction = total_weight / self.num_nodes
            else:
                vote_fraction = 0.0

            # Byzantine tolerance: need > 2/3 agreement
            if self.byzantine_tolerance:
                threshold = max(self.approval_threshold, 2.0 / 3.0)
            else:
                threshold = self.approval_threshold

            approved = vote_fraction >= threshold

            if approved:
                # Compute weighted merged vector
                vectors = np.stack([p.vector for p in proposals])
                weights = np.array([p.weight for p in proposals])
                weights /= weights.sum() + 1e-8
                merged = (vectors * weights[:, None]).sum(axis=0)
                merged /= (np.linalg.norm(merged) + 1e-8)

                # Merge metadata
                merged_meta = {}
                for p in proposals:
                    for k, v in p.metadata.items():
                        if k not in merged_meta:
                            merged_meta[k] = v

                result = ConsensusResult(
                    proposal_key=group_key,
                    approved=True,
                    vote_count=vote_count,
                    total_weight=total_weight,
                    threshold_met=vote_fraction,
                    merged_vector=merged,
                    metadata=merged_meta,
                )

                newly_approved.append(result)
                self._approved.append(result)

                # Remove from pending
                del self._proposal_groups[group_key]
                del self._group_centroids[group_key]

                logger.info(
                    f"Consensus reached for {group_key}: "
                    f"{vote_count} votes, weight={total_weight:.1f}"
                )

        return newly_approved

    def get_pending_proposals(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of pending proposal groups."""
        result = {}
        for key, proposals in self._proposal_groups.items():
            result[key] = {
                'vote_count': len(proposals),
                'total_weight': sum(p.weight for p in proposals),
                'voters': [p.node_id for p in proposals],
                'vote_fraction': sum(p.weight for p in proposals) / max(self.num_nodes, 1),
                'metadata_samples': [p.metadata for p in proposals[:3]],
            }
        return result

    def get_approved_primitives(self) -> List[Dict[str, Any]]:
        """Get list of approved primitive summaries."""
        return [
            {
                'group': r.proposal_key,
                'vote_count': r.vote_count,
                'total_weight': r.total_weight,
                'threshold_met': round(r.threshold_met, 3),
                'metadata': r.metadata,
            }
            for r in self._approved
        ]

    def reset_round(self) -> None:
        """Clear all pending proposals for a new round."""
        self._proposal_groups.clear()
        self._group_centroids.clear()
