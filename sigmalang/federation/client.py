"""
Federation Client - Phase 7 Track 2

Local node that participates in federated codebook learning.
Maintains a local codebook, computes updates from local data,
privatizes them, and communicates with the aggregation server.

Workflow:
    1. Train local codebook on local data
    2. Compute delta (update) from previous round
    3. Privatize delta via differential privacy
    4. Submit to aggregation server
    5. Receive global update and merge into local codebook

Usage:
    client = FederationClient(node_id="node-1")
    client.train_local(local_data)
    update = client.compute_update()
    # Send update to server, receive global
    client.apply_global_update(global_update)
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from sigmalang.federation.privacy import DifferentialPrivacy, PrivacyConfig

logger = logging.getLogger(__name__)


class LocalCodebook:
    """Local codebook maintained by a federation client."""

    def __init__(self, size: int = 256, dim: int = 64, seed: int = 42):
        self.size = size
        self.dim = dim

        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(size, dim).astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings /= norms

        self.usage_count = np.zeros(size, dtype=np.int64)
        self._previous_embeddings: Optional[np.ndarray] = None

    def snapshot(self) -> None:
        """Save current state for computing deltas."""
        self._previous_embeddings = self.embeddings.copy()

    def compute_delta(self) -> Optional[np.ndarray]:
        """Compute update delta since last snapshot."""
        if self._previous_embeddings is None:
            return None
        return self.embeddings - self._previous_embeddings

    def train_on_batch(self, features: np.ndarray, lr: float = 0.01) -> Dict[str, float]:
        """
        Train local codebook on a batch of features using online VQ.

        Args:
            features: (N, dim) feature vectors
            lr: learning rate for codebook updates

        Returns:
            Training metrics
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

        # Find nearest codebook entries
        dists = (
            (features ** 2).sum(axis=1, keepdims=True) -
            2.0 * features @ self.embeddings.T +
            (self.embeddings ** 2).sum(axis=1, keepdims=True).T
        )
        indices = dists.argmin(axis=1)

        # Update assigned entries
        total_dist = 0.0
        for i in range(self.size):
            mask = indices == i
            if not mask.any():
                continue
            assigned = features[mask]
            centroid = assigned.mean(axis=0)
            self.embeddings[i] = (1 - lr) * self.embeddings[i] + lr * centroid
            self.usage_count[i] += mask.sum()
            total_dist += dists[mask, i].sum()

        # Re-normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings /= norms

        return {
            'avg_distance': float(total_dist / max(len(features), 1)),
            'utilization': float((self.usage_count > 0).sum()) / self.size,
        }


class FederationClient:
    """
    Client node for federated codebook learning.

    Manages local training, privacy, and communication with
    the aggregation server.
    """

    def __init__(
        self,
        node_id: str,
        codebook_size: int = 256,
        codebook_dim: int = 64,
        privacy_config: Optional[PrivacyConfig] = None,
    ):
        self.node_id = node_id
        self.codebook = LocalCodebook(codebook_size, codebook_dim)
        self.privacy = DifferentialPrivacy(privacy_config)

        self._round = 0
        self._total_samples = 0
        self._metrics_history: List[Dict[str, float]] = []

    def train_local(
        self,
        data: np.ndarray,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        Train local codebook on local data.

        Args:
            data: (N, dim) local feature data
            epochs: training epochs
            batch_size: mini-batch size
            lr: learning rate

        Returns:
            Training summary
        """
        self.codebook.snapshot()
        n = data.shape[0]
        self._total_samples += n

        all_metrics = []
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = data[perm[start:start + batch_size]]
                metrics = self.codebook.train_on_batch(batch, lr=lr)
                all_metrics.append(metrics)

        avg_dist = float(np.mean([m['avg_distance'] for m in all_metrics]))
        utilization = all_metrics[-1]['utilization'] if all_metrics else 0.0

        summary = {
            'node_id': self.node_id,
            'round': self._round,
            'samples': n,
            'epochs': epochs,
            'avg_distance': round(avg_dist, 6),
            'utilization': round(utilization, 3),
        }
        self._metrics_history.append(summary)
        return summary

    def compute_update(self) -> Optional[np.ndarray]:
        """
        Compute privatized update to share with server.

        Returns:
            Privatized codebook delta, or None if no changes
        """
        delta = self.codebook.compute_delta()
        if delta is None:
            logger.warning(f"[{self.node_id}] No snapshot - cannot compute delta")
            return None

        # Privatize
        privatized = self.privacy.privatize_update(delta, self._total_samples)
        self._round += 1

        logger.info(
            f"[{self.node_id}] Round {self._round}: "
            f"delta_norm={np.linalg.norm(delta):.4f}, "
            f"privatized_norm={np.linalg.norm(privatized):.4f}"
        )

        return privatized

    def apply_global_update(
        self,
        global_update: np.ndarray,
        merge_weight: float = 0.5
    ) -> None:
        """
        Merge global aggregated update into local codebook.

        Args:
            global_update: (codebook_size, dim) aggregated delta
            merge_weight: weight for global vs local (0=keep local, 1=use global)
        """
        self.codebook.embeddings = (
            (1 - merge_weight) * self.codebook.embeddings +
            merge_weight * (self.codebook.embeddings + global_update)
        )

        # Re-normalize
        norms = np.linalg.norm(
            self.codebook.embeddings, axis=1, keepdims=True
        ) + 1e-8
        self.codebook.embeddings /= norms

        logger.info(f"[{self.node_id}] Applied global update (weight={merge_weight})")

    def get_status(self) -> Dict[str, Any]:
        """Get client status summary."""
        return {
            'node_id': self.node_id,
            'round': self._round,
            'total_samples': self._total_samples,
            'codebook_utilization': round(
                float((self.codebook.usage_count > 0).sum()) / self.codebook.size, 3
            ),
            'privacy': self.privacy.get_privacy_report(),
            'recent_metrics': self._metrics_history[-5:] if self._metrics_history else [],
        }

    def fingerprint(self) -> str:
        """Compute a fingerprint of current codebook state."""
        return hashlib.sha256(self.codebook.embeddings.tobytes()).hexdigest()[:16]
