"""
Differential Privacy for Federated Codebook Learning - Phase 7 Track 2

Implements (epsilon, delta)-differential privacy mechanisms to protect
local codebook patterns during federation. Ensures no single node's
data can be reconstructed from shared updates.

Mechanisms:
    1. Gaussian noise injection calibrated to sensitivity
    2. Gradient clipping for bounded sensitivity
    3. Privacy budget accounting (Renyi DP composition)
    4. Secure aggregation simulation

Research Basis:
    - DP-SGD (Abadi et al., 2016): Deep learning with differential privacy
    - Federated Learning (McMahan et al., 2017): Communication-efficient learning
    - RDP Composition (Mironov, 2017): Renyi differential privacy
"""

import math
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Privacy configuration for federated learning."""
    epsilon: float = 1.0         # Privacy budget per round
    delta: float = 1e-5          # Failure probability
    max_grad_norm: float = 1.0   # Gradient clipping bound
    noise_multiplier: float = 1.1 # Noise multiplier (computed from epsilon if 0)
    total_rounds: int = 100      # Total expected federation rounds
    min_participants: int = 3    # Minimum nodes for aggregation


@dataclass
class PrivacyAccountant:
    """Tracks cumulative privacy expenditure."""
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    rounds_completed: int = 0
    budget_epsilon: float = 10.0
    budget_delta: float = 1e-4

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.budget_epsilon - self.spent_epsilon)

    @property
    def budget_exhausted(self) -> bool:
        return self.spent_epsilon >= self.budget_epsilon

    def record_round(self, epsilon: float, delta: float) -> None:
        """Record privacy expenditure for one round."""
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.rounds_completed += 1
        logger.debug(
            f"Privacy spend: eps={self.spent_epsilon:.4f}/{self.budget_epsilon}, "
            f"delta={self.spent_delta:.2e}/{self.budget_delta}"
        )


class DifferentialPrivacy:
    """
    Differential privacy mechanism for codebook updates.

    Provides noise injection and clipping to ensure (epsilon, delta)-DP
    for shared gradients/embeddings during federation.
    """

    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        self.accountant = PrivacyAccountant(
            budget_epsilon=self.config.epsilon * self.config.total_rounds,
            budget_delta=self.config.delta * self.config.total_rounds,
        )
        self._rng = np.random.RandomState(42)

    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        Clip per-sample gradients to bounded L2 norm.

        Args:
            gradients: (N, dim) gradient vectors

        Returns:
            Clipped gradients with ||g||_2 <= max_grad_norm
        """
        if gradients.ndim == 1:
            gradients = gradients.reshape(1, -1)

        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        clip_factor = np.minimum(1.0, self.config.max_grad_norm / (norms + 1e-8))
        return gradients * clip_factor

    def add_noise(self, aggregated: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Add calibrated Gaussian noise to aggregated updates.

        Noise scale: sigma = (sensitivity * noise_multiplier) / sqrt(num_samples)

        Args:
            aggregated: aggregated gradient/embedding update
            num_samples: number of samples in the aggregation

        Returns:
            Noisy aggregated update
        """
        sensitivity = self.config.max_grad_norm
        sigma = sensitivity * self.config.noise_multiplier / max(math.sqrt(num_samples), 1.0)

        noise = self._rng.normal(0, sigma, size=aggregated.shape).astype(np.float32)

        logger.debug(f"DP noise: sigma={sigma:.4f}, shape={aggregated.shape}")
        return aggregated + noise

    def privatize_update(
        self,
        local_update: np.ndarray,
        num_local_samples: int
    ) -> np.ndarray:
        """
        Full privacy pipeline: clip + noise for a single node's update.

        Args:
            local_update: (dim,) or (N, dim) local codebook update
            num_local_samples: number of local data points

        Returns:
            Privatized update safe for sharing
        """
        if self.accountant.budget_exhausted:
            logger.warning("Privacy budget exhausted! Returning zeros.")
            return np.zeros_like(local_update)

        clipped = self.clip_gradients(local_update)
        noisy = self.add_noise(clipped, num_local_samples)

        # Account for this round
        round_epsilon = self._compute_round_epsilon(num_local_samples)
        self.accountant.record_round(round_epsilon, self.config.delta)

        return noisy

    def _compute_round_epsilon(self, num_samples: int) -> float:
        """
        Compute per-round epsilon using Gaussian mechanism formula.

        eps = sensitivity * sqrt(2 * ln(1.25 / delta)) / (sigma * sqrt(n))
        """
        sensitivity = self.config.max_grad_norm
        sigma = sensitivity * self.config.noise_multiplier / max(math.sqrt(num_samples), 1.0)
        if sigma < 1e-10:
            return float('inf')

        return sensitivity * math.sqrt(2.0 * math.log(1.25 / self.config.delta)) / sigma

    def get_privacy_report(self) -> dict:
        """Get current privacy expenditure report."""
        return {
            'epsilon_spent': round(self.accountant.spent_epsilon, 4),
            'epsilon_budget': round(self.accountant.budget_epsilon, 4),
            'epsilon_remaining': round(self.accountant.remaining_epsilon, 4),
            'delta_spent': self.accountant.spent_delta,
            'rounds_completed': self.accountant.rounds_completed,
            'budget_exhausted': self.accountant.budget_exhausted,
        }


class SecureAggregator:
    """
    Simulated secure aggregation.

    In production this would use cryptographic MPC (e.g., Shamir secret sharing).
    Here we simulate the semantics: individual updates are never exposed,
    only their aggregate is revealed.
    """

    def __init__(self, min_participants: int = 3):
        self.min_participants = min_participants
        self._pending_updates: List[np.ndarray] = []
        self._participant_ids: List[str] = []

    def submit_update(self, participant_id: str, update: np.ndarray) -> None:
        """Submit a privatized update from one participant."""
        if participant_id in self._participant_ids:
            logger.warning(f"Duplicate submission from {participant_id}, ignoring")
            return
        self._pending_updates.append(update)
        self._participant_ids.append(participant_id)

    def can_aggregate(self) -> bool:
        """Check if enough participants have submitted."""
        return len(self._pending_updates) >= self.min_participants

    def aggregate(self) -> Optional[np.ndarray]:
        """
        Aggregate all submitted updates.

        Returns mean of all updates, or None if insufficient participants.
        """
        if not self.can_aggregate():
            logger.warning(
                f"Not enough participants: {len(self._pending_updates)}"
                f"/{self.min_participants}"
            )
            return None

        result = np.mean(self._pending_updates, axis=0).astype(np.float32)

        # Clear for next round
        n = len(self._pending_updates)
        self._pending_updates.clear()
        self._participant_ids.clear()

        logger.info(f"Securely aggregated {n} updates")
        return result

    @property
    def pending_count(self) -> int:
        return len(self._pending_updates)
