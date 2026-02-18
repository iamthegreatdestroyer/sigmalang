"""
Cascaded Codebook Architecture - Sprint 5 Task 5.2

Implements multi-layer codebook with frozen and trainable components based on
the UniCode² paper (June 2025): https://hf.co/papers/2506.20214

Key Concepts:
- Cascaded architecture: Multiple codebook layers with different update policies
- Frozen codebooks: Tier 0-1 primitives (existential/domain) - never updated
- Trainable codebooks: Tier 2 primitives (learned patterns) - continuously refined
- Layer coordination: Efficient lookup across all layers

Benefits:
- Faster training convergence (stable foundation)
- Better gradient flow (only train what needs learning)
- Modular architecture (easy to extend)
- Reduced catastrophic forgetting
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


# =============================================================================
# Codebook Layer Types
# =============================================================================

class CodebookLayerType(Enum):
    """Types of codebook layers with different update policies."""

    FROZEN = "frozen"  # Never updated (Tier 0-1: existential/domain)
    TRAINABLE = "trainable"  # Continuously refined (Tier 2: learned)
    ADAPTIVE = "adaptive"  # Updated based on usage patterns


@dataclass
class CodebookLayer:
    """A single layer in the cascaded codebook."""

    name: str
    layer_type: CodebookLayerType
    tier: int  # 0, 1, or 2
    embeddings: torch.Tensor
    indices: torch.Tensor  # Primitive indices
    metadata: Dict[str, Any] = field(default_factory=dict)
    frozen: bool = True
    usage_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        """Initialize layer state."""
        if self.layer_type == CodebookLayerType.TRAINABLE:
            self.frozen = False

    @property
    def size(self) -> int:
        """Get number of primitives in this layer."""
        return len(self.indices)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 1

    def lookup(self, index: int) -> Optional[torch.Tensor]:
        """Look up embedding by index."""
        if index < 0 or index >= self.size:
            return None
        return self.embeddings[index]

    def update_embedding(self, index: int, new_embedding: torch.Tensor) -> bool:
        """Update an embedding (only if not frozen)."""
        if self.frozen:
            return False

        if index < 0 or index >= self.size:
            return False

        self.embeddings[index] = new_embedding
        return True

    def record_usage(self, index: int) -> None:
        """Record usage of a primitive."""
        if 0 <= index < self.size:
            self.usage_counts[index] += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this layer."""
        if not self.usage_counts:
            return {
                'total_usage': 0,
                'unique_primitives_used': 0,
                'avg_usage_per_primitive': 0.0,
                'most_used_indices': []
            }

        total = sum(self.usage_counts.values())
        unique = len(self.usage_counts)
        most_used = sorted(self.usage_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total_usage': total,
            'unique_primitives_used': unique,
            'avg_usage_per_primitive': total / max(1, unique),
            'most_used_indices': [idx for idx, _ in most_used],
            'usage_coverage': unique / max(1, self.size)
        }


# =============================================================================
# Cascaded Codebook
# =============================================================================

class CascadedCodebook(nn.Module):
    """
    Multi-layer cascaded codebook with frozen and trainable components.

    Architecture:
        Layer 0 (Frozen): Tier 0 Existential Primitives (0-15)
        Layer 1 (Frozen): Tier 1 Domain Primitives (16-127)
        Layer 2 (Trainable): Tier 2 Learned Primitives (128-255)

    Lookup Order:
        1. Try Layer 0 (fastest, frozen)
        2. Try Layer 1 (fast, frozen)
        3. Try Layer 2 (trainable, updated)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        tier0_size: int = 16,
        tier1_size: int = 112,
        tier2_size: int = 128,
        enable_usage_tracking: bool = True
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.enable_usage_tracking = enable_usage_tracking

        # Create layers
        self.layers: Dict[str, CodebookLayer] = {}

        # Layer 0: Tier 0 Existential (Frozen)
        self.layers['tier0'] = CodebookLayer(
            name='tier0_existential',
            layer_type=CodebookLayerType.FROZEN,
            tier=0,
            embeddings=torch.randn(tier0_size, embedding_dim),
            indices=torch.arange(0, tier0_size),
            frozen=True,
            metadata={'description': 'Existential primitives (being, causality, etc.)'}
        )

        # Layer 1: Tier 1 Domain (Frozen)
        self.layers['tier1'] = CodebookLayer(
            name='tier1_domain',
            layer_type=CodebookLayerType.FROZEN,
            tier=1,
            embeddings=torch.randn(tier1_size, embedding_dim),
            indices=torch.arange(tier0_size, tier0_size + tier1_size),
            frozen=True,
            metadata={'description': 'Domain primitives (math, language, etc.)'}
        )

        # Layer 2: Tier 2 Learned (Trainable)
        self.layers['tier2'] = CodebookLayer(
            name='tier2_learned',
            layer_type=CodebookLayerType.TRAINABLE,
            tier=2,
            embeddings=nn.Parameter(torch.randn(tier2_size, embedding_dim)),
            indices=torch.arange(tier0_size + tier1_size, tier0_size + tier1_size + tier2_size),
            frozen=False,
            metadata={'description': 'Learned primitives (patterns discovered from data)'}
        )

        # Register layer 2 embeddings as a parameter
        self.register_parameter('tier2_embeddings', self.layers['tier2'].embeddings)

        # Lookup table for fast index -> layer mapping
        self._build_index_map()

    def _build_index_map(self) -> None:
        """Build fast lookup map from primitive index to layer."""
        self.index_to_layer: Dict[int, str] = {}

        for layer_name, layer in self.layers.items():
            for idx in layer.indices.tolist():
                self.index_to_layer[idx] = layer_name

    def lookup(self, index: int) -> Optional[torch.Tensor]:
        """
        Look up embedding by primitive index.

        Args:
            index: Primitive index (0-255)

        Returns:
            Embedding tensor or None if not found
        """
        layer_name = self.index_to_layer.get(index)
        if layer_name is None:
            return None

        layer = self.layers[layer_name]

        # Record usage
        if self.enable_usage_tracking:
            layer.record_usage(index)

        # Get embedding
        local_index = (layer.indices == index).nonzero(as_tuple=True)[0]
        if len(local_index) == 0:
            return None

        return layer.embeddings[local_index[0]]

    def lookup_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for a batch of indices.

        Args:
            indices: Tensor of primitive indices

        Returns:
            Tensor of embeddings [batch_size, embedding_dim]
        """
        batch_size = indices.shape[0]
        embeddings = torch.zeros(batch_size, self.embedding_dim)

        for i, idx in enumerate(indices.tolist()):
            emb = self.lookup(idx)
            if emb is not None:
                embeddings[i] = emb

        return embeddings

    def update_trainable(
        self,
        indices: torch.Tensor,
        new_embeddings: torch.Tensor,
        learning_rate: float = 0.01
    ) -> int:
        """
        Update trainable layer embeddings.

        Args:
            indices: Indices to update
            new_embeddings: New embedding values
            learning_rate: Learning rate for update

        Returns:
            Number of embeddings updated
        """
        updated = 0
        tier2_layer = self.layers['tier2']

        if tier2_layer.frozen:
            return 0

        for idx, new_emb in zip(indices.tolist(), new_embeddings):
            # Only update if in tier 2 range
            layer_name = self.index_to_layer.get(idx)
            if layer_name != 'tier2':
                continue

            # Get local index
            local_indices = (tier2_layer.indices == idx).nonzero(as_tuple=True)[0]
            if len(local_indices) == 0:
                continue

            local_idx = local_indices[0]

            # Gradient update
            with torch.no_grad():
                tier2_layer.embeddings[local_idx] = (
                    (1 - learning_rate) * tier2_layer.embeddings[local_idx] +
                    learning_rate * new_emb
                )

            updated += 1

        return updated

    def freeze_layer(self, layer_name: str) -> bool:
        """Freeze a layer (prevent updates)."""
        if layer_name not in self.layers:
            return False

        self.layers[layer_name].frozen = True
        return True

    def unfreeze_layer(self, layer_name: str) -> bool:
        """Unfreeze a layer (allow updates)."""
        if layer_name not in self.layers:
            return False

        if self.layers[layer_name].layer_type == CodebookLayerType.FROZEN:
            return False  # Cannot unfreeze a FROZEN type layer

        self.layers[layer_name].frozen = False
        return True

    def get_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all layers."""
        stats = {}

        for layer_name, layer in self.layers.items():
            stats[layer_name] = {
                'type': layer.layer_type.value,
                'tier': layer.tier,
                'size': layer.size,
                'frozen': layer.frozen,
                'embedding_dim': layer.embedding_dim,
                **layer.get_usage_stats()
            }

        return stats

    def get_total_size(self) -> int:
        """Get total number of primitives across all layers."""
        return sum(layer.size for layer in self.layers.values())

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = []

        for layer in self.layers.values():
            if not layer.frozen and isinstance(layer.embeddings, nn.Parameter):
                params.append(layer.embeddings)

        return params

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch lookup.

        Args:
            indices: Batch of primitive indices

        Returns:
            Batch of embeddings
        """
        return self.lookup_batch(indices)


# =============================================================================
# Cascaded Codebook Trainer
# =============================================================================

class CascadedCodebookTrainer:
    """
    Trainer for cascaded codebook with layer-specific learning.

    Only trains Tier 2 (learned primitives) while keeping Tier 0-1 frozen.
    """

    def __init__(
        self,
        codebook: CascadedCodebook,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.codebook = codebook
        self.learning_rate = learning_rate

        # Only optimize trainable parameters (Tier 2)
        trainable_params = codebook.get_trainable_parameters()
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        self.training_stats = {
            'steps': 0,
            'total_loss': 0.0,
            'tier2_updates': 0
        }

    def train_step(
        self,
        indices: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            indices: Primitive indices to update
            target_embeddings: Target embedding values

        Returns:
            Training metrics
        """
        self.optimizer.zero_grad()

        # Get current embeddings
        current_embeddings = self.codebook.lookup_batch(indices)

        # Calculate loss (only for Tier 2 indices)
        tier2_mask = indices >= 128  # Tier 2 starts at 128
        if tier2_mask.sum() == 0:
            return {'loss': 0.0, 'tier2_updates': 0}

        loss = nn.functional.mse_loss(
            current_embeddings[tier2_mask],
            target_embeddings[tier2_mask]
        )

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Update stats
        self.training_stats['steps'] += 1
        self.training_stats['total_loss'] += loss.item()
        self.training_stats['tier2_updates'] += tier2_mask.sum().item()

        return {
            'loss': loss.item(),
            'tier2_updates': tier2_mask.sum().item(),
            'avg_loss': self.training_stats['total_loss'] / self.training_stats['steps']
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.training_stats,
            'learning_rate': self.learning_rate,
            **self.codebook.get_layer_stats()
        }


# =============================================================================
# Integration Functions
# =============================================================================

def create_cascaded_codebook_from_existing(
    existing_primitives: Dict[int, Any],
    embedding_dim: int = 512
) -> CascadedCodebook:
    """
    Create a cascaded codebook from existing ΣLANG primitives.

    Args:
        existing_primitives: Dict mapping primitive IDs to primitive objects
        embedding_dim: Embedding dimension

    Returns:
        Initialized CascadedCodebook
    """
    # Count primitives per tier
    tier0_size = sum(1 for idx in existing_primitives if idx < 16)
    tier1_size = sum(1 for idx in existing_primitives if 16 <= idx < 128)
    tier2_size = sum(1 for idx in existing_primitives if idx >= 128)

    codebook = CascadedCodebook(
        embedding_dim=embedding_dim,
        tier0_size=max(16, tier0_size),
        tier1_size=max(112, tier1_size),
        tier2_size=max(128, tier2_size)
    )

    return codebook


# =============================================================================
# Global Cascaded Codebook
# =============================================================================

_global_cascaded_codebook: Optional[CascadedCodebook] = None


def get_cascaded_codebook() -> CascadedCodebook:
    """Get or create the global cascaded codebook."""
    global _global_cascaded_codebook
    if _global_cascaded_codebook is None:
        _global_cascaded_codebook = CascadedCodebook()
    return _global_cascaded_codebook


def initialize_cascaded_codebook(
    embedding_dim: int = 512,
    tier0_size: int = 16,
    tier1_size: int = 112,
    tier2_size: int = 128
) -> CascadedCodebook:
    """
    Initialize the global cascaded codebook.

    Usage:
        from sigmalang.core.cascaded_codebook import initialize_cascaded_codebook

        codebook = initialize_cascaded_codebook(embedding_dim=512)

        # Look up primitive
        embedding = codebook.lookup(42)

        # Train only Tier 2
        trainer = CascadedCodebookTrainer(codebook)
        metrics = trainer.train_step(indices, targets)
    """
    global _global_cascaded_codebook
    _global_cascaded_codebook = CascadedCodebook(
        embedding_dim=embedding_dim,
        tier0_size=tier0_size,
        tier1_size=tier1_size,
        tier2_size=tier2_size
    )
    return _global_cascaded_codebook
