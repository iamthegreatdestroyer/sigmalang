"""
Multi-Modal Vector Quantization - Phase 7 Track 1

Unified codebook layer that maps text, image, and audio modalities into
a shared Sigma-Primitive space. Enables cross-modal retrieval and
analogy (e.g., "this image sounds like...").

Architecture:
    Text tokens  ──→ TextProjector  ──→ shared_dim ──→ ┐
    Image feats  ──→ ImageProjector ──→ shared_dim ──→ ├──→ VQ Codebook ──→ primitive_id
    Audio feats  ──→ AudioProjector ──→ shared_dim ──→ ┘

The VQ codebook is modality-agnostic: entries are trained/updated from
all modalities so that semantically similar concepts (regardless of
source modality) map to nearby primitives.

Research Basis:
    - OmniTokenizer (Jan 2025): Joint image-video tokenizer
    - MMVQ (2024): Multi-modal VQ-VAE alignment
    - SigmaLang analogy system: cross_modal_analogies.py

Usage:
    from sigmalang.core.multimodal_vq import MultiModalVQ

    vq = MultiModalVQ(codebook_size=256, shared_dim=64)

    # Encode from different modalities
    text_ids = vq.encode_text(text_features)
    image_ids = vq.encode_image(image_features)
    audio_ids = vq.encode_audio(audio_features)

    # Cross-modal search
    matches = vq.search_across_modalities(query_vec, top_k=5)
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiModalVQConfig:
    """Configuration for multi-modal VQ."""
    codebook_size: int = 256      # Number of shared primitives
    shared_dim: int = 64          # Dimension of shared embedding space
    text_input_dim: int = 128     # Raw text feature dimension
    image_input_dim: int = 96     # Raw image feature dimension
    audio_input_dim: int = 80     # Raw audio feature dimension
    ema_decay: float = 0.99       # Exponential moving average for codebook update
    commitment_weight: float = 0.25
    diversity_weight: float = 0.1  # Encourage codebook utilization
    temperature: float = 1.0       # Soft quantization temperature


# =============================================================================
# Linear Projector (no torch dependency)
# =============================================================================

class LinearProjector:
    """
    Simple linear projection: y = x @ W + b

    Initialized with Xavier uniform for stable training.
    """

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        limit = math.sqrt(6.0 / (input_dim + output_dim))
        self.weight = rng.uniform(-limit, limit, (input_dim, output_dim)).astype(np.float32)
        self.bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project input to output space."""
        if x.ndim == 1:
            return x @ self.weight + self.bias
        return x @ self.weight + self.bias  # (batch, out_dim)

    def update(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> None:
        """Gradient-free pseudo-update: nudge weights toward reducing error."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            target = target.reshape(1, -1)
        projected = x @ self.weight + self.bias
        error = target - projected  # (batch, out_dim)
        # Approximate gradient: dW = x^T @ error
        grad_w = x.T @ error / x.shape[0]
        grad_b = error.mean(axis=0)
        self.weight += lr * grad_w
        self.bias += lr * grad_b


# =============================================================================
# VQ Codebook
# =============================================================================

class VQCodebook:
    """
    Vector Quantization codebook with EMA updates.

    Maps continuous vectors to discrete codebook entries using
    nearest-neighbor lookup and updates entries via exponential
    moving averages of assigned vectors.
    """

    def __init__(self, config: MultiModalVQConfig):
        self.config = config
        self.size = config.codebook_size
        self.dim = config.shared_dim

        # Initialize codebook with unit-norm random vectors
        rng = np.random.RandomState(42)
        self.embeddings = rng.randn(self.size, self.dim).astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings /= norms

        # EMA tracking
        self._ema_count = np.ones(self.size, dtype=np.float32)
        self._ema_weight = self.embeddings.copy()

        # Usage statistics
        self.usage_count = np.zeros(self.size, dtype=np.int64)
        self.modality_usage: Dict[str, np.ndarray] = {}

    def quantize(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize vectors to nearest codebook entries.

        Args:
            vectors: (N, shared_dim) continuous vectors

        Returns:
            (quantized, indices, distances)
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # L2 distance: ||x - e||^2 = ||x||^2 - 2*x@e^T + ||e||^2
        x_sq = (vectors ** 2).sum(axis=1, keepdims=True)
        e_sq = (self.embeddings ** 2).sum(axis=1, keepdims=True).T
        dists = x_sq - 2.0 * vectors @ self.embeddings.T + e_sq

        indices = dists.argmin(axis=1)
        quantized = self.embeddings[indices]
        min_dists = dists[np.arange(len(indices)), indices]

        # Track usage
        for idx in indices:
            self.usage_count[idx] += 1

        return quantized, indices, min_dists

    def soft_quantize(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft quantization using temperature-scaled distances.

        Returns weighted combination of codebook entries.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        x_sq = (vectors ** 2).sum(axis=1, keepdims=True)
        e_sq = (self.embeddings ** 2).sum(axis=1, keepdims=True).T
        dists = x_sq - 2.0 * vectors @ self.embeddings.T + e_sq

        # Softmin: convert distances to weights
        neg_dists = -dists / max(self.config.temperature, 1e-6)
        neg_dists -= neg_dists.max(axis=1, keepdims=True)  # Stability
        weights = np.exp(neg_dists)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8

        quantized = weights @ self.embeddings
        return quantized, weights

    def update_ema(self, vectors: np.ndarray, indices: np.ndarray) -> None:
        """Update codebook entries using exponential moving average."""
        decay = self.config.ema_decay

        for i in range(self.size):
            mask = indices == i
            if not mask.any():
                continue

            assigned = vectors[mask]
            count = assigned.shape[0]

            self._ema_count[i] = decay * self._ema_count[i] + (1 - decay) * count
            self._ema_weight[i] = (
                decay * self._ema_weight[i] +
                (1 - decay) * assigned.sum(axis=0)
            )

            # Laplace smoothing
            n = self._ema_count[i] + 1e-5
            self.embeddings[i] = self._ema_weight[i] / n

    def utilization(self) -> float:
        """Fraction of codebook entries that have been used."""
        return float((self.usage_count > 0).sum()) / self.size

    def reset_usage(self) -> None:
        """Reset usage counters."""
        self.usage_count[:] = 0

    def find_nearest(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find top-k nearest codebook entries."""
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        dists = ((self.embeddings - vector) ** 2).sum(axis=1)
        top_indices = np.argsort(dists)[:top_k]

        return [(int(idx), float(dists[idx])) for idx in top_indices]


# =============================================================================
# Multi-Modal VQ System
# =============================================================================

class MultiModalVQ:
    """
    Unified multi-modal vector quantization.

    Projects text, image, and audio features into a shared embedding
    space and quantizes them through a single codebook. This enables:

    1. Cross-modal retrieval (find images matching text descriptions)
    2. Unified compression (all modalities share one primitive set)
    3. Cross-modal analogies (transfer semantics across modalities)
    """

    def __init__(self, config: Optional[MultiModalVQConfig] = None):
        self.config = config or MultiModalVQConfig()

        # Modality-specific projectors
        self.text_projector = LinearProjector(
            self.config.text_input_dim, self.config.shared_dim, seed=42
        )
        self.image_projector = LinearProjector(
            self.config.image_input_dim, self.config.shared_dim, seed=43
        )
        self.audio_projector = LinearProjector(
            self.config.audio_input_dim, self.config.shared_dim, seed=44
        )

        # Shared codebook
        self.codebook = VQCodebook(self.config)

        # Cross-modal index: modality -> list of (vector, primitive_id, metadata)
        self._index: Dict[str, List[Tuple[np.ndarray, int, Dict]]] = {
            'text': [], 'image': [], 'audio': []
        }

        self._projectors = {
            'text': self.text_projector,
            'image': self.image_projector,
            'audio': self.audio_projector,
        }

    def encode(
        self,
        features: np.ndarray,
        modality: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode features from any modality into primitive IDs.

        Args:
            features: (N, modality_dim) raw features
            modality: 'text', 'image', or 'audio'
            metadata: optional metadata to store in index

        Returns:
            (primitive_ids, quantized_vectors)
        """
        if modality not in self._projectors:
            raise ValueError(f"Unknown modality: {modality}. Use text/image/audio.")

        projector = self._projectors[modality]
        projected = projector.forward(features)

        # L2 normalize before quantization
        norms = np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8
        projected = projected / norms

        quantized, indices, _ = self.codebook.quantize(projected)

        # Update codebook
        self.codebook.update_ema(projected, indices)

        # Track modality usage
        if modality not in self.codebook.modality_usage:
            self.codebook.modality_usage[modality] = np.zeros(
                self.config.codebook_size, dtype=np.int64
            )
        for idx in indices:
            self.codebook.modality_usage[modality][idx] += 1

        # Store in cross-modal index
        if metadata is None:
            metadata = {}
        if projected.ndim == 1:
            self._index[modality].append((projected, int(indices[0]), metadata))
        else:
            for i in range(len(indices)):
                self._index[modality].append(
                    (projected[i], int(indices[i]), metadata)
                )

        return indices, quantized

    def encode_text(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience: encode text features."""
        indices, _ = self.encode(features, 'text', **kwargs)
        return indices

    def encode_image(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience: encode image features."""
        indices, _ = self.encode(features, 'image', **kwargs)
        return indices

    def encode_audio(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience: encode audio features."""
        indices, _ = self.encode(features, 'audio', **kwargs)
        return indices

    def decode(self, primitive_ids: np.ndarray) -> np.ndarray:
        """Look up codebook vectors for primitive IDs."""
        return self.codebook.embeddings[primitive_ids]

    def search_across_modalities(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        target_modality: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Cross-modal search: find entries from other modalities
        nearest to the query vector in the shared space.

        Args:
            query_vector: (shared_dim,) query in shared space
            top_k: number of results
            target_modality: restrict to specific modality, or None for all

        Returns:
            List of {modality, primitive_id, distance, metadata}
        """
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()

        # Normalize
        norm = np.linalg.norm(query_vector) + 1e-8
        query_vector = query_vector / norm

        results = []
        modalities = [target_modality] if target_modality else ['text', 'image', 'audio']

        for modality in modalities:
            for vec, prim_id, meta in self._index.get(modality, []):
                dist = float(np.sum((query_vector - vec) ** 2))
                results.append({
                    'modality': modality,
                    'primitive_id': prim_id,
                    'distance': dist,
                    'metadata': meta,
                })

        results.sort(key=lambda r: r['distance'])
        return results[:top_k]

    def cross_modal_analogy(
        self,
        source_vector: np.ndarray,
        source_modality: str,
        target_modality: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Cross-modal analogy: "What in target_modality is like this in source_modality?"

        Projects source through its projector, then searches target modality index.
        """
        projector = self._projectors.get(source_modality)
        if projector is None:
            raise ValueError(f"Unknown source modality: {source_modality}")

        shared = projector.forward(source_vector)
        norm = np.linalg.norm(shared) + 1e-8
        shared = shared / norm

        return self.search_across_modalities(shared, top_k, target_modality)

    def get_statistics(self) -> Dict[str, Any]:
        """Get multi-modal VQ statistics."""
        stats = {
            'codebook_size': self.config.codebook_size,
            'shared_dim': self.config.shared_dim,
            'codebook_utilization': round(self.codebook.utilization(), 3),
            'index_sizes': {m: len(entries) for m, entries in self._index.items()},
        }

        # Per-modality codebook usage
        for modality, usage in self.codebook.modality_usage.items():
            active = int((usage > 0).sum())
            stats[f'{modality}_active_primitives'] = active
            stats[f'{modality}_utilization'] = round(active / self.config.codebook_size, 3)

        # Cross-modal overlap: primitives used by multiple modalities
        if len(self.codebook.modality_usage) >= 2:
            masks = [usage > 0 for usage in self.codebook.modality_usage.values()]
            overlap = masks[0]
            for m in masks[1:]:
                overlap = overlap & m
            stats['cross_modal_overlap'] = int(overlap.sum())
            stats['cross_modal_overlap_pct'] = round(
                float(overlap.sum()) / self.config.codebook_size * 100, 1
            )

        return stats

    def serialize(self) -> bytes:
        """Serialize the VQ system to bytes."""
        import struct

        parts = []
        # Magic + version
        parts.append(b'\xCE\xA3\x4D\x56')  # Sigma-MV (multi-modal VQ)
        parts.append(struct.pack('<B', 1))    # version

        # Config
        parts.append(struct.pack('<IIII',
            self.config.codebook_size,
            self.config.shared_dim,
            self.config.text_input_dim,
            self.config.image_input_dim
        ))

        # Codebook embeddings
        cb_bytes = self.codebook.embeddings.astype(np.float32).tobytes()
        parts.append(struct.pack('<I', len(cb_bytes)))
        parts.append(cb_bytes)

        return b''.join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> 'MultiModalVQ':
        """Deserialize from bytes."""
        import struct

        offset = 0
        magic = data[offset:offset + 4]
        offset += 4
        if magic != b'\xCE\xA3\x4D\x56':
            raise ValueError("Invalid MultiModalVQ magic bytes")

        version = struct.unpack_from('<B', data, offset)[0]
        offset += 1

        cb_size, shared_dim, text_dim, image_dim = struct.unpack_from('<IIII', data, offset)
        offset += 16

        config = MultiModalVQConfig(
            codebook_size=cb_size,
            shared_dim=shared_dim,
            text_input_dim=text_dim,
            image_input_dim=image_dim,
        )

        vq = cls(config)

        # Load codebook
        cb_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        cb_array = np.frombuffer(data[offset:offset + cb_len], dtype=np.float32)
        vq.codebook.embeddings = cb_array.reshape(cb_size, shared_dim).copy()
        vq.codebook._ema_weight = vq.codebook.embeddings.copy()

        return vq
