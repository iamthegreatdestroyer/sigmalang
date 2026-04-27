"""
Product Quantization for Codebook Storage - Phase 7 Track 9

Compresses the SigmaLang codebook itself using product quantization (PQ),
reducing memory footprint for edge/mobile deployment. A 256x512 codebook
(512KB at FP32) can be compressed to <16KB with PQ.

Algorithm:
    1. Split each embedding vector into M sub-vectors
    2. Train K centroids per sub-space using k-means
    3. Replace each sub-vector with its centroid index (1 byte)
    4. Lookup: reconstruct by concatenating sub-centroids
    5. Distance: asymmetric distance computation (ADC) for fast search

Research Basis:
    - Product Quantization (Jegou et al., 2011): foundational PQ paper
    - OPQ (Ge et al., 2013): optimized rotation for PQ
    - FastText.zip (2016): PQ for word embeddings
    - JPQ (2021): joint query encoder + PQ optimization

Usage:
    from sigmalang.core.pq_codebook import ProductQuantizer

    pq = ProductQuantizer(num_subspaces=8, num_centroids=256)
    pq.train(codebook_embeddings)
    codes = pq.encode(codebook_embeddings)
    reconstructed = pq.decode(codes)

    # Fast search
    distances = pq.asymmetric_distance(query_vector, codes)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PQConfig:
    """Product quantization configuration."""
    num_subspaces: int = 8       # M: number of sub-vector partitions
    num_centroids: int = 256     # K: centroids per sub-space (max 256 for uint8)
    max_iterations: int = 25     # k-means iterations
    seed: int = 42


class ProductQuantizer:
    """
    Product Quantization for codebook compression.

    Splits D-dimensional vectors into M sub-vectors of dimension D/M,
    then independently quantizes each sub-vector using K centroids.

    Memory: M * K * (D/M) * 4 bytes for centroids + N * M bytes for codes
    For M=8, K=256, D=512, N=256:
        Centroids: 8 * 256 * 64 * 4 = 512KB (same as original... stored once)
        Codes: 256 * 8 = 2KB (vs 512KB original)
        But centroids are shared across many lookups, making this worthwhile
        when the codebook is used in multiple contexts.
    """

    def __init__(self, config: Optional[PQConfig] = None):
        self.config = config or PQConfig()
        self.M = self.config.num_subspaces
        self.K = self.config.num_centroids

        # Centroids: (M, K, D/M) — trained via k-means
        self._centroids: Optional[np.ndarray] = None
        self._sub_dim: int = 0
        self._trained = False
        self._rng = np.random.RandomState(self.config.seed)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, vectors: np.ndarray) -> Dict[str, Any]:
        """
        Train PQ centroids using k-means on each sub-space.

        Args:
            vectors: (N, D) training vectors

        Returns:
            Training statistics
        """
        N, D = vectors.shape

        if D % self.M != 0:
            raise ValueError(
                f"Vector dimension {D} must be divisible by num_subspaces {self.M}"
            )

        self._sub_dim = D // self.M
        self._centroids = np.zeros(
            (self.M, self.K, self._sub_dim), dtype=np.float32
        )

        total_distortion = 0.0

        for m in range(self.M):
            # Extract sub-vectors for this subspace
            start = m * self._sub_dim
            end = start + self._sub_dim
            sub_vectors = vectors[:, start:end].astype(np.float32)

            # Run k-means
            centroids, distortion = self._kmeans(
                sub_vectors,
                min(self.K, N),
                self.config.max_iterations,
            )
            self._centroids[m, :len(centroids)] = centroids
            total_distortion += distortion

        self._trained = True

        logger.info(
            f"PQ trained: M={self.M}, K={self.K}, sub_dim={self._sub_dim}, "
            f"distortion={total_distortion:.4f}"
        )

        return {
            'num_subspaces': self.M,
            'num_centroids': self.K,
            'sub_dim': self._sub_dim,
            'total_distortion': round(total_distortion, 4),
            'training_vectors': N,
        }

    def _kmeans(
        self,
        data: np.ndarray,
        k: int,
        max_iter: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Simple k-means clustering.

        Returns (centroids, distortion).
        """
        N = data.shape[0]
        k = min(k, N)

        # Initialize with random samples
        indices = self._rng.choice(N, size=k, replace=False)
        centroids = data[indices].copy()

        distortion = 0.0

        for iteration in range(max_iter):
            # Assign each point to nearest centroid
            dists = self._pairwise_l2(data, centroids)
            assignments = dists.argmin(axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)

            for i in range(N):
                c = assignments[i]
                new_centroids[c] += data[i]
                counts[c] += 1

            # Handle empty clusters
            for c in range(k):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    # Re-initialize empty cluster with random point
                    new_centroids[c] = data[self._rng.randint(N)]

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < 1e-6:
                break

        # Final distortion
        dists = self._pairwise_l2(data, centroids)
        distortion = float(dists.min(axis=1).sum())

        return centroids, distortion

    def _pairwise_l2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute pairwise L2 distances: (N, K)."""
        a_sq = (a ** 2).sum(axis=1, keepdims=True)
        b_sq = (b ** 2).sum(axis=1, keepdims=True).T
        return np.maximum(0, a_sq - 2.0 * a @ b.T + b_sq)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to PQ codes.

        Args:
            vectors: (N, D) vectors to encode

        Returns:
            (N, M) uint8 centroid indices
        """
        if not self._trained:
            raise RuntimeError("PQ not trained. Call train() first.")

        N, D = vectors.shape
        codes = np.zeros((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            start = m * self._sub_dim
            end = start + self._sub_dim
            sub_vectors = vectors[:, start:end].astype(np.float32)

            dists = self._pairwise_l2(sub_vectors, self._centroids[m])
            codes[:, m] = dists.argmin(axis=1).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to approximate vectors.

        Args:
            codes: (N, M) uint8 centroid indices

        Returns:
            (N, D) reconstructed vectors
        """
        if not self._trained:
            raise RuntimeError("PQ not trained. Call train() first.")

        N = codes.shape[0]
        D = self.M * self._sub_dim
        vectors = np.zeros((N, D), dtype=np.float32)

        for m in range(self.M):
            start = m * self._sub_dim
            end = start + self._sub_dim
            vectors[:, start:end] = self._centroids[m, codes[:, m]]

        return vectors

    def asymmetric_distance(
        self,
        query: np.ndarray,
        codes: np.ndarray,
    ) -> np.ndarray:
        """
        Asymmetric Distance Computation (ADC).

        Compute approximate L2 distances between a full-precision query
        and PQ-encoded database vectors without decoding.

        This is O(M*K + N*M) instead of O(N*D) for brute force.

        Args:
            query: (D,) query vector
            codes: (N, M) PQ codes of database vectors

        Returns:
            (N,) approximate distances
        """
        if not self._trained:
            raise RuntimeError("PQ not trained. Call train() first.")

        # Precompute distance table: (M, K) distances from query sub-vectors to centroids
        dist_table = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            start = m * self._sub_dim
            end = start + self._sub_dim
            q_sub = query[start:end]
            # Distance from q_sub to each centroid in subspace m
            diff = self._centroids[m] - q_sub
            dist_table[m] = (diff ** 2).sum(axis=1)

        # Lookup and sum: for each database vector, sum distances per subspace
        N = codes.shape[0]
        distances = np.zeros(N, dtype=np.float32)
        for m in range(self.M):
            distances += dist_table[m, codes[:, m]]

        return distances

    def search(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors using ADC.

        Returns list of (index, distance) tuples.
        """
        distances = self.asymmetric_distance(query, codes)
        top_indices = np.argsort(distances)[:top_k]
        return [(int(idx), float(distances[idx])) for idx in top_indices]

    def compression_stats(self, original_vectors: np.ndarray) -> Dict[str, Any]:
        """Compute compression statistics."""
        N, D = original_vectors.shape
        original_bytes = N * D * 4  # FP32
        code_bytes = N * self.M     # uint8 codes
        centroid_bytes = self.M * self.K * self._sub_dim * 4  # FP32 centroids

        # Reconstruction error
        if self._trained:
            codes = self.encode(original_vectors)
            reconstructed = self.decode(codes)
            mse = float(((original_vectors - reconstructed) ** 2).mean())
            rmse = math.sqrt(mse)
        else:
            mse = 0.0
            rmse = 0.0

        return {
            'original_bytes': original_bytes,
            'code_bytes': code_bytes,
            'centroid_bytes': centroid_bytes,
            'total_compressed_bytes': code_bytes + centroid_bytes,
            'compression_ratio': round(original_bytes / max(1, code_bytes + centroid_bytes), 1),
            'code_only_ratio': round(original_bytes / max(1, code_bytes), 1),
            'mse': round(mse, 6),
            'rmse': round(rmse, 6),
        }

    def serialize(self) -> bytes:
        """Serialize PQ to bytes."""
        import struct

        parts = []
        parts.append(b'\xCE\xA3\x50\x51')  # Magic: Sigma-PQ
        parts.append(struct.pack('<BBH', 1, self.M, self.K))  # version, M, K
        parts.append(struct.pack('<I', self._sub_dim))

        if self._centroids is not None:
            cb = self._centroids.astype(np.float32).tobytes()
            parts.append(struct.pack('<I', len(cb)))
            parts.append(cb)
        else:
            parts.append(struct.pack('<I', 0))

        return b''.join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> 'ProductQuantizer':
        """Deserialize PQ from bytes."""
        import struct

        offset = 0
        magic = data[offset:offset + 4]
        offset += 4
        if magic != b'\xCE\xA3\x50\x51':
            raise ValueError("Invalid PQ magic bytes")

        version, M, K = struct.unpack_from('<BBH', data, offset)
        offset += 4

        sub_dim = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        config = PQConfig(num_subspaces=M, num_centroids=K)
        pq = cls(config)
        pq._sub_dim = sub_dim

        cb_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        if cb_len > 0:
            cb = np.frombuffer(data[offset:offset + cb_len], dtype=np.float32)
            pq._centroids = cb.reshape(M, K, sub_dim).copy()
            pq._trained = True

        return pq
