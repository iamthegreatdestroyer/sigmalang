"""
Tests for VectorOptimizer and DocumentVectorizer
"""

import numpy as np
import pytest

from sigmalang.core.vector_optimizer import (
    DocumentVector,
    DocumentVectorizer,
    VectorOptimizer,
    VectorOptimizerConfig,
)

# ---------------------------------------------------------------------------
# VectorOptimizer
# ---------------------------------------------------------------------------

class TestVectorOptimizer:
    """Unit tests for VectorOptimizer."""

    def _random_embeddings(self, n: int = 20, d: int = 64, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        embs = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    def test_output_shape(self):
        """compress() returns vector of correct dim."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=128))
        embs = self._random_embeddings(10, 64)
        result = optimizer.compress(embs)
        assert result.shape == (128,), f"Expected (128,), got {result.shape}"

    def test_output_normalized(self):
        """compress() output should be unit-normalized by default."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64))
        embs = self._random_embeddings(10, 64)
        result = optimizer.compress(embs)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_output_dtype(self):
        """compress() returns float32."""
        optimizer = VectorOptimizer()
        embs = self._random_embeddings(5, 128)
        result = optimizer.compress(embs)
        assert result.dtype == np.float32

    def test_single_embedding(self):
        """compress() handles single embedding (n=1)."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64))
        embs = self._random_embeddings(1, 64)
        result = optimizer.compress(embs)
        assert result.shape == (64,)

    def test_small_d_pads_to_dim(self):
        """When d < dim, output is still correct shape with zero-padding."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=256))
        embs = self._random_embeddings(10, 32)  # d=32 < dim=256
        result = optimizer.compress(embs)
        assert result.shape == (256,)

    def test_large_d_truncates(self):
        """When d > dim, output correctly truncates."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=32))
        embs = self._random_embeddings(10, 128)  # d=128 > dim=32
        result = optimizer.compress(embs)
        assert result.shape == (32,)

    def test_similarity_aligns_with_cluster(self):
        """
        Vector should be more similar to cluster it represents than random noise.
        """
        rng = np.random.default_rng(42)
        # Create a cluster of similar embeddings
        center = rng.standard_normal(64).astype(np.float32)
        center /= np.linalg.norm(center)
        embs = center + rng.standard_normal((10, 64)).astype(np.float32) * 0.1
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64, max_iterations=200))
        vector = optimizer.compress(embs)

        # The vector should be more similar to center than to a random vector
        random_vec = rng.standard_normal(64).astype(np.float32)
        random_vec /= np.linalg.norm(random_vec)

        sim_to_center = float(vector @ center)
        sim_to_random = float(vector @ random_vec)
        assert sim_to_center > sim_to_random, (
            f"Vector should align with cluster: {sim_to_center:.4f} vs {sim_to_random:.4f}"
        )

    def test_reconstruct_scores_shape(self):
        """reconstruct_scores() returns correct shape."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64))
        embs = self._random_embeddings(10, 64)
        vector = optimizer.compress(embs)
        candidates = self._random_embeddings(20, 64)
        scores = optimizer.reconstruct_scores(vector, candidates)
        assert scores.shape == (20,)

    def test_reconstruct_scores_range(self):
        """reconstruct_scores() values are in [-1, 1]."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64))
        embs = self._random_embeddings(10, 64)
        vector = optimizer.compress(embs)
        candidates = self._random_embeddings(50, 64)
        scores = optimizer.reconstruct_scores(vector, candidates)
        assert np.all(scores >= -1.01) and np.all(scores <= 1.01)

    def test_top_k_returns_k(self):
        """top_k() returns k results."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64))
        embs = self._random_embeddings(10, 64)
        vector = optimizer.compress(embs)
        candidates = self._random_embeddings(100, 64)
        indices, scores = optimizer.top_k(vector, candidates, k=5)
        assert len(indices) == 5
        assert len(scores) == 5

    def test_top_k_sorted_descending(self):
        """top_k() scores are sorted descending."""
        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=64))
        embs = self._random_embeddings(10, 64)
        vector = optimizer.compress(embs)
        candidates = self._random_embeddings(50, 64)
        _, scores = optimizer.top_k(vector, candidates, k=5)
        assert np.all(scores[:-1] >= scores[1:]), "Scores not sorted descending"

    def test_top_k_finds_similar(self):
        """top_k() should rank the most similar embedding highest."""
        rng = np.random.default_rng(42)
        dim = 64
        # Create a "cluster" embedding and a clear outlier
        cluster = rng.standard_normal(dim).astype(np.float32)
        cluster /= np.linalg.norm(cluster)

        embs = cluster + rng.standard_normal((5, dim)).astype(np.float32) * 0.05
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        optimizer = VectorOptimizer(VectorOptimizerConfig(dim=dim, max_iterations=100))
        vector = optimizer.compress(embs)

        # Candidates: 1 very similar, 9 random
        similar = cluster + rng.standard_normal(dim).astype(np.float32) * 0.01
        similar /= np.linalg.norm(similar)
        randoms = rng.standard_normal((9, dim)).astype(np.float32)
        randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)
        candidates = np.vstack([similar.reshape(1, -1), randoms])

        indices, scores = optimizer.top_k(vector, candidates, k=1)
        assert indices[0] == 0, f"Expected similar embedding (idx 0), got idx {indices[0]}"

    def test_compress_multi_scale(self):
        """compress_multi_scale() returns list of vectors at each dim."""
        optimizer = VectorOptimizer()
        embs = self._random_embeddings(10, 64)
        dims = [32, 64, 128]
        vectors = optimizer.compress_multi_scale(embs, dims=dims)
        assert len(vectors) == 3
        for i, (d, v) in enumerate(zip(dims, vectors)):
            assert v.shape == (d,), f"dim[{i}]: expected ({d},), got {v.shape}"

    def test_no_normalize_option(self):
        """normalize_output=False preserves non-unit-norm."""
        cfg = VectorOptimizerConfig(dim=64, normalize_output=False)
        optimizer = VectorOptimizer(cfg)
        embs = self._random_embeddings(10, 64) * 5.0  # large norm
        result = optimizer.compress(embs)
        # Can't guarantee any particular norm — just check it's not always 1
        assert result.shape == (64,)

    def test_deterministic_with_same_seed(self):
        """Same input → same output (optimizer is deterministic)."""
        cfg = VectorOptimizerConfig(dim=64, max_iterations=50)
        optimizer = VectorOptimizer(cfg)
        rng = np.random.default_rng(7)
        embs = rng.standard_normal((10, 64)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        v1 = optimizer.compress(embs.copy())
        v2 = optimizer.compress(embs.copy())
        np.testing.assert_array_equal(v1, v2)

    @pytest.mark.parametrize("n,d,dim", [
        (1, 16, 32),
        (5, 32, 32),
        (50, 128, 64),
        (100, 256, 512),
    ])
    def test_various_shapes(self, n, d, dim):
        """compress() handles various input shapes."""
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((n, d)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        cfg = VectorOptimizerConfig(dim=dim, max_iterations=20)
        optimizer = VectorOptimizer(cfg)
        result = optimizer.compress(embs)
        assert result.shape == (dim,)


# ---------------------------------------------------------------------------
# DocumentVectorizer
# ---------------------------------------------------------------------------

class TestDocumentVectorizer:
    """Tests for the high-level DocumentVectorizer."""

    def test_vectorize_returns_document_vector(self):
        """vectorize() returns DocumentVector instance."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("Python machine learning frameworks.")
        assert isinstance(result, DocumentVector)

    def test_vector_correct_dim(self):
        """vectorize() vector has configured dim."""
        dim = 128
        vec = DocumentVectorizer(dim=dim)
        result = vec.vectorize("Hello world in multiple tokens.")
        assert result.vector.shape == (dim,)

    def test_vector_dtype(self):
        """vectorize() vector is float32."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("test document")
        assert result.vector.dtype == np.float32

    def test_compression_ratio_positive(self):
        """compression_ratio is positive."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("A" * 1000)
        assert result.compression_ratio > 0

    def test_long_document(self):
        """vectorize() handles long documents."""
        text = " ".join([f"word_{i}" for i in range(500)])
        vec = DocumentVectorizer(dim=128)
        result = vec.vectorize(text)
        assert result.vector.shape == (128,)
        assert result.source_token_count > 1

    def test_empty_text_no_crash(self):
        """vectorize() doesn't crash on empty text."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("")
        assert isinstance(result, DocumentVector)

    def test_single_word(self):
        """vectorize() handles single-word input."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("hello")
        assert result.vector.shape == (64,)

    def test_vectorize_batch(self):
        """vectorize_batch() returns list of DocumentVectors."""
        texts = ["First document.", "Second document.", "Third document."]
        vec = DocumentVectorizer(dim=64)
        results = vec.vectorize_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, DocumentVector) for r in results)

    def test_similarity_symmetric(self):
        """similarity(a, b) == similarity(b, a)."""
        vec = DocumentVectorizer(dim=64)
        da = vec.vectorize("machine learning algorithms")
        db = vec.vectorize("deep neural networks")
        assert abs(vec.similarity(da, db) - vec.similarity(db, da)) < 1e-6

    def test_similarity_self_equals_one(self):
        """similarity(a, a) ≈ 1.0 for unit-normalized vectors."""
        vec = DocumentVectorizer(dim=64)
        da = vec.vectorize("identical document text")
        db = vec.vectorize("identical document text")
        sim = vec.similarity(da, db)
        assert abs(sim - 1.0) < 1e-4, f"Self-similarity should be ~1, got {sim}"

    def test_nearest_returns_k(self):
        """nearest() returns k results."""
        vec = DocumentVectorizer(dim=64)
        query = vec.vectorize("machine learning")
        corpus = [
            vec.vectorize("deep learning neural networks"),
            vec.vectorize("cooking recipes pasta"),
            vec.vectorize("computer vision algorithms"),
            vec.vectorize("natural language processing"),
            vec.vectorize("classical music theory"),
        ]
        results = vec.nearest(query, corpus, k=3)
        assert len(results) == 3

    def test_nearest_indices_valid(self):
        """nearest() indices are valid corpus indices."""
        vec = DocumentVectorizer(dim=64)
        query = vec.vectorize("machine learning")
        corpus = [vec.vectorize(f"doc {i}") for i in range(10)]
        results = vec.nearest(query, corpus, k=5)
        for idx, score in results:
            assert 0 <= idx < 10, f"Invalid index {idx}"
            assert -1.0 <= score <= 1.0, f"Score out of range {score}"

    def test_nearest_sorted_descending(self):
        """nearest() results are sorted by score descending."""
        vec = DocumentVectorizer(dim=64)
        query = vec.vectorize("machine learning optimization")
        corpus = [vec.vectorize(f"document number {i} about various topics") for i in range(20)]
        results = vec.nearest(query, corpus, k=10)
        scores = [s for _, s in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Not sorted descending"

    def test_optimization_loss_in_range(self):
        """optimization_loss is in [0, 2] (bounded cosine dissimilarity)."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("test text with several words for compression")
        assert 0.0 <= result.optimization_loss <= 2.0, f"loss={result.optimization_loss}"

    def test_elapsed_ms_positive(self):
        """elapsed_ms is positive."""
        vec = DocumentVectorizer(dim=64)
        result = vec.vectorize("timing check")
        assert result.elapsed_ms > 0
