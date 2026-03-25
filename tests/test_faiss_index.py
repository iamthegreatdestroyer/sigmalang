"""
Tests for FAISS-compatible vector index (faiss_index.py)
"""

import pytest
import numpy as np
from sigmalang.core.faiss_index import (
    BaseIndex,
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    LSHIndex,
    IndexType,
    SearchResult,
    VectorIndex,
    _l2_distances,
    _cosine_distances,
)


def _rng_vecs(n, d, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) * scale).astype(np.float32)


DIM = 32
LARGE_N = 200


# ===========================================================================
# Utility functions
# ===========================================================================

class TestDistanceFunctions:
    def test_l2_zero_distance_to_self(self):
        x = _rng_vecs(5, DIM)
        d = _l2_distances(x, x)
        for i in range(5):
            assert d[i, i] < 1e-4, f"L2 self-distance should be ~0, got {d[i,i]}"

    def test_l2_symmetry(self):
        x = _rng_vecs(5, DIM)
        d = _l2_distances(x, x)
        np.testing.assert_allclose(d, d.T, atol=1e-4)

    def test_cosine_range(self):
        x = _rng_vecs(10, DIM)
        d = _cosine_distances(x, x)
        # Cosine distance should be in [-1, 2]
        assert d.min() >= -1.1 and d.max() <= 2.1

    def test_cosine_self_is_zero(self):
        x = _rng_vecs(5, DIM)
        d = _cosine_distances(x, x)
        for i in range(5):
            assert abs(d[i, i]) < 1e-4


# ===========================================================================
# FlatIndex
# ===========================================================================

class TestFlatIndex:
    def test_empty_search_returns_inf(self):
        idx = FlatIndex(dim=DIM)
        D, I = idx.search(_rng_vecs(1, DIM), k=5)
        assert np.all(np.isinf(D))
        assert np.all(I == -1)

    def test_add_increases_ntotal(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(10, DIM))
        assert idx.ntotal == 10
        idx.add(_rng_vecs(5, DIM))
        assert idx.ntotal == 15

    def test_search_returns_correct_shape(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(50, DIM))
        D, I = idx.search(_rng_vecs(3, DIM), k=5)
        assert D.shape == (3, 5)
        assert I.shape == (3, 5)

    def test_nearest_neighbor_exact(self):
        """The nearest neighbor of a vector to itself should be itself."""
        idx = FlatIndex(dim=DIM)
        vecs = _rng_vecs(20, DIM)
        idx.add(vecs)
        D, I = idx.search(vecs[:3], k=1)
        for i in range(3):
            assert I[i, 0] == i, f"NN of vector {i} should be {i}, got {I[i,0]}"

    def test_k_larger_than_database(self):
        """k larger than n should return at most n results."""
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(5, DIM))
        D, I = idx.search(_rng_vecs(1, DIM), k=20)
        # Valid results should be 5
        valid = (I[0] >= 0).sum()
        assert valid == 5

    def test_add_1d_vector(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(1, DIM).squeeze())  # 1D
        assert idx.ntotal == 1

    def test_reset_clears_data(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(10, DIM))
        idx.reset()
        assert idx.ntotal == 0
        D, I = idx.search(_rng_vecs(1, DIM), k=3)
        assert np.all(np.isinf(D))

    def test_is_trained(self):
        idx = FlatIndex(dim=DIM)
        assert idx.is_trained

    def test_cosine_metric(self):
        idx = FlatIndex(dim=DIM, metric="cosine")
        vecs = _rng_vecs(20, DIM)
        idx.add(vecs)
        D, I = idx.search(vecs[:3], k=1)
        for i in range(3):
            assert I[i, 0] == i

    def test_repr(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(5, DIM))
        r = repr(idx)
        assert "FlatIndex" in r and "dim=32" in r


# ===========================================================================
# IVFIndex
# ===========================================================================

class TestIVFIndex:
    def test_add_auto_trains(self):
        idx = IVFIndex(dim=DIM, nlist=4, nprobe=2)
        idx.add(_rng_vecs(40, DIM))
        assert idx.ntotal == 40
        assert idx.is_trained

    def test_manual_train_then_add(self):
        idx = IVFIndex(dim=DIM, nlist=4, nprobe=2)
        train_vecs = _rng_vecs(100, DIM)
        idx.train(train_vecs)
        assert idx.is_trained
        idx.add(train_vecs[:20])
        assert idx.ntotal == 20

    def test_search_shape(self):
        idx = IVFIndex(dim=DIM, nlist=4, nprobe=2)
        idx.add(_rng_vecs(40, DIM))
        D, I = idx.search(_rng_vecs(5, DIM), k=3)
        assert D.shape == (5, 3)
        assert I.shape == (5, 3)

    def test_empty_search(self):
        idx = IVFIndex(dim=DIM, nlist=4)
        D, I = idx.search(_rng_vecs(1, DIM), k=5)
        assert np.all(np.isinf(D))
        assert np.all(I == -1)

    def test_recall_reasonable(self):
        """IVF should have decent recall against exact search."""
        n, k = LARGE_N, 5
        vecs = _rng_vecs(n, DIM, seed=1)
        exact = FlatIndex(dim=DIM)
        exact.add(vecs)
        approx = IVFIndex(dim=DIM, nlist=10, nprobe=5)
        approx.add(vecs)

        queries = _rng_vecs(20, DIM, seed=99)
        _, I_exact = exact.search(queries, k=k)
        _, I_approx = approx.search(queries, k=k)

        # Compute recall@k
        hits = sum(
            len(set(I_exact[i]).intersection(set(I_approx[i])))
            for i in range(len(queries))
        )
        recall = hits / (len(queries) * k)
        assert recall >= 0.6, f"IVF recall too low: {recall:.2f}"

    def test_reset(self):
        idx = IVFIndex(dim=DIM, nlist=4)
        idx.add(_rng_vecs(20, DIM))
        idx.reset()
        assert idx.ntotal == 0


# ===========================================================================
# HNSWIndex
# ===========================================================================

class TestHNSWIndex:
    def test_add_ntotal(self):
        idx = HNSWIndex(dim=DIM, M=8)
        idx.add(_rng_vecs(30, DIM))
        assert idx.ntotal == 30

    def test_search_shape(self):
        idx = HNSWIndex(dim=DIM, M=8)
        idx.add(_rng_vecs(50, DIM))
        D, I = idx.search(_rng_vecs(3, DIM), k=5)
        assert D.shape == (3, 5)
        assert I.shape == (3, 5)

    def test_nearest_to_self(self):
        idx = HNSWIndex(dim=DIM, M=8)
        vecs = _rng_vecs(30, DIM)
        idx.add(vecs)
        D, I = idx.search(vecs[:5], k=1)
        for i in range(5):
            assert I[i, 0] == i, f"HNSW NN of vector {i} should be self, got {I[i,0]}"

    def test_empty_search(self):
        idx = HNSWIndex(dim=DIM)
        D, I = idx.search(_rng_vecs(1, DIM), k=5)
        assert np.all(np.isinf(D))
        assert np.all(I == -1)

    def test_is_trained(self):
        idx = HNSWIndex(dim=DIM)
        assert idx.is_trained

    def test_reset(self):
        idx = HNSWIndex(dim=DIM)
        idx.add(_rng_vecs(10, DIM))
        idx.reset()
        assert idx.ntotal == 0

    def test_recall_vs_flat(self):
        """HNSW (numpy fallback = exact) should match FlatIndex."""
        n, k = 50, 5
        vecs = _rng_vecs(n, DIM)
        flat = FlatIndex(dim=DIM)
        flat.add(vecs)
        hnsw = HNSWIndex(dim=DIM, M=8)
        hnsw.add(vecs)

        queries = _rng_vecs(10, DIM, seed=77)
        _, I_flat = flat.search(queries, k=k)

        # If HNSW uses numpy fallback, results should exactly match
        if hnsw._faiss_index is None:
            _, I_hnsw = hnsw.search(queries, k=k)
            np.testing.assert_array_equal(I_flat, I_hnsw)


# ===========================================================================
# LSHIndex
# ===========================================================================

class TestLSHIndex:
    def test_add_ntotal(self):
        idx = LSHIndex(dim=DIM, n_bits=16, n_tables=4)
        idx.add(_rng_vecs(50, DIM))
        assert idx.ntotal == 50

    def test_search_shape(self):
        idx = LSHIndex(dim=DIM, n_bits=16, n_tables=4)
        idx.add(_rng_vecs(50, DIM))
        D, I = idx.search(_rng_vecs(3, DIM), k=5)
        assert D.shape == (3, 5)
        assert I.shape == (3, 5)

    def test_empty_search(self):
        idx = LSHIndex(dim=DIM, n_bits=16, n_tables=4)
        D, I = idx.search(_rng_vecs(1, DIM), k=5)
        assert np.all(np.isinf(D))
        assert np.all(I == -1)

    def test_nearest_to_self(self):
        """With multiple tables, self should be retrievable."""
        n = 30
        idx = LSHIndex(dim=DIM, n_bits=32, n_tables=8)
        vecs = _rng_vecs(n, DIM)
        idx.add(vecs)
        D, I = idx.search(vecs[:5], k=1)
        # Self should appear in results (either as NN or in top-k)
        for i in range(5):
            # With multiple tables self should hash to same bucket
            # (not guaranteed but very likely)
            assert i in I[i], f"Vector {i} not found in its own search results"

    def test_reset(self):
        idx = LSHIndex(dim=DIM, n_bits=16, n_tables=4)
        idx.add(_rng_vecs(20, DIM))
        idx.reset()
        assert idx.ntotal == 0


# ===========================================================================
# VectorIndex Factory
# ===========================================================================

class TestVectorIndexFactory:
    @pytest.mark.parametrize("index_type", [
        IndexType.FLAT,
        IndexType.HNSW,
        IndexType.LSH,
    ])
    def test_create_basic_types(self, index_type):
        idx = VectorIndex.create(index_type, dim=DIM)
        assert isinstance(idx, BaseIndex)
        assert idx.dim == DIM

    def test_create_ivf(self):
        idx = VectorIndex.create(IndexType.IVF, dim=DIM, nlist=4, nprobe=2)
        assert isinstance(idx, IVFIndex)
        assert idx.nlist == 4

    def test_create_hnsw_params(self):
        idx = VectorIndex.create(IndexType.HNSW, dim=DIM, M=16)
        assert isinstance(idx, HNSWIndex)
        assert idx.M == 16

    def test_create_lsh_params(self):
        idx = VectorIndex.create(IndexType.LSH, dim=DIM, n_bits=32, n_tables=4)
        assert isinstance(idx, LSHIndex)
        assert idx.n_bits == 32

    def test_create_invalid_type(self):
        with pytest.raises((ValueError, AttributeError)):
            VectorIndex.create("invalid_type", dim=DIM)

    def test_auto_small(self):
        idx = VectorIndex.auto(n_vectors=100, dim=DIM)
        assert isinstance(idx, FlatIndex)

    def test_auto_medium(self):
        idx = VectorIndex.auto(n_vectors=50_000, dim=DIM, recall_target=0.90)
        assert isinstance(idx, IVFIndex)

    def test_auto_high_recall(self):
        idx = VectorIndex.auto(n_vectors=50_000, dim=DIM, recall_target=0.97)
        assert isinstance(idx, HNSWIndex)

    def test_auto_large(self):
        idx = VectorIndex.auto(n_vectors=600_000, dim=DIM, recall_target=0.80)
        assert isinstance(idx, LSHIndex)


# ===========================================================================
# SearchResult
# ===========================================================================

class TestSearchResult:
    def test_search_result_via_method(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(20, DIM))
        sr = idx.search_result(_rng_vecs(3, DIM), k=5)
        assert isinstance(sr, SearchResult)
        assert sr.query_count == 3
        assert sr.distances.shape == (3, 5)
        assert sr.indices.shape == (3, 5)
        assert sr.elapsed_ms > 0

    def test_search_result_1d_query(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(10, DIM))
        sr = idx.search_result(_rng_vecs(1, DIM).squeeze(), k=3)
        assert sr.query_count == 1

    def test_len(self):
        idx = FlatIndex(dim=DIM)
        idx.add(_rng_vecs(10, DIM))
        assert len(idx) == 10
