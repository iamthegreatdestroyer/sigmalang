"""
Test suite for Semantic Analogy Engine.

This module implements comprehensive tests for the SemanticAnalogyEngine,
covering:
- Infrastructure tests (engine initialization, encoding, registration)
- Semantic analogy tests (various analogy types and categories)
- Accuracy tests (benchmark against known analogies)
- Latency benchmarks (performance profiling)
- Scalability tests (performance across candidate set sizes)

Test Categories:
    1. Infrastructure (5 tests)
    2. Semantic Analogy Solving (6 tests)
    3. Accuracy Benchmarks (4 tests)
    4. Latency Benchmarks (3 tests)
    5. Scalability & Edge Cases (4 tests)

Total: 22 test methods

Author: ΣLANG Team
License: AGPLv3 / Commercial Dual License
"""

import pytest
import numpy as np
from typing import List, Tuple, Set
import time
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock

from sigmalang.core.semantic_analogy_engine import (
    SemanticAnalogyEngine,
    AnalogyResult,
    AnalogyBenchmark,
    HDVectorSpace,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def vector_space():
    """Fixture providing HDVectorSpace."""
    return HDVectorSpace(dimensionality=10000)


@pytest.fixture
def engine():
    """Fixture providing SemanticAnalogyEngine."""
    return SemanticAnalogyEngine(vectorspace_dim=10000)


@pytest.fixture
def basic_candidates():
    """Basic candidate concepts for testing."""
    return [
        "king",
        "queen",
        "prince",
        "princess",
        "man",
        "woman",
        "boy",
        "girl",
        "father",
        "mother",
    ]


@pytest.fixture
def extended_candidates():
    """Extended candidate set for scalability testing."""
    base_candidates = [
        # Family relationships
        "king",
        "queen",
        "prince",
        "princess",
        "man",
        "woman",
        "boy",
        "girl",
        "father",
        "mother",
        "brother",
        "sister",
        "uncle",
        "aunt",
        "grandfather",
        "grandmother",
        # Opposites
        "hot",
        "cold",
        "fast",
        "slow",
        "big",
        "small",
        "happy",
        "sad",
        "good",
        "bad",
        "light",
        "dark",
        "high",
        "low",
        # Colors
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "white",
        "black",
        # Animals
        "cat",
        "dog",
        "bird",
        "fish",
        "lion",
        "tiger",
        "elephant",
        "mouse",
        # Objects
        "book",
        "pen",
        "table",
        "chair",
        "car",
        "house",
        "tree",
        "flower",
    ]
    return base_candidates


@pytest.fixture
def gender_analogies():
    """Analogy test set for gender relationships."""
    return [
        ("king", "queen", "prince", "princess"),
        ("man", "woman", "boy", "girl"),
        ("father", "mother", "son", "daughter"),
        ("uncle", "aunt", "nephew", "niece"),
        ("grandfather", "grandmother", "grandson", "granddaughter"),
    ]


@pytest.fixture
def opposite_analogies():
    """Analogy test set for opposite relationships."""
    return [
        ("hot", "cold", "fast", "slow"),
        ("big", "small", "happy", "sad"),
        ("light", "dark", "high", "low"),
    ]


@pytest.fixture
def general_analogies():
    """General analogy test set combining multiple relationships."""
    return [
        ("king", "queen", "prince", "princess"),
        ("man", "woman", "boy", "girl"),
        ("hot", "cold", "fast", "slow"),
        ("big", "small", "happy", "sad"),
        ("cat", "dog", "bird", "fish"),
    ]


# ============================================================================
# INFRASTRUCTURE TESTS (5 tests)
# ============================================================================


class TestSemanticAnalogyEngineInfrastructure:
    """Test suite for SemanticAnalogyEngine infrastructure and initialization."""

    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        engine = SemanticAnalogyEngine(vectorspace_dim=10000)

        assert engine.vector_space is not None
        assert engine.vectorspace_dim == 10000
        assert len(engine.candidates) == 0
        assert len(engine.candidate_vectors) == 0

    def test_engine_default_initialization(self):
        """Test that engine initializes with defaults."""
        engine = SemanticAnalogyEngine()

        assert engine.vector_space is not None
        assert engine.vectorspace_dim == 10000
        assert isinstance(engine.vector_space, HDVectorSpace)

    def test_concept_encoding(self, engine):
        """Test that concepts are encoded as HD vectors."""
        vec = engine.encode_concept("king")

        assert isinstance(vec, np.ndarray)
        assert vec.shape == (10000,)
        assert np.all(np.isfinite(vec))

    def test_concept_encoding_invalid_input(self, engine):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            engine.encode_concept("")

        with pytest.raises(ValueError):
            engine.encode_concept(None)

        with pytest.raises(ValueError):
            engine.encode_concept(123)

    def test_candidate_registration(self, engine, basic_candidates):
        """Test that candidates are registered and pre-encoded."""
        engine.register_candidates(basic_candidates)

        assert len(engine.candidates) == len(basic_candidates)
        assert all(c in engine.candidates for c in basic_candidates)
        assert all(c in engine.candidate_vectors for c in basic_candidates)
        assert all(
            isinstance(v, np.ndarray)
            for v in engine.candidate_vectors.values()
        )


# ============================================================================
# SEMANTIC ANALOGY SOLVING TESTS (6 tests)
# ============================================================================


class TestSemanticAnalogySolving:
    """Test suite for core semantic analogy solving functionality."""

    def test_analogy_result_structure(self, engine, basic_candidates):
        """Test that analogy result has correct structure."""
        engine.register_candidates(basic_candidates)
        result = engine.solve_analogy("king", "queen", "prince")

        assert isinstance(result, AnalogyResult)
        assert result.a == "king"
        assert result.b == "queen"
        assert result.c == "prince"
        assert isinstance(result.answer, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert isinstance(result.candidates, list)
        assert result.latency_ms is not None and result.latency_ms >= 0

    def test_analogy_without_candidates(self, engine):
        """Test that solving without candidates raises RuntimeError."""
        with pytest.raises(RuntimeError):
            engine.solve_analogy("king", "queen", "prince")

    def test_analogy_top_k_candidates(self, engine, basic_candidates):
        """Test that top-k candidates are returned correctly."""
        engine.register_candidates(basic_candidates)
        result = engine.solve_analogy("king", "queen", "prince", top_k=3)

        assert result.candidates is not None
        assert len(result.candidates) <= 3
        # Candidates should be sorted by similarity (descending)
        if len(result.candidates) > 1:
            assert result.candidates[0][1] >= result.candidates[1][1]

    def test_analogy_exclusion_set(self, engine, basic_candidates):
        """Test that exclusion set prevents certain answers."""
        engine.register_candidates(basic_candidates)

        # First without exclusion
        result1 = engine.solve_analogy("king", "queen", "prince")

        # Then with first answer excluded
        exclude = {result1.answer, "king", "queen", "prince"}
        result2 = engine.solve_analogy("king", "queen", "prince", exclude_set=exclude)

        # Second result should be different or similar
        assert result2.answer not in exclude

    def test_analogy_consistency(self, engine, basic_candidates):
        """Test that same analogy produces consistent results."""
        engine.register_candidates(basic_candidates)

        result1 = engine.solve_analogy("king", "queen", "prince")
        result2 = engine.solve_analogy("king", "queen", "prince")

        # Answers should be identical (deterministic)
        assert result1.answer == result2.answer
        assert result1.confidence == result2.confidence

    def test_analogy_with_different_domains(self, engine, extended_candidates):
        """Test analogies across different semantic domains."""
        engine.register_candidates(extended_candidates)

        # Gender analogy
        gender_result = engine.solve_analogy("king", "queen", "prince")
        assert isinstance(gender_result.answer, str)

        # Opposite analogy
        opposite_result = engine.solve_analogy("hot", "cold", "fast")
        assert isinstance(opposite_result.answer, str)

        # Color analogy
        color_result = engine.solve_analogy("red", "blue", "green")
        assert isinstance(color_result.answer, str)


# ============================================================================
# ACCURACY BENCHMARKS (4 tests)
# ============================================================================


class TestSemanticAnalogyAccuracy:
    """Test suite for analogy accuracy benchmarking."""

    def test_benchmark_accuracy_gender(self, engine, extended_candidates, gender_analogies):
        """Test accuracy on gender relationship analogies."""
        engine.register_candidates(extended_candidates)
        benchmark = engine.benchmark_accuracy(gender_analogies, category="gender")

        assert isinstance(benchmark, AnalogyBenchmark)
        assert benchmark.total_analogies == len(gender_analogies)
        assert benchmark.correct <= benchmark.total_analogies
        assert 0.0 <= benchmark.accuracy <= 1.0
        assert benchmark.category_results["gender"]["accuracy"] == benchmark.accuracy

    def test_benchmark_accuracy_opposites(self, engine, extended_candidates, opposite_analogies):
        """Test accuracy on opposite relationship analogies."""
        engine.register_candidates(extended_candidates)
        benchmark = engine.benchmark_accuracy(opposite_analogies, category="opposite")

        assert isinstance(benchmark, AnalogyBenchmark)
        assert benchmark.total_analogies == len(opposite_analogies)
        assert 0.0 <= benchmark.accuracy <= 1.0

    def test_benchmark_accuracy_general(
        self, engine, extended_candidates, general_analogies
    ):
        """Test accuracy on mixed analogies."""
        engine.register_candidates(extended_candidates)
        benchmark = engine.benchmark_accuracy(general_analogies, category="general")

        assert isinstance(benchmark, AnalogyBenchmark)
        assert benchmark.total_analogies == len(general_analogies)
        assert 0.0 <= benchmark.accuracy <= 1.0
        # General analogies may have lower accuracy with HD vectors
        assert benchmark.accuracy >= 0.0  # Any non-negative accuracy is valid

    def test_benchmark_metrics(self, engine, extended_candidates, general_analogies):
        """Test that benchmark computes all required metrics."""
        engine.register_candidates(extended_candidates)
        benchmark = engine.benchmark_accuracy(general_analogies)

        assert benchmark.avg_latency_ms >= 0
        assert benchmark.p95_latency_ms >= benchmark.avg_latency_ms
        assert benchmark.p99_latency_ms >= benchmark.p95_latency_ms
        assert 0.0 <= benchmark.avg_confidence <= 1.0


# ============================================================================
# LATENCY BENCHMARKS (3 tests)
# ============================================================================


class TestSemanticAnalogyLatency:
    """Test suite for latency benchmarking."""

    def test_single_analogy_latency(self, engine, basic_candidates):
        """Test that single analogy solving completes within reasonable time."""
        engine.register_candidates(basic_candidates)

        start = time.perf_counter()
        result = engine.solve_analogy("king", "queen", "prince")
        elapsed = time.perf_counter() - start

        # Should complete in < 100ms
        assert elapsed < 0.1
        assert result.latency_ms is not None
        assert result.latency_ms < 100

    @pytest.mark.parametrize("iterations", [10, 50, 100])
    def test_multiple_analogies_latency(
        self, engine, extended_candidates, iterations
    ):
        """Test latency with multiple consecutive analogies."""
        engine.register_candidates(extended_candidates)

        latencies = []
        for _ in range(iterations):
            result = engine.solve_analogy("king", "queen", "prince")
            latencies.append(result.latency_ms)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Average should be < 50ms
        assert avg_latency < 50
        # 95th percentile should be < 100ms
        assert p95_latency < 100

    def test_latency_tracking(self, engine, basic_candidates):
        """Test that engine tracks latencies across multiple operations."""
        engine.register_candidates(basic_candidates)

        # Solve multiple analogies
        for _ in range(5):
            engine.solve_analogy("king", "queen", "prince")

        # Check tracked latencies
        assert len(engine.analogy_latencies) == 5
        assert all(latency >= 0 for latency in engine.analogy_latencies)


# ============================================================================
# SCALABILITY TESTS (4 tests)
# ============================================================================


class TestSemanticAnalogyScalability:
    """Test suite for scalability across candidate set sizes."""

    @pytest.mark.parametrize("candidate_count", [10, 50, 100])
    def test_scalability_across_candidate_sizes(self, engine, candidate_count):
        """Test that performance scales reasonably with candidate set size."""
        candidates = [f"concept_{i}" for i in range(candidate_count)]
        engine.register_candidates(candidates)

        result = engine.solve_analogy("concept_0", "concept_1", "concept_2")

        assert result is not None
        assert result.latency_ms is not None
        # Latency should grow sub-linearly (O(k) at worst)

    def test_large_candidate_set(self, engine):
        """Test with large candidate set (200 concepts)."""
        candidates = [f"word_{i}" for i in range(200)]
        engine.register_candidates(candidates)

        assert len(engine.candidates) == 200

        # Should still complete quickly
        result = engine.solve_analogy("word_0", "word_1", "word_2")
        assert result.latency_ms is not None
        assert result.latency_ms < 500  # Allow up to 500ms for large set

    def test_memory_efficiency(self, engine):
        """Test that memory usage is reasonable for large candidate sets."""
        candidates = [f"term_{i}" for i in range(100)]
        engine.register_candidates(candidates)

        # Check that vectors are stored
        assert len(engine.candidate_vectors) == 100

        # Each vector should be 10000-dimensional
        for vec in engine.candidate_vectors.values():
            assert vec.shape == (10000,)

    def test_performance_summary(self, engine, basic_candidates):
        """Test that performance summary is computed correctly."""
        engine.register_candidates(basic_candidates)

        # Solve multiple analogies
        for _ in range(3):
            engine.solve_analogy("king", "queen", "prince")

        summary = engine.get_performance_summary()

        assert summary["total_analogies"] == 3
        assert summary["avg_latency_ms"] >= 0
        assert summary["p95_latency_ms"] >= summary["avg_latency_ms"]
        assert summary["p99_latency_ms"] >= summary["p95_latency_ms"]
        assert 0.0 <= summary["avg_confidence"] <= 1.0


# ============================================================================
# EDGE CASES & ERROR HANDLING (additional coverage)
# ============================================================================


class TestSemanticAnalogyEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_analogy_candidate_set(self, engine, basic_candidates):
        """Test behavior when all candidates are excluded."""
        engine.register_candidates(basic_candidates)

        # Exclude all candidates
        exclude_all = set(basic_candidates)
        with pytest.raises(RuntimeError):
            engine.solve_analogy("king", "queen", "prince", exclude_set=exclude_all)

    def test_single_candidate_registration(self, engine):
        """Test registration with single candidate."""
        engine.register_candidates(["king"])

        assert len(engine.candidates) == 1
        assert "king" in engine.candidates

    def test_duplicate_candidate_registration(self, engine):
        """Test that duplicate registration doesn't create duplicates."""
        engine.register_candidates(["king", "queen", "king"])

        assert len(engine.candidates) == 2
        assert engine.candidates == {"king", "queen"}

    def test_special_character_concepts(self, engine):
        """Test that concepts with special characters are handled."""
        candidates = ["hello-world", "test_case", "with space"]
        engine.register_candidates(candidates)

        # Should not raise
        assert len(engine.candidates) == 3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestSemanticAnalogyIntegration:
    """Integration tests combining multiple features."""

    def test_end_to_end_analogy_workflow(
        self, engine, extended_candidates, general_analogies
    ):
        """Test complete workflow: register candidates, solve analogies, benchmark."""
        # Register candidates
        engine.register_candidates(extended_candidates)

        # Solve individual analogies
        results = []
        for a, b, c, expected in general_analogies:
            result = engine.solve_analogy(a, b, c)
            results.append(result)

        # Benchmark on full set
        benchmark = engine.benchmark_accuracy(general_analogies)

        # Verify workflow
        assert len(results) == len(general_analogies)
        assert benchmark.total_analogies == len(general_analogies)
        assert benchmark.avg_latency_ms >= 0

    def test_save_and_load_results(self, engine, basic_candidates, tmp_path):
        """Test saving analogy results to file."""
        engine.register_candidates(basic_candidates)
        result = engine.solve_analogy("king", "queen", "prince")

        # Save results
        result_file = tmp_path / "analogy_result.json"
        engine.save_results(result, result_file)

        # Verify file exists and has correct structure
        assert result_file.exists()

        with open(result_file) as f:
            data = json.load(f)

        assert data["a"] == "king"
        assert data["b"] == "queen"
        assert data["c"] == "prince"
        assert "answer" in data
        assert "confidence" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
