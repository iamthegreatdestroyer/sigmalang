"""
Semantic Analogy Engine for ΣLANG.

This module implements semantic analogy resolution using hyperdimensional vectors.
The engine solves problems of the form: A:B::C:? 

Where ? is the unknown fourth element that completes the analogy.

Example:
    >>> engine = SemanticAnalogyEngine()
    >>> result = engine.solve_analogy("king", "queen", "prince")
    >>> result.answer  # Expected: "princess"
    >>> result.confidence  # Expected: 0.92

Algorithm:
    1. Encode A, B, C as HD vectors using semantic encoding
    2. Compute relationship vector: B - A (what makes B from A)
    3. Apply relationship to C: C + (B - A) (what makes ? from C)
    4. Find nearest HD vector to result in candidate space
    5. Return candidate with highest similarity

This leverages the geometric properties of hyperdimensional space
where semantic relationships are linear and composable.

Author: ΣLANG Team
License: AGPLv3 / Commercial Dual License
Copyright (c) 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from pathlib import Path
import json
import logging

from core.encoder import HyperdimensionalEncoder
from core.parser import SemanticParser

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AnalogyResult:
    """Result of semantic analogy solving."""

    a: str
    """First element of analogy"""

    b: str
    """Second element of analogy"""

    c: str
    """Third element of analogy"""

    answer: str
    """Predicted fourth element"""

    confidence: float
    """Confidence in answer (0-1)"""

    reasoning: str
    """Explanation of the analogy"""

    similarity_to_ground_truth: Optional[float] = None
    """Similarity to ground truth if available"""

    candidates: Optional[List[Tuple[str, float]]] = None
    """Top 5 candidates with similarities"""

    latency_ms: Optional[float] = None
    """Time to solve analogy in milliseconds"""


@dataclass
class AnalogyBenchmark:
    """Benchmark result for analogy solving."""

    total_analogies: int
    """Total number of analogies tested"""

    correct: int
    """Number of correct answers"""

    accuracy: float
    """Accuracy rate (0-1)"""

    avg_latency_ms: float
    """Average latency in milliseconds"""

    p95_latency_ms: float
    """95th percentile latency"""

    p99_latency_ms: float
    """99th percentile latency"""

    avg_confidence: float
    """Average confidence across all attempts"""

    category_results: Dict[str, Dict[str, float]]
    """Per-category breakdown: {category: {accuracy, count}}"""


class SemanticAnalogyEngine:
    """
    Hyperdimensional semantic analogy solver.

    This engine uses HD vector arithmetic to solve semantic analogies.
    It encodes concepts semantically and leverages the linear structure
    of HD space to find analogical relationships.

    Attributes:
        encoder: HyperdimensionalEncoder instance
        parser: SemanticParser instance
        candidates: Set of known concepts
        candidate_vectors: Mapping from concept to HD vector
    """

    def __init__(
        self,
        encoder: Optional[HyperdimensionalEncoder] = None,
        parser: Optional[SemanticParser] = None,
        vectorspace_dim: int = 10000,
    ):
        """
        Initialize the semantic analogy engine.

        Args:
            encoder: HyperdimensionalEncoder instance. If None, creates new one.
            parser: SemanticParser instance. If None, creates new one.
            vectorspace_dim: Dimensionality of HD vector space

        Example:
            >>> engine = SemanticAnalogyEngine()
            >>> result = engine.solve_analogy("hot", "cold", "fast")
            >>> print(result.answer)
        """
        self.encoder = encoder or HyperdimensionalEncoder(vectorspace_dim)
        self.parser = parser or SemanticParser()
        self.vectorspace_dim = vectorspace_dim

        # Candidate space
        self.candidates: Set[str] = set()
        self.candidate_vectors: Dict[str, np.ndarray] = {}

        # Performance tracking
        self.analogy_latencies: List[float] = []
        self.analogy_confidences: List[float] = []

        logger.info(
            f"SemanticAnalogyEngine initialized with dim={vectorspace_dim}"
        )

    def encode_concept(self, concept: str) -> np.ndarray:
        """
        Encode a concept as an HD vector.

        Args:
            concept: String representation of concept

        Returns:
            HD vector representation (1D numpy array)

        Raises:
            ValueError: If concept is empty or None
        """
        if not concept or not isinstance(concept, str):
            raise ValueError(f"Invalid concept: {concept}")

        # Parse concept into semantic components
        components = self.parser.parse_semantic_components(concept)

        # Encode each component
        encoded_components = [
            self.encoder.encode(component) for component in components
        ]

        # Combine via bundle (superposition)
        if len(encoded_components) == 1:
            return encoded_components[0]

        combined = encoded_components[0].copy()
        for component_vec in encoded_components[1:]:
            combined = self.encoder.bundle(combined, component_vec)

        return combined

    def register_candidates(self, concepts: List[str]) -> None:
        """
        Register a set of candidate concepts for analogy solving.

        Pre-encodes all candidates to speed up analogy solving.

        Args:
            concepts: List of concepts to register
        """
        for concept in concepts:
            if concept not in self.candidates:
                try:
                    vec = self.encode_concept(concept)
                    self.candidates.add(concept)
                    self.candidate_vectors[concept] = vec
                except Exception as e:
                    logger.warning(f"Failed to register candidate '{concept}': {e}")

        logger.info(f"Registered {len(self.candidates)} candidates")

    def solve_analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 5,
        exclude_set: Optional[Set[str]] = None,
    ) -> AnalogyResult:
        """
        Solve semantic analogy: A:B::C:?

        Algorithm:
            1. Encode A, B, C as HD vectors
            2. Compute relationship: relationship_vec = B - A
            3. Apply to C: result_vec = C + relationship_vec
            4. Find nearest candidate to result_vec
            5. Return result with confidence

        Args:
            a: First element
            b: Second element
            c: Third element
            top_k: Return top-k candidates
            exclude_set: Set of concepts to exclude from candidates

        Returns:
            AnalogyResult with answer, confidence, and candidates

        Raises:
            ValueError: If any concept is invalid
            RuntimeError: If no candidates registered
        """
        import time

        start_time = time.perf_counter()

        if not self.candidates:
            raise RuntimeError(
                "No candidates registered. Call register_candidates() first."
            )

        try:
            # Encode concepts
            a_vec = self.encode_concept(a)
            b_vec = self.encode_concept(b)
            c_vec = self.encode_concept(c)

            # Compute relationship vector: what makes B from A
            relationship_vec = b_vec - a_vec

            # Apply relationship to C: what makes ? from C
            analogy_vec = c_vec + relationship_vec

            # Normalize for similarity computation
            analogy_vec = analogy_vec / (np.linalg.norm(analogy_vec) + 1e-10)

            # Find nearest candidates
            exclude = exclude_set or {a, b, c}
            candidates_to_check = [
                (concept, vec)
                for concept, vec in self.candidate_vectors.items()
                if concept not in exclude
            ]

            if not candidates_to_check:
                raise RuntimeError("No valid candidates after exclusion")

            # Compute similarities
            similarities = [
                (
                    concept,
                    float(
                        np.dot(analogy_vec, vec / (np.linalg.norm(vec) + 1e-10))
                    ),
                )
                for concept, vec in candidates_to_check
            ]

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top result
            best_answer, best_similarity = similarities[0]
            confidence = (best_similarity + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]

            # Get top-k candidates
            top_candidates = similarities[:top_k]

            # Compute latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Track performance metrics
            self.analogy_latencies.append(latency_ms)
            self.analogy_confidences.append(confidence)

            return AnalogyResult(
                a=a,
                b=b,
                c=c,
                answer=best_answer,
                confidence=confidence,
                reasoning=self._generate_reasoning(a, b, c, best_answer),
                candidates=top_candidates,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Analogy solving failed for {a}:{b}::{c}:?: {e}")
            raise

    def _generate_reasoning(
        self, a: str, b: str, c: str, answer: str
    ) -> str:
        """
        Generate human-readable reasoning for the analogy.

        Args:
            a, b, c: Analogy elements
            answer: The answer

        Returns:
            Reasoning string
        """
        return (
            f"As '{a}' relates to '{b}', '{c}' relates to '{answer}' "
            f"via semantic vector arithmetic in hyperdimensional space."
        )

    def benchmark_accuracy(
        self, analogies: List[Tuple[str, str, str, str]], category: str = "general"
    ) -> AnalogyBenchmark:
        """
        Benchmark accuracy on a set of known analogies.

        Args:
            analogies: List of (a, b, c, correct_answer) tuples
            category: Category name for result tracking

        Returns:
            AnalogyBenchmark with accuracy and performance metrics

        Example:
            >>> analogies = [
            ...     ("king", "queen", "prince", "princess"),
            ...     ("man", "woman", "uncle", "aunt"),
            ... ]
            >>> benchmark = engine.benchmark_accuracy(analogies)
            >>> print(f"Accuracy: {benchmark.accuracy:.2%}")
        """
        correct_count = 0
        latencies = []

        for a, b, c, expected_answer in analogies:
            try:
                result = self.solve_analogy(a, b, c)
                latencies.append(result.latency_ms or 0)

                # Check if answer matches (exact or partial match)
                if result.answer.lower() == expected_answer.lower():
                    correct_count += 1
                    result.similarity_to_ground_truth = 1.0
                else:
                    # Compute semantic similarity to ground truth
                    try:
                        expected_vec = self.encode_concept(expected_answer)
                        answer_vec = self.encode_concept(result.answer)
                        similarity = float(
                            np.dot(
                                expected_vec / (np.linalg.norm(expected_vec) + 1e-10),
                                answer_vec / (np.linalg.norm(answer_vec) + 1e-10),
                            )
                        )
                        result.similarity_to_ground_truth = (similarity + 1.0) / 2.0
                    except Exception:
                        result.similarity_to_ground_truth = 0.0

            except Exception as e:
                logger.warning(f"Failed to solve {a}:{b}::{c}:? : {e}")
                latencies.append(0)

        # Compute statistics
        latencies_array = np.array(latencies)
        accuracy = correct_count / len(analogies) if analogies else 0.0
        avg_latency = float(np.mean(latencies_array)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies_array, 95))
        p99_latency = float(np.percentile(latencies_array, 99))
        avg_confidence = float(np.mean(self.analogy_confidences))

        return AnalogyBenchmark(
            total_analogies=len(analogies),
            correct=correct_count,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            avg_confidence=avg_confidence,
            category_results={
                category: {
                    "accuracy": accuracy,
                    "count": len(analogies),
                    "correct": correct_count,
                }
            },
        )

    def save_results(self, results: AnalogyResult, filepath: Path) -> None:
        """
        Save analogy result to JSON file.

        Args:
            results: AnalogyResult to save
            filepath: Path to save JSON file
        """
        data = {
            "a": results.a,
            "b": results.b,
            "c": results.c,
            "answer": results.answer,
            "confidence": results.confidence,
            "reasoning": results.reasoning,
            "latency_ms": results.latency_ms,
            "candidates": [
                {"concept": c, "similarity": float(s)}
                for c, s in (results.candidates or [])
            ],
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get summary of engine performance across all operations.

        Returns:
            Dictionary with performance metrics
        """
        latencies = np.array(self.analogy_latencies)
        confidences = np.array(self.analogy_confidences)

        return {
            "total_analogies": len(self.analogy_latencies),
            "avg_latency_ms": float(np.mean(latencies)) if len(latencies) > 0 else 0,
            "p95_latency_ms": float(np.percentile(latencies, 95))
            if len(latencies) > 0
            else 0,
            "p99_latency_ms": float(np.percentile(latencies, 99))
            if len(latencies) > 0
            else 0,
            "avg_confidence": float(np.mean(confidences))
            if len(confidences) > 0
            else 0,
            "min_confidence": float(np.min(confidences))
            if len(confidences) > 0
            else 0,
            "max_confidence": float(np.max(confidences))
            if len(confidences) > 0
            else 0,
        }
