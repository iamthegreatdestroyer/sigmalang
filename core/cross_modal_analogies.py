"""
Cross-Modal Analogies for sigmalang.

This module provides cross-modal analogy reasoning, enabling reasoning
like "A is to B as C is to D" across different modalities (text, code,
mathematical concepts).

Classes:
    Modality: Enumeration of supported modalities
    ModalityConfig: Configuration for modality handling
    ModalityBridge: Bridge between different modalities
    AnalogyPair: Represents an analogy pair (source, target)
    AnalogyResult: Result of analogy solving
    CrossModalEncoder: Encodes items across modalities
    RelationExtractor: Extracts relations between concepts
    AnalogySolver: Solves analogy problems
    CrossModalAnalogy: Main cross-modal analogy interface

Example:
    >>> analogy = CrossModalAnalogy()
    >>> # king - man + woman = queen
    >>> result = analogy.solve("king", "man", "woman")
    >>> print(f"Result: {result.answer}")
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .transformer_embeddings import (
    TransformerEncoder,
    EmbeddingConfig,
    EmbeddingCache,
)


class Modality(Enum):
    """Supported modalities for cross-modal reasoning."""
    TEXT = auto()        # Natural language text
    CODE = auto()        # Programming code
    MATH = auto()        # Mathematical expressions
    SYMBOL = auto()      # Symbolic representations
    SEMANTIC = auto()    # Semantic concepts
    NUMERIC = auto()     # Numeric values


@dataclass
class ModalityConfig:
    """Configuration for modality handling.
    
    Attributes:
        modality: The modality type
        embedding_dim: Dimension of embeddings
        projection_dim: Dimension after projection
        normalize: Whether to normalize embeddings
        use_cache: Whether to cache embeddings
    """
    modality: Modality = Modality.TEXT
    embedding_dim: int = 384
    projection_dim: int = 256
    normalize: bool = True
    use_cache: bool = True


@dataclass
class AnalogyPair:
    """Represents a pair in an analogy.
    
    For analogy A:B::C:D, this represents one pair (A:B or C:D).
    
    Attributes:
        source: Source concept
        target: Target concept
        modality: Modality of the pair
        relation_vector: Computed relation vector
        confidence: Confidence in the relation
    """
    source: str
    target: str
    modality: Modality = Modality.TEXT
    relation_vector: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class AnalogyResult:
    """Result of analogy solving.
    
    Attributes:
        query: The query (A, B, C) triple
        answer: The computed answer D
        confidence: Confidence score [0, 1]
        candidates: List of candidate answers with scores
        computation_time: Time to solve (seconds)
        method: Method used to solve
        metadata: Additional metadata
    """
    query: Tuple[str, str, str]
    answer: str
    confidence: float
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    computation_time: float = 0.0
    method: str = "vector_arithmetic"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationResult:
    """Result of relation extraction.
    
    Attributes:
        source: Source concept
        target: Target concept
        relation_type: Type of relation
        relation_vector: Vector representation
        strength: Relation strength [0, 1]
        bidirectional: Whether relation is bidirectional
    """
    source: str
    target: str
    relation_type: str
    relation_vector: np.ndarray
    strength: float
    bidirectional: bool = False


class ModalityProjector:
    """Projects embeddings between modalities.
    
    Learns projections between different modality spaces to enable
    cross-modal comparisons.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        shared_dim: int = 256
    ):
        """Initialize projector.
        
        Args:
            source_dim: Source embedding dimension
            target_dim: Target embedding dimension
            shared_dim: Shared space dimension
        """
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.shared_dim = shared_dim
        
        # Initialize random projection matrices
        np.random.seed(42)  # Deterministic
        self._source_projection = self._init_projection(source_dim, shared_dim)
        self._target_projection = self._init_projection(target_dim, shared_dim)
        self._lock = threading.Lock()
    
    def _init_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Initialize projection matrix."""
        # Random orthogonal projection
        matrix = np.random.randn(input_dim, output_dim)
        # Normalize columns
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return (matrix / norms).astype(np.float32)
    
    def project_source(self, embedding: np.ndarray) -> np.ndarray:
        """Project source embedding to shared space.
        
        Args:
            embedding: Source embedding
            
        Returns:
            Projected embedding
        """
        with self._lock:
            projected = embedding @ self._source_projection
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm
            return projected
    
    def project_target(self, embedding: np.ndarray) -> np.ndarray:
        """Project target embedding to shared space.
        
        Args:
            embedding: Target embedding
            
        Returns:
            Projected embedding
        """
        with self._lock:
            projected = embedding @ self._target_projection
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm
            return projected
    
    def align(
        self,
        source_pairs: List[Tuple[np.ndarray, np.ndarray]],
        target_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ):
        """Align projections using parallel pairs.
        
        Args:
            source_pairs: Pairs of (source_emb, aligned_target_emb)
            target_pairs: Same pairs from target perspective
        """
        if not source_pairs or not target_pairs:
            return
        
        # Simple Procrustes alignment
        # This is a placeholder for more sophisticated alignment
        with self._lock:
            # Average the pairs to get alignment direction
            for (s, t) in source_pairs[:10]:
                s_proj = self.project_source(s)
                t_proj = self.project_target(t)
                
                # Adjust projection slightly towards alignment
                self._source_projection += 0.01 * np.outer(s, t_proj - s_proj)


class CrossModalEncoder:
    """Encodes items across different modalities.
    
    Uses modality-specific preprocessing and shared embedding space.
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None
    ):
        """Initialize encoder.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig(fallback_dim=256)
        self._encoder = TransformerEncoder(
            config=self.config,
            use_cache=True
        )
        self._modality_processors: Dict[Modality, Callable] = {
            Modality.TEXT: self._process_text,
            Modality.CODE: self._process_code,
            Modality.MATH: self._process_math,
            Modality.SYMBOL: self._process_symbol,
            Modality.SEMANTIC: self._process_semantic,
            Modality.NUMERIC: self._process_numeric,
        }
        self._projectors: Dict[Tuple[Modality, Modality], ModalityProjector] = {}
        self._lock = threading.Lock()
    
    def _process_text(self, item: str) -> str:
        """Process text modality."""
        return item.lower().strip()
    
    def _process_code(self, item: str) -> str:
        """Process code modality."""
        # Add code markers for better embedding
        return f"[CODE] {item.strip()} [/CODE]"
    
    def _process_math(self, item: str) -> str:
        """Process math modality."""
        return f"[MATH] {item.strip()} [/MATH]"
    
    def _process_symbol(self, item: str) -> str:
        """Process symbolic modality."""
        return f"[SYMBOL] {item.strip()} [/SYMBOL]"
    
    def _process_semantic(self, item: str) -> str:
        """Process semantic modality."""
        return f"[CONCEPT] {item.strip()} [/CONCEPT]"
    
    def _process_numeric(self, item: str) -> str:
        """Process numeric modality."""
        return f"[NUM] {item.strip()} [/NUM]"
    
    def encode(
        self,
        item: str,
        modality: Modality = Modality.TEXT
    ) -> np.ndarray:
        """Encode an item to embedding.
        
        Args:
            item: Item to encode
            modality: Modality of the item
            
        Returns:
            Embedding vector
        """
        # Preprocess based on modality
        processor = self._modality_processors.get(
            modality,
            self._process_text
        )
        processed = processor(item)
        
        # Encode
        return self._encoder.encode_text(processed)
    
    def encode_batch(
        self,
        items: List[str],
        modality: Modality = Modality.TEXT
    ) -> np.ndarray:
        """Encode multiple items.
        
        Args:
            items: List of items to encode
            modality: Modality of all items
            
        Returns:
            Array of embeddings
        """
        processor = self._modality_processors.get(
            modality,
            self._process_text
        )
        processed = [processor(item) for item in items]
        return self._encoder.encode_batch(processed)
    
    def get_projector(
        self,
        source_modality: Modality,
        target_modality: Modality
    ) -> ModalityProjector:
        """Get or create projector between modalities.
        
        Args:
            source_modality: Source modality
            target_modality: Target modality
            
        Returns:
            ModalityProjector instance
        """
        key = (source_modality, target_modality)
        
        with self._lock:
            if key not in self._projectors:
                dim = self._encoder.get_dimensionality()
                self._projectors[key] = ModalityProjector(
                    source_dim=dim,
                    target_dim=dim,
                    shared_dim=min(256, dim)
                )
            return self._projectors[key]
    
    def cross_modal_similarity(
        self,
        item1: str,
        modality1: Modality,
        item2: str,
        modality2: Modality
    ) -> float:
        """Compute similarity across modalities.
        
        Args:
            item1: First item
            modality1: Modality of first item
            item2: Second item
            modality2: Modality of second item
            
        Returns:
            Similarity score
        """
        emb1 = self.encode(item1, modality1)
        emb2 = self.encode(item2, modality2)
        
        if modality1 != modality2:
            projector = self.get_projector(modality1, modality2)
            emb1 = projector.project_source(emb1)
            emb2 = projector.project_target(emb2)
        
        return self._cosine_similarity(emb1, emb2)
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class RelationExtractor:
    """Extracts semantic relations between concepts.
    
    Identifies and quantifies relationships like synonymy, antonymy,
    hypernymy, meronymy, etc.
    """
    
    # Known relation types
    RELATION_TYPES = {
        "synonymy": ["same_as", "similar_to", "equivalent"],
        "antonymy": ["opposite_of", "contrary_to"],
        "hypernymy": ["is_a", "type_of", "subclass_of"],
        "hyponymy": ["includes", "has_type"],
        "meronymy": ["part_of", "component_of"],
        "holonymy": ["has_part", "contains"],
        "causation": ["causes", "leads_to", "results_in"],
        "transformation": ["becomes", "transforms_to"],
        "association": ["related_to", "associated_with"],
    }
    
    def __init__(self, encoder: CrossModalEncoder):
        """Initialize extractor.
        
        Args:
            encoder: Cross-modal encoder instance
        """
        self._encoder = encoder
        self._relation_cache: Dict[str, RelationResult] = {}
        self._relation_prototypes: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        
        # Initialize relation prototypes
        self._init_prototypes()
    
    def _init_prototypes(self):
        """Initialize prototype relation vectors."""
        # Define archetypal pairs for each relation type
        archetypes = {
            "synonymy": [("big", "large"), ("small", "little")],
            "antonymy": [("good", "bad"), ("hot", "cold")],
            "hypernymy": [("dog", "animal"), ("car", "vehicle")],
            "transformation": [("ice", "water"), ("caterpillar", "butterfly")],
            "causation": [("fire", "smoke"), ("rain", "flood")],
        }
        
        for relation_type, pairs in archetypes.items():
            vectors = []
            for source, target in pairs:
                emb_source = self._encoder.encode(source, Modality.TEXT)
                emb_target = self._encoder.encode(target, Modality.TEXT)
                vectors.append(emb_target - emb_source)
            
            if vectors:
                # Average of relation vectors
                self._relation_prototypes[relation_type] = np.mean(
                    vectors, axis=0
                )
    
    def _cache_key(self, source: str, target: str) -> str:
        """Generate cache key for relation."""
        return hashlib.md5(f"{source}:{target}".encode()).hexdigest()[:16]
    
    def extract_relation(
        self,
        source: str,
        target: str,
        modality: Modality = Modality.TEXT
    ) -> RelationResult:
        """Extract relation between two concepts.
        
        Args:
            source: Source concept
            target: Target concept
            modality: Modality of concepts
            
        Returns:
            RelationResult with extracted relation
        """
        cache_key = self._cache_key(source, target)
        
        with self._lock:
            if cache_key in self._relation_cache:
                return self._relation_cache[cache_key]
        
        # Compute embeddings
        emb_source = self._encoder.encode(source, modality)
        emb_target = self._encoder.encode(target, modality)
        
        # Compute relation vector
        relation_vector = emb_target - emb_source
        
        # Classify relation type
        relation_type, strength = self._classify_relation(relation_vector)
        
        # Check bidirectionality
        reverse_vector = emb_source - emb_target
        reverse_type, reverse_strength = self._classify_relation(reverse_vector)
        bidirectional = (
            relation_type == reverse_type and
            abs(strength - reverse_strength) < 0.2
        )
        
        result = RelationResult(
            source=source,
            target=target,
            relation_type=relation_type,
            relation_vector=relation_vector,
            strength=strength,
            bidirectional=bidirectional
        )
        
        with self._lock:
            self._relation_cache[cache_key] = result
        
        return result
    
    def _classify_relation(
        self,
        relation_vector: np.ndarray
    ) -> Tuple[str, float]:
        """Classify relation vector to type.
        
        Args:
            relation_vector: Relation vector
            
        Returns:
            Tuple of (relation_type, strength)
        """
        best_type = "association"
        best_similarity = 0.0
        
        for rel_type, prototype in self._relation_prototypes.items():
            similarity = self._cosine_similarity(relation_vector, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = rel_type
        
        # Convert similarity to strength
        strength = (best_similarity + 1) / 2  # Map [-1, 1] to [0, 1]
        
        return best_type, strength
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def find_similar_relations(
        self,
        relation: RelationResult,
        candidates: List[Tuple[str, str]],
        top_k: int = 5
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Find pairs with similar relations.
        
        Args:
            relation: Reference relation
            candidates: List of (source, target) pairs
            top_k: Number of results to return
            
        Returns:
            List of ((source, target), similarity) sorted by similarity
        """
        results = []
        
        for source, target in candidates:
            cand_relation = self.extract_relation(source, target)
            similarity = self._cosine_similarity(
                relation.relation_vector,
                cand_relation.relation_vector
            )
            results.append(((source, target), similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class AnalogySolver:
    """Solves analogy problems using vector arithmetic.
    
    Implements A:B::C:? where D = B - A + C
    """
    
    def __init__(
        self,
        encoder: CrossModalEncoder,
        vocabulary: Optional[List[str]] = None
    ):
        """Initialize solver.
        
        Args:
            encoder: Cross-modal encoder
            vocabulary: Optional vocabulary for answer lookup
        """
        self._encoder = encoder
        self._vocabulary = vocabulary or []
        self._vocabulary_embeddings: Optional[np.ndarray] = None
        self._lock = threading.Lock()
    
    def set_vocabulary(self, vocabulary: List[str]):
        """Set vocabulary for answer lookup.
        
        Args:
            vocabulary: List of possible answers
        """
        with self._lock:
            self._vocabulary = vocabulary
            if vocabulary:
                self._vocabulary_embeddings = self._encoder.encode_batch(
                    vocabulary
                )
            else:
                self._vocabulary_embeddings = None
    
    def add_to_vocabulary(self, items: List[str]):
        """Add items to vocabulary.
        
        Args:
            items: Items to add
        """
        with self._lock:
            new_items = [item for item in items if item not in self._vocabulary]
            if not new_items:
                return
            
            self._vocabulary.extend(new_items)
            new_embeddings = self._encoder.encode_batch(new_items)
            
            if self._vocabulary_embeddings is None:
                self._vocabulary_embeddings = new_embeddings
            else:
                self._vocabulary_embeddings = np.vstack([
                    self._vocabulary_embeddings,
                    new_embeddings
                ])
    
    def solve(
        self,
        a: str,
        b: str,
        c: str,
        modality: Modality = Modality.TEXT,
        method: str = "vector_arithmetic"
    ) -> AnalogyResult:
        """Solve analogy A:B::C:?
        
        Args:
            a: First element of source pair
            b: Second element of source pair
            c: First element of target pair
            modality: Modality of elements
            method: Solving method
            
        Returns:
            AnalogyResult with answer and candidates
        """
        start_time = time.time()
        
        if method == "vector_arithmetic":
            result = self._solve_vector_arithmetic(a, b, c, modality)
        elif method == "relation_transfer":
            result = self._solve_relation_transfer(a, b, c, modality)
        else:
            # Default to vector arithmetic
            result = self._solve_vector_arithmetic(a, b, c, modality)
        
        result.computation_time = time.time() - start_time
        result.method = method
        
        return result
    
    def _solve_vector_arithmetic(
        self,
        a: str,
        b: str,
        c: str,
        modality: Modality
    ) -> AnalogyResult:
        """Solve using vector arithmetic: D = B - A + C."""
        # Encode
        emb_a = self._encoder.encode(a, modality)
        emb_b = self._encoder.encode(b, modality)
        emb_c = self._encoder.encode(c, modality)
        
        # Compute target vector
        emb_d = emb_b - emb_a + emb_c
        
        # Normalize
        norm = np.linalg.norm(emb_d)
        if norm > 0:
            emb_d = emb_d / norm
        
        # Find closest in vocabulary
        candidates = self._find_closest(emb_d, exclude=[a, b, c])
        
        if candidates:
            answer = candidates[0][0]
            confidence = candidates[0][1]
        else:
            answer = f"[{c}]"  # Fallback to marked C
            confidence = 0.0
        
        return AnalogyResult(
            query=(a, b, c),
            answer=answer,
            confidence=confidence,
            candidates=candidates
        )
    
    def _solve_relation_transfer(
        self,
        a: str,
        b: str,
        c: str,
        modality: Modality
    ) -> AnalogyResult:
        """Solve by transferring relation from A:B to C:?."""
        # Extract relation from A:B
        emb_a = self._encoder.encode(a, modality)
        emb_b = self._encoder.encode(b, modality)
        relation = emb_b - emb_a
        
        # Apply to C
        emb_c = self._encoder.encode(c, modality)
        emb_d = emb_c + relation
        
        # Normalize
        norm = np.linalg.norm(emb_d)
        if norm > 0:
            emb_d = emb_d / norm
        
        # Find closest
        candidates = self._find_closest(emb_d, exclude=[a, b, c])
        
        if candidates:
            answer = candidates[0][0]
            confidence = candidates[0][1]
        else:
            answer = f"[{c}]"
            confidence = 0.0
        
        return AnalogyResult(
            query=(a, b, c),
            answer=answer,
            confidence=confidence,
            candidates=candidates
        )
    
    def _find_closest(
        self,
        target: np.ndarray,
        exclude: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find closest words in vocabulary.
        
        Args:
            target: Target embedding
            exclude: Words to exclude
            top_k: Number of results
            
        Returns:
            List of (word, similarity) tuples
        """
        with self._lock:
            if self._vocabulary_embeddings is None or len(self._vocabulary) == 0:
                return []
            
            # Compute similarities
            similarities = self._vocabulary_embeddings @ target
            
            # Create results excluding specified words
            results = []
            exclude_set = set(w.lower() for w in exclude)
            
            for idx, sim in enumerate(similarities):
                word = self._vocabulary[idx]
                if word.lower() not in exclude_set:
                    results.append((word, float(sim)))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
    
    def evaluate_analogy(
        self,
        a: str,
        b: str,
        c: str,
        expected_d: str,
        modality: Modality = Modality.TEXT
    ) -> Dict[str, Any]:
        """Evaluate analogy solution against expected answer.
        
        Args:
            a: First element of source pair
            b: Second element of source pair
            c: First element of target pair
            expected_d: Expected answer
            modality: Modality of elements
            
        Returns:
            Evaluation metrics
        """
        result = self.solve(a, b, c, modality)
        
        # Check if correct
        correct = result.answer.lower() == expected_d.lower()
        
        # Find rank of expected answer
        rank = -1
        for i, (cand, _) in enumerate(result.candidates):
            if cand.lower() == expected_d.lower():
                rank = i + 1
                break
        
        return {
            "correct": correct,
            "predicted": result.answer,
            "expected": expected_d,
            "confidence": result.confidence,
            "rank": rank,
            "mrr": 1.0 / rank if rank > 0 else 0.0
        }


class RelationTransfer:
    """Transfers relations between concept pairs.
    
    Given a relation extracted from one pair, applies it to generate
    new pairs with the same relation.
    """
    
    def __init__(
        self,
        encoder: CrossModalEncoder,
        relation_extractor: RelationExtractor
    ):
        """Initialize transfer module.
        
        Args:
            encoder: Cross-modal encoder
            relation_extractor: Relation extractor
        """
        self._encoder = encoder
        self._extractor = relation_extractor
        self._transfer_cache: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
    
    def transfer_relation(
        self,
        source_pair: Tuple[str, str],
        target_source: str,
        vocabulary: List[str],
        top_k: int = 5,
        modality: Modality = Modality.TEXT
    ) -> List[Tuple[str, float]]:
        """Transfer relation to find target.
        
        Given (A, B) and C, find D such that A:B::C:D.
        
        Args:
            source_pair: (A, B) pair defining the relation
            target_source: C - source of target pair
            vocabulary: Vocabulary to search for D
            top_k: Number of results
            modality: Modality of concepts
            
        Returns:
            List of (D, confidence) tuples
        """
        a, b = source_pair
        
        # Extract relation vector
        relation = self._extractor.extract_relation(a, b, modality)
        
        # Encode C
        emb_c = self._encoder.encode(target_source, modality)
        
        # Compute expected D embedding
        emb_d = emb_c + relation.relation_vector
        
        # Normalize
        norm = np.linalg.norm(emb_d)
        if norm > 0:
            emb_d = emb_d / norm
        
        # Find closest in vocabulary
        vocab_embeddings = self._encoder.encode_batch(vocabulary, modality)
        
        similarities = vocab_embeddings @ emb_d
        
        results = []
        exclude = {a.lower(), b.lower(), target_source.lower()}
        
        for idx, sim in enumerate(similarities):
            word = vocabulary[idx]
            if word.lower() not in exclude:
                results.append((word, float(sim)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def generate_analogies(
        self,
        source_pair: Tuple[str, str],
        concept_pool: List[str],
        top_k: int = 5,
        modality: Modality = Modality.TEXT
    ) -> List[Tuple[str, str, float]]:
        """Generate new analogies with same relation.
        
        Args:
            source_pair: (A, B) defining the relation
            concept_pool: Pool of concepts to form pairs
            top_k: Number of analogies to generate
            modality: Modality of concepts
            
        Returns:
            List of (C, D, confidence) tuples
        """
        a, b = source_pair
        
        # Extract relation
        relation = self._extractor.extract_relation(a, b, modality)
        
        # For each concept, compute expected pair
        results = []
        
        for c in concept_pool:
            if c.lower() in {a.lower(), b.lower()}:
                continue
            
            # Compute expected D
            emb_c = self._encoder.encode(c, modality)
            emb_d = emb_c + relation.relation_vector
            
            # Find closest D in pool
            best_d = None
            best_sim = -1.0
            
            for d in concept_pool:
                if d.lower() in {a.lower(), b.lower(), c.lower()}:
                    continue
                
                emb_d_cand = self._encoder.encode(d, modality)
                norm1 = np.linalg.norm(emb_d)
                norm2 = np.linalg.norm(emb_d_cand)
                
                if norm1 > 0 and norm2 > 0:
                    sim = float(np.dot(emb_d, emb_d_cand) / (norm1 * norm2))
                    if sim > best_sim:
                        best_sim = sim
                        best_d = d
            
            if best_d is not None:
                results.append((c, best_d, best_sim))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


class CrossModalAnalogy:
    """Main cross-modal analogy interface.
    
    Unified interface for cross-modal analogy reasoning.
    
    Example:
        >>> cma = CrossModalAnalogy()
        >>> result = cma.solve("king", "man", "woman")
        >>> print(result.answer)  # queen (if in vocabulary)
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        vocabulary: Optional[List[str]] = None
    ):
        """Initialize cross-modal analogy system.
        
        Args:
            config: Embedding configuration
            vocabulary: Initial vocabulary
        """
        self.config = config or EmbeddingConfig(fallback_dim=256)
        self._encoder = CrossModalEncoder(self.config)
        self._relation_extractor = RelationExtractor(self._encoder)
        self._solver = AnalogySolver(self._encoder, vocabulary)
        self._transfer = RelationTransfer(self._encoder, self._relation_extractor)
        self._lock = threading.Lock()
    
    def set_vocabulary(self, vocabulary: List[str]):
        """Set vocabulary for answer lookup.
        
        Args:
            vocabulary: List of possible answers
        """
        self._solver.set_vocabulary(vocabulary)
    
    def add_vocabulary(self, items: List[str]):
        """Add items to vocabulary.
        
        Args:
            items: Items to add
        """
        self._solver.add_to_vocabulary(items)
    
    def solve(
        self,
        a: str,
        b: str,
        c: str,
        modality: Modality = Modality.TEXT,
        method: str = "vector_arithmetic"
    ) -> AnalogyResult:
        """Solve analogy A:B::C:?
        
        Args:
            a: First element of source pair
            b: Second element of source pair
            c: First element of target pair
            modality: Modality of elements
            method: Solving method
            
        Returns:
            AnalogyResult with answer
        """
        return self._solver.solve(a, b, c, modality, method)
    
    def solve_cross_modal(
        self,
        a: str,
        a_modality: Modality,
        b: str,
        b_modality: Modality,
        c: str,
        c_modality: Modality,
        target_modality: Modality = Modality.TEXT
    ) -> AnalogyResult:
        """Solve cross-modal analogy.
        
        Each element can be from a different modality.
        
        Args:
            a: First element of source pair
            a_modality: Modality of a
            b: Second element of source pair
            b_modality: Modality of b
            c: First element of target pair
            c_modality: Modality of c
            target_modality: Expected modality of answer
            
        Returns:
            AnalogyResult with answer
        """
        start_time = time.time()
        
        # Encode each with its modality
        emb_a = self._encoder.encode(a, a_modality)
        emb_b = self._encoder.encode(b, b_modality)
        emb_c = self._encoder.encode(c, c_modality)
        
        # Get projectors for cross-modal alignment
        proj_ab = self._encoder.get_projector(a_modality, b_modality)
        proj_cd = self._encoder.get_projector(c_modality, target_modality)
        
        # Project to shared space
        emb_a_proj = proj_ab.project_source(emb_a)
        emb_b_proj = proj_ab.project_target(emb_b)
        emb_c_proj = proj_cd.project_source(emb_c)
        
        # Compute relation and apply
        relation = emb_b_proj - emb_a_proj
        emb_d_proj = emb_c_proj + relation
        
        # Find answer (simplified - would need target modality vocabulary)
        # For now, return C with relation marker
        return AnalogyResult(
            query=(a, b, c),
            answer=f"[{c}+relation]",
            confidence=0.5,
            computation_time=time.time() - start_time,
            method="cross_modal",
            metadata={
                "modalities": [a_modality, b_modality, c_modality, target_modality]
            }
        )
    
    def extract_relation(
        self,
        source: str,
        target: str,
        modality: Modality = Modality.TEXT
    ) -> RelationResult:
        """Extract relation between concepts.
        
        Args:
            source: Source concept
            target: Target concept
            modality: Modality of concepts
            
        Returns:
            RelationResult with extracted relation
        """
        return self._relation_extractor.extract_relation(source, target, modality)
    
    def transfer_relation(
        self,
        source_pair: Tuple[str, str],
        target_source: str,
        vocabulary: List[str],
        top_k: int = 5,
        modality: Modality = Modality.TEXT
    ) -> List[Tuple[str, float]]:
        """Transfer relation to new context.
        
        Args:
            source_pair: (A, B) defining relation
            target_source: C for target pair
            vocabulary: Vocabulary to search
            top_k: Number of results
            modality: Modality of concepts
            
        Returns:
            List of (D, confidence) tuples
        """
        return self._transfer.transfer_relation(
            source_pair, target_source, vocabulary, top_k, modality
        )
    
    def generate_analogies(
        self,
        source_pair: Tuple[str, str],
        concept_pool: List[str],
        top_k: int = 5,
        modality: Modality = Modality.TEXT
    ) -> List[Tuple[str, str, float]]:
        """Generate new analogies with same relation.
        
        Args:
            source_pair: (A, B) defining relation
            concept_pool: Concepts to form pairs
            top_k: Number to generate
            modality: Modality of concepts
            
        Returns:
            List of (C, D, confidence) tuples
        """
        return self._transfer.generate_analogies(
            source_pair, concept_pool, top_k, modality
        )
    
    def evaluate(
        self,
        analogies: List[Tuple[str, str, str, str]],
        modality: Modality = Modality.TEXT
    ) -> Dict[str, float]:
        """Evaluate on a set of analogies.
        
        Args:
            analogies: List of (A, B, C, D) tuples
            modality: Modality of analogies
            
        Returns:
            Evaluation metrics
        """
        correct = 0
        total = len(analogies)
        mrr_sum = 0.0
        
        for a, b, c, expected_d in analogies:
            eval_result = self._solver.evaluate_analogy(
                a, b, c, expected_d, modality
            )
            if eval_result["correct"]:
                correct += 1
            mrr_sum += eval_result["mrr"]
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "mrr": mrr_sum / total if total > 0 else 0.0
        }
    
    def similarity(
        self,
        item1: str,
        item2: str,
        modality1: Modality = Modality.TEXT,
        modality2: Optional[Modality] = None
    ) -> float:
        """Compute similarity between items.
        
        Args:
            item1: First item
            item2: Second item
            modality1: Modality of first item
            modality2: Modality of second item (defaults to modality1)
            
        Returns:
            Similarity score
        """
        if modality2 is None:
            modality2 = modality1
        
        return self._encoder.cross_modal_similarity(
            item1, modality1, item2, modality2
        )
    
    def get_encoder(self) -> CrossModalEncoder:
        """Get the cross-modal encoder."""
        return self._encoder
    
    def get_relation_extractor(self) -> RelationExtractor:
        """Get the relation extractor."""
        return self._relation_extractor


# Convenience functions

def create_analogy_solver(
    vocabulary: Optional[List[str]] = None
) -> CrossModalAnalogy:
    """Create analogy solver with optional vocabulary.
    
    Args:
        vocabulary: Optional vocabulary for answers
        
    Returns:
        CrossModalAnalogy instance
    """
    return CrossModalAnalogy(vocabulary=vocabulary)


def solve_analogy(a: str, b: str, c: str) -> AnalogyResult:
    """Solve single analogy A:B::C:?
    
    Convenience function using default configuration.
    
    Args:
        a: First element of source pair
        b: Second element of source pair
        c: First element of target pair
        
    Returns:
        AnalogyResult with answer
    """
    solver = CrossModalAnalogy()
    return solver.solve(a, b, c)
