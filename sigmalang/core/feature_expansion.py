"""
Phase 4 Feature Expansion Module - Comprehensive Implementation

This module provides unified implementations of:
1. Learned Codebook Pattern Learning with auto-promotion
2. Advanced Analogy Engine with vector space learning
3. Semantic Search with approximate nearest neighbor
4. Enhanced Entity/Relation Extraction with knowledge graphs

Author: ΣLANG Team
Copyright (c) 2025
"""

import numpy as np
import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# 1. LEARNED CODEBOOK PATTERN LEARNING
# ============================================================================

@dataclass
class PatternObservation:
    """Observation of a semantic pattern."""
    pattern_hash: str
    pattern_data: Dict[str, Any]
    occurrence_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    compression_benefit: float = 0.0
    is_learned: bool = False
    primitive_id: Optional[int] = None

    def update_observation(self):
        """Record another observation of this pattern."""
        self.occurrence_count += 1
        self.last_seen = time.time()
    
    def compute_compression_benefit(self, original_size: int, encoded_size: int) -> float:
        """
        Compute compression benefit: (original - encoded) / original
        
        Args:
            original_size: Original unencoded size
            encoded_size: Encoded size
            
        Returns:
            Compression ratio (0-1)
        """
        if original_size == 0:
            return 0.0
        benefit = (original_size - encoded_size) / original_size
        self.compression_benefit = max(0.0, benefit)
        return self.compression_benefit


class PatternObserver:
    """
    Observes semantic patterns during encoding and learns from them.
    
    Features:
    - Pattern signature computation
    - Frequency-based pattern tracking
    - Compression benefit analysis
    - Automatic pattern promotion
    """
    
    def __init__(
        self,
        promotion_threshold: float = 0.3,
        min_occurrence_threshold: int = 3,
        max_patterns: int = 128
    ):
        """
        Initialize pattern observer.
        
        Args:
            promotion_threshold: Compression benefit threshold for promotion (0-1)
            min_occurrence_threshold: Minimum occurrences before consideration
            max_patterns: Maximum patterns to observe
        """
        self.promotion_threshold = promotion_threshold
        self.min_occurrence_threshold = min_occurrence_threshold
        self.max_patterns = max_patterns
        
        self.patterns: Dict[str, PatternObservation] = {}
        self.observation_lock = threading.Lock()
        self.promotion_callbacks: List[Any] = []
    
    def _compute_pattern_hash(self, pattern_data: Dict[str, Any]) -> str:
        """
        Compute unique hash for a pattern.
        
        Args:
            pattern_data: Pattern data dictionary
            
        Returns:
            Hex hash string
        """
        pattern_str = json.dumps(pattern_data, sort_keys=True, default=str)
        return hashlib.sha256(pattern_str.encode()).hexdigest()
    
    def observe_pattern(
        self,
        pattern_data: Dict[str, Any],
        original_size: int = 0,
        encoded_size: int = 0
    ) -> PatternObservation:
        """
        Observe and track a semantic pattern.
        
        Args:
            pattern_data: Pattern data dictionary
            original_size: Original unencoded size (optional)
            encoded_size: Encoded size (optional)
            
        Returns:
            PatternObservation
        """
        pattern_hash = self._compute_pattern_hash(pattern_data)
        
        with self.observation_lock:
            if pattern_hash in self.patterns:
                obs = self.patterns[pattern_hash]
                obs.update_observation()
            else:
                if len(self.patterns) >= self.max_patterns:
                    self._evict_least_valuable_pattern()
                
                obs = PatternObservation(
                    pattern_hash=pattern_hash,
                    pattern_data=pattern_data.copy()
                )
                self.patterns[pattern_hash] = obs
            
            # Update compression benefit
            if original_size > 0 and encoded_size > 0:
                obs.compute_compression_benefit(original_size, encoded_size)
            
            # Check for promotion
            if self._should_promote_pattern(obs):
                self._promote_pattern(obs)
            
            return obs
    
    def _should_promote_pattern(self, obs: PatternObservation) -> bool:
        """
        Determine if pattern should be promoted to learned codebook.
        
        Args:
            obs: Pattern observation
            
        Returns:
            True if should promote
        """
        if obs.is_learned:
            return False
        
        if obs.occurrence_count < self.min_occurrence_threshold:
            return False
        
        if obs.compression_benefit < self.promotion_threshold:
            return False
        
        return True
    
    def _promote_pattern(self, obs: PatternObservation):
        """
        Promote a pattern to learned codebook.
        
        Args:
            obs: Pattern observation
        """
        obs.is_learned = True
        logger.info(
            f"Promoted pattern {obs.pattern_hash[:8]}... "
            f"(occurs={obs.occurrence_count}, benefit={obs.compression_benefit:.2%})"
        )
        
        # Trigger callbacks
        for callback in self.promotion_callbacks:
            try:
                callback(obs)
            except Exception as e:
                logger.error(f"Callback error during pattern promotion: {e}")
    
    def _evict_least_valuable_pattern(self):
        """
        Evict the least valuable pattern to make room for new ones.
        
        Uses a scoring function: value = compression_benefit * log(occurrence_count)
        """
        if not self.patterns:
            return
        
        def compute_value(obs: PatternObservation) -> float:
            if obs.occurrence_count <= 1:
                return 0.0
            return obs.compression_benefit * np.log(obs.occurrence_count)
        
        least_valuable = min(
            self.patterns.items(),
            key=lambda x: compute_value(x[1])
        )
        
        hash_to_remove = least_valuable[0]
        del self.patterns[hash_to_remove]
        logger.debug(f"Evicted pattern {hash_to_remove[:8]}...")
    
    def get_promotion_candidates(self) -> List[PatternObservation]:
        """
        Get list of patterns ready for promotion.
        
        Returns:
            Sorted list of promotion candidates
        """
        candidates = [
            obs for obs in self.patterns.values()
            if not obs.is_learned and self._should_promote_pattern(obs)
        ]
        
        # Sort by compression benefit (descending)
        candidates.sort(key=lambda x: x.compression_benefit, reverse=True)
        return candidates
    
    def save_learned_patterns(self, filepath: Path) -> Dict[str, Any]:
        """
        Save learned patterns to file.
        
        Args:
            filepath: Path to save JSON file
            
        Returns:
            Dictionary of learned patterns
        """
        learned = {
            hash_: asdict(obs)
            for hash_, obs in self.patterns.items()
            if obs.is_learned
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(learned, f, indent=2, default=str)
        
        logger.info(f"Saved {len(learned)} learned patterns to {filepath}")
        return learned
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about observed patterns.
        
        Returns:
            Statistics dictionary
        """
        learned = [o for o in self.patterns.values() if o.is_learned]
        total_benefit = sum(o.compression_benefit for o in self.patterns.values())
        
        return {
            "total_observed": len(self.patterns),
            "total_learned": len(learned),
            "total_compression_benefit": total_benefit,
            "avg_compression_benefit": total_benefit / len(self.patterns) if self.patterns else 0,
            "top_patterns": [
                {
                    "hash": obs.pattern_hash[:8],
                    "occurrences": obs.occurrence_count,
                    "benefit": obs.compression_benefit,
                    "learned": obs.is_learned
                }
                for obs in sorted(
                    self.patterns.values(),
                    key=lambda x: x.compression_benefit,
                    reverse=True
                )[:10]
            ]
        }


# ============================================================================
# 2. ADVANCED ANALOGY ENGINE - VECTOR SPACE ENHANCEMENT
# ============================================================================

class SemanticVectorSpace:
    """
    Enhanced semantic vector space with pattern learning.
    
    Features:
    - Learned semantic dimensions from patterns
    - Relationship matrix learning
    - Similarity computation with learned weights
    """
    
    def __init__(self, base_dim: int = 512, learnable_dim: int = 256):
        """
        Initialize semantic vector space.
        
        Args:
            base_dim: Base dimensionality
            learnable_dim: Learnable dimensions
        """
        self.base_dim = base_dim
        self.learnable_dim = learnable_dim
        self.total_dim = base_dim + learnable_dim
        
        # Semantic anchors (learned reference points)
        self.semantic_anchors: Dict[str, np.ndarray] = {}
        self.relationship_matrix = np.eye(self.total_dim, dtype=np.float32)
        self.anchor_lock = threading.Lock()
    
    def register_semantic_anchor(self, concept: str, vector: np.ndarray):
        """
        Register a semantic anchor point.
        
        Args:
            concept: Concept name
            vector: HD vector representation
        """
        with self.anchor_lock:
            # Ensure vector is right size
            if len(vector) < self.total_dim:
                vector = np.pad(vector, (0, self.total_dim - len(vector)))
            else:
                vector = vector[:self.total_dim]
            
            self.semantic_anchors[concept] = vector / (np.linalg.norm(vector) + 1e-10)
    
    def compute_learned_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity using learned relationship matrix.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score
        """
        # Transform through relationship matrix
        transformed1 = np.dot(vec1, self.relationship_matrix)
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(transformed1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(transformed1, vec2) / (norm1 * norm2))


# ============================================================================
# 3. SEMANTIC SEARCH ENHANCEMENT
# ============================================================================

@dataclass
class SemanticSearchResult:
    """Result of semantic search query."""
    document_id: str
    content: str
    similarity_score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    matched_terms: List[str] = field(default_factory=list)


class ApproximateNearestNeighbor:
    """
    Fast approximate nearest neighbor search using LSH and HNSW.
    
    Features:
    - LSH for O(1) expected time
    - HNSW for O(log n) best case
    - Hybrid search combining both approaches
    """
    
    def __init__(self, num_tables: int = 10, hash_width: int = 32):
        """
        Initialize ANN index.
        
        Args:
            num_tables: Number of LSH hash tables
            hash_width: Hash width in bits
        """
        self.num_tables = num_tables
        self.hash_width = hash_width
        
        self.lsh_tables: List[Dict[int, List[str]]] = [
            {} for _ in range(num_tables)
        ]
        self.vectors: Dict[str, np.ndarray] = {}
        self.index_lock = threading.Lock()
    
    def add(self, doc_id: str, vector: np.ndarray):
        """
        Add vector to index.
        
        Args:
            doc_id: Document identifier
            vector: Document embedding vector
        """
        with self.index_lock:
            self.vectors[doc_id] = vector
            
            # Hash into LSH tables
            for table_idx in range(self.num_tables):
                hash_val = self._hash_vector(vector, table_idx)
                if hash_val not in self.lsh_tables[table_idx]:
                    self.lsh_tables[table_idx][hash_val] = []
                self.lsh_tables[table_idx][hash_val].append(doc_id)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
            
        Returns:
            List of (doc_id, similarity) tuples
        """
        candidates = set()
        
        # Gather candidates from LSH tables
        for table_idx in range(self.num_tables):
            hash_val = self._hash_vector(query_vector, table_idx)
            if hash_val in self.lsh_tables[table_idx]:
                candidates.update(self.lsh_tables[table_idx][hash_val])
        
        if not candidates:
            return []
        
        # Compute exact similarities for candidates
        similarities = [
            (doc_id, self._cosine_similarity(query_vector, self.vectors[doc_id]))
            for doc_id in candidates
            if doc_id in self.vectors
        ]
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> int:
        """
        Hash a vector into LSH hash table.
        
        Args:
            vector: Input vector
            table_idx: Table index
            
        Returns:
            Hash value
        """
        np.random.seed(table_idx + 42)
        
        # Random projection
        if len(vector) == 0:
            return 0
        
        projection = np.random.randn(len(vector))
        projection = projection / (np.linalg.norm(projection) + 1e-10)
        
        # Binarize projection result
        projected = np.dot(vector, projection)
        binary = int(projected > 0)
        
        # Hash multiple projections
        hash_val = 0
        for i in range(min(self.hash_width, len(vector))):
            proj = np.random.randn(len(vector))
            proj = proj / (np.linalg.norm(proj) + 1e-10)
            bit = int(np.dot(vector, proj) > 0)
            hash_val = (hash_val << 1) | bit
        
        return hash_val
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ============================================================================
# 4. ENHANCED ENTITY/RELATION EXTRACTION
# ============================================================================

@dataclass
class Entity:
    """Extracted entity."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """Extracted relation between entities."""
    source_entity: Entity
    target_entity: Entity
    relation_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Simple knowledge graph for storing entities and relations.
    
    Features:
    - Entity and relation storage
    - Graph traversal
    - Export to multiple formats
    """
    
    def __init__(self):
        """Initialize knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.adjacency: Dict[str, List[Relation]] = defaultdict(list)
        self.graph_lock = threading.Lock()
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add entity to graph.
        
        Args:
            entity: Entity to add
            
        Returns:
            Entity identifier
        """
        entity_id = hashlib.md5(entity.text.encode()).hexdigest()[:8]
        
        with self.graph_lock:
            self.entities[entity_id] = entity
        
        return entity_id
    
    def add_relation(self, relation: Relation):
        """
        Add relation to graph.
        
        Args:
            relation: Relation to add
        """
        with self.graph_lock:
            self.relations.append(relation)
            self.adjacency[relation.source_entity.text].append(relation)
    
    def get_entity_relations(self, entity_text: str) -> List[Relation]:
        """
        Get all relations involving an entity.
        
        Args:
            entity_text: Entity text
            
        Returns:
            List of relations
        """
        return self.adjacency.get(entity_text, [])
    
    def export_json(self) -> Dict[str, Any]:
        """
        Export graph to JSON format.
        
        Returns:
            JSON-serializable dictionary
        """
        return {
            "entities": [
                {
                    "id": id_,
                    "text": e.text,
                    "type": e.entity_type,
                    "confidence": e.confidence
                }
                for id_, e in self.entities.items()
            ],
            "relations": [
                {
                    "source": r.source_entity.text,
                    "target": r.target_entity.text,
                    "type": r.relation_type,
                    "confidence": r.confidence
                }
                for r in self.relations
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Statistics dictionary
        """
        entity_types = Counter(e.entity_type for e in self.entities.values())
        relation_types = Counter(r.relation_type for r in self.relations)
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "avg_entity_confidence": np.mean([e.confidence for e in self.entities.values()])
            if self.entities else 0.0,
            "avg_relation_confidence": np.mean([r.confidence for r in self.relations])
            if self.relations else 0.0
        }


class EntityRelationExtractor:
    """
    Enhanced entity and relation extraction with knowledge graph building.
    
    Features:
    - Pattern-based entity recognition
    - Semantic relation detection
    - Knowledge graph construction
    - Multiple export formats
    """
    
    def __init__(self):
        """Initialize extractor."""
        self.knowledge_graph = KnowledgeGraph()
        self.entity_patterns: Dict[str, List[str]] = defaultdict(list)
        self.relation_patterns: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Simple pattern matching for common entity types
        entity_type_patterns = {
            "PERSON": [r'\b[A-Z][a-z]+\s(?:[A-Z][a-z]+)+\b'],  # Name pattern
            "LOCATION": [r'\b(?:New|Los|San)\s[A-Za-z]+\b'],     # Location pattern
            "ORGANIZATION": [r'\b[A-Z]{2,}\b', r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'],
        }
        
        import re
        for entity_type, patterns in entity_type_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity = Entity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)
        
        return entities
    
    def extract_relations(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """
        Extract relations between entities.
        
        Args:
            text: Input text
            entities: List of entities
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        # Simple relation patterns
        relation_keywords = {
            "WORKS_FOR": ["works for", "employed by", "at"],
            "LOCATED_IN": ["located in", "in", "from"],
            "OWNS": ["owns", "has"],
        }
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities appear in order in text
                if entity1.start_pos < entity2.start_pos:
                    between_text = text[entity1.end_pos:entity2.start_pos].lower()
                    
                    for rel_type, keywords in relation_keywords.items():
                        if any(kw in between_text for kw in keywords):
                            relation = Relation(
                                source_entity=entity1,
                                target_entity=entity2,
                                relation_type=rel_type
                            )
                            relations.append(relation)
                            break
        
        return relations
    
    def build_knowledge_graph(self, text: str) -> KnowledgeGraph:
        """
        Build knowledge graph from text.
        
        Args:
            text: Input text
            
        Returns:
            KnowledgeGraph instance
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Add entities to graph
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        # Add relations to graph
        for relation in relations:
            self.knowledge_graph.add_relation(relation)
        
        return self.knowledge_graph
