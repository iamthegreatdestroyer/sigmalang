"""
Entity and Relation Extraction Module for SigmaLang.

This module provides Named Entity Recognition (NER), relation extraction,
and knowledge graph construction capabilities for the SigmaLang encoding system.

Features:
- Rule-based and pattern-based NER
- Entity type classification (PERSON, ORG, LOCATION, DATE, etc.)
- Relation extraction between entities
- Knowledge graph construction and querying
- Entity linking and coreference resolution
- Graph export (NetworkX, JSON, Cypher)

Author: SigmaLang Team
"""

from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Set, Tuple, Any, Iterator, 
    Callable, Union, Pattern
)
from collections import defaultdict
import threading


class EntityType(Enum):
    """Types of named entities."""
    PERSON = auto()
    ORGANIZATION = auto()
    LOCATION = auto()
    DATE = auto()
    TIME = auto()
    MONEY = auto()
    PERCENT = auto()
    PRODUCT = auto()
    EVENT = auto()
    WORK_OF_ART = auto()
    LANGUAGE = auto()
    QUANTITY = auto()
    ORDINAL = auto()
    CARDINAL = auto()
    CONCEPT = auto()
    UNKNOWN = auto()
    
    @classmethod
    def from_string(cls, s: str) -> "EntityType":
        """Convert string to EntityType."""
        mapping = {
            "person": cls.PERSON,
            "per": cls.PERSON,
            "org": cls.ORGANIZATION,
            "organization": cls.ORGANIZATION,
            "loc": cls.LOCATION,
            "location": cls.LOCATION,
            "gpe": cls.LOCATION,
            "date": cls.DATE,
            "time": cls.TIME,
            "money": cls.MONEY,
            "percent": cls.PERCENT,
            "product": cls.PRODUCT,
            "event": cls.EVENT,
            "work_of_art": cls.WORK_OF_ART,
            "language": cls.LANGUAGE,
            "quantity": cls.QUANTITY,
            "ordinal": cls.ORDINAL,
            "cardinal": cls.CARDINAL,
            "concept": cls.CONCEPT,
        }
        return mapping.get(s.lower(), cls.UNKNOWN)


class RelationType(Enum):
    """Types of relations between entities."""
    WORKS_FOR = auto()
    LOCATED_IN = auto()
    BORN_IN = auto()
    FOUNDED_BY = auto()
    CEO_OF = auto()
    PART_OF = auto()
    OWNS = auto()
    CREATED_BY = auto()
    MARRIED_TO = auto()
    PARENT_OF = auto()
    SIBLING_OF = auto()
    FRIEND_OF = auto()
    MEMBER_OF = auto()
    AFFILIATED_WITH = auto()
    OCCURRED_AT = auto()
    OCCURRED_ON = auto()
    HAS_PROPERTY = auto()
    IS_A = auto()
    RELATED_TO = auto()
    UNKNOWN = auto()
    
    @classmethod
    def from_string(cls, s: str) -> "RelationType":
        """Convert string to RelationType."""
        mapping = {
            "works_for": cls.WORKS_FOR,
            "employed_by": cls.WORKS_FOR,
            "located_in": cls.LOCATED_IN,
            "based_in": cls.LOCATED_IN,
            "born_in": cls.BORN_IN,
            "founded_by": cls.FOUNDED_BY,
            "founded": cls.FOUNDED_BY,
            "ceo_of": cls.CEO_OF,
            "leads": cls.CEO_OF,
            "part_of": cls.PART_OF,
            "owns": cls.OWNS,
            "created_by": cls.CREATED_BY,
            "authored_by": cls.CREATED_BY,
            "married_to": cls.MARRIED_TO,
            "spouse_of": cls.MARRIED_TO,
            "parent_of": cls.PARENT_OF,
            "sibling_of": cls.SIBLING_OF,
            "friend_of": cls.FRIEND_OF,
            "member_of": cls.MEMBER_OF,
            "affiliated_with": cls.AFFILIATED_WITH,
            "occurred_at": cls.OCCURRED_AT,
            "occurred_on": cls.OCCURRED_ON,
            "has_property": cls.HAS_PROPERTY,
            "is_a": cls.IS_A,
            "related_to": cls.RELATED_TO,
        }
        return mapping.get(s.lower().replace(" ", "_"), cls.UNKNOWN)


@dataclass
class Entity:
    """Represents a named entity."""
    text: str
    entity_type: EntityType
    start: int = 0
    end: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.end == 0:
            self.end = self.start + len(self.text)
        if self._id is None:
            self._id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for entity."""
        content = f"{self.text}:{self.entity_type.name}:{self.start}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def id(self) -> str:
        """Get entity ID."""
        return self._id
    
    def __hash__(self):
        return hash((self.text, self.entity_type, self.start))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.text == other.text and 
                self.entity_type == other.entity_type and
                self.start == other.start)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "type": self.entity_type.name,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            entity_type=EntityType[data["type"]],
            start=data.get("start", 0),
            end=data.get("end", 0),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            _id=data.get("id")
        )


@dataclass
class Relation:
    """Represents a relation between two entities."""
    source: Entity
    target: Entity
    relation_type: RelationType
    confidence: float = 1.0
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self._id is None:
            self._id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for relation."""
        content = f"{self.source.id}:{self.relation_type.name}:{self.target.id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def id(self) -> str:
        """Get relation ID."""
        return self._id
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (self.source == other.source and 
                self.target == other.target and
                self.relation_type == other.relation_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "type": self.relation_type.name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            source=Entity.from_dict(data["source"]),
            target=Entity.from_dict(data["target"]),
            relation_type=RelationType[data["type"]],
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence", ""),
            metadata=data.get("metadata", {}),
            _id=data.get("id")
        )


@dataclass
class ExtractionResult:
    """Result of entity/relation extraction."""
    text: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[Relation]:
        """Get all relations of a specific type."""
        return [r for r in self.relations if r.relation_type == relation_type]
    
    def get_relations_for_entity(self, entity: Entity) -> List[Relation]:
        """Get all relations involving an entity."""
        return [r for r in self.relations 
                if r.source == entity or r.target == entity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "metadata": self.metadata
        }


class EntityPattern:
    """Pattern for matching entities."""
    
    def __init__(
        self,
        pattern: Union[str, Pattern],
        entity_type: EntityType,
        priority: int = 0,
        processor: Optional[Callable[[re.Match], str]] = None
    ):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.entity_type = entity_type
        self.priority = priority
        self.processor = processor or (lambda m: m.group(0))
    
    def find_matches(self, text: str) -> List[Entity]:
        """Find all matches in text."""
        entities = []
        for match in self.pattern.finditer(text):
            entity_text = self.processor(match)
            entities.append(Entity(
                text=entity_text,
                entity_type=self.entity_type,
                start=match.start(),
                end=match.end(),
                confidence=0.8
            ))
        return entities


class RelationPattern:
    """Pattern for extracting relations."""
    
    def __init__(
        self,
        pattern: Union[str, Pattern],
        relation_type: RelationType,
        source_group: int = 1,
        target_group: int = 2,
        source_type: Optional[EntityType] = None,
        target_type: Optional[EntityType] = None
    ):
        self.pattern = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        self.relation_type = relation_type
        self.source_group = source_group
        self.target_group = target_group
        self.source_type = source_type
        self.target_type = target_type
    
    def find_matches(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Relation]:
        """Find all relation matches in text."""
        relations = []
        entity_map = {e.text.lower(): e for e in entities}
        
        for match in self.pattern.finditer(text):
            try:
                source_text = match.group(self.source_group).strip()
                target_text = match.group(self.target_group).strip()
                
                # Try to link to existing entities
                source_entity = entity_map.get(source_text.lower())
                target_entity = entity_map.get(target_text.lower())
                
                # Create new entities if not found
                if source_entity is None:
                    source_entity = Entity(
                        text=source_text,
                        entity_type=self.source_type or EntityType.UNKNOWN,
                        start=match.start(self.source_group),
                        confidence=0.6
                    )
                
                if target_entity is None:
                    target_entity = Entity(
                        text=target_text,
                        entity_type=self.target_type or EntityType.UNKNOWN,
                        start=match.start(self.target_group),
                        confidence=0.6
                    )
                
                relations.append(Relation(
                    source=source_entity,
                    target=target_entity,
                    relation_type=self.relation_type,
                    confidence=0.7,
                    evidence=match.group(0)
                ))
            except (IndexError, AttributeError):
                continue
        
        return relations


class EntityRecognizer:
    """Rule-based Named Entity Recognizer."""
    
    # Common patterns for entity recognition
    DEFAULT_PATTERNS = [
        # Dates
        EntityPattern(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            EntityType.DATE
        ),
        EntityPattern(
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            EntityType.DATE
        ),
        EntityPattern(
            r'\b\d{4}-\d{2}-\d{2}\b',
            EntityType.DATE
        ),
        # Time
        EntityPattern(
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            EntityType.TIME
        ),
        # Money
        EntityPattern(
            r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|trillion)?',
            EntityType.MONEY,
            priority=1
        ),
        EntityPattern(
            r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP)\b',
            EntityType.MONEY
        ),
        # Percentages
        EntityPattern(
            r'\b\d+(?:\.\d+)?%\b',
            EntityType.PERCENT
        ),
        # Quantities
        EntityPattern(
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|oz|ml|L|GB|MB|TB)\b',
            EntityType.QUANTITY
        ),
        # Ordinals
        EntityPattern(
            r'\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth)\b',
            EntityType.ORDINAL
        ),
        # Cardinals (large numbers)
        EntityPattern(
            r'\b\d{1,3}(?:,\d{3})+\b',
            EntityType.CARDINAL
        ),
    ]
    
    # Common name patterns (simplified)
    NAME_PREFIXES = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lord', 'lady'}
    
    def __init__(self, patterns: Optional[List[EntityPattern]] = None):
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self._custom_entities: Dict[str, EntityType] = {}
        self._lock = threading.Lock()
    
    def add_pattern(self, pattern: EntityPattern):
        """Add a custom pattern."""
        with self._lock:
            self.patterns.append(pattern)
            self.patterns.sort(key=lambda p: -p.priority)
    
    def add_entity(self, text: str, entity_type: EntityType):
        """Add a known entity."""
        with self._lock:
            self._custom_entities[text.lower()] = entity_type
    
    def recognize(self, text: str) -> List[Entity]:
        """Recognize entities in text."""
        entities = []
        used_spans = set()
        
        # Check custom entities first
        for entity_text, entity_type in self._custom_entities.items():
            pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if not self._overlaps(span, used_spans):
                    entities.append(Entity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0
                    ))
                    used_spans.add(span)
        
        # Apply pattern matching
        for pattern in self.patterns:
            for entity in pattern.find_matches(text):
                span = (entity.start, entity.end)
                if not self._overlaps(span, used_spans):
                    entities.append(entity)
                    used_spans.add(span)
        
        # Detect potential person names (capitalized words)
        entities.extend(self._detect_names(text, used_spans))
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities
    
    def _overlaps(
        self, 
        span: Tuple[int, int], 
        used_spans: Set[Tuple[int, int]]
    ) -> bool:
        """Check if span overlaps with any used span."""
        for used_start, used_end in used_spans:
            if span[0] < used_end and span[1] > used_start:
                return True
        return False
    
    def _detect_names(
        self, 
        text: str, 
        used_spans: Set[Tuple[int, int]]
    ) -> List[Entity]:
        """Detect potential person names."""
        entities = []
        # Match capitalized word sequences (potential names)
        name_pattern = re.compile(
            r'\b(?:(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        )
        
        for match in name_pattern.finditer(text):
            span = (match.start(), match.end())
            if not self._overlaps(span, used_spans):
                # Check if it looks like a name (not all caps, not a sentence start)
                name_text = match.group(0)
                if self._looks_like_name(name_text, match.start(), text):
                    entities.append(Entity(
                        text=name_text,
                        entity_type=EntityType.PERSON,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.6
                    ))
        
        return entities
    
    def _looks_like_name(self, text: str, start: int, full_text: str) -> bool:
        """Heuristic check if text looks like a name."""
        words = text.split()
        
        # Check for name prefixes
        if words[0].lower().rstrip('.') in self.NAME_PREFIXES:
            return True
        
        # At least 2 capitalized words
        if len(words) >= 2:
            # Not at sentence start (unless with prefix)
            if start > 0 and full_text[start-1] not in '.!?\n':
                return True
            elif start == 0 or full_text[start-1] in '.!?\n':
                # Could be sentence start, lower confidence
                return len(words) >= 2 and all(w[0].isupper() for w in words)
        
        return False


class RelationExtractor:
    """Extract relations between entities."""
    
    DEFAULT_PATTERNS = [
        # Person works for organization
        RelationPattern(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works?\s+(?:for|at)|is\s+employed\s+by|joined)\s+([A-Z][a-zA-Z\s&]+)',
            RelationType.WORKS_FOR,
            source_type=EntityType.PERSON,
            target_type=EntityType.ORGANIZATION
        ),
        # Organization located in location
        RelationPattern(
            r'([A-Z][a-zA-Z\s&]+)\s+(?:is\s+)?(?:located|based|headquartered)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            RelationType.LOCATED_IN,
            source_type=EntityType.ORGANIZATION,
            target_type=EntityType.LOCATION
        ),
        # Person born in location
        RelationPattern(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+born\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            RelationType.BORN_IN,
            source_type=EntityType.PERSON,
            target_type=EntityType.LOCATION
        ),
        # Organization founded by person
        RelationPattern(
            r'([A-Z][a-zA-Z\s&]+)\s+was\s+founded\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            RelationType.FOUNDED_BY,
            source_type=EntityType.ORGANIZATION,
            target_type=EntityType.PERSON
        ),
        # Person is CEO of organization
        RelationPattern(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?:CEO|chief\s+executive|president|founder)\s+of\s+([A-Z][a-zA-Z\s&]+)',
            RelationType.CEO_OF,
            source_type=EntityType.PERSON,
            target_type=EntityType.ORGANIZATION
        ),
        # Entity is part of entity
        RelationPattern(
            r'([A-Z][a-zA-Z\s&]+)\s+is\s+(?:a\s+)?part\s+of\s+([A-Z][a-zA-Z\s&]+)',
            RelationType.PART_OF
        ),
        # Person married to person
        RelationPattern(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?married\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            RelationType.MARRIED_TO,
            source_type=EntityType.PERSON,
            target_type=EntityType.PERSON
        ),
    ]
    
    def __init__(self, patterns: Optional[List[RelationPattern]] = None):
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self._lock = threading.Lock()
    
    def add_pattern(self, pattern: RelationPattern):
        """Add a custom pattern."""
        with self._lock:
            self.patterns.append(pattern)
    
    def extract(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations from text."""
        relations = []
        seen = set()
        
        for pattern in self.patterns:
            for relation in pattern.find_matches(text, entities):
                # Deduplicate
                key = (relation.source.text.lower(), 
                       relation.relation_type, 
                       relation.target.text.lower())
                if key not in seen:
                    relations.append(relation)
                    seen.add(key)
        
        return relations
    
    def extract_from_entities(
        self, 
        entities: List[Entity],
        co_occurrence_window: int = 50
    ) -> List[Relation]:
        """Extract relations based on entity co-occurrence."""
        relations = []
        
        # Group entities by proximity
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                distance = abs(e2.start - e1.end)
                if distance <= co_occurrence_window:
                    # Infer relation type from entity types
                    rel_type = self._infer_relation_type(e1, e2)
                    if rel_type != RelationType.UNKNOWN:
                        relations.append(Relation(
                            source=e1,
                            target=e2,
                            relation_type=rel_type,
                            confidence=0.5,
                            evidence=f"Co-occurrence within {distance} chars"
                        ))
        
        return relations
    
    def _infer_relation_type(self, e1: Entity, e2: Entity) -> RelationType:
        """Infer relation type from entity types."""
        type_pair = (e1.entity_type, e2.entity_type)
        
        inference_map = {
            (EntityType.PERSON, EntityType.ORGANIZATION): RelationType.AFFILIATED_WITH,
            (EntityType.ORGANIZATION, EntityType.LOCATION): RelationType.LOCATED_IN,
            (EntityType.EVENT, EntityType.DATE): RelationType.OCCURRED_ON,
            (EntityType.EVENT, EntityType.LOCATION): RelationType.OCCURRED_AT,
            (EntityType.PERSON, EntityType.LOCATION): RelationType.LOCATED_IN,
            (EntityType.PRODUCT, EntityType.ORGANIZATION): RelationType.CREATED_BY,
        }
        
        return inference_map.get(type_pair, RelationType.UNKNOWN)


class KnowledgeGraph:
    """Graph-based knowledge representation."""
    
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def add_entity(self, entity: Entity) -> str:
        """Add entity to graph."""
        with self._lock:
            self._entities[entity.id] = entity
            self._type_index[entity.entity_type].add(entity.id)
            return entity.id
    
    def add_relation(self, relation: Relation) -> str:
        """Add relation to graph."""
        with self._lock:
            # Ensure entities exist
            if relation.source.id not in self._entities:
                self.add_entity(relation.source)
            if relation.target.id not in self._entities:
                self.add_entity(relation.target)
            
            self._relations[relation.id] = relation
            self._adjacency[relation.source.id].add(relation.id)
            self._adjacency[relation.target.id].add(relation.id)
            return relation.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self._relations.get(relation_id)
    
    def get_entities(
        self, 
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """Get all entities, optionally filtered by type."""
        with self._lock:
            if entity_type is None:
                return list(self._entities.values())
            return [
                self._entities[eid] 
                for eid in self._type_index.get(entity_type, set())
            ]
    
    def get_relations(
        self,
        relation_type: Optional[RelationType] = None
    ) -> List[Relation]:
        """Get all relations, optionally filtered by type."""
        with self._lock:
            if relation_type is None:
                return list(self._relations.values())
            return [
                r for r in self._relations.values() 
                if r.relation_type == relation_type
            ]
    
    def get_neighbors(
        self, 
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[Relation, Entity]]:
        """Get neighboring entities connected by relations."""
        with self._lock:
            neighbors = []
            for rel_id in self._adjacency.get(entity_id, set()):
                relation = self._relations.get(rel_id)
                if relation is None:
                    continue
                if relation_type and relation.relation_type != relation_type:
                    continue
                
                # Determine the neighbor
                if relation.source.id == entity_id:
                    neighbors.append((relation, relation.target))
                else:
                    neighbors.append((relation, relation.source))
            
            return neighbors
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[Tuple[Entity, Optional[Relation]]]]:
        """Find shortest path between two entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None
        
        if source_id == target_id:
            return [(self._entities[source_id], None)]
        
        # BFS
        from collections import deque
        visited = {source_id}
        queue = deque([(source_id, [(self._entities[source_id], None)])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for relation, neighbor in self.get_neighbors(current_id):
                if neighbor.id == target_id:
                    return path + [(neighbor, relation)]
                
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [(neighbor, relation)]))
        
        return None
    
    def query(
        self,
        entity_type: Optional[EntityType] = None,
        relation_type: Optional[RelationType] = None,
        text_contains: Optional[str] = None
    ) -> List[Union[Entity, Relation]]:
        """Query the knowledge graph."""
        results = []
        
        with self._lock:
            # Query entities
            if relation_type is None:
                for entity in self._entities.values():
                    if entity_type and entity.entity_type != entity_type:
                        continue
                    if text_contains and text_contains.lower() not in entity.text.lower():
                        continue
                    results.append(entity)
            
            # Query relations
            if entity_type is None:
                for relation in self._relations.values():
                    if relation_type and relation.relation_type != relation_type:
                        continue
                    if text_contains:
                        if (text_contains.lower() not in relation.source.text.lower() and
                            text_contains.lower() not in relation.target.text.lower()):
                            continue
                    results.append(relation)
        
        return results
    
    @property
    def num_entities(self) -> int:
        """Number of entities in graph."""
        return len(self._entities)
    
    @property
    def num_relations(self) -> int:
        """Number of relations in graph."""
        return len(self._relations)
    
    def merge(self, other: "KnowledgeGraph"):
        """Merge another knowledge graph into this one."""
        with self._lock:
            for entity in other._entities.values():
                self.add_entity(entity)
            for relation in other._relations.values():
                self.add_relation(relation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        with self._lock:
            return {
                "entities": [e.to_dict() for e in self._entities.values()],
                "relations": [r.to_dict() for r in self._relations.values()]
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls()
        for entity_data in data.get("entities", []):
            graph.add_entity(Entity.from_dict(entity_data))
        for relation_data in data.get("relations", []):
            graph.add_relation(Relation.from_dict(relation_data))
        return graph
    
    def to_json(self, indent: int = 2) -> str:
        """Export graph as JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_cypher(self) -> str:
        """Export graph as Cypher statements (Neo4j)."""
        statements = []
        
        # Create entities
        for entity in self._entities.values():
            props = {
                "id": entity.id,
                "text": entity.text,
                "confidence": entity.confidence
            }
            props_str = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' 
                                  for k, v in props.items())
            statements.append(
                f'CREATE (:{entity.entity_type.name} {{{props_str}}})'
            )
        
        # Create relations
        for relation in self._relations.values():
            statements.append(
                f'MATCH (a {{id: "{relation.source.id}"}}), (b {{id: "{relation.target.id}"}}) '
                f'CREATE (a)-[:{relation.relation_type.name} {{confidence: {relation.confidence}}}]->(b)'
            )
        
        return ";\n".join(statements) + ";"
    
    def to_networkx(self):
        """Export to NetworkX graph (if available)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required for graph export")
        
        G = nx.DiGraph()
        
        for entity in self._entities.values():
            G.add_node(
                entity.id,
                text=entity.text,
                type=entity.entity_type.name,
                confidence=entity.confidence
            )
        
        for relation in self._relations.values():
            G.add_edge(
                relation.source.id,
                relation.target.id,
                type=relation.relation_type.name,
                confidence=relation.confidence,
                evidence=relation.evidence
            )
        
        return G


class EntityRelationExtractor:
    """
    Main class for entity and relation extraction.
    
    Combines entity recognition and relation extraction with
    optional knowledge graph construction.
    """
    
    def __init__(
        self,
        entity_recognizer: Optional[EntityRecognizer] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        build_graph: bool = True
    ):
        self.entity_recognizer = entity_recognizer or EntityRecognizer()
        self.relation_extractor = relation_extractor or RelationExtractor()
        self.build_graph = build_graph
        self._graph = KnowledgeGraph() if build_graph else None
        self._lock = threading.Lock()
    
    def extract(
        self,
        text: str,
        extract_relations: bool = True,
        use_cooccurrence: bool = False,
        cooccurrence_window: int = 50
    ) -> ExtractionResult:
        """
        Extract entities and relations from text.
        
        Args:
            text: Input text
            extract_relations: Whether to extract relations
            use_cooccurrence: Use co-occurrence for relation inference
            cooccurrence_window: Window size for co-occurrence
            
        Returns:
            ExtractionResult with entities and relations
        """
        # Recognize entities
        entities = self.entity_recognizer.recognize(text)
        
        # Extract relations
        relations = []
        if extract_relations and entities:
            relations = self.relation_extractor.extract(text, entities)
            
            if use_cooccurrence:
                cooc_relations = self.relation_extractor.extract_from_entities(
                    entities, cooccurrence_window
                )
                # Add non-duplicate co-occurrence relations
                existing = {(r.source.id, r.relation_type, r.target.id) 
                           for r in relations}
                for rel in cooc_relations:
                    key = (rel.source.id, rel.relation_type, rel.target.id)
                    if key not in existing:
                        relations.append(rel)
        
        result = ExtractionResult(
            text=text,
            entities=entities,
            relations=relations
        )
        
        # Update knowledge graph
        if self._graph:
            with self._lock:
                for entity in entities:
                    self._graph.add_entity(entity)
                for relation in relations:
                    self._graph.add_relation(relation)
        
        return result
    
    def extract_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[ExtractionResult]:
        """Extract from multiple texts."""
        return [self.extract(text, **kwargs) for text in texts]
    
    @property
    def graph(self) -> Optional[KnowledgeGraph]:
        """Get the knowledge graph."""
        return self._graph
    
    def reset_graph(self):
        """Reset the knowledge graph."""
        if self._graph:
            with self._lock:
                self._graph = KnowledgeGraph()
    
    def add_custom_entity(self, text: str, entity_type: EntityType):
        """Add a custom entity for recognition."""
        self.entity_recognizer.add_entity(text, entity_type)
    
    def add_entity_pattern(self, pattern: EntityPattern):
        """Add custom entity pattern."""
        self.entity_recognizer.add_pattern(pattern)
    
    def add_relation_pattern(self, pattern: RelationPattern):
        """Add custom relation pattern."""
        self.relation_extractor.add_pattern(pattern)


# Convenience functions
def create_extractor(build_graph: bool = True) -> EntityRelationExtractor:
    """Create a default entity/relation extractor."""
    return EntityRelationExtractor(build_graph=build_graph)


def extract_entities(text: str) -> List[Entity]:
    """Quick entity extraction."""
    recognizer = EntityRecognizer()
    return recognizer.recognize(text)


def extract_relations(text: str) -> List[Relation]:
    """Quick relation extraction."""
    extractor = EntityRelationExtractor(build_graph=False)
    result = extractor.extract(text)
    return result.relations


def build_knowledge_graph(texts: List[str]) -> KnowledgeGraph:
    """Build knowledge graph from multiple texts."""
    extractor = EntityRelationExtractor(build_graph=True)
    for text in texts:
        extractor.extract(text)
    return extractor.graph


__all__ = [
    "EntityType",
    "RelationType", 
    "Entity",
    "Relation",
    "ExtractionResult",
    "EntityPattern",
    "RelationPattern",
    "EntityRecognizer",
    "RelationExtractor",
    "KnowledgeGraph",
    "EntityRelationExtractor",
    "create_extractor",
    "extract_entities",
    "extract_relations",
    "build_knowledge_graph",
]
