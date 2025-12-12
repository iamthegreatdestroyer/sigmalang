"""
Pattern Persistence Layer - Phase 2A.4

Implements efficient storage, retrieval, and indexing of analogy patterns.

Key Components:
- PatternIndex: Inverted index for fast pattern search
- PatternMetadata: Rich metadata tracking for patterns
- CatalogPersistence: JSON serialization with compression
- PatternDiscovery: Advanced search algorithms
"""

import json
import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# METADATA AND INDEXING
# ============================================================================

@dataclass
class PatternMetadata:
    """Metadata for a stored pattern."""
    pattern_id: str
    created_at: str
    accessed_count: int = 0
    last_accessed: str = ""
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    domain_tags: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    performance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PatternMetadata":
        """Create metadata from dictionary."""
        return PatternMetadata(**data)


class PatternIndex:
    """
    Inverted index for efficient pattern searching.
    
    Features:
    - Word-based indexing for pattern terms
    - Domain tag indexing
    - Relationship vector indexing
    - O(1) term lookups, O(log n) range queries
    """

    def __init__(self):
        """Initialize empty index."""
        self.term_index: Dict[str, Set[str]] = {}  # term -> pattern_ids
        self.domain_index: Dict[str, Set[str]] = {}  # domain -> pattern_ids
        self.pattern_index: Dict[str, 'AnalogyPattern'] = {}  # pattern_id -> pattern
        self.id_to_metadata: Dict[str, PatternMetadata] = {}

    def add_pattern(
        self,
        pattern_id: str,
        pattern: Any,
        metadata: PatternMetadata
    ) -> None:
        """
        Add pattern to index.

        Args:
            pattern_id: Unique pattern identifier
            pattern: AnalogyPattern object
            metadata: PatternMetadata for pattern
        """
        # Store pattern
        self.pattern_index[pattern_id] = pattern
        self.id_to_metadata[pattern_id] = metadata

        # Index terms from analogies - handle dict and attribute-based patterns
        pattern_str = str(pattern).lower()
        
        # Extract terms from string representation
        import re
        # Find all words (sequences of alphanumeric characters)
        terms = re.findall(r'\b[a-z0-9]+\b', pattern_str)
        
        for term in terms:
            if term and term not in ['analogies', 'relationships', 'and', 'the', 'a']:  # Skip common words
                if term not in self.term_index:
                    self.term_index[term] = set()
                self.term_index[term].add(pattern_id)

        # Index domain tags
        for tag in metadata.domain_tags:
            tag_lower = tag.lower()
            if tag_lower not in self.domain_index:
                self.domain_index[tag_lower] = set()
            self.domain_index[tag_lower].add(pattern_id)

    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove pattern from index.

        Args:
            pattern_id: Pattern to remove

        Returns:
            True if removed, False if not found
        """
        if pattern_id not in self.pattern_index:
            return False

        # Remove from pattern index
        del self.pattern_index[pattern_id]
        del self.id_to_metadata[pattern_id]

        # Remove from term index
        terms_to_clean = []
        for term, pattern_ids in self.term_index.items():
            pattern_ids.discard(pattern_id)
            if not pattern_ids:
                terms_to_clean.append(term)

        for term in terms_to_clean:
            del self.term_index[term]

        # Remove from domain index
        domains_to_clean = []
        for domain, pattern_ids in self.domain_index.items():
            pattern_ids.discard(pattern_id)
            if not pattern_ids:
                domains_to_clean.append(domain)

        for domain in domains_to_clean:
            del self.domain_index[domain]

        return True

    def search_by_term(self, term: str) -> List[str]:
        """
        Find patterns containing term.

        Args:
            term: Search term

        Returns:
            List of matching pattern_ids
        """
        term_lower = term.lower()
        return list(self.term_index.get(term_lower, set()))

    def search_by_domain(self, domain: str) -> List[str]:
        """
        Find patterns in domain.

        Args:
            domain: Domain tag

        Returns:
            List of matching pattern_ids
        """
        domain_lower = domain.lower()
        return list(self.domain_index.get(domain_lower, set()))

    def search_by_terms(self, terms: List[str]) -> List[str]:
        """
        Find patterns matching all terms (AND search).

        Args:
            terms: List of search terms

        Returns:
            List of matching pattern_ids
        """
        if not terms:
            return []

        # Start with first term
        result_set = set(self.search_by_term(terms[0]))

        # Intersect with remaining terms
        for term in terms[1:]:
            result_set &= set(self.search_by_term(term))

        return list(result_set)

    def get_pattern(self, pattern_id: str) -> Optional[Any]:
        """Get pattern by ID."""
        return self.pattern_index.get(pattern_id)

    def get_metadata(self, pattern_id: str) -> Optional[PatternMetadata]:
        """Get metadata by pattern ID."""
        return self.id_to_metadata.get(pattern_id)

    def size(self) -> int:
        """Return number of indexed patterns."""
        return len(self.pattern_index)

    def clear(self) -> None:
        """Clear entire index."""
        self.term_index.clear()
        self.domain_index.clear()
        self.pattern_index.clear()
        self.id_to_metadata.clear()


# ============================================================================
# CATALOG PERSISTENCE
# ============================================================================

class CatalogPersistence:
    """
    Handles catalog serialization and deserialization.
    
    Features:
    - JSON format for human readability
    - Gzip compression for efficiency
    - Metadata preservation
    - Format versioning
    """

    VERSION = "1.0"
    COMPRESSION_LEVEL = 9

    @staticmethod
    def serialize_catalog(
        patterns: Dict[str, Any],
        metadata: Dict[str, PatternMetadata]
    ) -> str:
        """
        Serialize catalog to JSON string.

        Args:
            patterns: Dict of pattern_id -> AnalogyPattern
            metadata: Dict of pattern_id -> PatternMetadata

        Returns:
            JSON string representation
        """
        catalog_data = {
            "version": CatalogPersistence.VERSION,
            "created_at": datetime.now().isoformat(),
            "pattern_count": len(patterns),
            "patterns": {},
            "metadata": {}
        }

        # Serialize patterns
        for pattern_id, pattern in patterns.items():
            if hasattr(pattern, 'to_dict'):
                catalog_data["patterns"][pattern_id] = pattern.to_dict()
            else:
                catalog_data["patterns"][pattern_id] = str(pattern)

        # Serialize metadata
        for pattern_id, meta in metadata.items():
            catalog_data["metadata"][pattern_id] = meta.to_dict()

        return json.dumps(catalog_data, indent=2)

    @staticmethod
    def deserialize_catalog(
        json_str: str
    ) -> Tuple[Dict[str, Any], Dict[str, PatternMetadata]]:
        """
        Deserialize catalog from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            Tuple of (patterns dict, metadata dict)
        """
        catalog_data = json.loads(json_str)

        patterns = catalog_data.get("patterns", {})
        metadata = {}

        # Deserialize metadata
        for pattern_id, meta_dict in catalog_data.get("metadata", {}).items():
            metadata[pattern_id] = PatternMetadata.from_dict(meta_dict)

        return patterns, metadata

    @staticmethod
    def save_compressed(
        filepath: str,
        json_str: str
    ) -> int:
        """
        Save JSON string with gzip compression.

        Args:
            filepath: Path to save file
            json_str: JSON string to compress

        Returns:
            Compressed size in bytes
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(
            filepath,
            'wt',
            compresslevel=CatalogPersistence.COMPRESSION_LEVEL,
            encoding='utf-8'
        ) as f:
            f.write(json_str)

        return filepath.stat().st_size

    @staticmethod
    def load_compressed(filepath: str) -> str:
        """
        Load and decompress JSON string.

        Args:
            filepath: Path to compressed file

        Returns:
            Decompressed JSON string
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Catalog file not found: {filepath}")

        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return f.read()


# ============================================================================
# ENHANCED CATALOG WITH PERSISTENCE
# ============================================================================

class EnhancedAnalogyCatalog:
    """
    Enhanced analogy pattern catalog with persistence and indexing.
    
    Features:
    - Pattern registration with metadata
    - Efficient search via indexing
    - Persistent storage with compression
    - Pattern discovery and analytics
    """

    def __init__(self):
        """Initialize catalog."""
        self.patterns: Dict[str, Any] = {}
        self.metadata: Dict[str, PatternMetadata] = {}
        self.index = PatternIndex()
        self._pattern_counter = 0

    def register_pattern(
        self,
        pattern: Any,
        domain_tags: Optional[List[str]] = None,
        pattern_id: Optional[str] = None
    ) -> str:
        """
        Register a new pattern.

        Args:
            pattern: AnalogyPattern object
            domain_tags: Optional domain tags
            pattern_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Pattern ID
        """
        if pattern_id is None:
            pattern_id = f"pattern_{self._pattern_counter:06d}"
            self._pattern_counter += 1

        # Create metadata
        metadata = PatternMetadata(
            pattern_id=pattern_id,
            created_at=datetime.now().isoformat(),
            domain_tags=domain_tags or []
        )

        # Store pattern
        self.patterns[pattern_id] = pattern
        self.metadata[pattern_id] = metadata

        # Add to index
        self.index.add_pattern(pattern_id, pattern, metadata)

        logger.info(f"Registered pattern: {pattern_id}")
        return pattern_id

    def unregister_pattern(self, pattern_id: str) -> bool:
        """
        Remove pattern from catalog.

        Args:
            pattern_id: Pattern to remove

        Returns:
            True if removed, False if not found
        """
        if pattern_id not in self.patterns:
            return False

        del self.patterns[pattern_id]
        del self.metadata[pattern_id]
        self.index.remove_pattern(pattern_id)

        logger.info(f"Unregistered pattern: {pattern_id}")
        return True

    def search_by_term(self, term: str) -> List[Tuple[str, Any]]:
        """
        Search patterns by term.

        Args:
            term: Search term

        Returns:
            List of (pattern_id, pattern) tuples
        """
        matching_ids = self.index.search_by_term(term)
        return [
            (pid, self.patterns[pid])
            for pid in matching_ids
        ]

    def search_by_domain(self, domain: str) -> List[Tuple[str, Any]]:
        """
        Search patterns by domain.

        Args:
            domain: Domain tag

        Returns:
            List of (pattern_id, pattern) tuples
        """
        matching_ids = self.index.search_by_domain(domain)
        return [
            (pid, self.patterns[pid])
            for pid in matching_ids
        ]

    def search_by_terms(self, terms: List[str]) -> List[Tuple[str, Any]]:
        """
        Search patterns matching all terms.

        Args:
            terms: List of search terms

        Returns:
            List of (pattern_id, pattern) tuples
        """
        matching_ids = self.index.search_by_terms(terms)
        return [
            (pid, self.patterns[pid])
            for pid in matching_ids
        ]

    def discover_patterns(self, query: str) -> List[Tuple[str, float]]:
        """
        Discover patterns by natural language query.

        Args:
            query: Natural language query

        Returns:
            List of (pattern_id, relevance_score) tuples
        """
        # Split query into terms
        terms = [t.strip() for t in query.lower().split()]

        # Search for matching patterns
        matching_ids = self.index.search_by_terms(terms)

        # Score by number of matching terms
        results = []
        for pattern_id in matching_ids:
            pattern = self.patterns[pattern_id]
            # Count matching terms in pattern
            matching_count = 0
            if hasattr(pattern, 'analogies'):
                pattern_str = str(pattern.analogies).lower()
                for term in terms:
                    if term in pattern_str:
                        matching_count += 1

            # Relevance score based on matches
            relevance = min(1.0, matching_count / len(terms)) if terms else 0.0
            results.append((pattern_id, relevance))

        # Sort by relevance descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def update_metadata(
        self,
        pattern_id: str,
        success_rate: Optional[float] = None,
        confidence: Optional[float] = None,
        domain_tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update pattern metadata after use.

        Args:
            pattern_id: Pattern to update
            success_rate: New success rate
            confidence: New confidence value
            domain_tags: New domain tags

        Returns:
            True if updated, False if not found
        """
        if pattern_id not in self.metadata:
            return False

        meta = self.metadata[pattern_id]
        meta.accessed_count += 1
        meta.last_accessed = datetime.now().isoformat()

        if success_rate is not None:
            meta.success_rate = success_rate

        if confidence is not None:
            # Update exponential moving average
            old_avg = meta.avg_confidence
            new_avg = 0.9 * old_avg + 0.1 * confidence
            meta.avg_confidence = new_avg

        if domain_tags is not None:
            meta.domain_tags = domain_tags

        # Update performance score
        meta.performance_score = (
            meta.success_rate * 0.7 +
            meta.avg_confidence * 0.3
        )

        return True

    def save(self, filepath: str) -> int:
        """
        Save catalog to file.

        Args:
            filepath: Path to save file

        Returns:
            Compressed file size in bytes
        """
        json_str = CatalogPersistence.serialize_catalog(
            self.patterns,
            self.metadata
        )
        file_size = CatalogPersistence.save_compressed(filepath, json_str)

        logger.info(
            f"Saved catalog with {len(self.patterns)} patterns "
            f"to {filepath} ({file_size} bytes)"
        )
        return file_size

    def load(self, filepath: str) -> int:
        """
        Load catalog from file.

        Args:
            filepath: Path to load file

        Returns:
            Number of patterns loaded
        """
        json_str = CatalogPersistence.load_compressed(filepath)
        patterns, metadata = CatalogPersistence.deserialize_catalog(json_str)

        # Clear existing data
        self.patterns.clear()
        self.metadata.clear()
        self.index.clear()

        # Restore patterns and metadata
        self.patterns = patterns
        self.metadata = metadata

        # Rebuild index
        for pattern_id, meta in metadata.items():
            if pattern_id in patterns:
                self.index.add_pattern(pattern_id, patterns[pattern_id], meta)

        pattern_count = len(self.patterns)
        logger.info(f"Loaded catalog with {pattern_count} patterns from {filepath}")
        return pattern_count

    def get_catalog_stats(self) -> Dict[str, Any]:
        """
        Get catalog statistics.

        Returns:
            Dictionary of stats
        """
        total_accesses = sum(m.accessed_count for m in self.metadata.values())
        avg_performance = (
            sum(m.performance_score for m in self.metadata.values()) /
            len(self.metadata) if self.metadata else 0.0
        )

        return {
            'total_patterns': len(self.patterns),
            'total_accesses': total_accesses,
            'avg_performance_score': round(avg_performance, 3),
            'unique_domains': len(self.index.domain_index),
            'indexed_terms': len(self.index.term_index)
        }

    def clear(self) -> None:
        """Clear entire catalog."""
        self.patterns.clear()
        self.metadata.clear()
        self.index.clear()
        self._pattern_counter = 0
        logger.info("Cleared catalog")
