"""
Test suite for pattern persistence layer - Phase 2A.4

Tests cover:
- PatternIndex functionality (add, remove, search)
- PatternMetadata management
- CatalogPersistence (serialize, compress, decompress)
- EnhancedAnalogyCatalog (register, search, discovery)
- Performance and compression validation
"""

import pytest
import json
import gzip
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from core.pattern_persistence import (
    PatternMetadata,
    PatternIndex,
    CatalogPersistence,
    EnhancedAnalogyCatalog
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_metadata() -> PatternMetadata:
    """Create sample pattern metadata."""
    return PatternMetadata(
        pattern_id="test_001",
        created_at=datetime.now().isoformat(),
        accessed_count=5,
        success_rate=0.85,
        avg_confidence=0.92,
        domain_tags=["logic", "reasoning"],
        performance_score=0.868
    )


@pytest.fixture
def sample_pattern() -> dict:
    """Create sample pattern object."""
    return {
        "analogies": [
            ["A", "B"],
            ["C", "D"]
        ],
        "relationships": ["equals", "transforms"]
    }


@pytest.fixture
def sample_index() -> PatternIndex:
    """Create populated pattern index."""
    index = PatternIndex()

    patterns = {
        "p1": {"analogies": [["sun", "moon"], ["day", "night"]]},
        "p2": {"analogies": [["hot", "cold"], ["fire", "ice"]]},
        "p3": {"analogies": [["sun", "star"], ["planet", "orbit"]]}
    }

    metadata_list = [
        PatternMetadata(
            pattern_id="p1",
            created_at=datetime.now().isoformat(),
            domain_tags=["astronomy", "natural"]
        ),
        PatternMetadata(
            pattern_id="p2",
            created_at=datetime.now().isoformat(),
            domain_tags=["physics", "temperature"]
        ),
        PatternMetadata(
            pattern_id="p3",
            created_at=datetime.now().isoformat(),
            domain_tags=["astronomy", "space"]
        )
    ]

    for (pattern_id, pattern), metadata in zip(patterns.items(), metadata_list):
        index.add_pattern(pattern_id, pattern, metadata)

    return index


@pytest.fixture
def sample_catalog() -> EnhancedAnalogyCatalog:
    """Create populated catalog."""
    catalog = EnhancedAnalogyCatalog()

    patterns = [
        {"name": "analogy1", "analogies": [["A", "B"], ["C", "D"]]},
        {"name": "analogy2", "analogies": [["X", "Y"], ["Z", "W"]]},
        {"name": "analogy3", "analogies": [["sun", "moon"], ["day", "night"]]}
    ]

    domains = [
        ["math", "logic"],
        ["linguistics"],
        ["astronomy"]
    ]

    for pattern, domain_tags in zip(patterns, domains):
        catalog.register_pattern(pattern, domain_tags=domain_tags)

    return catalog


# ============================================================================
# PatternMetadata Tests
# ============================================================================

class TestPatternMetadata:
    """Tests for PatternMetadata class."""

    def test_metadata_creation(self, sample_metadata: PatternMetadata):
        """Test metadata object creation."""
        assert sample_metadata.pattern_id == "test_001"
        assert sample_metadata.accessed_count == 5
        assert sample_metadata.success_rate == 0.85
        assert "logic" in sample_metadata.domain_tags

    def test_metadata_to_dict(self, sample_metadata: PatternMetadata):
        """Test metadata serialization to dict."""
        data = sample_metadata.to_dict()

        assert isinstance(data, dict)
        assert data["pattern_id"] == "test_001"
        assert data["success_rate"] == 0.85
        assert data["domain_tags"] == ["logic", "reasoning"]

    def test_metadata_from_dict(self):
        """Test metadata deserialization from dict."""
        data = {
            "pattern_id": "test_002",
            "created_at": datetime.now().isoformat(),
            "accessed_count": 10,
            "success_rate": 0.9,
            "avg_confidence": 0.95,
            "domain_tags": ["test"],
            "related_patterns": [],
            "performance_score": 0.92,
            "last_accessed": datetime.now().isoformat()
        }

        metadata = PatternMetadata.from_dict(data)

        assert metadata.pattern_id == "test_002"
        assert metadata.accessed_count == 10
        assert metadata.success_rate == 0.9

    def test_metadata_roundtrip(self, sample_metadata: PatternMetadata):
        """Test metadata serialization roundtrip."""
        data = sample_metadata.to_dict()
        restored = PatternMetadata.from_dict(data)

        assert restored.pattern_id == sample_metadata.pattern_id
        assert restored.success_rate == sample_metadata.success_rate
        assert restored.domain_tags == sample_metadata.domain_tags


# ============================================================================
# PatternIndex Tests
# ============================================================================

class TestPatternIndex:
    """Tests for PatternIndex class."""

    def test_index_creation(self):
        """Test index initialization."""
        index = PatternIndex()

        assert len(index.pattern_index) == 0
        assert len(index.term_index) == 0
        assert len(index.domain_index) == 0

    def test_add_pattern(self, sample_index: PatternIndex):
        """Test adding pattern to index."""
        assert sample_index.size() == 3

    def test_search_by_term(self, sample_index: PatternIndex):
        """Test term-based search."""
        results = sample_index.search_by_term("sun")

        assert len(results) == 2  # p1 and p3
        assert "p1" in results
        assert "p3" in results

    def test_search_by_domain(self, sample_index: PatternIndex):
        """Test domain-based search."""
        results = sample_index.search_by_domain("astronomy")

        assert len(results) == 2  # p1 and p3
        assert "p1" in results
        assert "p3" in results

    def test_search_by_multiple_domains(self, sample_index: PatternIndex):
        """Test finding patterns across multiple domains."""
        astronomy = sample_index.search_by_domain("astronomy")
        physics = sample_index.search_by_domain("physics")

        assert len(astronomy) == 2
        assert len(physics) == 1

    def test_search_by_terms_and(self, sample_index: PatternIndex):
        """Test AND search with multiple terms."""
        results = sample_index.search_by_terms(["sun", "moon"])

        assert len(results) == 1  # Only p1 has both
        assert "p1" in results

    def test_search_nonexistent_term(self, sample_index: PatternIndex):
        """Test search for nonexistent term."""
        results = sample_index.search_by_term("unicorn")

        assert len(results) == 0

    def test_remove_pattern(self, sample_index: PatternIndex):
        """Test pattern removal."""
        assert sample_index.size() == 3

        removed = sample_index.remove_pattern("p1")
        assert removed is True
        assert sample_index.size() == 2

        # Verify term index cleaned up
        results = sample_index.search_by_term("sun")
        assert "p1" not in results

    def test_remove_nonexistent_pattern(self, sample_index: PatternIndex):
        """Test removing nonexistent pattern."""
        removed = sample_index.remove_pattern("nonexistent")

        assert removed is False
        assert sample_index.size() == 3

    def test_get_pattern(self, sample_index: PatternIndex):
        """Test retrieving pattern by ID."""
        pattern = sample_index.get_pattern("p1")

        assert pattern is not None
        assert "sun" in str(pattern)

    def test_get_metadata(self, sample_index: PatternIndex):
        """Test retrieving metadata by pattern ID."""
        metadata = sample_index.get_metadata("p1")

        assert metadata is not None
        assert metadata.pattern_id == "p1"
        assert "astronomy" in metadata.domain_tags

    def test_clear_index(self, sample_index: PatternIndex):
        """Test clearing entire index."""
        assert sample_index.size() == 3

        sample_index.clear()

        assert sample_index.size() == 0
        assert len(sample_index.term_index) == 0
        assert len(sample_index.domain_index) == 0


# ============================================================================
# CatalogPersistence Tests
# ============================================================================

class TestCatalogPersistence:
    """Tests for CatalogPersistence class."""

    def test_serialize_catalog(self, sample_pattern: dict, sample_metadata: PatternMetadata):
        """Test catalog serialization."""
        patterns = {"p1": sample_pattern}
        metadata = {"p1": sample_metadata}

        json_str = CatalogPersistence.serialize_catalog(patterns, metadata)

        assert isinstance(json_str, str)
        data = json.loads(json_str)

        assert data["version"] == "1.0"
        assert data["pattern_count"] == 1
        assert "p1" in data["patterns"]
        assert "p1" in data["metadata"]

    def test_deserialize_catalog(self, sample_pattern: dict, sample_metadata: PatternMetadata):
        """Test catalog deserialization."""
        patterns = {"p1": sample_pattern}
        metadata = {"p1": sample_metadata}

        json_str = CatalogPersistence.serialize_catalog(patterns, metadata)
        restored_patterns, restored_metadata = CatalogPersistence.deserialize_catalog(json_str)

        assert "p1" in restored_patterns
        assert "p1" in restored_metadata

    def test_save_and_load_compressed(self, sample_pattern: dict, sample_metadata: PatternMetadata):
        """Test saving and loading compressed catalog."""
        patterns = {"p1": sample_pattern}
        metadata = {"p1": sample_metadata}

        json_str = CatalogPersistence.serialize_catalog(patterns, metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_catalog.json.gz"

            # Save
            saved_size = CatalogPersistence.save_compressed(str(filepath), json_str)
            assert filepath.exists()
            assert saved_size > 0

            # Verify compression
            uncompressed_size = len(json_str.encode('utf-8'))
            compression_ratio = saved_size / uncompressed_size
            assert compression_ratio < 1.0  # Should be compressed

            # Load
            loaded_json = CatalogPersistence.load_compressed(str(filepath))
            assert loaded_json == json_str

    def test_compression_efficiency(self, sample_pattern: dict):
        """Test compression provides efficiency gains."""
        patterns = {f"p{i}": sample_pattern for i in range(100)}
        metadata = {
            f"p{i}": PatternMetadata(
                pattern_id=f"p{i}",
                created_at=datetime.now().isoformat(),
                domain_tags=["test", "compression"]
            )
            for i in range(100)
        }

        json_str = CatalogPersistence.serialize_catalog(patterns, metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large_catalog.json.gz"

            saved_size = CatalogPersistence.save_compressed(str(filepath), json_str)
            uncompressed_size = len(json_str.encode('utf-8'))

            # Should have significant compression (>30%)
            compression_ratio = saved_size / uncompressed_size
            assert compression_ratio < 0.7  # At least 30% compression

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            CatalogPersistence.load_compressed("/nonexistent/path/file.json.gz")


# ============================================================================
# EnhancedAnalogyCatalog Tests
# ============================================================================

class TestEnhancedAnalogyCatalog:
    """Tests for EnhancedAnalogyCatalog class."""

    def test_catalog_creation(self):
        """Test catalog initialization."""
        catalog = EnhancedAnalogyCatalog()

        assert len(catalog.patterns) == 0
        assert len(catalog.metadata) == 0
        assert catalog.index.size() == 0

    def test_register_pattern(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test registering pattern."""
        assert len(sample_catalog.patterns) == 3

    def test_register_pattern_with_custom_id(self, sample_pattern: dict):
        """Test registering pattern with custom ID."""
        catalog = EnhancedAnalogyCatalog()
        pattern_id = catalog.register_pattern(
            sample_pattern,
            domain_tags=["test"],
            pattern_id="custom_id"
        )

        assert pattern_id == "custom_id"
        assert "custom_id" in catalog.patterns

    def test_unregister_pattern(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test unregistering pattern."""
        pattern_ids = list(sample_catalog.patterns.keys())
        pattern_id = pattern_ids[0]

        result = sample_catalog.unregister_pattern(pattern_id)

        assert result is True
        assert pattern_id not in sample_catalog.patterns
        assert pattern_id not in sample_catalog.metadata

    def test_unregister_nonexistent_pattern(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test unregistering nonexistent pattern."""
        result = sample_catalog.unregister_pattern("nonexistent")

        assert result is False

    def test_search_by_term(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test searching catalog by term."""
        results = sample_catalog.search_by_term("sun")

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_by_domain(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test searching catalog by domain."""
        results = sample_catalog.search_by_domain("astronomy")

        assert len(results) == 1
        assert results[0][0]  # pattern_id exists

    def test_search_by_multiple_terms(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test searching by multiple terms."""
        results = sample_catalog.search_by_terms(["A", "B"])

        assert len(results) > 0

    def test_discover_patterns(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test pattern discovery via query."""
        results = sample_catalog.discover_patterns("sun moon night")

        # Results should be tuples of (pattern_id, relevance_score)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(0 <= r[1] <= 1.0 for r in results)

        # Should be sorted by relevance (descending)
        if len(results) > 1:
            relevances = [r[1] for r in results]
            assert relevances == sorted(relevances, reverse=True)

    def test_update_metadata(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test updating pattern metadata."""
        pattern_id = list(sample_catalog.patterns.keys())[0]

        result = sample_catalog.update_metadata(
            pattern_id,
            success_rate=0.95,
            confidence=0.88,
            domain_tags=["updated"]
        )

        assert result is True

        metadata = sample_catalog.metadata[pattern_id]
        assert metadata.success_rate == 0.95
        assert "updated" in metadata.domain_tags
        assert metadata.accessed_count == 1

    def test_metadata_exponential_moving_average(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test confidence EMA update."""
        pattern_id = list(sample_catalog.patterns.keys())[0]

        # First update
        sample_catalog.update_metadata(pattern_id, confidence=0.8)
        meta = sample_catalog.metadata[pattern_id]
        first_avg = meta.avg_confidence

        # Second update
        sample_catalog.update_metadata(pattern_id, confidence=0.9)
        second_avg = meta.avg_confidence

        # Should be between old and new values
        assert 0.8 <= second_avg <= 0.9

    def test_save_and_load_roundtrip(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test save/load roundtrip preserves catalog."""
        original_count = len(sample_catalog.patterns)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "catalog.json.gz"

            # Save
            saved_size = sample_catalog.save(str(filepath))
            assert filepath.exists()
            assert saved_size > 0

            # Load into new catalog
            new_catalog = EnhancedAnalogyCatalog()
            loaded_count = new_catalog.load(str(filepath))

            assert loaded_count == original_count
            assert len(new_catalog.patterns) == original_count

    def test_catalog_stats(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test catalog statistics generation."""
        # Update some metadata to generate meaningful stats
        pattern_ids = list(sample_catalog.patterns.keys())
        for pattern_id in pattern_ids:
            sample_catalog.update_metadata(
                pattern_id,
                success_rate=0.85,
                confidence=0.9
            )

        stats = sample_catalog.get_catalog_stats()

        assert stats["total_patterns"] == 3
        assert stats["total_accesses"] >= 3  # At least one access each
        assert 0 <= stats["avg_performance_score"] <= 1.0
        assert stats["unique_domains"] >= 2

    def test_clear_catalog(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test clearing entire catalog."""
        assert len(sample_catalog.patterns) > 0

        sample_catalog.clear()

        assert len(sample_catalog.patterns) == 0
        assert len(sample_catalog.metadata) == 0
        assert sample_catalog.index.size() == 0

    def test_catalog_counter_increment(self, sample_pattern: dict):
        """Test pattern ID auto-increment."""
        catalog = EnhancedAnalogyCatalog()

        p1 = catalog.register_pattern(sample_pattern)
        p2 = catalog.register_pattern(sample_pattern)
        p3 = catalog.register_pattern(sample_pattern)

        # IDs should increment
        assert p1 != p2 != p3
        assert "pattern_" in p1


# ============================================================================
# Integration Tests
# ============================================================================

class TestPersistenceIntegration:
    """Integration tests for persistence layer."""

    def test_full_workflow(self, sample_pattern: dict):
        """Test complete workflow: register, update, save, load, search."""
        catalog1 = EnhancedAnalogyCatalog()

        # Register patterns
        p1 = catalog1.register_pattern(
            sample_pattern,
            domain_tags=["test", "logic"]
        )
        p2 = catalog1.register_pattern(
            sample_pattern,
            domain_tags=["test", "math"]
        )

        # Update metadata
        catalog1.update_metadata(p1, success_rate=0.9, confidence=0.85)
        catalog1.update_metadata(p2, success_rate=0.95, confidence=0.92)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "workflow_test.json.gz"

            # Save
            catalog1.save(str(filepath))

            # Load into new catalog
            catalog2 = EnhancedAnalogyCatalog()
            catalog2.load(str(filepath))

            # Verify search still works
            results = catalog2.search_by_domain("test")
            assert len(results) == 2

            # Verify metadata preserved
            meta = catalog2.metadata[p1]
            assert meta.success_rate == 0.9

    def test_large_catalog_persistence(self, sample_pattern: dict):
        """Test persistence with large catalog."""
        catalog = EnhancedAnalogyCatalog()

        # Add many patterns
        for i in range(1000):
            pattern = {
                **sample_pattern,
                "id": i
            }
            catalog.register_pattern(
                pattern,
                domain_tags=[f"domain_{i % 10}", "large_test"]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large_catalog.json.gz"

            saved_size = catalog.save(str(filepath))

            # Load and verify
            new_catalog = EnhancedAnalogyCatalog()
            loaded = new_catalog.load(str(filepath))

            assert loaded == 1000
            assert new_catalog.index.size() == 1000

    def test_search_performance_with_indexing(self, sample_catalog: EnhancedAnalogyCatalog):
        """Test that indexing enables fast searches."""
        import time

        # Add many domain tags
        for pattern_id in sample_catalog.patterns.keys():
            sample_catalog.update_metadata(
                pattern_id,
                domain_tags=[f"tag_{i}" for i in range(10)]
            )

        # Search should be fast (O(log n) or better)
        start = time.time()
        for _ in range(100):
            sample_catalog.search_by_domain("tag_5")
        elapsed = time.time() - start

        # 100 searches should be very fast (< 100ms on modern hardware)
        assert elapsed < 0.1, f"Search too slow: {elapsed}s"
