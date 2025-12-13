"""
Comprehensive tests for Pattern Evolution Layer (Phase 2A.4 Task 2).

Tests cover:
- PatternClusterer: Hierarchical clustering with silhouette scoring
- PatternAbstractor: LCS extraction for pattern abstraction
- EmergentPatternDiscoverer: Novelty and utility-based pattern discovery
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from sigmalang.core.pattern_evolution import (
    PatternClusterer,
    PatternAbstractor,
    EmergentPatternDiscoverer,
    ClusterResult,
    EmergentPattern,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_patterns() -> Dict[str, Any]:
    """Fixture providing sample pattern dictionary."""
    return {
        "logic_001": {"type": "logical", "content": "A implies B implies C"},
        "logic_002": {"type": "logical", "content": "B implies C implies D"},
        "logic_003": {"type": "logical", "content": "A implies B implies D"},
        "arithmetic_001": {"type": "arithmetic", "content": "1 plus 2 plus 3"},
        "arithmetic_002": {"type": "arithmetic", "content": "2 plus 3 plus 4"},
        "language_001": {"type": "language", "content": "cat is animal"},
        "language_002": {"type": "language", "content": "dog is animal"},
        "language_003": {"type": "language", "content": "bird is animal"},
    }


@pytest.fixture
def frequency_data() -> Dict[str, int]:
    """Fixture providing pattern frequency data."""
    return {
        "logic_001": 15,
        "logic_002": 12,
        "logic_003": 10,
        "arithmetic_001": 25,
        "arithmetic_002": 20,
        "language_001": 30,
        "language_002": 28,
        "language_003": 5,
    }


# ============================================================================
# PATTERN CLUSTERER TESTS
# ============================================================================

class TestPatternClusterer:
    """Test suite for PatternClusterer."""

    def test_clusterer_creation(self):
        """Test clusterer initialization."""
        clusterer = PatternClusterer()
        assert clusterer is not None
        assert clusterer.distance_metric == "euclidean"

    def test_clusterer_with_custom_metric(self):
        """Test clusterer with custom distance metric."""
        clusterer = PatternClusterer(distance_metric="cosine")
        assert clusterer.distance_metric == "cosine"

    def test_pattern_distance_identical_patterns(self):
        """Test distance between identical patterns."""
        clusterer = PatternClusterer()
        pattern = {"content": "A implies B"}
        distance = clusterer.compute_pattern_distance(pattern, pattern)
        assert distance == 0.0  # Identical patterns have distance 0

    def test_pattern_distance_different_patterns(self):
        """Test distance between different patterns."""
        clusterer = PatternClusterer()
        p1 = {"content": "A implies B"}
        p2 = {"content": "C implies D"}
        distance = clusterer.compute_pattern_distance(p1, p2)
        assert 0.0 <= distance <= 1.0

    def test_pattern_distance_similar_patterns(self):
        """Test distance between similar patterns."""
        clusterer = PatternClusterer()
        p1 = {"content": "A implies B implies C"}
        p2 = {"content": "A implies B implies D"}
        distance = clusterer.compute_pattern_distance(p1, p2)
        assert distance < 0.5  # Similar patterns should have smaller distance

    def test_cluster_single_pattern(self, sample_patterns):
        """Test clustering with single pattern."""
        clusterer = PatternClusterer()
        single = {"p1": sample_patterns["logic_001"]}
        results = clusterer.cluster_patterns(single)
        
        assert len(results) == 1
        assert results[0].size == 1
        assert results[0].silhouette_score == 1.0

    def test_cluster_two_patterns(self, sample_patterns):
        """Test clustering with two patterns."""
        clusterer = PatternClusterer()
        two = {
            "p1": sample_patterns["logic_001"],
            "p2": sample_patterns["logic_002"],
        }
        results = clusterer.cluster_patterns(two)
        
        assert len(results) <= 2
        assert sum(c.size for c in results) == 2

    def test_cluster_multiple_patterns(self, sample_patterns):
        """Test clustering with multiple patterns."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        
        assert len(results) == 3
        assert sum(c.size for c in results) == len(sample_patterns)

    def test_cluster_result_properties(self, sample_patterns):
        """Test that cluster results have correct properties."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        
        for cluster in results:
            assert cluster.cluster_id >= 0
            assert len(cluster.patterns) > 0
            assert cluster.size == len(cluster.patterns)
            assert 0.0 <= cluster.silhouette_score <= 1.0

    def test_cluster_silhouette_score_validity(self, sample_patterns):
        """Test that silhouette scores are valid."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(sample_patterns, num_clusters=2)
        
        for cluster in results:
            # Silhouette scores should be between -1 and 1
            assert -1.0 <= cluster.silhouette_score <= 1.0

    def test_cluster_consistency(self, sample_patterns):
        """Test that clustering is deterministic."""
        clusterer = PatternClusterer()
        
        results1 = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        results2 = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        
        # Should get same clustering
        assert len(results1) == len(results2)
        for c1, c2 in zip(results1, results2):
            assert c1.size == c2.size

    def test_cluster_cohesion_valid_range(self, sample_patterns):
        """Test that cohesion scores are in valid range."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(sample_patterns, num_clusters=2)
        
        for cluster in results:
            assert 0.0 <= cluster.cohesion <= 1.0

    def test_cluster_separation_valid_range(self, sample_patterns):
        """Test that separation scores are in valid range."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(sample_patterns, num_clusters=2)
        
        for cluster in results:
            assert 0.0 <= cluster.separation <= 1.0

    def test_cluster_ordering_by_size(self, sample_patterns):
        """Test that clusters are ordered by size."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(sample_patterns, num_clusters=2)
        
        # Results should be sorted by size descending
        for i in range(len(results) - 1):
            assert results[i].size >= results[i + 1].size


# ============================================================================
# PATTERN ABSTRACTOR TESTS
# ============================================================================

class TestPatternAbstractor:
    """Test suite for PatternAbstractor."""

    def test_extract_common_pattern_single_pattern(self):
        """Test extraction with single pattern."""
        patterns = ["A implies B implies C"]
        result = PatternAbstractor.extract_common_pattern(patterns)
        assert result == "A implies B implies C"

    def test_extract_common_pattern_empty_list(self):
        """Test extraction with empty list."""
        result = PatternAbstractor.extract_common_pattern([])
        assert result == ""

    def test_extract_common_pattern_identical(self):
        """Test extraction with identical patterns."""
        patterns = [
            "A implies B",
            "A implies B",
            "A implies B",
        ]
        result = PatternAbstractor.extract_common_pattern(patterns)
        assert result == "A implies B"

    def test_lcs_basic(self):
        """Test basic LCS computation."""
        s1 = "AGGTAB"
        s2 = "GXTXAYB"
        result = PatternAbstractor.lcs(s1, s2)
        # LCS should be non-empty for these strings
        assert len(result) > 0

    def test_lcs_identical_strings(self):
        """Test LCS with identical strings."""
        s = "ABCDEF"
        result = PatternAbstractor.lcs(s, s)
        assert result == s

    def test_lcs_no_common_subsequence(self):
        """Test LCS with no common characters."""
        s1 = "ABC"
        s2 = "DEF"
        result = PatternAbstractor.lcs(s1, s2)
        assert result == ""

    def test_lcs_single_character_overlap(self):
        """Test LCS with single character overlap."""
        s1 = "A"
        s2 = "A"
        result = PatternAbstractor.lcs(s1, s2)
        assert result == "A"

    def test_extract_parameters_single_pattern(self):
        """Test parameter extraction from single pattern."""
        patterns = ["A implies B"]
        template = "A implies B"
        params = PatternAbstractor.extract_parameters(patterns, template)
        
        assert len(params) == 1
        assert params[0] == {}

    def test_extract_parameters_multiple_patterns(self):
        """Test parameter extraction from multiple patterns."""
        patterns = [
            "A implies B",
            "C implies D",
            "E implies F",
        ]
        template = "X implies Y"
        params = PatternAbstractor.extract_parameters(patterns, template)
        
        assert len(params) == 3

    def test_extract_common_pattern_similarity(self):
        """Test that common pattern captures similarity."""
        patterns = [
            "A implies B implies C",
            "A implies B implies D",
            "A implies B implies E",
        ]
        result = PatternAbstractor.extract_common_pattern(patterns)
        # Should capture common prefix
        assert "A" in result or "implies" in result


# ============================================================================
# EMERGENT PATTERN DISCOVERER TESTS
# ============================================================================

class TestEmergentPatternDiscoverer:
    """Test suite for EmergentPatternDiscoverer."""

    def test_discoverer_creation(self):
        """Test discoverer initialization."""
        discoverer = EmergentPatternDiscoverer()
        assert discoverer is not None
        assert discoverer.novelty_threshold == 0.7

    def test_discoverer_custom_threshold(self):
        """Test discoverer with custom threshold."""
        discoverer = EmergentPatternDiscoverer(novelty_threshold=0.8)
        assert discoverer.novelty_threshold == 0.8

    def test_novelty_calculation_basic(self):
        """Test basic novelty calculation."""
        discoverer = EmergentPatternDiscoverer()
        p1 = "A implies B"
        p2 = "C implies D"
        patterns = [p1, p2]
        
        novelty = discoverer._compute_cluster_novelty(patterns)
        assert 0.0 <= novelty <= 1.0

    def test_novelty_identical_patterns(self):
        """Test novelty for identical patterns."""
        discoverer = EmergentPatternDiscoverer()
        p = "A implies B"
        patterns = [p, p, p]
        
        novelty = discoverer._compute_cluster_novelty(patterns)
        assert novelty < 0.5  # Identical patterns should have low novelty

    def test_utility_calculation_with_frequency(self):
        """Test utility calculation with frequency data."""
        discoverer = EmergentPatternDiscoverer()
        pattern_ids = ["p1", "p2"]
        frequency = {"p1": 50, "p2": 60}
        discoverer.pattern_frequency = frequency
        
        utility = discoverer._compute_cluster_utility(pattern_ids, 0.3)
        assert 0.0 <= utility <= 1.0

    def test_utility_calculation_no_frequency(self):
        """Test utility calculation without frequency data."""
        discoverer = EmergentPatternDiscoverer()
        pattern_ids = ["p1", "p2"]
        
        utility = discoverer._compute_cluster_utility(pattern_ids, 0.3)
        assert 0.0 <= utility <= 1.0

    def test_discover_patterns_empty_clusters(self):
        """Test discovery with empty clusters."""
        discoverer = EmergentPatternDiscoverer()
        patterns = {}
        clusters = []
        
        results = discoverer.discover_patterns(patterns, clusters)
        assert results == []

    def test_discover_patterns_single_pattern_clusters(self, sample_patterns):
        """Test discovery with single-pattern clusters."""
        discoverer = EmergentPatternDiscoverer()
        clusters = [
            ClusterResult(cluster_id=0, patterns=["logic_001"]),
            ClusterResult(cluster_id=1, patterns=["logic_002"]),
        ]
        
        results = discoverer.discover_patterns(sample_patterns, clusters)
        # Single-pattern clusters typically don't produce emergent patterns
        assert isinstance(results, list)

    def test_discover_patterns_with_frequency(self, sample_patterns, frequency_data):
        """Test discovery with frequency data."""
        discoverer = EmergentPatternDiscoverer(novelty_threshold=0.6)
        clusters = [
            ClusterResult(
                cluster_id=0,
                patterns=["logic_001", "logic_002", "logic_003"],
                silhouette_score=0.8
            ),
        ]
        
        results = discoverer.discover_patterns(
            sample_patterns,
            clusters,
            frequency_data
        )
        
        assert isinstance(results, list)

    def test_emergent_pattern_properties(self, sample_patterns, frequency_data):
        """Test that emergent patterns have required properties."""
        discoverer = EmergentPatternDiscoverer(novelty_threshold=0.5)
        clusters = [
            ClusterResult(
                cluster_id=0,
                patterns=["logic_001", "logic_002"],
                silhouette_score=0.7
            ),
        ]
        
        results = discoverer.discover_patterns(
            sample_patterns,
            clusters,
            frequency_data
        )
        
        for pattern in results:
            assert isinstance(pattern, EmergentPattern)
            assert 0.0 <= pattern.novelty_score <= 1.0
            assert 0.0 <= pattern.utility_score <= 1.0
            assert 0.0 <= pattern.emergence_score <= 1.0

    def test_emergence_reason_generation(self):
        """Test emergence reason generation."""
        discoverer = EmergentPatternDiscoverer()
        
        reason1 = discoverer._get_emergence_reason(0.9, 0.3)
        assert "High novelty" in reason1
        
        reason2 = discoverer._get_emergence_reason(0.3, 0.9)
        assert "High utility" in reason2
        
        reason3 = discoverer._get_emergence_reason(0.75, 0.75)
        assert "Balanced" in reason3 or reason3 != ""

    def test_discovery_sorting_by_emergence_score(self, sample_patterns, frequency_data):
        """Test that discovered patterns are sorted by emergence score."""
        discoverer = EmergentPatternDiscoverer(novelty_threshold=0.4)
        clusters = [
            ClusterResult(
                cluster_id=0,
                patterns=["logic_001", "logic_002"],
                silhouette_score=0.7
            ),
            ClusterResult(
                cluster_id=1,
                patterns=["language_001", "language_002", "language_003"],
                silhouette_score=0.8
            ),
        ]
        
        results = discoverer.discover_patterns(
            sample_patterns,
            clusters,
            frequency_data
        )
        
        # Should be sorted by emergence score
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].emergence_score >= results[i + 1].emergence_score


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPatternEvolutionIntegration:
    """Integration tests for pattern evolution layer."""

    def test_full_evolution_workflow(self, sample_patterns, frequency_data):
        """Test complete clustering → abstraction → discovery workflow."""
        # Step 1: Cluster patterns
        clusterer = PatternClusterer()
        clusters = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        
        assert len(clusters) > 0
        assert sum(c.size for c in clusters) == len(sample_patterns)
        
        # Step 2: Discover emergent patterns
        discoverer = EmergentPatternDiscoverer(novelty_threshold=0.5)
        emergent = discoverer.discover_patterns(
            sample_patterns,
            clusters,
            frequency_data
        )
        
        assert isinstance(emergent, list)
        for pattern in emergent:
            assert isinstance(pattern, EmergentPattern)

    def test_large_pattern_set_clustering(self):
        """Test clustering with larger pattern set."""
        # Create 50 patterns
        patterns = {}
        for i in range(50):
            patterns[f"p{i}"] = {
                "type": "type" + str(i % 5),
                "content": f"pattern content {i} with some similarity"
            }
        
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(patterns, num_clusters=5)
        
        assert len(results) == 5
        assert sum(c.size for c in results) == 50

    def test_clustering_with_auto_cluster_detection(self, sample_patterns):
        """Test automatic cluster number detection."""
        clusterer = PatternClusterer()
        results_auto = clusterer.cluster_patterns(
            sample_patterns,
            num_clusters=None  # Auto-detect
        )
        
        assert len(results_auto) > 0
        assert sum(c.size for c in results_auto) == len(sample_patterns)

    def test_pattern_abstraction_from_clusters(self, sample_patterns):
        """Test pattern abstraction from cluster results."""
        clusterer = PatternClusterer()
        clusters = clusterer.cluster_patterns(sample_patterns, num_clusters=2)
        
        # Extract patterns from largest cluster
        if clusters:
            largest = clusters[0]
            cluster_patterns = [
                str(sample_patterns[pid])
                for pid in largest.patterns
            ]
            
            abstract = PatternAbstractor.extract_common_pattern(cluster_patterns)
            assert isinstance(abstract, str)

    def test_evolution_pipeline_performance(self, sample_patterns, frequency_data):
        """Test performance of complete evolution pipeline."""
        import time
        
        start = time.time()
        
        # Full pipeline
        clusterer = PatternClusterer()
        clusters = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        
        discoverer = EmergentPatternDiscoverer(novelty_threshold=0.5)
        emergent = discoverer.discover_patterns(
            sample_patterns,
            clusters,
            frequency_data
        )
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds for 8 patterns)
        assert elapsed < 5.0

    def test_pattern_evolution_no_regressions(self, sample_patterns, frequency_data):
        """Test that pattern evolution doesn't lose original patterns."""
        clusterer = PatternClusterer()
        clusters = clusterer.cluster_patterns(sample_patterns, num_clusters=3)
        
        # Verify all original patterns are in clusters
        clustered_pids = set()
        for cluster in clusters:
            clustered_pids.update(cluster.patterns)
        
        assert clustered_pids == set(sample_patterns.keys())


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestPatternEvolutionEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pattern_dict(self):
        """Test handling of empty pattern dictionary."""
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns({})
        assert results == []

    def test_very_similar_patterns(self):
        """Test clustering with very similar patterns."""
        patterns = {
            "p1": "A implies B",
            "p2": "A implies B",
            "p3": "A implies B",
        }
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(patterns, num_clusters=1)
        
        assert len(results) == 1
        assert results[0].size == 3

    def test_very_different_patterns(self):
        """Test clustering with very different patterns."""
        patterns = {
            "p1": "AAAAAAAA",
            "p2": "BBBBBBBB",
            "p3": "CCCCCCCC",
        }
        clusterer = PatternClusterer()
        results = clusterer.cluster_patterns(patterns, num_clusters=3)
        
        assert len(results) <= 3

    def test_lcs_very_long_strings(self):
        """Test LCS with very long strings."""
        s1 = "A" * 100 + "B" * 50
        s2 = "A" * 100 + "C" * 50
        result = PatternAbstractor.lcs(s1, s2)
        
        # Should find common "A"s
        assert "A" in result or len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
