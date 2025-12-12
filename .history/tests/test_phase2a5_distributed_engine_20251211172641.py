import pytest

from core.pattern_persistence import EnhancedAnalogyCatalog
from core.distributed_catalog import CatalogSynchronizer
from core.pattern_evolution import PatternClusterer


class TestCatalogSynchronizer:
    def test_delta_roundtrip_adds_patterns(self):
        src = EnhancedAnalogyCatalog()
        src.register_pattern({"content": "A implies B"}, domain_tags=["logic"], pattern_id="pattern_000010")
        src.register_pattern({"content": "X plus Y"}, domain_tags=["math"], pattern_id="pattern_000011")

        delta = CatalogSynchronizer.create_delta(src)
        payload = delta.to_json()

        dst = EnhancedAnalogyCatalog()
        added, updated = CatalogSynchronizer.apply_delta(dst, payload)

        assert added == 2
        assert updated == 0
        assert "pattern_000010" in dst.patterns
        assert "pattern_000011" in dst.patterns
        assert dst.index.size() == 2

    def test_conflict_skip(self):
        src = EnhancedAnalogyCatalog()
        src.register_pattern({"content": "A implies B"}, domain_tags=["logic"], pattern_id="pattern_000010")

        dst = EnhancedAnalogyCatalog()
        dst.register_pattern({"content": "DIFFERENT"}, domain_tags=["logic"], pattern_id="pattern_000010")

        payload = CatalogSynchronizer.create_delta(src).to_json()
        added, updated = CatalogSynchronizer.apply_delta(dst, payload, conflict="skip")

        assert added == 0
        assert updated == 0
        assert dst.patterns["pattern_000010"] == {"content": "DIFFERENT"}

    def test_conflict_overwrite(self):
        src = EnhancedAnalogyCatalog()
        src.register_pattern({"content": "A implies B"}, domain_tags=["logic"], pattern_id="pattern_000010")

        dst = EnhancedAnalogyCatalog()
        dst.register_pattern({"content": "DIFFERENT"}, domain_tags=["logic"], pattern_id="pattern_000010")

        payload = CatalogSynchronizer.create_delta(src).to_json()
        added, updated = CatalogSynchronizer.apply_delta(dst, payload, conflict="overwrite")

        assert added == 0
        assert updated == 1
        assert dst.patterns["pattern_000010"] == {"content": "A implies B"}


class TestParallelClustering:
    @pytest.fixture
    def patterns(self):
        # Two loose groups to reduce flakiness.
        return {
            "p1": {"content": "A implies B"},
            "p2": {"content": "A implies C"},
            "p3": {"content": "X plus Y"},
            "p4": {"content": "X minus Y"},
        }

    def _cluster_signature(self, results):
        # Compare clusters as sets of pattern IDs, ignoring cluster_id ordering.
        return sorted([tuple(sorted(r.patterns)) for r in results])

    def test_parallel_matches_sequential(self, patterns):
        clusterer = PatternClusterer()

        seq = clusterer.cluster_patterns(patterns, num_clusters=2, n_jobs=1)
        par = clusterer.cluster_patterns(patterns, num_clusters=2, n_jobs=2, backend="process")

        assert self._cluster_signature(seq) == self._cluster_signature(par)
