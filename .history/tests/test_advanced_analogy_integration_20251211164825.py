"""Integration Tests for Advanced Analogy Patterns - Task 4 (System Intelligence).

Tests for:
- Analytics collection and reporting
- Feedback loops and optimization triggers
- System intelligence and health metrics
- End-to-end workflows
- Integration of persistence, evolution, and intelligence layers
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
from core.advanced_analogy_patterns import (
    AnalyticsCollector,
    QueryMethod,
    QueryAnalytic,
    FeedbackLoop,
    SystemIntelligence,
)


# ============================================================================
# ANALYTICS COLLECTOR TESTS
# ============================================================================

class TestAnalyticsCollector:
    """Test suite for AnalyticsCollector."""

    @pytest.fixture
    def collector(self):
        """Provide a fresh analytics collector."""
        return AnalyticsCollector(max_history=1000)

    def test_record_single_query(self, collector):
        """Test recording a single query."""
        analytic = collector.record_query(
            query_key="king:man::queen:?",
            method_used=QueryMethod.CACHING,
            result="queen",
            confidence=0.95,
            latency_ms=5.0,
            cache_hit=True,
            success=True,
        )

        assert analytic.query_key == "king:man::queen:?"
        assert analytic.method_used == QueryMethod.CACHING
        assert analytic.result == "queen"
        assert analytic.confidence == 0.95
        assert analytic.latency_ms == 5.0
        assert analytic.cache_hit is True

    def test_analytics_deque_respects_max_history(self, collector):
        """Test that max_history limit is respected."""
        for i in range(1500):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.FUZZY,
                result="result",
                confidence=0.8,
                latency_ms=10.0,
                cache_hit=False,
            )

        assert len(collector.analytics) == 1000
        # Oldest should be from index 500 onwards
        assert collector.analytics[0].query_key == "query_500"

    def test_get_snapshot_empty(self, collector):
        """Test getting snapshot from empty collector."""
        snapshot = collector.get_snapshot()

        assert snapshot.total_queries == 0
        assert snapshot.cache_hit_rate == 0.0
        assert snapshot.avg_latency_ms == 0.0
        assert snapshot.success_rate == 0.0
        assert snapshot.avg_confidence == 0.0

    def test_get_snapshot_with_queries(self, collector):
        """Test getting snapshot with multiple queries."""
        # Record 10 queries
        for i in range(10):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.CACHING if i % 2 == 0 else QueryMethod.FUZZY,
                result=f"result_{i}",
                confidence=0.8 + (i * 0.01),
                latency_ms=10.0 + (i * 2.0),
                cache_hit=(i % 3 == 0),
                success=True,
            )

        snapshot = collector.get_snapshot()

        assert snapshot.total_queries == 10
        assert snapshot.cache_hit_rate > 0  # Some cache hits
        assert snapshot.avg_latency_ms > 10.0
        assert snapshot.success_rate == 1.0  # All successful
        assert snapshot.method_distribution['caching'] == 5
        assert snapshot.method_distribution['fuzzy'] == 5

    def test_cache_statistics_tracking(self, collector):
        """Test that cache hit/miss statistics are tracked."""
        # 7 hits, 3 misses
        for i in range(10):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.CACHING,
                result="result",
                confidence=0.9,
                latency_ms=5.0,
                cache_hit=(i < 7),
            )

        assert collector.cache_stats['hits'] == 7
        assert collector.cache_stats['misses'] == 3

    def test_method_statistics_tracking(self, collector):
        """Test that method-specific statistics are tracked."""
        # Record queries using different methods
        for i in range(5):
            collector.record_query(
                query_key=f"query_caching_{i}",
                method_used=QueryMethod.CACHING,
                result="result",
                confidence=0.95,
                latency_ms=5.0,
                cache_hit=True,
                success=True,
            )
            collector.record_query(
                query_key=f"query_fuzzy_{i}",
                method_used=QueryMethod.FUZZY,
                result="result",
                confidence=0.8,
                latency_ms=50.0,
                cache_hit=False,
                success=i > 0,  # One fails
            )

        caching_stats = collector.get_method_stats(QueryMethod.CACHING)
        fuzzy_stats = collector.get_method_stats(QueryMethod.FUZZY)

        assert caching_stats['count'] == 5
        assert caching_stats['success_rate'] == 1.0
        assert fuzzy_stats['count'] == 5
        assert fuzzy_stats['success_rate'] == 0.8

    def test_export_analytics(self, collector, tmp_path):
        """Test exporting analytics to JSON file."""
        # Record some queries
        for i in range(20):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.CACHING,
                result="result",
                confidence=0.9,
                latency_ms=10.0,
                cache_hit=(i % 2 == 0),
            )

        export_file = tmp_path / "analytics.json"
        collector.export_analytics(export_file)

        assert export_file.exists()
        data = json.loads(export_file.read_text())

        assert 'snapshot' in data
        assert 'method_stats' in data
        assert 'sample_queries' in data
        assert data['snapshot']['total_queries'] == 20

    def test_clear_analytics(self, collector):
        """Test clearing analytics."""
        # Record queries
        for i in range(10):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.CACHING,
                result="result",
                confidence=0.9,
                latency_ms=10.0,
                cache_hit=False,
            )

        assert len(collector.analytics) == 10
        assert collector.cache_stats['hits'] == 0

        collector.clear()

        assert len(collector.analytics) == 0
        assert collector.cache_stats['hits'] == 0


# ============================================================================
# FEEDBACK LOOP TESTS
# ============================================================================

class TestFeedbackLoop:
    """Test suite for FeedbackLoop."""

    @pytest.fixture
    def feedback(self):
        """Provide a fresh feedback loop."""
        return FeedbackLoop()

    def test_record_feedback_positive(self, feedback):
        """Test recording positive feedback."""
        feedback.record_feedback(
            query_key="king:man::queen:?",
            result="queen",
            user_rating=1.0,  # Excellent
            confidence=0.95,
        )

        assert len(feedback.feedback_buffer) == 1
        assert feedback.feedback_buffer[0]['user_rating'] == 1.0
        assert feedback.feedback_buffer[0]['matches_expected'] is True

    def test_record_feedback_negative(self, feedback):
        """Test recording negative feedback."""
        feedback.record_feedback(
            query_key="king:man::queen:?",
            result="princess",
            user_rating=0.2,  # Poor
            confidence=0.5,
        )

        assert len(feedback.feedback_buffer) == 1
        assert feedback.feedback_buffer[0]['user_rating'] == 0.2
        assert feedback.feedback_buffer[0]['matches_expected'] is False

    def test_feedback_buffer_max_size(self, feedback):
        """Test that feedback buffer respects max size."""
        # Record 600 feedbacks (max is 500)
        for i in range(600):
            feedback.record_feedback(
                query_key=f"query_{i}",
                result="result",
                user_rating=0.5,
                confidence=0.7,
            )

        assert len(feedback.feedback_buffer) == 500

    def test_performance_history_tracking(self, feedback):
        """Test that performance history is tracked."""
        ratings = [1.0, 0.8, 0.6, 0.9, 0.7]
        for rating in ratings:
            feedback.record_feedback(
                query_key="query",
                result="result",
                user_rating=rating,
                confidence=0.8,
            )

        assert len(feedback.performance_history) == 5
        assert list(feedback.performance_history) == ratings

    def test_trigger_optimization_with_callbacks(self, feedback):
        """Test that optimization is triggered with callbacks."""
        weight_learner = Mock()
        threshold_learner = Mock()
        feedback.pattern_weight_learner = weight_learner
        feedback.threshold_learner = threshold_learner

        # Record 100 feedbacks to trigger optimization
        for i in range(100):
            feedback.record_feedback(
                query_key=f"query_{i}",
                result="result",
                user_rating=0.8,
                confidence=0.8,
            )

        # Callbacks should have been called
        weight_learner.assert_called()
        threshold_learner.assert_called()

    def test_get_effectiveness_empty(self, feedback):
        """Test getting effectiveness metrics when empty."""
        effectiveness = feedback.get_effectiveness()

        assert effectiveness['effectiveness'] == 0.0
        assert effectiveness['feedback_count'] == 0

    def test_get_effectiveness_with_data(self, feedback):
        """Test getting effectiveness metrics with data."""
        # Mix of good and bad feedback
        for i in range(10):
            rating = 0.9 if i < 7 else 0.3  # 7 good, 3 bad
            feedback.record_feedback(
                query_key=f"query_{i}",
                result="result",
                user_rating=rating,
                confidence=0.8,
            )

        effectiveness = feedback.get_effectiveness()

        assert effectiveness['feedback_count'] == 10
        assert effectiveness['good_feedback_rate'] == 0.7  # 7/10


# ============================================================================
# SYSTEM INTELLIGENCE TESTS
# ============================================================================

class TestSystemIntelligence:
    """Test suite for SystemIntelligence."""

    @pytest.fixture
    def system(self):
        """Provide a fresh system intelligence instance."""
        return SystemIntelligence()

    def test_record_query_execution(self, system):
        """Test recording a query execution."""
        system.record_query_execution(
            query_key="king:man::queen:?",
            method=QueryMethod.CACHING,
            result="queen",
            confidence=0.95,
            latency_ms=5.0,
            cache_hit=True,
        )

        assert system.analytics.cache_stats['hits'] == 1

    def test_submit_user_feedback(self, system):
        """Test submitting user feedback."""
        system.submit_user_feedback(
            query_key="king:man::queen:?",
            result="queen",
            rating=1.0,
            confidence=0.95,
        )

        assert len(system.feedback.feedback_buffer) == 1

    def test_generate_recommendations_empty(self, system):
        """Test generating recommendations from empty system."""
        recommendations = system.generate_recommendations()

        assert isinstance(recommendations, list)

    def test_generate_recommendations_low_cache_hit(self, system):
        """Test recommendation for low cache hit rate."""
        # Record 100+ queries with low cache hit rate
        for i in range(120):
            system.record_query_execution(
                query_key=f"query_{i}",
                method=QueryMethod.FUZZY,
                result="result",
                confidence=0.8,
                latency_ms=50.0,
                cache_hit=(i < 30),  # Only 25% hit rate
            )

        recommendations = system.generate_recommendations()
        cache_recs = [r for r in recommendations if r['type'] == 'cache_efficiency']

        assert len(cache_recs) > 0

    def test_generate_recommendations_high_latency(self, system):
        """Test recommendation for high latency."""
        # Record queries with high latency
        for i in range(50):
            system.record_query_execution(
                query_key=f"query_{i}",
                method=QueryMethod.FUZZY,
                result="result",
                confidence=0.8,
                latency_ms=150.0,  # High latency
                cache_hit=False,
            )

        recommendations = system.generate_recommendations()
        latency_recs = [r for r in recommendations if r['type'] == 'latency']

        assert len(latency_recs) > 0

    def test_get_system_health_empty(self, system):
        """Test getting system health from empty system."""
        health = system.get_system_health()

        assert 'overall_score' in health
        assert 'factors' in health
        assert 'snapshot' in health
        assert 'feedback_effectiveness' in health

    def test_get_system_health_good(self, system):
        """Test getting system health with good metrics."""
        # Record queries with good performance
        for i in range(50):
            system.record_query_execution(
                query_key=f"query_{i}",
                method=QueryMethod.CACHING,
                result="result",
                confidence=0.95,
                latency_ms=5.0,
                cache_hit=(i % 2 == 0),  # 50% hit rate
            )

        health = system.get_system_health()

        assert health['overall_score'] > 10  # Reasonable health score
        assert len(health['factors']) == 4

    def test_export_system_report(self, system, tmp_path):
        """Test exporting system report."""
        # Record some queries
        for i in range(30):
            system.record_query_execution(
                query_key=f"query_{i}",
                method=QueryMethod.CACHING,
                result="result",
                confidence=0.9,
                latency_ms=10.0,
                cache_hit=(i % 2 == 0),
            )

        report_file = tmp_path / "system_report.json"
        system.export_system_report(report_file)

        assert report_file.exists()
        data = json.loads(report_file.read_text())

        assert 'timestamp' in data
        assert 'system_health' in data
        assert 'recommendations' in data
        assert 'analytics' in data
        assert 'feedback_effectiveness' in data


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end tests for complete workflows."""

    def test_full_workflow_with_analytics_and_feedback(self):
        """Test complete workflow: query → analytics → feedback → recommendations."""
        system = SystemIntelligence()

        # Simulate 100 queries with varying cache hit rates
        for i in range(100):
            cache_hit = i < 50  # First 50 hit, rest miss
            system.record_query_execution(
                query_key=f"query_{i}",
                method=QueryMethod.CACHING if i % 2 == 0 else QueryMethod.FUZZY,
                result=f"result_{i}",
                confidence=0.85 + (i % 15) / 100,
                latency_ms=5.0 if cache_hit else 50.0,
                cache_hit=cache_hit,
            )

        # Get analytics
        snapshot = system.analytics.get_snapshot()
        assert snapshot.total_queries == 100

        # Submit some user feedback
        for i in range(20):
            system.submit_user_feedback(
                query_key=f"query_{i}",
                result=f"result_{i}",
                rating=0.95,  # Good feedback
                confidence=0.9,
            )

        # Get recommendations
        recommendations = system.generate_recommendations()
        assert len(recommendations) > 0

        # Get system health
        health = system.get_system_health()
        assert health['overall_score'] > 0

    def test_analytics_driven_optimization(self):
        """Test that analytics guide optimization decisions."""
        system = SystemIntelligence()

        # Phase 1: Poor performance (high latency, low cache hits)
        for i in range(50):
            system.record_query_execution(
                query_key=f"query_phase1_{i}",
                method=QueryMethod.FUZZY,
                result="result",
                confidence=0.7,
                latency_ms=200.0,  # Very high
                cache_hit=False,  # No hits
            )

        # Phase 2: Better performance (lower latency, higher cache hits)
        for i in range(50):
            system.record_query_execution(
                query_key=f"query_phase2_{i}",
                method=QueryMethod.CACHING,
                result="result",
                confidence=0.9,
                latency_ms=10.0,  # Much lower
                cache_hit=(i % 2 == 0),  # 50% hits
            )

        # System should show improvement
        health = system.get_system_health()
        assert health['overall_score'] > 0

        # Recommendations should reflect optimization opportunities
        recommendations = system.generate_recommendations()
        # Should have recommendations for improvement
        assert isinstance(recommendations, list)

    def test_feedback_triggered_relearning(self):
        """Test that accumulated feedback triggers relearning."""
        relearning_triggered = {'weight': False, 'threshold': False}

        def mock_weight_learner(feedback):
            relearning_triggered['weight'] = True

        def mock_threshold_learner(feedback):
            relearning_triggered['threshold'] = True

        feedback_loop = FeedbackLoop(
            pattern_weight_learner=mock_weight_learner,
            threshold_learner=mock_threshold_learner,
        )

        # Record 100 feedbacks to trigger optimization
        for i in range(100):
            feedback_loop.record_feedback(
                query_key=f"query_{i}",
                result="result",
                user_rating=0.9 if i % 2 == 0 else 0.7,
                confidence=0.8,
            )

        assert relearning_triggered['weight'] is True
        assert relearning_triggered['threshold'] is True

    def test_multi_method_analytics_tracking(self):
        """Test analytics tracking across multiple methods."""
        system = SystemIntelligence()

        methods = [
            QueryMethod.CACHING,
            QueryMethod.FUZZY,
            QueryMethod.INVERSE,
            QueryMethod.CHAINING,
            QueryMethod.COMPOSITION,
        ]

        # Record queries using each method
        for method in methods:
            for i in range(10):
                system.record_query_execution(
                    query_key=f"query_{method.value}_{i}",
                    method=method,
                    result="result",
                    confidence=0.85,
                    latency_ms=50.0,
                    cache_hit=method == QueryMethod.CACHING,
                )

        # Verify all methods are tracked
        snapshot = system.analytics.get_snapshot()
        assert len(snapshot.method_distribution) == 5
        assert all(count == 10 for count in snapshot.method_distribution.values())

    def test_system_persistence_across_sessions(self, tmp_path):
        """Test that system state can be exported and analyzed across sessions."""
        system = SystemIntelligence()

        # Session 1: Record queries
        for i in range(50):
            system.record_query_execution(
                query_key=f"session1_query_{i}",
                method=QueryMethod.CACHING,
                result="result",
                confidence=0.9,
                latency_ms=10.0,
                cache_hit=(i % 2 == 0),
            )

        # Export session 1
        report_file = tmp_path / "session1_report.json"
        system.export_system_report(report_file)

        # Session 2: Create new system and continue
        system2 = SystemIntelligence()
        for i in range(50):
            system2.record_query_execution(
                query_key=f"session2_query_{i}",
                method=QueryMethod.CACHING,
                result="result",
                confidence=0.92,
                latency_ms=12.0,
                cache_hit=(i % 3 == 0),
            )

        # Export session 2
        report_file2 = tmp_path / "session2_report.json"
        system2.export_system_report(report_file2)

        # Both reports should exist
        assert report_file.exists()
        assert report_file2.exists()

        # Compare them
        data1 = json.loads(report_file.read_text())
        data2 = json.loads(report_file2.read_text())

        assert data1['analytics']['total_queries'] == 50
        assert data2['analytics']['total_queries'] == 50


# ============================================================================
# STRESS AND EDGE CASE TESTS
# ============================================================================

class TestStressAndEdgeCases:
    """Stress tests and edge case handling."""

    def test_high_volume_analytics(self):
        """Test analytics under high query volume."""
        system = SystemIntelligence()

        # Process 5000 queries
        start_time = time.time()
        for i in range(5000):
            system.record_query_execution(
                query_key=f"query_{i}",
                method=QueryMethod.CACHING if i % 5 == 0 else QueryMethod.FUZZY,
                result="result",
                confidence=0.8 + (i % 20) / 100,
                latency_ms=10.0 + (i % 100) / 10,
                cache_hit=(i % 3 == 0),
            )

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 10.0  # 10 seconds for 5000 queries

        # Analytics should work
        snapshot = system.analytics.get_snapshot()
        assert snapshot.total_queries == 1000  # Only last 1000 kept

    def test_extreme_confidence_values(self):
        """Test analytics with extreme confidence values."""
        system = SystemIntelligence()

        # Very low confidence
        system.record_query_execution(
            query_key="query_low",
            method=QueryMethod.FUZZY,
            result="result",
            confidence=0.01,
            latency_ms=50.0,
            cache_hit=False,
        )

        # Very high confidence
        system.record_query_execution(
            query_key="query_high",
            method=QueryMethod.CACHING,
            result="result",
            confidence=0.99,
            latency_ms=5.0,
            cache_hit=True,
        )

        snapshot = system.analytics.get_snapshot()
        assert 0.0 <= snapshot.avg_confidence <= 1.0

    def test_extreme_latency_values(self):
        """Test analytics with extreme latency values."""
        system = SystemIntelligence()

        # Very fast
        system.record_query_execution(
            query_key="query_fast",
            method=QueryMethod.CACHING,
            result="result",
            confidence=0.95,
            latency_ms=0.1,
            cache_hit=True,
        )

        # Very slow
        system.record_query_execution(
            query_key="query_slow",
            method=QueryMethod.COMPOSITION,
            result="result",
            confidence=0.8,
            latency_ms=10000.0,
            cache_hit=False,
        )

        snapshot = system.analytics.get_snapshot()
        assert snapshot.avg_latency_ms > 0

    def test_all_failed_queries(self):
        """Test analytics when all queries fail."""
        collector = AnalyticsCollector()

        for i in range(20):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.FUZZY,
                result="result",
                confidence=0.5,
                latency_ms=100.0,
                cache_hit=False,
                success=False,  # All fail
            )

        snapshot = collector.get_snapshot()
        assert snapshot.success_rate == 0.0

    def test_all_successful_queries(self):
        """Test analytics when all queries succeed."""
        collector = AnalyticsCollector()

        for i in range(20):
            collector.record_query(
                query_key=f"query_{i}",
                method_used=QueryMethod.CACHING,
                result="result",
                confidence=0.95,
                latency_ms=5.0,
                cache_hit=True,
                success=True,  # All succeed
            )

        snapshot = collector.get_snapshot()
        assert snapshot.success_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
