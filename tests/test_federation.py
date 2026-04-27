"""
Tests for Federated Codebook Learning - Phase 7 Track 2

Tests privacy mechanisms, client training, server aggregation,
and consensus protocol.
"""

import numpy as np
import pytest

from sigmalang.federation.aggregation_server import AggregationServer
from sigmalang.federation.client import FederationClient, LocalCodebook
from sigmalang.federation.consensus import ConsensusProtocol
from sigmalang.federation.privacy import DifferentialPrivacy, PrivacyConfig, SecureAggregator

# =============================================================================
# Privacy Tests
# =============================================================================

class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""

    def test_clip_gradients(self):
        dp = DifferentialPrivacy(PrivacyConfig(max_grad_norm=1.0))
        grads = np.array([[3.0, 4.0]], dtype=np.float32)  # norm = 5
        clipped = dp.clip_gradients(grads)
        norm = np.linalg.norm(clipped[0])
        assert norm <= 1.0 + 1e-6

    def test_clip_already_small(self):
        dp = DifferentialPrivacy(PrivacyConfig(max_grad_norm=10.0))
        grads = np.array([[0.1, 0.2]], dtype=np.float32)
        clipped = dp.clip_gradients(grads)
        np.testing.assert_allclose(clipped, grads, atol=1e-6)

    def test_add_noise_changes_values(self):
        dp = DifferentialPrivacy(PrivacyConfig(noise_multiplier=1.0))
        original = np.ones(64, dtype=np.float32)
        noisy = dp.add_noise(original, num_samples=10)
        assert not np.allclose(original, noisy)

    def test_privatize_update(self):
        dp = DifferentialPrivacy(PrivacyConfig(
            epsilon=1.0, noise_multiplier=0.5, max_grad_norm=1.0
        ))
        update = np.random.randn(256, 64).astype(np.float32) * 5
        privatized = dp.privatize_update(update, num_local_samples=100)

        assert privatized.shape == update.shape
        assert dp.accountant.rounds_completed == 1
        assert dp.accountant.spent_epsilon > 0

    def test_budget_exhaustion(self):
        config = PrivacyConfig(epsilon=0.01, total_rounds=1, noise_multiplier=0.1)
        dp = DifferentialPrivacy(config)

        # Exhaust budget
        for _ in range(10):
            dp.privatize_update(np.ones(10, dtype=np.float32), 1)

        # Should return zeros when exhausted
        result = dp.privatize_update(np.ones(10, dtype=np.float32), 1)
        if dp.accountant.budget_exhausted:
            np.testing.assert_array_equal(result, np.zeros(10))

    def test_privacy_report(self):
        dp = DifferentialPrivacy()
        dp.privatize_update(np.ones(10, dtype=np.float32), 1)
        report = dp.get_privacy_report()
        assert 'epsilon_spent' in report
        assert 'rounds_completed' in report
        assert report['rounds_completed'] == 1


class TestSecureAggregator:
    """Test secure aggregation."""

    def test_minimum_participants(self):
        agg = SecureAggregator(min_participants=3)
        agg.submit_update("a", np.ones(10))
        agg.submit_update("b", np.ones(10))
        assert not agg.can_aggregate()
        assert agg.aggregate() is None

    def test_successful_aggregation(self):
        agg = SecureAggregator(min_participants=2)
        agg.submit_update("a", np.ones(10) * 2)
        agg.submit_update("b", np.ones(10) * 4)
        assert agg.can_aggregate()

        result = agg.aggregate()
        assert result is not None
        np.testing.assert_allclose(result, np.ones(10) * 3)

    def test_duplicate_submission_ignored(self):
        agg = SecureAggregator(min_participants=2)
        agg.submit_update("a", np.ones(10))
        agg.submit_update("a", np.ones(10) * 2)  # Duplicate
        assert agg.pending_count == 1


# =============================================================================
# Client Tests
# =============================================================================

class TestLocalCodebook:
    """Test local codebook training."""

    def test_train_on_batch(self):
        cb = LocalCodebook(size=32, dim=16)
        data = np.random.randn(100, 16).astype(np.float32)
        metrics = cb.train_on_batch(data, lr=0.1)
        assert 'avg_distance' in metrics
        assert 'utilization' in metrics
        assert metrics['utilization'] > 0

    def test_compute_delta(self):
        cb = LocalCodebook(size=32, dim=16)
        cb.snapshot()
        data = np.random.randn(100, 16).astype(np.float32)
        cb.train_on_batch(data, lr=0.5)
        delta = cb.compute_delta()
        assert delta is not None
        assert np.linalg.norm(delta) > 0


class TestFederationClient:
    """Test federation client."""

    def test_train_local(self):
        client = FederationClient(node_id="test-1", codebook_size=32, codebook_dim=16)
        data = np.random.randn(200, 16).astype(np.float32)
        result = client.train_local(data, epochs=2, batch_size=32)
        assert result['node_id'] == "test-1"
        assert result['samples'] == 200

    def test_compute_update(self):
        client = FederationClient(node_id="test-1", codebook_size=32, codebook_dim=16)
        data = np.random.randn(100, 16).astype(np.float32)
        client.train_local(data)
        update = client.compute_update()
        assert update is not None
        assert update.shape == (32, 16)

    def test_apply_global_update(self):
        client = FederationClient(node_id="test-1", codebook_size=32, codebook_dim=16)
        before = client.codebook.embeddings.copy()
        global_update = np.random.randn(32, 16).astype(np.float32) * 0.01
        client.apply_global_update(global_update, merge_weight=0.5)
        assert not np.allclose(before, client.codebook.embeddings)

    def test_fingerprint(self):
        client = FederationClient(node_id="test-1", codebook_size=32, codebook_dim=16)
        fp = client.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_status(self):
        client = FederationClient(node_id="test-1", codebook_size=32, codebook_dim=16)
        status = client.get_status()
        assert status['node_id'] == "test-1"
        assert 'privacy' in status


# =============================================================================
# Aggregation Server Tests
# =============================================================================

class TestAggregationServer:
    """Test aggregation server."""

    def test_receive_update(self):
        server = AggregationServer(min_clients=2)
        update = np.random.randn(32, 16).astype(np.float32)
        assert server.receive_update("node-1", update, weight=100)
        assert not server.can_aggregate()

    def test_aggregate_round(self):
        server = AggregationServer(min_clients=2, use_secure_aggregation=False)
        u1 = np.ones((32, 16), dtype=np.float32) * 0.1
        u2 = np.ones((32, 16), dtype=np.float32) * 0.3
        server.receive_update("node-1", u1, weight=1.0)
        server.receive_update("node-2", u2, weight=1.0)

        result = server.aggregate_round()
        assert result is not None
        assert result.num_participants == 2

    def test_reject_stale_update(self):
        server = AggregationServer(min_clients=1, max_staleness=1)
        update = np.ones(10, dtype=np.float32)
        # Advance server round
        server._current_round = 5
        assert not server.receive_update("node-1", update, client_round=1)

    def test_convergence_check(self):
        server = AggregationServer(min_clients=1, use_secure_aggregation=False)
        for i in range(10):
            tiny = np.ones(10, dtype=np.float32) * 1e-6
            server.receive_update(f"node-{i}", tiny, weight=1.0)
            server.aggregate_round()
        assert server.is_converged(threshold=1e-3)

    def test_status(self):
        server = AggregationServer(min_clients=2)
        status = server.get_status()
        assert 'current_round' in status
        assert 'pending_updates' in status


# =============================================================================
# Consensus Tests
# =============================================================================

class TestConsensusProtocol:
    """Test consensus protocol."""

    def test_propose_groups_similar(self):
        protocol = ConsensusProtocol(num_nodes=5, similarity_threshold=0.9)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similar = np.array([0.99, 0.1, 0.0], dtype=np.float32)

        g1 = protocol.propose("node-1", vec, {"pattern": "hello"})
        g2 = protocol.propose("node-2", similar, {"pattern": "hello"})
        assert g1 == g2  # Should be same group

    def test_consensus_reached(self):
        protocol = ConsensusProtocol(num_nodes=5, approval_threshold=0.6)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for i in range(3):
            protocol.propose(f"node-{i}", vec, weight=1.0)

        approved = protocol.check_consensus()
        assert len(approved) == 1
        assert approved[0].approved

    def test_consensus_not_reached(self):
        protocol = ConsensusProtocol(num_nodes=10, approval_threshold=0.6)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        protocol.propose("node-1", vec, weight=1.0)
        protocol.propose("node-2", vec, weight=1.0)

        approved = protocol.check_consensus()
        assert len(approved) == 0

    def test_byzantine_tolerance(self):
        protocol = ConsensusProtocol(
            num_nodes=6, approval_threshold=0.5, byzantine_tolerance=True
        )
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Need > 2/3 for byzantine tolerance
        for i in range(4):
            protocol.propose(f"node-{i}", vec, weight=1.0)

        approved = protocol.check_consensus()
        assert len(approved) == 1

    def test_duplicate_vote_updates(self):
        protocol = ConsensusProtocol(num_nodes=3, approval_threshold=0.6)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        protocol.propose("node-1", vec, weight=1.0)
        protocol.propose("node-1", vec * 0.99, weight=2.0)  # Update

        pending = protocol.get_pending_proposals()
        for group_info in pending.values():
            assert group_info['vote_count'] == 1  # Not 2

    def test_reset_round(self):
        protocol = ConsensusProtocol(num_nodes=5)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        protocol.propose("node-1", vec)
        protocol.reset_round()
        assert len(protocol.get_pending_proposals()) == 0


# =============================================================================
# Integration Test: Full Federation Round
# =============================================================================

class TestFederationIntegration:
    """End-to-end federation test."""

    def test_full_federation_round(self):
        """Simulate a complete federation round with 3 clients."""
        # Setup
        server = AggregationServer(min_clients=3, use_secure_aggregation=False)
        clients = [
            FederationClient(f"node-{i}", codebook_size=32, codebook_dim=16)
            for i in range(3)
        ]

        # Each client trains on local data
        for client in clients:
            local_data = np.random.randn(100, 16).astype(np.float32)
            client.train_local(local_data, epochs=2, batch_size=32)

        # Each client computes and submits update
        for client in clients:
            update = client.compute_update()
            assert update is not None
            server.receive_update(client.node_id, update, weight=100)

        # Server aggregates
        assert server.can_aggregate()
        result = server.aggregate_round()
        assert result is not None

        # Clients apply global update
        global_update = result.global_update
        for client in clients:
            client.apply_global_update(global_update, merge_weight=0.3)

        # Verify clients can continue training
        for client in clients:
            new_data = np.random.randn(50, 16).astype(np.float32)
            summary = client.train_local(new_data)
            assert summary['round'] == 1
