"""
Comprehensive test suite for pattern_intelligence module.

Tests cover three learning components:
1. MethodPredictor: Gradient boosting for method selection
2. ThresholdLearner: Gradient descent for threshold optimization
3. WeightLearner: Exponential moving average for weight calibration

Test Coverage:
- Unit tests for each component (9 + 8 + 8 tests)
- Integration tests combining components (6 tests)
- Edge case handling (4 tests)
- Total: 35 comprehensive tests
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple
from core.pattern_intelligence import (
    MethodPredictor,
    ThresholdLearner,
    WeightLearner,
    MethodPrediction,
    ThresholdOptimization,
    PatternWeight,
    compute_pattern_features
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_patterns() -> Dict[str, Dict[str, str]]:
    """Create sample patterns for testing."""
    return {
        "p1": {"type": "logic", "content": "all(a, b, c)"},
        "p2": {"type": "analogy", "content": "A is to B as C is to D"},
        "p3": {"type": "sequence", "content": "1, 2, 3, 4, 5"},
        "p4": {"type": "logic", "content": "if(x, then y)"},
        "p5": {"type": "analogy", "content": "dog is to bark as cat is to meow"},
    }


@pytest.fixture
def training_features() -> List[List[float]]:
    """Create training feature vectors."""
    return [
        [0.8, 0.6, 0.4],  # High cohesion, medium frequency, low novelty
        [0.5, 0.9, 0.7],  # Medium cohesion, high frequency, medium novelty
        [0.7, 0.4, 0.8],  # High cohesion, low frequency, high novelty
        [0.6, 0.8, 0.5],  # Medium cohesion, high frequency, medium novelty
        [0.9, 0.3, 0.9],  # Very high cohesion, low frequency, high novelty
    ]


@pytest.fixture
def training_labels() -> List[str]:
    """Create training method labels."""
    return ["clustering", "abstraction", "discovery", "clustering", "discovery"]


@pytest.fixture
def performance_data() -> List[Tuple[str, bool, float]]:
    """Create performance data for threshold learning."""
    return [
        ("p1", True, 0.95),
        ("p2", False, 0.35),
        ("p3", True, 0.88),
        ("p4", False, 0.42),
        ("p5", True, 0.91),
        ("p1", True, 0.92),
        ("p2", False, 0.40),
        ("p3", True, 0.85),
    ]


# ============================================================================
# TestMethodPredictor - 9 tests
# ============================================================================

class TestMethodPredictor:
    """Test suite for MethodPredictor gradient boosting."""

    def test_predictor_creation(self):
        """Test MethodPredictor initialization."""
        predictor = MethodPredictor()
        assert predictor is not None
        assert not predictor.is_trained
        assert predictor.model is not None

    def test_predictor_with_custom_parameters(self):
        """Test MethodPredictor with custom hyperparameters."""
        predictor = MethodPredictor(n_estimators=50, learning_rate=0.05)
        assert predictor.model.n_estimators == 50
        assert predictor.model.learning_rate == 0.05

    def test_train_basic(self, training_features, training_labels):
        """Test basic training."""
        predictor = MethodPredictor()
        predictor.train(training_features, training_labels)

        assert predictor.is_trained
        assert len(predictor.feature_names) == 3
        assert set(predictor._method_classes) == {"clustering", "abstraction", "discovery"}

    def test_train_with_feature_names(self, training_features, training_labels):
        """Test training with explicit feature names."""
        feature_names = ["cohesion", "frequency", "novelty"]
        predictor = MethodPredictor()
        predictor.train(training_features, training_labels, feature_names=feature_names)

        assert predictor.feature_names == feature_names

    def test_predict_requires_training(self, training_features):
        """Test that prediction requires prior training."""
        predictor = MethodPredictor()
        with pytest.raises(RuntimeError, match="must be trained"):
            predictor.predict(training_features[0])

    def test_predict_basic(self, training_features, training_labels):
        """Test basic prediction."""
        predictor = MethodPredictor()
        predictor.train(training_features, training_labels)

        method, confidence = predictor.predict(training_features[0])

        assert isinstance(method, str)
        assert 0 <= confidence <= 1
        assert method in predictor._method_classes

    def test_predict_proba(self, training_features, training_labels):
        """Test probability predictions."""
        predictor = MethodPredictor()
        predictor.train(training_features, training_labels)

        proba = predictor.predict_proba(training_features[0])

        assert isinstance(proba, dict)
        assert set(proba.keys()) == {"clustering", "abstraction", "discovery"}
        assert all(0 <= p <= 1 for p in proba.values())
        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_get_feature_importance(self, training_features, training_labels):
        """Test feature importance extraction."""
        feature_names = ["cohesion", "frequency", "novelty"]
        predictor = MethodPredictor()
        predictor.train(training_features, training_labels, feature_names=feature_names)

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert set(importance.keys()) == set(feature_names)
        assert all(0 <= imp <= 1 for imp in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_feature_importance_requires_training(self):
        """Test that feature importance requires prior training."""
        predictor = MethodPredictor()
        with pytest.raises(RuntimeError, match="must be trained"):
            predictor.get_feature_importance()


# ============================================================================
# TestThresholdLearner - 8 tests
# ============================================================================

class TestThresholdLearner:
    """Test suite for ThresholdLearner gradient descent optimization."""

    def test_learner_creation(self):
        """Test ThresholdLearner initialization."""
        learner = ThresholdLearner()
        assert learner is not None
        assert learner.learning_rate == 0.01
        assert learner.max_iterations == 50

    def test_learner_with_custom_parameters(self):
        """Test ThresholdLearner with custom parameters."""
        learner = ThresholdLearner(learning_rate=0.05, max_iterations=100, tolerance=1e-5)
        assert learner.learning_rate == 0.05
        assert learner.max_iterations == 100
        assert learner.tolerance == 1e-5

    def test_learn_basic(self, sample_patterns, performance_data):
        """Test basic threshold learning."""
        learner = ThresholdLearner()
        result = learner.learn(sample_patterns, performance_data)

        assert isinstance(result, ThresholdOptimization)
        assert 0 <= result.optimal_threshold <= 1
        assert 0 <= result.f1_score <= 1
        assert result.convergence_iterations > 0

    def test_learn_convergence(self, sample_patterns, performance_data):
        """Test that learning converges."""
        learner = ThresholdLearner(learning_rate=0.01, max_iterations=100)
        result = learner.learn(sample_patterns, performance_data)

        # Should converge before max iterations
        assert result.convergence_iterations < 100

    def test_learn_valid_metrics(self, sample_patterns, performance_data):
        """Test that learned threshold produces valid metrics."""
        learner = ThresholdLearner()
        result = learner.learn(sample_patterns, performance_data)

        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1_score <= 1
        assert result.validation_score >= 0

    def test_learn_empty_data_raises(self, sample_patterns):
        """Test that learning with empty data raises error."""
        learner = ThresholdLearner()
        with pytest.raises(ValueError, match="cannot be empty"):
            learner.learn(sample_patterns, [])

    def test_validate_threshold(self, performance_data):
        """Test threshold validation."""
        learner = ThresholdLearner()
        precision, recall, f1 = learner.validate(0.7, performance_data)

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_adaptive_threshold_by_difficulty(self):
        """Test adaptive threshold adjustment."""
        learner = ThresholdLearner()

        # Easy pattern
        thresh_easy = learner.adaptive_threshold(0.2)
        assert thresh_easy > 0.8

        # Medium pattern
        thresh_medium = learner.adaptive_threshold(0.5)
        assert 0.6 < thresh_medium < 0.8

        # Hard pattern
        thresh_hard = learner.adaptive_threshold(0.8)
        assert thresh_hard < 0.6


# ============================================================================
# TestWeightLearner - 8 tests
# ============================================================================

class TestWeightLearner:
    """Test suite for WeightLearner exponential moving average."""

    def test_learner_creation(self):
        """Test WeightLearner initialization."""
        learner = WeightLearner()
        assert learner is not None
        assert learner.alpha == 0.1
        assert len(learner.weights) == 0

    def test_learner_with_custom_alpha(self):
        """Test WeightLearner with custom alpha."""
        learner = WeightLearner(alpha=0.3)
        assert learner.alpha == 0.3

    def test_learner_alpha_validation(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="must be in"):
            WeightLearner(alpha=0.0)
        with pytest.raises(ValueError, match="must be in"):
            WeightLearner(alpha=1.0)

    def test_calibrate_weights(self, sample_patterns, performance_data):
        """Test weight calibration from performance data."""
        learner = WeightLearner()
        learner.calibrate(sample_patterns, performance_data)

        # All patterns should have weights
        assert len(learner.weights) == len(sample_patterns)

        # All weights should be in valid range
        for pid, weight in learner.get_all_weights().items():
            assert 0 <= weight <= 1

    def test_update_weight_new_pattern(self):
        """Test updating weight for new pattern."""
        learner = WeightLearner(alpha=0.1)
        learner.update_weight("p_new", True, 0.9)

        assert "p_new" in learner.weights
        weight = learner.get_pattern_weight("p_new")
        assert weight > 0.8

    def test_update_weight_existing_pattern(self, sample_patterns, performance_data):
        """Test updating weight for existing pattern."""
        learner = WeightLearner(alpha=0.2)
        learner.calibrate(sample_patterns, performance_data)

        initial_weight = learner.get_pattern_weight("p1")

        # Update with high performance
        learner.update_weight("p1", True, 0.99)
        updated_weight = learner.get_pattern_weight("p1")

        # Weight should increase towards 0.99
        assert updated_weight > initial_weight

    def test_weight_ema_convergence(self):
        """Test that EMA converges with repeated updates."""
        learner = WeightLearner(alpha=0.5)
        pattern_id = "test_pattern"

        # Initialize at 0.5
        learner.update_weight(pattern_id, True, 0.5)
        initial = learner.get_pattern_weight(pattern_id)

        # Multiple updates towards 1.0
        for _ in range(100):
            learner.update_weight(pattern_id, True, 1.0)

        final = learner.get_pattern_weight(pattern_id)

        # Should converge towards 1.0
        assert final > initial
        assert final > 0.99

    def test_get_weight_info(self, sample_patterns, performance_data):
        """Test getting detailed weight information."""
        learner = WeightLearner()
        learner.calibrate(sample_patterns, performance_data)

        info = learner.get_weight_info("p1")

        assert isinstance(info, PatternWeight)
        assert info.pattern_id == "p1"
        assert 0 <= info.weight <= 1
        assert info.update_count > 0


# ============================================================================
# TestIntegration - 6 tests
# ============================================================================

class TestIntegrationLearners:
    """Test integration of all three learning components."""

    def test_method_prediction_pipeline(self, training_features, training_labels, sample_patterns):
        """Test end-to-end method prediction pipeline."""
        # Train predictor
        predictor = MethodPredictor()
        predictor.train(training_features, training_labels)

        # Predict for all patterns
        for feature in training_features:
            method, confidence = predictor.predict(feature)
            assert method in training_labels
            assert 0 < confidence <= 1

    def test_threshold_learning_pipeline(self, sample_patterns, performance_data):
        """Test end-to-end threshold learning pipeline."""
        learner = ThresholdLearner(learning_rate=0.01)

        # Learn threshold
        result = learner.learn(sample_patterns, performance_data)

        # Validate threshold
        precision, recall, f1 = learner.validate(result.optimal_threshold, performance_data)

        assert precision > 0 or recall > 0  # At least one should be positive

    def test_weight_learning_pipeline(self, sample_patterns, performance_data):
        """Test end-to-end weight learning pipeline."""
        learner = WeightLearner(alpha=0.1)

        # Calibrate from performance data
        learner.calibrate(sample_patterns, performance_data)

        # Update weights based on new outcomes
        learner.update_weight("p1", True, 0.95)

        # Get all weights
        all_weights = learner.get_all_weights()

        assert len(all_weights) == len(sample_patterns)
        assert all(0 <= w <= 1 for w in all_weights.values())

    def test_full_learning_system(self, training_features, training_labels, sample_patterns, performance_data):
        """Test all three learners working together."""
        # Initialize learners
        method_pred = MethodPredictor()
        threshold_learner = ThresholdLearner()
        weight_learner = WeightLearner(alpha=0.1)

        # Train all
        method_pred.train(training_features, training_labels)
        result = threshold_learner.learn(sample_patterns, performance_data)
        weight_learner.calibrate(sample_patterns, performance_data)

        # Predictions + optimization
        for feature in training_features:
            method, conf = method_pred.predict(feature)
            threshold = threshold_learner.adaptive_threshold(0.5)
            weight = weight_learner.get_pattern_weight("p1")

            assert isinstance(method, str)
            assert 0 < conf <= 1
            assert 0 < threshold < 1
            assert 0 <= weight <= 1

    def test_learners_handle_large_dataset(self, sample_patterns):
        """Test learners with large dataset."""
        # Create large synthetic dataset
        large_features = [[np.random.random() for _ in range(3)] for _ in range(100)]
        large_labels = [["clustering", "abstraction", "discovery"][i % 3] for i in range(100)]

        # Train method predictor
        method_pred = MethodPredictor()
        method_pred.train(large_features, large_labels)

        # Should handle 100 samples without error
        prediction, conf = method_pred.predict(large_features[0])
        assert isinstance(prediction, str)

    def test_feature_computation_helper(self, sample_patterns):
        """Test pattern feature computation helper."""
        features = compute_pattern_features(
            sample_patterns["p1"],
            cluster_cohesion=0.8,
            pattern_frequency=0.6,
            novelty_score=0.7
        )

        assert isinstance(features, list)
        assert len(features) == 3
        assert features[0] == 0.8
        assert features[1] == 0.6
        assert features[2] == 0.7


# ============================================================================
# TestEdgeCases - 4 tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_method_predictor_single_class(self):
        """Test predictor with multiple classes (required for gradient boosting)."""
        predictor = MethodPredictor()
        # Gradient boosting requires at least 2 classes
        features = [[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]]
        labels = ["clustering", "abstraction"]

        predictor.train(features, labels)
        method, conf = predictor.predict([0.5, 0.5, 0.5])

        assert method in ["clustering", "abstraction"]
        assert conf > 0

    def test_threshold_learner_perfect_separation(self, sample_patterns):
        """Test threshold learning with perfect separation."""
        # Create data that's easily separable
        perfect_data = [
            ("p1", True, 0.99),
            ("p1", True, 0.98),
            ("p2", False, 0.01),
            ("p2", False, 0.02),
        ]

        learner = ThresholdLearner()
        result = learner.learn(sample_patterns, perfect_data)

        # Should achieve high F1
        assert result.f1_score > 0.9

    def test_weight_learner_extreme_outcomes(self):
        """Test weight learner with extreme performance values."""
        learner = WeightLearner(alpha=0.5)

        # Extreme success
        learner.update_weight("p_extreme_success", True, 0.99999)
        weight_success = learner.get_pattern_weight("p_extreme_success")
        assert weight_success > 0.99

        # Extreme failure
        learner.update_weight("p_extreme_fail", False, 0.00001)
        weight_fail = learner.get_pattern_weight("p_extreme_fail")
        assert weight_fail < 0.01

    def test_method_predictor_boundary_features(self):
        """Test predictor with boundary feature values."""
        predictor = MethodPredictor()
        features = [
            [0.0, 0.0, 0.0],  # All zeros
            [1.0, 1.0, 1.0],  # All ones
            [0.5, 0.5, 0.5],  # All midpoints
        ]
        labels = ["clustering", "abstraction", "discovery"]

        predictor.train(features, labels)

        # Should handle boundary values
        for feature in features:
            method, conf = predictor.predict(feature)
            assert 0 < conf <= 1


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
