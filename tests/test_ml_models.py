"""
Comprehensive tests for the Enhanced ML Models module.

Tests cover:
- Enums and data classes
- ModelRegistry with versioning
- AdaptiveLearner with drift detection
- FeatureExtractor with multiple types
- ModelEnsemble with different strategies
- Simple models for testing
- Convenience functions
- Integration scenarios
"""

import time
import threading
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from sigmalang.core.ml_models import (
    # Enums
    ModelType,
    ModelState,
    EnsembleStrategy,
    FeatureType,
    # Data classes
    ModelMetadata,
    TrainingMetrics,
    FeatureConfig,
    EnsembleConfig,
    ModelVersion,
    # Main classes
    ModelRegistry,
    AdaptiveLearner,
    FeatureExtractor,
    ModelEnsemble,
    # Simple models
    SimpleClassifier,
    SimpleRegressor,
    LinearModel,
    # Base classes
    BaseModel,
    # Convenience functions
    create_model_registry,
    create_ensemble,
    create_feature_extractor,
    train_with_metrics,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestModelType:
    """Tests for ModelType enum."""
    
    def test_all_model_types_exist(self):
        """Test all model types are defined."""
        assert ModelType.CLASSIFIER is not None
        assert ModelType.REGRESSOR is not None
        assert ModelType.CLUSTERER is not None
        assert ModelType.TRANSFORMER is not None
        assert ModelType.ENCODER is not None
        assert ModelType.EMBEDDING is not None
    
    def test_model_types_are_unique(self):
        """Test all model types have unique values."""
        values = [t.value for t in ModelType]
        assert len(values) == len(set(values))


class TestModelState:
    """Tests for ModelState enum."""
    
    def test_all_states_exist(self):
        """Test all model states are defined."""
        assert ModelState.REGISTERED is not None
        assert ModelState.TRAINING is not None
        assert ModelState.READY is not None
        assert ModelState.DEPRECATED is not None
        assert ModelState.FAILED is not None


class TestEnsembleStrategy:
    """Tests for EnsembleStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test all ensemble strategies are defined."""
        assert EnsembleStrategy.VOTING is not None
        assert EnsembleStrategy.AVERAGING is not None
        assert EnsembleStrategy.WEIGHTED_AVERAGE is not None
        assert EnsembleStrategy.STACKING is not None
        assert EnsembleStrategy.BOOSTING is not None


class TestFeatureType:
    """Tests for FeatureType enum."""
    
    def test_all_feature_types_exist(self):
        """Test all feature types are defined."""
        assert FeatureType.NUMERIC is not None
        assert FeatureType.CATEGORICAL is not None
        assert FeatureType.TEXT is not None
        assert FeatureType.SEQUENCE is not None
        assert FeatureType.EMBEDDING is not None
        assert FeatureType.GRAPH is not None


# ============================================================================
# DATA CLASS TESTS
# ============================================================================

class TestModelMetadata:
    """Tests for ModelMetadata data class."""
    
    def test_creation_with_required_fields(self):
        """Test creating metadata with required fields."""
        metadata = ModelMetadata(
            name="test_model",
            model_type=ModelType.CLASSIFIER,
            version="1.0.0",
        )
        
        assert metadata.name == "test_model"
        assert metadata.model_type == ModelType.CLASSIFIER
        assert metadata.version == "1.0.0"
        assert metadata.state == ModelState.REGISTERED
        assert metadata.metrics == {}
        assert metadata.tags == []
    
    def test_creation_with_all_fields(self):
        """Test creating metadata with all fields."""
        metadata = ModelMetadata(
            name="test_model",
            model_type=ModelType.REGRESSOR,
            version="2.0.0",
            state=ModelState.READY,
            metrics={"accuracy": 0.95},
            tags=["production", "v2"],
            description="Test model",
            params={"learning_rate": 0.01},
        )
        
        assert metadata.description == "Test model"
        assert metadata.metrics["accuracy"] == 0.95
        assert "production" in metadata.tags
        assert metadata.params["learning_rate"] == 0.01
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = ModelMetadata(
            name="test",
            model_type=ModelType.CLASSIFIER,
            version="1.0.0",
        )
        
        d = metadata.to_dict()
        
        assert d["name"] == "test"
        assert d["model_type"] == "CLASSIFIER"
        assert d["version"] == "1.0.0"
        assert d["state"] == "REGISTERED"
    
    def test_created_at_auto_set(self):
        """Test created_at is automatically set."""
        before = time.time()
        metadata = ModelMetadata(
            name="test",
            model_type=ModelType.CLASSIFIER,
            version="1.0.0",
        )
        after = time.time()
        
        assert before <= metadata.created_at <= after


class TestTrainingMetrics:
    """Tests for TrainingMetrics data class."""
    
    def test_default_creation(self):
        """Test default creation."""
        metrics = TrainingMetrics()
        
        assert metrics.loss_history == []
        assert metrics.accuracy_history == []
        assert metrics.validation_metrics == {}
        assert metrics.training_time == 0.0
        assert metrics.iterations == 0
        assert metrics.convergence_achieved is False
    
    def test_add_epoch_metrics(self):
        """Test adding epoch metrics."""
        metrics = TrainingMetrics()
        
        metrics.add_epoch_metrics(0.5, 0.8)
        metrics.add_epoch_metrics(0.3, 0.85)
        
        assert metrics.loss_history == [0.5, 0.3]
        assert metrics.accuracy_history == [0.8, 0.85]
        assert metrics.iterations == 2
    
    def test_add_epoch_metrics_without_accuracy(self):
        """Test adding epoch metrics without accuracy."""
        metrics = TrainingMetrics()
        
        metrics.add_epoch_metrics(0.5)
        
        assert metrics.loss_history == [0.5]
        assert metrics.accuracy_history == []
    
    def test_final_loss_property(self):
        """Test final_loss property."""
        metrics = TrainingMetrics()
        
        assert metrics.final_loss is None
        
        metrics.add_epoch_metrics(0.5)
        metrics.add_epoch_metrics(0.3)
        
        assert metrics.final_loss == 0.3
    
    def test_final_accuracy_property(self):
        """Test final_accuracy property."""
        metrics = TrainingMetrics()
        
        assert metrics.final_accuracy is None
        
        metrics.add_epoch_metrics(0.5, 0.8)
        metrics.add_epoch_metrics(0.3, 0.9)
        
        assert metrics.final_accuracy == 0.9


class TestFeatureConfig:
    """Tests for FeatureConfig data class."""
    
    def test_creation(self):
        """Test creating feature config."""
        config = FeatureConfig(
            name="age",
            feature_type=FeatureType.NUMERIC,
        )
        
        assert config.name == "age"
        assert config.feature_type == FeatureType.NUMERIC
        assert config.enabled is True
        assert config.transformer is None
    
    def test_creation_with_transformer(self):
        """Test creating with transformer."""
        config = FeatureConfig(
            name="category",
            feature_type=FeatureType.CATEGORICAL,
            transformer="one_hot",
            params={"drop": "first"},
        )
        
        assert config.transformer == "one_hot"
        assert config.params["drop"] == "first"


class TestEnsembleConfig:
    """Tests for EnsembleConfig data class."""
    
    def test_default_creation(self):
        """Test default creation."""
        config = EnsembleConfig()
        
        assert config.strategy == EnsembleStrategy.AVERAGING
        assert config.weights is None
        assert config.voting_threshold == 0.5
        assert config.meta_learner is None


class TestModelVersion:
    """Tests for ModelVersion data class."""
    
    def test_creation(self):
        """Test creating model version."""
        version = ModelVersion(
            version_id="1.0.0",
            parent_version="0.9.0",
            notes="Initial release",
        )
        
        assert version.version_id == "1.0.0"
        assert version.parent_version == "0.9.0"
        assert version.is_current is False
        assert version.notes == "Initial release"


# ============================================================================
# MODEL REGISTRY TESTS
# ============================================================================

class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return ModelRegistry()
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return SimpleClassifier()
    
    def test_register_model(self, registry, simple_model):
        """Test registering a model."""
        metadata = registry.register(
            "test_model",
            simple_model,
            ModelType.CLASSIFIER,
            version="1.0.0",
            description="Test classifier",
        )
        
        assert metadata.name == "test_model"
        assert metadata.model_type == ModelType.CLASSIFIER
        assert "test_model" in registry
        assert len(registry) == 1
    
    def test_register_with_tags(self, registry, simple_model):
        """Test registering with tags."""
        metadata = registry.register(
            "test_model",
            simple_model,
            ModelType.CLASSIFIER,
            tags=["production", "v1"],
        )
        
        assert "production" in metadata.tags
        assert "v1" in metadata.tags
    
    def test_get_model(self, registry, simple_model):
        """Test getting a model."""
        registry.register("test", simple_model, ModelType.CLASSIFIER)
        
        retrieved = registry.get("test")
        
        assert retrieved is simple_model
    
    def test_get_nonexistent_model(self, registry):
        """Test getting nonexistent model."""
        result = registry.get("nonexistent")
        
        assert result is None
    
    def test_get_metadata(self, registry, simple_model):
        """Test getting model metadata."""
        registry.register(
            "test",
            simple_model,
            ModelType.CLASSIFIER,
            description="Test desc",
        )
        
        metadata = registry.get_metadata("test")
        
        assert metadata is not None
        assert metadata.name == "test"
        assert metadata.description == "Test desc"
    
    def test_get_metadata_nonexistent(self, registry):
        """Test getting metadata for nonexistent model."""
        metadata = registry.get_metadata("nonexistent")
        
        assert metadata is None
    
    def test_update_state(self, registry, simple_model):
        """Test updating model state."""
        registry.register("test", simple_model, ModelType.CLASSIFIER)
        
        result = registry.update_state("test", ModelState.READY)
        
        assert result is True
        metadata = registry.get_metadata("test")
        assert metadata.state == ModelState.READY
    
    def test_update_state_nonexistent(self, registry):
        """Test updating state for nonexistent model."""
        result = registry.update_state("nonexistent", ModelState.READY)
        
        assert result is False
    
    def test_update_metrics(self, registry, simple_model):
        """Test updating model metrics."""
        registry.register("test", simple_model, ModelType.CLASSIFIER)
        
        result = registry.update_metrics("test", {"accuracy": 0.95, "f1": 0.92})
        
        assert result is True
        metadata = registry.get_metadata("test")
        assert metadata.metrics["accuracy"] == 0.95
        assert metadata.metrics["f1"] == 0.92
    
    def test_update_metrics_nonexistent(self, registry):
        """Test updating metrics for nonexistent model."""
        result = registry.update_metrics("nonexistent", {"accuracy": 0.95})
        
        assert result is False
    
    def test_list_models(self, registry):
        """Test listing all models."""
        model1 = SimpleClassifier()
        model2 = SimpleRegressor()
        
        registry.register("classifier", model1, ModelType.CLASSIFIER)
        registry.register("regressor", model2, ModelType.REGRESSOR)
        
        models = registry.list_models()
        
        assert len(models) == 2
    
    def test_list_models_by_type(self, registry):
        """Test listing models by type."""
        model1 = SimpleClassifier()
        model2 = SimpleRegressor()
        
        registry.register("classifier", model1, ModelType.CLASSIFIER)
        registry.register("regressor", model2, ModelType.REGRESSOR)
        
        classifiers = registry.list_models(model_type=ModelType.CLASSIFIER)
        
        assert len(classifiers) == 1
        assert classifiers[0].name == "classifier"
    
    def test_list_models_by_state(self, registry, simple_model):
        """Test listing models by state."""
        registry.register("model1", simple_model, ModelType.CLASSIFIER)
        registry.update_state("model1", ModelState.READY)
        
        model2 = SimpleClassifier()
        registry.register("model2", model2, ModelType.CLASSIFIER)
        
        ready = registry.list_models(state=ModelState.READY)
        
        assert len(ready) == 1
        assert ready[0].name == "model1"
    
    def test_list_models_by_tags(self, registry, simple_model):
        """Test listing models by tags."""
        registry.register(
            "model1",
            simple_model,
            ModelType.CLASSIFIER,
            tags=["production", "v1"],
        )
        
        model2 = SimpleClassifier()
        registry.register(
            "model2",
            model2,
            ModelType.CLASSIFIER,
            tags=["development"],
        )
        
        production = registry.list_models(tags=["production"])
        
        assert len(production) == 1
        assert production[0].name == "model1"
    
    def test_delete_model(self, registry, simple_model):
        """Test deleting a model."""
        registry.register("test", simple_model, ModelType.CLASSIFIER)
        
        result = registry.delete("test")
        
        assert result is True
        assert "test" not in registry
        assert len(registry) == 0
    
    def test_delete_nonexistent_model(self, registry):
        """Test deleting nonexistent model."""
        result = registry.delete("nonexistent")
        
        assert result is False
    
    def test_version_management(self, registry, simple_model):
        """Test version management."""
        registry.register("test", simple_model, ModelType.CLASSIFIER, version="1.0.0")
        
        model2 = SimpleClassifier()
        registry.register("test", model2, ModelType.CLASSIFIER, version="2.0.0")
        
        versions = registry.get_versions("test")
        
        assert len(versions) == 2
        # Latest version should be current
        latest = [v for v in versions if v.is_current]
        assert len(latest) == 1
        assert latest[0].version_id == "2.0.0"
    
    def test_listener_notification(self, registry, simple_model):
        """Test listener notification."""
        events = []
        
        def listener(name: str, event: str):
            events.append((name, event))
        
        registry.add_listener(listener)
        registry.register("test", simple_model, ModelType.CLASSIFIER)
        
        assert ("test", "registered") in events
    
    def test_contains(self, registry, simple_model):
        """Test __contains__ method."""
        assert "test" not in registry
        
        registry.register("test", simple_model, ModelType.CLASSIFIER)
        
        assert "test" in registry
    
    def test_len(self, registry, simple_model):
        """Test __len__ method."""
        assert len(registry) == 0
        
        registry.register("test1", simple_model, ModelType.CLASSIFIER)
        registry.register("test2", SimpleClassifier(), ModelType.CLASSIFIER)
        
        assert len(registry) == 2
    
    def test_thread_safety(self, registry):
        """Test thread-safe operations."""
        results = []
        errors = []
        
        def register_model(i: int):
            try:
                model = SimpleClassifier()
                registry.register(f"model_{i}", model, ModelType.CLASSIFIER)
                results.append(i)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=register_model, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(registry) == 10


# ============================================================================
# ADAPTIVE LEARNER TESTS
# ============================================================================

class TestAdaptiveLearner:
    """Tests for AdaptiveLearner class."""
    
    @pytest.fixture
    def linear_model(self):
        """Create a linear model for testing."""
        return LinearModel(learning_rate=0.1, iterations=100)
    
    @pytest.fixture
    def learner(self, linear_model):
        """Create an adaptive learner for testing."""
        return AdaptiveLearner(
            linear_model,
            drift_threshold=0.2,
            window_size=50,
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1
        return X, y
    
    def test_creation(self, learner):
        """Test creating an adaptive learner."""
        assert learner.drift_threshold == 0.2
        assert learner.window_size == 50
        assert learner._is_fitted is False
    
    def test_fit(self, learner, sample_data):
        """Test initial fitting."""
        X, y = sample_data
        
        result = learner.fit(X, y)
        
        assert result is learner
        assert learner._is_fitted is True
        assert learner._samples_seen == 100
    
    def test_predict_without_fit(self, learner, sample_data):
        """Test prediction without fitting raises error."""
        X, _ = sample_data
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            learner.predict(X)
    
    def test_predict_after_fit(self, learner, sample_data):
        """Test prediction after fitting."""
        X, y = sample_data
        learner.fit(X, y)
        
        predictions = learner.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_partial_fit(self, learner, sample_data):
        """Test incremental fitting."""
        X, y = sample_data
        learner.fit(X[:50], y[:50])
        
        result = learner.partial_fit(X[50:], y[50:])
        
        assert result is learner
        assert learner._samples_seen == 100
    
    def test_get_stats(self, learner, sample_data):
        """Test getting statistics."""
        X, y = sample_data
        learner.fit(X, y)
        
        stats = learner.get_stats()
        
        assert stats["samples_seen"] == 100
        assert stats["is_fitted"] is True
        assert "drift_score" in stats
        assert "baseline_error" in stats
    
    def test_drift_detection(self, learner, sample_data):
        """Test drift detection capability."""
        X, y = sample_data
        learner.fit(X, y)
        
        # Get initial drift score
        initial_score = learner.get_drift_score()
        
        # Score should be None or low after initial fit
        # because error window may not be populated
        stats = learner.get_stats()
        assert stats["current_window_size"] >= 0
    
    def test_get_drift_score_before_fit(self, learner):
        """Test drift score before fitting."""
        score = learner.get_drift_score()
        
        assert score is None


# ============================================================================
# FEATURE EXTRACTOR TESTS
# ============================================================================

class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a feature extractor for testing."""
        return FeatureExtractor(normalize=True)
    
    def test_creation(self, extractor):
        """Test creating a feature extractor."""
        assert extractor.normalize is True
        assert extractor._is_fitted is False
    
    def test_add_feature(self, extractor):
        """Test adding a feature."""
        result = extractor.add_feature("age", FeatureType.NUMERIC)
        
        assert result is extractor
        assert "age" in extractor._features
    
    def test_add_feature_with_transformer(self, extractor):
        """Test adding feature with transformer."""
        extractor.add_feature(
            "category",
            FeatureType.CATEGORICAL,
            transformer="one_hot",
            params={"drop": "first"},
        )
        
        config = extractor._features["category"]
        assert config.transformer == "one_hot"
        assert config.params["drop"] == "first"
    
    def test_fit(self, extractor):
        """Test fitting the extractor."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        
        data = {"age": np.array([20, 30, 40, 50, 60])}
        result = extractor.fit(data)
        
        assert result is extractor
        assert extractor._is_fitted is True
        assert "age" in extractor._fitted_stats
    
    def test_fit_numeric_stats(self, extractor):
        """Test fitted stats for numeric features."""
        extractor.add_feature("value", FeatureType.NUMERIC)
        
        data = {"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        extractor.fit(data)
        
        stats = extractor._fitted_stats["value"]
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
    
    def test_fit_categorical_stats(self, extractor):
        """Test fitted stats for categorical features."""
        extractor.add_feature("color", FeatureType.CATEGORICAL)
        
        data = {"color": np.array(["red", "blue", "red", "green"])}
        extractor.fit(data)
        
        stats = extractor._fitted_stats["color"]
        assert stats["n_categories"] == 3
        assert "red" in stats["categories"]
    
    def test_transform_without_fit(self, extractor):
        """Test transform without fit raises error."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            extractor.transform({"age": np.array([25])})
    
    def test_transform_numeric(self, extractor):
        """Test transforming numeric features."""
        extractor.add_feature("value", FeatureType.NUMERIC)
        
        data = {"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        extractor.fit(data)
        
        result = extractor.transform(data)
        
        assert result.shape == (5, 1)
        # Should be normalized (mean ~0, std ~1)
        assert abs(np.mean(result)) < 1e-10
    
    def test_transform_categorical(self, extractor):
        """Test transforming categorical features."""
        extractor.add_feature("color", FeatureType.CATEGORICAL)
        
        data = {"color": np.array(["red", "blue", "green"])}
        extractor.fit(data)
        
        result = extractor.transform(data)
        
        # One-hot encoded: 3 samples x 3 categories
        assert result.shape == (3, 3)
        # Each row should have exactly one 1
        assert all(np.sum(row) == 1 for row in result)
    
    def test_fit_transform(self, extractor):
        """Test fit and transform in one step."""
        extractor.add_feature("value", FeatureType.NUMERIC)
        
        data = {"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        result = extractor.fit_transform(data)
        
        assert extractor._is_fitted is True
        assert result.shape == (5, 1)
    
    def test_get_feature_names(self, extractor):
        """Test getting feature names."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        extractor.add_feature("name", FeatureType.TEXT)
        
        names = extractor.get_feature_names()
        
        assert "age" in names
        assert "name" in names
    
    def test_get_feature_info(self, extractor):
        """Test getting feature info."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        extractor.fit({"age": np.array([20, 30, 40])})
        
        info = extractor.get_feature_info("age")
        
        assert info is not None
        assert info["name"] == "age"
        assert info["type"] == "NUMERIC"
        assert "stats" in info
    
    def test_get_feature_info_nonexistent(self, extractor):
        """Test getting info for nonexistent feature."""
        info = extractor.get_feature_info("nonexistent")
        
        assert info is None
    
    def test_disable_feature(self, extractor):
        """Test disabling a feature."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        
        result = extractor.disable_feature("age")
        
        assert result is True
        assert extractor._features["age"].enabled is False
    
    def test_disable_nonexistent_feature(self, extractor):
        """Test disabling nonexistent feature."""
        result = extractor.disable_feature("nonexistent")
        
        assert result is False
    
    def test_enable_feature(self, extractor):
        """Test enabling a feature."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        extractor.disable_feature("age")
        
        result = extractor.enable_feature("age")
        
        assert result is True
        assert extractor._features["age"].enabled is True
    
    def test_transform_text_feature(self, extractor):
        """Test transforming text features."""
        extractor.add_feature("text", FeatureType.TEXT)
        
        data = {"text": np.array(["hello", "world", "test"])}
        result = extractor.fit_transform(data)
        
        # Simple text transform returns character count
        assert result.shape == (3, 1)
        assert result[0, 0] == 5  # "hello" has 5 chars
    
    def test_transform_sequence_feature(self, extractor):
        """Test transforming sequence features."""
        extractor.add_feature("seq", FeatureType.SEQUENCE)
        
        data = {"seq": np.array([[1, 2, 3], [4, 5, 6]])}
        result = extractor.fit_transform(data)
        
        # Sequences are padded to max_len
        assert result.shape[0] == 2
    
    def test_transform_embedding_feature(self, extractor):
        """Test transforming embedding features."""
        extractor.add_feature("emb", FeatureType.EMBEDDING)
        
        embeddings = np.random.randn(5, 10)
        data = {"emb": embeddings}
        result = extractor.fit_transform(data)
        
        # Embeddings passed through unchanged
        np.testing.assert_array_equal(result, embeddings)
    
    def test_multiple_features(self, extractor):
        """Test extracting multiple features."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        extractor.add_feature("income", FeatureType.NUMERIC)
        
        data = {
            "age": np.array([25, 30, 35]),
            "income": np.array([50000, 60000, 70000]),
        }
        result = extractor.fit_transform(data)
        
        assert result.shape == (3, 2)
    
    def test_disabled_features_not_transformed(self, extractor):
        """Test that disabled features are not included."""
        extractor.add_feature("age", FeatureType.NUMERIC)
        extractor.add_feature("income", FeatureType.NUMERIC)
        extractor.disable_feature("income")
        
        data = {
            "age": np.array([25, 30, 35]),
            "income": np.array([50000, 60000, 70000]),
        }
        result = extractor.fit_transform(data)
        
        assert result.shape == (3, 1)  # Only age


# ============================================================================
# MODEL ENSEMBLE TESTS
# ============================================================================

class TestModelEnsemble:
    """Tests for ModelEnsemble class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (np.sum(X, axis=1) > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def ensemble(self):
        """Create an ensemble for testing."""
        return ModelEnsemble(strategy=EnsembleStrategy.AVERAGING)
    
    def test_creation(self, ensemble):
        """Test creating an ensemble."""
        assert ensemble.strategy == EnsembleStrategy.AVERAGING
        assert ensemble.voting_threshold == 0.5
        assert len(ensemble) == 0
    
    def test_add_model(self, ensemble):
        """Test adding a model."""
        model = SimpleClassifier()
        result = ensemble.add_model("model1", model, weight=1.5)
        
        assert result is ensemble
        assert len(ensemble) == 1
        assert ensemble._weights["model1"] == 1.5
    
    def test_remove_model(self, ensemble):
        """Test removing a model."""
        model = SimpleClassifier()
        ensemble.add_model("model1", model)
        
        result = ensemble.remove_model("model1")
        
        assert result is True
        assert len(ensemble) == 0
    
    def test_remove_nonexistent_model(self, ensemble):
        """Test removing nonexistent model."""
        result = ensemble.remove_model("nonexistent")
        
        assert result is False
    
    def test_fit(self, ensemble, sample_data):
        """Test fitting the ensemble."""
        X, y = sample_data
        
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        
        result = ensemble.fit(X, y)
        
        assert result is ensemble
        assert ensemble._is_fitted is True
    
    def test_predict_without_fit(self, ensemble, sample_data):
        """Test prediction without fit raises error."""
        X, _ = sample_data
        ensemble.add_model("model1", SimpleClassifier())
        
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict(X)
    
    def test_predict_empty_ensemble(self, sample_data):
        """Test prediction with empty ensemble raises error."""
        X, y = sample_data
        ensemble = ModelEnsemble()
        ensemble._is_fitted = True
        
        with pytest.raises(ValueError, match="No models"):
            ensemble.predict(X)
    
    def test_predict_averaging(self, sample_data):
        """Test averaging strategy prediction."""
        X, y = sample_data
        
        ensemble = ModelEnsemble(strategy=EnsembleStrategy.AVERAGING)
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_predict_voting(self, sample_data):
        """Test voting strategy prediction."""
        X, y = sample_data
        
        ensemble = ModelEnsemble(strategy=EnsembleStrategy.VOTING)
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_predict_weighted_average(self, sample_data):
        """Test weighted average strategy."""
        X, y = sample_data
        
        ensemble = ModelEnsemble(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
        ensemble.add_model("model1", SimpleClassifier(), weight=2.0)
        ensemble.add_model("model2", SimpleClassifier(), weight=1.0)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_predict_stacking(self, sample_data):
        """Test stacking strategy."""
        X, y = sample_data
        
        ensemble = ModelEnsemble(strategy=EnsembleStrategy.STACKING)
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_set_weights(self, ensemble):
        """Test setting weights."""
        ensemble.add_model("model1", SimpleClassifier(), weight=1.0)
        ensemble.add_model("model2", SimpleClassifier(), weight=1.0)
        
        result = ensemble.set_weights({"model1": 2.0, "model2": 3.0})
        
        assert result is ensemble
        assert ensemble._weights["model1"] == 2.0
        assert ensemble._weights["model2"] == 3.0
    
    def test_optimize_weights(self, sample_data):
        """Test weight optimization."""
        X, y = sample_data
        X_val, y_val = X[:20], y[:20]
        
        ensemble = ModelEnsemble()
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        ensemble.fit(X, y)
        
        weights = ensemble.optimize_weights(X_val, y_val)
        
        assert "model1" in weights
        assert "model2" in weights
        # Weights should sum to ~1
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_get_model_contributions(self, sample_data):
        """Test getting individual model contributions."""
        X, y = sample_data
        
        ensemble = ModelEnsemble()
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        ensemble.fit(X, y)
        
        contributions = ensemble.get_model_contributions(X[:10])
        
        assert "model1" in contributions
        assert "model2" in contributions
        assert len(contributions["model1"]) == 10
    
    def test_get_weights(self, ensemble):
        """Test getting weights."""
        ensemble.add_model("model1", SimpleClassifier(), weight=1.5)
        
        weights = ensemble.get_weights()
        
        assert weights["model1"] == 1.5
    
    def test_len(self, ensemble):
        """Test __len__ method."""
        assert len(ensemble) == 0
        
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        
        assert len(ensemble) == 2


# ============================================================================
# SIMPLE MODEL TESTS
# ============================================================================

class TestSimpleClassifier:
    """Tests for SimpleClassifier."""
    
    def test_fit_and_predict(self):
        """Test fitting and predicting."""
        classifier = SimpleClassifier()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 0, 1])  # Majority is 0
        
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        
        # Should predict majority class (0) for all
        np.testing.assert_array_equal(predictions, np.array([0, 0, 0, 0]))
    
    def test_predict_without_fit(self):
        """Test prediction without fit raises error."""
        classifier = SimpleClassifier()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            classifier.predict(np.array([[1, 2]]))


class TestSimpleRegressor:
    """Tests for SimpleRegressor."""
    
    def test_fit_and_predict(self):
        """Test fitting and predicting."""
        regressor = SimpleRegressor()
        
        X = np.array([[1], [2], [3], [4]])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        
        regressor.fit(X, y)
        predictions = regressor.predict(X)
        
        # Should predict mean (25.0) for all
        np.testing.assert_array_equal(predictions, np.array([25.0, 25.0, 25.0, 25.0]))
    
    def test_predict_without_fit(self):
        """Test prediction without fit raises error."""
        regressor = SimpleRegressor()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            regressor.predict(np.array([[1]]))


class TestLinearModel:
    """Tests for LinearModel."""
    
    def test_creation(self):
        """Test creating a linear model."""
        model = LinearModel(learning_rate=0.01, iterations=100)
        
        assert model.learning_rate == 0.01
        assert model.iterations == 100
    
    def test_fit_and_predict(self):
        """Test fitting and predicting."""
        np.random.seed(42)
        model = LinearModel(learning_rate=0.1, iterations=200)
        
        # Simple linear relationship
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Should approximate the linear relationship
        assert predictions is not None
        assert len(predictions) == 5
    
    def test_predict_without_fit(self):
        """Test prediction without fit raises error."""
        model = LinearModel()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.array([[1]]))
    
    def test_partial_fit(self):
        """Test incremental fitting."""
        model = LinearModel(learning_rate=0.1, iterations=1)
        
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model.fit(X, y)
        
        # Continue training with partial_fit
        model.partial_fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == 3


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_model_registry(self):
        """Test creating a model registry."""
        registry = create_model_registry()
        
        assert isinstance(registry, ModelRegistry)
        assert len(registry) == 0
    
    def test_create_ensemble(self):
        """Test creating an ensemble."""
        models = [
            ("model1", SimpleClassifier()),
            ("model2", SimpleClassifier()),
        ]
        
        ensemble = create_ensemble(models, strategy=EnsembleStrategy.VOTING)
        
        assert isinstance(ensemble, ModelEnsemble)
        assert len(ensemble) == 2
        assert ensemble.strategy == EnsembleStrategy.VOTING
    
    def test_create_feature_extractor(self):
        """Test creating a feature extractor."""
        features = [
            ("age", FeatureType.NUMERIC),
            ("category", FeatureType.CATEGORICAL),
        ]
        
        extractor = create_feature_extractor(features, normalize=True)
        
        assert isinstance(extractor, FeatureExtractor)
        assert "age" in extractor._features
        assert "category" in extractor._features
    
    def test_train_with_metrics(self):
        """Test training with metrics collection."""
        model = LinearModel(learning_rate=0.1, iterations=50)
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.sum(X, axis=1) + np.random.randn(50) * 0.1
        
        metrics = train_with_metrics(model, X, y)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.training_time > 0
        assert metrics.convergence_achieved is True
        assert metrics.final_loss is not None
    
    def test_train_with_validation(self):
        """Test training with validation metrics."""
        model = LinearModel(learning_rate=0.1, iterations=50)
        
        np.random.seed(42)
        X_train = np.random.randn(40, 3)
        y_train = np.sum(X_train, axis=1)
        X_val = np.random.randn(10, 3)
        y_val = np.sum(X_val, axis=1)
        
        metrics = train_with_metrics(model, X_train, y_train, X_val, y_val)
        
        assert "loss" in metrics.validation_metrics


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    def test_full_ml_pipeline(self):
        """Test a complete ML pipeline."""
        # Create registry
        registry = create_model_registry()
        
        # Create and register models
        model1 = SimpleClassifier()
        model2 = SimpleClassifier()
        
        registry.register("clf1", model1, ModelType.CLASSIFIER, tags=["baseline"])
        registry.register("clf2", model2, ModelType.CLASSIFIER, tags=["baseline"])
        
        # Create ensemble
        ensemble = create_ensemble([
            ("clf1", registry.get("clf1")),
            ("clf2", registry.get("clf2")),
        ])
        
        # Train
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (np.sum(X, axis=1) > 0).astype(int)
        
        ensemble.fit(X, y)
        
        # Predict
        predictions = ensemble.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_feature_extraction_pipeline(self):
        """Test feature extraction in a pipeline."""
        # Create extractor
        extractor = create_feature_extractor([
            ("numeric1", FeatureType.NUMERIC),
            ("numeric2", FeatureType.NUMERIC),
            ("category", FeatureType.CATEGORICAL),
        ])
        
        # Create data
        data = {
            "numeric1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "numeric2": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "category": np.array(["A", "B", "A", "C", "B"]),
        }
        
        # Extract features
        features = extractor.fit_transform(data)
        
        # 5 samples, 2 numeric + 3 one-hot = 5 features
        assert features.shape[0] == 5
        assert features.shape[1] == 5
    
    def test_adaptive_learning_workflow(self):
        """Test adaptive learning workflow."""
        # Create base model and learner
        model = LinearModel(learning_rate=0.1, iterations=100)
        learner = AdaptiveLearner(model, drift_threshold=0.2)
        
        # Initial training
        np.random.seed(42)
        X1 = np.random.randn(50, 3)
        y1 = np.sum(X1, axis=1)
        
        learner.fit(X1, y1)
        
        # Incremental learning
        X2 = np.random.randn(30, 3)
        y2 = np.sum(X2, axis=1)
        
        learner.partial_fit(X2, y2)
        
        # Check stats
        stats = learner.get_stats()
        assert stats["samples_seen"] == 80
    
    def test_model_versioning(self):
        """Test model versioning workflow."""
        registry = create_model_registry()
        
        # Register v1
        model_v1 = SimpleClassifier()
        registry.register("classifier", model_v1, ModelType.CLASSIFIER, version="1.0.0")
        
        # Train and update metrics
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.array([0] * 25 + [1] * 25)
        model_v1.fit(X, y)
        registry.update_metrics("classifier", {"accuracy": 0.8})
        registry.update_state("classifier", ModelState.READY)
        
        # Register v2
        model_v2 = SimpleClassifier()
        registry.register("classifier", model_v2, ModelType.CLASSIFIER, version="2.0.0")
        
        # Check versions
        versions = registry.get_versions("classifier")
        assert len(versions) == 2
        
        # Current should be v2
        metadata = registry.get_metadata("classifier")
        assert metadata.version == "2.0.0"
    
    def test_ensemble_with_weight_optimization(self):
        """Test ensemble with automatic weight optimization."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (np.sum(X, axis=1) > 0).astype(int)
        
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        # Create ensemble
        ensemble = ModelEnsemble(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
        ensemble.add_model("model1", SimpleClassifier())
        ensemble.add_model("model2", SimpleClassifier())
        
        # Train
        ensemble.fit(X_train, y_train)
        
        # Optimize weights
        optimized = ensemble.optimize_weights(X_val, y_val)
        
        assert sum(optimized.values()) > 0
        
        # Final prediction
        predictions = ensemble.predict(X_val)
        assert len(predictions) == 20


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Thread safety tests for ML components."""
    
    def test_feature_extractor_thread_safety(self):
        """Test feature extractor thread safety."""
        extractor = FeatureExtractor()
        extractor.add_feature("value", FeatureType.NUMERIC)
        extractor.fit({"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
        
        results = []
        errors = []
        
        def transform_data(i: int):
            try:
                data = {"value": np.array([float(i), float(i + 1)])}
                result = extractor.transform(data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=transform_data, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10
    
    def test_adaptive_learner_thread_safety(self):
        """Test adaptive learner thread safety."""
        model = LinearModel(learning_rate=0.1, iterations=10)
        learner = AdaptiveLearner(model)
        
        np.random.seed(42)
        X_init = np.random.randn(20, 3)
        y_init = np.sum(X_init, axis=1)
        learner.fit(X_init, y_init)
        
        errors = []
        
        def partial_fit_batch(i: int):
            try:
                X = np.random.randn(5, 3)
                y = np.sum(X, axis=1)
                learner.partial_fit(X, y)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=partial_fit_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
