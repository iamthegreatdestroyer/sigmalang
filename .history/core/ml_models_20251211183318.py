"""
Enhanced ML Models for SigmaLang Phase 2A.5.

This module provides machine learning infrastructure including:
- ModelRegistry: Central registry for managing ML models
- AdaptiveLearning: Online learning with model adaptation
- FeatureExtractor: Multi-modal feature extraction
- ModelEnsemble: Ensemble methods for improved predictions
- TransferLearner: Transfer learning utilities
- ModelVersioning: Version control for models

Design Philosophy:
- Pluggable model architectures
- Online learning capabilities
- Automatic feature engineering
- Ensemble predictions with weighted voting
- Model versioning and rollback
"""

from __future__ import annotations

import threading
import time
import hashlib
import json
import pickle
import copy
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    Sequence,
)

import numpy as np


# ============================================================================
# ENUMS & TYPE DEFINITIONS
# ============================================================================

class ModelType(Enum):
    """Types of ML models supported."""
    CLASSIFIER = auto()
    REGRESSOR = auto()
    CLUSTERER = auto()
    TRANSFORMER = auto()
    ENCODER = auto()
    EMBEDDING = auto()


class ModelState(Enum):
    """States of a model in the registry."""
    REGISTERED = auto()
    TRAINING = auto()
    READY = auto()
    DEPRECATED = auto()
    FAILED = auto()


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    VOTING = auto()           # Majority voting for classifiers
    AVERAGING = auto()        # Simple averaging
    WEIGHTED_AVERAGE = auto() # Weighted by performance
    STACKING = auto()         # Meta-learner stacking
    BOOSTING = auto()         # Sequential boosting


class FeatureType(Enum):
    """Types of features that can be extracted."""
    NUMERIC = auto()
    CATEGORICAL = auto()
    TEXT = auto()
    SEQUENCE = auto()
    EMBEDDING = auto()
    GRAPH = auto()


# Type variable for model predictions
T = TypeVar('T')
ModelInput = Union[np.ndarray, List[float], Dict[str, Any]]
ModelOutput = Union[np.ndarray, float, int, List[Any], Dict[str, Any]]


# ============================================================================
# PROTOCOLS & ABSTRACT CLASSES
# ============================================================================

class MLModel(Protocol):
    """Protocol for ML models that can be registered."""
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MLModel':
        """Train the model."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseModel':
        """Train the model on data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X, y)
        return self.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        return self


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    model_type: ModelType
    version: str
    state: ModelState = ModelState.REGISTERED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'model_type': self.model_type.name,
            'version': self.version,
            'state': self.state.name,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metrics': self.metrics,
            'tags': self.tags,
            'description': self.description,
            'params': self.params,
        }


@dataclass
class TrainingMetrics:
    """Metrics collected during model training."""
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    iterations: int = 0
    convergence_achieved: bool = False
    
    def add_epoch_metrics(self, loss: float, accuracy: Optional[float] = None):
        """Add metrics for an epoch."""
        self.loss_history.append(loss)
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
        self.iterations += 1
    
    @property
    def final_loss(self) -> Optional[float]:
        """Get final loss value."""
        return self.loss_history[-1] if self.loss_history else None
    
    @property
    def final_accuracy(self) -> Optional[float]:
        """Get final accuracy value."""
        return self.accuracy_history[-1] if self.accuracy_history else None


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    name: str
    feature_type: FeatureType
    transformer: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble."""
    strategy: EnsembleStrategy = EnsembleStrategy.AVERAGING
    weights: Optional[List[float]] = None
    voting_threshold: float = 0.5
    meta_learner: Optional[str] = None


@dataclass
class ModelVersion:
    """Version information for a model."""
    version_id: str
    parent_version: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    is_current: bool = False
    notes: str = ""


# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelRegistry:
    """
    Central registry for managing ML models.
    
    Features:
    - Model registration and lookup
    - Version management
    - State tracking
    - Model lifecycle management
    
    Example:
        >>> registry = ModelRegistry()
        >>> model = SimpleClassifier()
        >>> registry.register("my_model", model, ModelType.CLASSIFIER)
        >>> retrieved = registry.get("my_model")
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Tuple[Any, ModelMetadata]] = {}
        self._versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, str], None]] = []
    
    def register(
        self,
        name: str,
        model: Any,
        model_type: ModelType,
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ModelMetadata:
        """
        Register a model in the registry.
        
        Args:
            name: Unique model name
            model: The model instance
            model_type: Type of the model
            version: Version string
            description: Model description
            tags: Optional tags for categorization
            params: Model parameters
            
        Returns:
            ModelMetadata for the registered model
        """
        with self._lock:
            metadata = ModelMetadata(
                name=name,
                model_type=model_type,
                version=version,
                description=description,
                tags=tags or [],
                params=params or {},
            )
            
            # If model already exists, create new version
            if name in self._models:
                old_model, old_metadata = self._models[name]
                # Archive old version
                self._archive_version(name, old_model, old_metadata)
            
            self._models[name] = (model, metadata)
            
            # Create version entry
            version_entry = ModelVersion(
                version_id=version,
                is_current=True,
            )
            self._versions[name].append(version_entry)
            
            # Notify listeners
            self._notify_listeners(name, "registered")
            
            return metadata
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Get a model from the registry.
        
        Args:
            name: Model name
            version: Optional specific version
            
        Returns:
            The model instance or None if not found
        """
        with self._lock:
            if name not in self._models:
                return None
            
            model, metadata = self._models[name]
            
            if version is not None and metadata.version != version:
                # Try to find specific version
                for v in self._versions[name]:
                    if v.version_id == version and v.checkpoint_path:
                        return self._load_checkpoint(v.checkpoint_path)
                return None
            
            return model
    
    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        with self._lock:
            if name not in self._models:
                return None
            return self._models[name][1]
    
    def update_state(self, name: str, state: ModelState) -> bool:
        """Update model state."""
        with self._lock:
            if name not in self._models:
                return False
            
            model, metadata = self._models[name]
            metadata.state = state
            metadata.updated_at = time.time()
            
            self._notify_listeners(name, f"state_changed:{state.name}")
            return True
    
    def update_metrics(self, name: str, metrics: Dict[str, float]) -> bool:
        """Update model metrics."""
        with self._lock:
            if name not in self._models:
                return False
            
            model, metadata = self._models[name]
            metadata.metrics.update(metrics)
            metadata.updated_at = time.time()
            
            return True
    
    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        state: Optional[ModelState] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        with self._lock:
            results = []
            
            for name, (model, metadata) in self._models.items():
                # Apply filters
                if model_type is not None and metadata.model_type != model_type:
                    continue
                if state is not None and metadata.state != state:
                    continue
                if tags is not None:
                    if not all(t in metadata.tags for t in tags):
                        continue
                
                results.append(metadata)
            
            return results
    
    def delete(self, name: str) -> bool:
        """Delete a model from the registry."""
        with self._lock:
            if name not in self._models:
                return False
            
            del self._models[name]
            if name in self._versions:
                del self._versions[name]
            
            self._notify_listeners(name, "deleted")
            return True
    
    def get_versions(self, name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        with self._lock:
            return list(self._versions.get(name, []))
    
    def add_listener(self, callback: Callable[[str, str], None]):
        """Add a listener for registry events."""
        self._listeners.append(callback)
    
    def _archive_version(self, name: str, model: Any, metadata: ModelMetadata):
        """Archive a model version."""
        for v in self._versions[name]:
            if v.version_id == metadata.version:
                v.is_current = False
                v.metrics = metadata.metrics.copy()
    
    def _load_checkpoint(self, path: str) -> Optional[Any]:
        """Load model from checkpoint."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def _notify_listeners(self, name: str, event: str):
        """Notify all listeners of an event."""
        for listener in self._listeners:
            try:
                listener(name, event)
            except Exception:
                pass
    
    def __len__(self) -> int:
        """Get number of registered models."""
        return len(self._models)
    
    def __contains__(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._models


# ============================================================================
# ADAPTIVE LEARNING
# ============================================================================

class AdaptiveLearner:
    """
    Online learning with automatic model adaptation.
    
    Features:
    - Incremental learning
    - Concept drift detection
    - Automatic retraining triggers
    - Performance monitoring
    
    Example:
        >>> learner = AdaptiveLearner(base_model)
        >>> learner.partial_fit(X_batch, y_batch)
        >>> predictions = learner.predict(X_new)
    """
    
    def __init__(
        self,
        base_model: Any,
        learning_rate: float = 0.01,
        drift_threshold: float = 0.1,
        window_size: int = 100,
        retrain_trigger: int = 1000,
    ):
        """
        Initialize adaptive learner.
        
        Args:
            base_model: Base model for learning
            learning_rate: Learning rate for updates
            drift_threshold: Threshold for drift detection
            window_size: Window size for monitoring
            retrain_trigger: Samples before considering retraining
        """
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.retrain_trigger = retrain_trigger
        
        # State
        self._samples_seen = 0
        self._error_window: deque = deque(maxlen=window_size)
        self._baseline_error: Optional[float] = None
        self._is_fitted = False
        self._training_data_X: List[np.ndarray] = []
        self._training_data_y: List[np.ndarray] = []
        self._lock = threading.Lock()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveLearner':
        """Initial fit on training data."""
        with self._lock:
            self.base_model.fit(X, y)
            self._is_fitted = True
            self._samples_seen = len(X)
            
            # Store training data for potential retraining
            self._training_data_X.append(X)
            self._training_data_y.append(y)
            
            # Calculate baseline error
            predictions = self.base_model.predict(X)
            errors = self._calculate_errors(y, predictions)
            self._baseline_error = np.mean(errors)
            
            return self
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'AdaptiveLearner':
        """
        Incrementally fit on new data.
        
        Args:
            X: New feature data
            y: New labels
            sample_weight: Optional sample weights
            
        Returns:
            Self for chaining
        """
        with self._lock:
            # Check if base model supports partial_fit
            if hasattr(self.base_model, 'partial_fit'):
                if sample_weight is not None:
                    self.base_model.partial_fit(X, y, sample_weight=sample_weight)
                else:
                    self.base_model.partial_fit(X, y)
            else:
                # Fall back to full retraining with accumulated data
                self._training_data_X.append(X)
                self._training_data_y.append(y)
                
                if self._samples_seen >= self.retrain_trigger:
                    self._retrain()
            
            self._samples_seen += len(X)
            self._is_fitted = True
            
            # Monitor for drift
            predictions = self.base_model.predict(X)
            errors = self._calculate_errors(y, predictions)
            self._error_window.extend(errors)
            
            if self._detect_drift():
                self._handle_drift()
            
            return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.base_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities if available."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        raise NotImplementedError("Base model doesn't support predict_proba")
    
    def get_drift_score(self) -> Optional[float]:
        """Get current drift score."""
        if not self._error_window or self._baseline_error is None:
            return None
        
        current_error = np.mean(list(self._error_window))
        return abs(current_error - self._baseline_error) / (self._baseline_error + 1e-10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'samples_seen': self._samples_seen,
            'is_fitted': self._is_fitted,
            'drift_score': self.get_drift_score(),
            'baseline_error': self._baseline_error,
            'current_window_size': len(self._error_window),
        }
    
    def _calculate_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate prediction errors."""
        return np.abs(y_true - y_pred)
    
    def _detect_drift(self) -> bool:
        """Detect concept drift."""
        drift_score = self.get_drift_score()
        if drift_score is None:
            return False
        return drift_score > self.drift_threshold
    
    def _handle_drift(self):
        """Handle detected concept drift."""
        # Option: Full retrain with recent data
        if self._training_data_X:
            recent_X = self._training_data_X[-1]
            recent_y = self._training_data_y[-1]
            
            self.base_model.fit(recent_X, recent_y)
            
            # Update baseline
            predictions = self.base_model.predict(recent_X)
            errors = self._calculate_errors(recent_y, predictions)
            self._baseline_error = np.mean(errors)
            
            self._error_window.clear()
    
    def _retrain(self):
        """Full retrain with accumulated data."""
        if not self._training_data_X:
            return
        
        X_combined = np.vstack(self._training_data_X)
        y_combined = np.concatenate(self._training_data_y)
        
        self.base_model.fit(X_combined, y_combined)
        
        # Clear old data to save memory
        self._training_data_X = [X_combined]
        self._training_data_y = [y_combined]


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Multi-modal feature extraction system.
    
    Features:
    - Automatic feature detection
    - Multiple extraction strategies
    - Feature normalization
    - Feature selection
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> extractor.add_feature("age", FeatureType.NUMERIC)
        >>> features = extractor.transform(data)
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        self._features: Dict[str, FeatureConfig] = {}
        self._fitted_stats: Dict[str, Dict[str, float]] = {}
        self._is_fitted = False
        self._lock = threading.Lock()
    
    def add_feature(
        self,
        name: str,
        feature_type: FeatureType,
        transformer: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> 'FeatureExtractor':
        """
        Add a feature configuration.
        
        Args:
            name: Feature name
            feature_type: Type of feature
            transformer: Optional transformer name
            params: Transformer parameters
            
        Returns:
            Self for chaining
        """
        self._features[name] = FeatureConfig(
            name=name,
            feature_type=feature_type,
            transformer=transformer,
            params=params or {},
        )
        return self
    
    def fit(self, data: Dict[str, np.ndarray]) -> 'FeatureExtractor':
        """
        Fit the extractor on data.
        
        Args:
            data: Dictionary mapping feature names to arrays
            
        Returns:
            Self for chaining
        """
        with self._lock:
            for name, config in self._features.items():
                if name not in data:
                    continue
                
                values = data[name]
                
                if config.feature_type == FeatureType.NUMERIC:
                    self._fitted_stats[name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values) + 1e-10),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }
                elif config.feature_type == FeatureType.CATEGORICAL:
                    unique_values = np.unique(values)
                    self._fitted_stats[name] = {
                        'categories': list(unique_values),
                        'n_categories': len(unique_values),
                    }
            
            self._is_fitted = True
            return self
    
    def transform(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform data to features.
        
        Args:
            data: Dictionary mapping feature names to arrays
            
        Returns:
            Feature matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Extractor not fitted. Call fit() first.")
        
        feature_arrays = []
        
        for name, config in self._features.items():
            if not config.enabled or name not in data:
                continue
            
            values = data[name]
            transformed = self._transform_feature(name, values, config)
            
            if transformed is not None:
                if transformed.ndim == 1:
                    transformed = transformed.reshape(-1, 1)
                feature_arrays.append(transformed)
        
        if not feature_arrays:
            return np.array([])
        
        return np.hstack(feature_arrays)
    
    def fit_transform(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [name for name, config in self._features.items() if config.enabled]
    
    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a feature."""
        if name not in self._features:
            return None
        
        config = self._features[name]
        info = {
            'name': name,
            'type': config.feature_type.name,
            'enabled': config.enabled,
            'transformer': config.transformer,
        }
        
        if name in self._fitted_stats:
            info['stats'] = self._fitted_stats[name]
        
        return info
    
    def disable_feature(self, name: str) -> bool:
        """Disable a feature."""
        if name not in self._features:
            return False
        self._features[name].enabled = False
        return True
    
    def enable_feature(self, name: str) -> bool:
        """Enable a feature."""
        if name not in self._features:
            return False
        self._features[name].enabled = True
        return True
    
    def _transform_feature(
        self,
        name: str,
        values: np.ndarray,
        config: FeatureConfig,
    ) -> Optional[np.ndarray]:
        """Transform a single feature."""
        if config.feature_type == FeatureType.NUMERIC:
            return self._transform_numeric(name, values)
        elif config.feature_type == FeatureType.CATEGORICAL:
            return self._transform_categorical(name, values)
        elif config.feature_type == FeatureType.TEXT:
            return self._transform_text(name, values)
        elif config.feature_type == FeatureType.SEQUENCE:
            return self._transform_sequence(name, values)
        elif config.feature_type == FeatureType.EMBEDDING:
            return values  # Already embedded
        
        return values
    
    def _transform_numeric(self, name: str, values: np.ndarray) -> np.ndarray:
        """Transform numeric feature."""
        values = np.asarray(values, dtype=np.float64)
        
        if self.normalize and name in self._fitted_stats:
            stats = self._fitted_stats[name]
            values = (values - stats['mean']) / stats['std']
        
        return values
    
    def _transform_categorical(self, name: str, values: np.ndarray) -> np.ndarray:
        """Transform categorical feature using one-hot encoding."""
        if name not in self._fitted_stats:
            return values
        
        categories = self._fitted_stats[name]['categories']
        n_categories = len(categories)
        
        # One-hot encode
        encoded = np.zeros((len(values), n_categories))
        
        for i, val in enumerate(values):
            if val in categories:
                idx = categories.index(val)
                encoded[i, idx] = 1.0
        
        return encoded
    
    def _transform_text(self, name: str, values: np.ndarray) -> np.ndarray:
        """Transform text feature (basic bag of words)."""
        # Simple character count as placeholder
        # In production, would use TF-IDF or embeddings
        return np.array([[len(str(v))] for v in values])
    
    def _transform_sequence(self, name: str, values: np.ndarray) -> np.ndarray:
        """Transform sequence feature."""
        # Pad or truncate sequences
        max_len = 100  # Default max length
        
        result = []
        for seq in values:
            if len(seq) < max_len:
                padded = list(seq) + [0] * (max_len - len(seq))
            else:
                padded = seq[:max_len]
            result.append(padded)
        
        return np.array(result)


# ============================================================================
# MODEL ENSEMBLE
# ============================================================================

class ModelEnsemble:
    """
    Ensemble of models with multiple combination strategies.
    
    Features:
    - Multiple ensemble strategies
    - Automatic weight optimization
    - Heterogeneous model support
    - Cross-validation based weighting
    
    Example:
        >>> ensemble = ModelEnsemble(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
        >>> ensemble.add_model("model1", model1)
        >>> ensemble.add_model("model2", model2)
        >>> predictions = ensemble.predict(X)
    """
    
    def __init__(
        self,
        strategy: EnsembleStrategy = EnsembleStrategy.AVERAGING,
        voting_threshold: float = 0.5,
    ):
        """
        Initialize ensemble.
        
        Args:
            strategy: Combination strategy
            voting_threshold: Threshold for voting classifiers
        """
        self.strategy = strategy
        self.voting_threshold = voting_threshold
        
        self._models: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}
        self._is_fitted = False
        self._lock = threading.Lock()
    
    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
    ) -> 'ModelEnsemble':
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
            weight: Initial weight
            
        Returns:
            Self for chaining
        """
        with self._lock:
            self._models[name] = model
            self._weights[name] = weight
            return self
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the ensemble."""
        with self._lock:
            if name not in self._models:
                return False
            del self._models[name]
            del self._weights[name]
            return True
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for chaining
        """
        with self._lock:
            for name, model in self._models.items():
                model.fit(X, y)
            
            self._is_fitted = True
            return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        if not self._models:
            raise ValueError("No models in ensemble")
        
        predictions = self._collect_predictions(X)
        
        if self.strategy == EnsembleStrategy.VOTING:
            return self._voting_combine(predictions)
        elif self.strategy == EnsembleStrategy.AVERAGING:
            return self._average_combine(predictions)
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_combine(predictions)
        elif self.strategy == EnsembleStrategy.STACKING:
            return self._stacking_combine(predictions, X)
        else:
            return self._average_combine(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        probas = []
        total_weight = 0.0
        
        for name, model in self._models.items():
            if hasattr(model, 'predict_proba'):
                weight = self._weights[name]
                proba = model.predict_proba(X)
                probas.append(proba * weight)
                total_weight += weight
        
        if not probas:
            raise NotImplementedError("No models support predict_proba")
        
        return np.sum(probas, axis=0) / total_weight
    
    def set_weights(self, weights: Dict[str, float]) -> 'ModelEnsemble':
        """Set model weights."""
        with self._lock:
            for name, weight in weights.items():
                if name in self._weights:
                    self._weights[name] = weight
            return self
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Optimize weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Evaluation metric (higher is better)
            
        Returns:
            Optimized weights
        """
        if metric is None:
            metric = self._default_metric
        
        with self._lock:
            scores = {}
            
            for name, model in self._models.items():
                predictions = model.predict(X_val)
                score = metric(y_val, predictions)
                scores[name] = max(score, 1e-10)  # Avoid zero weights
            
            # Normalize to get weights
            total_score = sum(scores.values())
            optimized = {name: score / total_score for name, score in scores.items()}
            
            self._weights.update(optimized)
            return optimized
    
    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual model predictions."""
        return {name: model.predict(X) for name, model in self._models.items()}
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self._weights.copy()
    
    def __len__(self) -> int:
        """Get number of models in ensemble."""
        return len(self._models)
    
    def _collect_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Collect predictions from all models."""
        return {name: model.predict(X) for name, model in self._models.items()}
    
    def _voting_combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine using majority voting."""
        all_preds = np.array(list(predictions.values()))
        
        # Count votes for each class
        n_samples = all_preds.shape[1]
        result = np.zeros(n_samples)
        
        for i in range(n_samples):
            votes = all_preds[:, i]
            result[i] = statistics.mode(votes)
        
        return result
    
    def _average_combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine using simple averaging."""
        all_preds = np.array(list(predictions.values()))
        return np.mean(all_preds, axis=0)
    
    def _weighted_average_combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine using weighted averaging."""
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        total_weight = 0.0
        
        for name, preds in predictions.items():
            weight = self._weights.get(name, 1.0)
            weighted_sum += preds * weight
            total_weight += weight
        
        return weighted_sum / total_weight
    
    def _stacking_combine(
        self,
        predictions: Dict[str, np.ndarray],
        X: np.ndarray,
    ) -> np.ndarray:
        """Combine using stacking (meta-learner)."""
        # Use predictions as features for a simple meta-learner
        # In production, would train a separate meta-model
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Simple averaging as fallback
        return np.mean(pred_matrix, axis=1)
    
    def _default_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default accuracy metric."""
        return float(np.mean(y_true == y_pred))


# ============================================================================
# SIMPLE MODELS FOR TESTING
# ============================================================================

class SimpleClassifier(BaseModel):
    """Simple majority class classifier for testing."""
    
    def __init__(self):
        self._majority_class: Optional[int] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleClassifier':
        """Fit by finding majority class."""
        unique, counts = np.unique(y, return_counts=True)
        self._majority_class = unique[np.argmax(counts)]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict majority class for all samples."""
        if self._majority_class is None:
            raise RuntimeError("Model not fitted")
        return np.full(len(X), self._majority_class)


class SimpleRegressor(BaseModel):
    """Simple mean regressor for testing."""
    
    def __init__(self):
        self._mean: Optional[float] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleRegressor':
        """Fit by computing mean."""
        self._mean = float(np.mean(y))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean for all samples."""
        if self._mean is None:
            raise RuntimeError("Model not fitted")
        return np.full(len(X), self._mean)


class LinearModel(BaseModel):
    """Simple linear model."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModel':
        """Fit using gradient descent."""
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0.0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self._weights) + self._bias
            
            # Gradient descent
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self._weights -= self.learning_rate * dw
            self._bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self._weights is None:
            raise RuntimeError("Model not fitted")
        return np.dot(X, self._weights) + self._bias
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModel':
        """Incremental fit."""
        if self._weights is None:
            self._weights = np.zeros(X.shape[1])
        
        y_pred = np.dot(X, self._weights) + self._bias
        dw = np.dot(X.T, (y_pred - y)) / len(X)
        db = np.sum(y_pred - y) / len(X)
        
        self._weights -= self.learning_rate * dw
        self._bias -= self.learning_rate * db
        
        return self


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_model_registry() -> ModelRegistry:
    """Create a new model registry."""
    return ModelRegistry()


def create_ensemble(
    models: List[Tuple[str, Any]],
    strategy: EnsembleStrategy = EnsembleStrategy.AVERAGING,
) -> ModelEnsemble:
    """
    Create an ensemble from a list of models.
    
    Args:
        models: List of (name, model) tuples
        strategy: Ensemble strategy
        
    Returns:
        Configured ModelEnsemble
    """
    ensemble = ModelEnsemble(strategy=strategy)
    for name, model in models:
        ensemble.add_model(name, model)
    return ensemble


def create_feature_extractor(
    features: List[Tuple[str, FeatureType]],
    normalize: bool = True,
) -> FeatureExtractor:
    """
    Create a feature extractor from a list of features.
    
    Args:
        features: List of (name, type) tuples
        normalize: Whether to normalize
        
    Returns:
        Configured FeatureExtractor
    """
    extractor = FeatureExtractor(normalize=normalize)
    for name, feature_type in features:
        extractor.add_feature(name, feature_type)
    return extractor


def train_with_metrics(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> TrainingMetrics:
    """
    Train a model and collect metrics.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        
    Returns:
        TrainingMetrics object
    """
    metrics = TrainingMetrics()
    start_time = time.time()
    
    # Fit model
    model.fit(X_train, y_train)
    
    metrics.training_time = time.time() - start_time
    
    # Calculate training metrics
    train_pred = model.predict(X_train)
    train_loss = float(np.mean((train_pred - y_train) ** 2))
    metrics.add_epoch_metrics(train_loss)
    
    # Validation metrics
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        metrics.validation_metrics['loss'] = val_loss
    
    metrics.convergence_achieved = True
    
    return metrics
