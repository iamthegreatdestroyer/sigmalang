"""
Transformer Embeddings Integration for sigmalang.

This module provides transformer-based embeddings with hybrid HD + transformer
similarity computation, efficient caching, and graceful fallback when
dependencies are unavailable.

Classes:
    PoolingStrategy: Pooling strategies for transformer outputs
    EmbeddingConfig: Configuration for embedding models
    TransformerEncoder: Main transformer encoding interface
    HybridEncoder: Combined HD + Transformer encoding
    EmbeddingCache: Efficient embedding caching
    TransformerEmbeddings: Unified embedding interface

Example:
    >>> encoder = TransformerEncoder()
    >>> embedding = encoder.encode_text("Hello world")
    >>> similarity = encoder.similarity("Hello", "Hi there")
    >>> print(f"Similarity: {similarity:.3f}")  # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class PoolingStrategy(Enum):
    """Pooling strategies for transformer outputs."""
    MEAN = auto()      # Mean pooling over tokens
    CLS = auto()       # Use [CLS] token
    MAX = auto()       # Max pooling over tokens
    FIRST_LAST = auto() # Concatenate first and last


class EmbeddingModelType(Enum):
    """Types of embedding models."""
    TRANSFORMER = auto()     # sentence-transformers
    TFIDF = auto()           # TF-IDF fallback
    HD = auto()              # Hyperdimensional only
    RANDOM = auto()          # Random projection (baseline)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.
    
    Attributes:
        model_name: Name of the transformer model
        dimensionality: Output embedding dimension
        pooling: Pooling strategy for transformer outputs
        normalize: Whether to L2-normalize embeddings
        batch_size: Batch size for encoding
        device: Device to run on ('cpu', 'cuda', 'auto')
        cache_size: Maximum number of cached embeddings
        fallback_dim: Dimension for fallback encoders
    """
    model_name: str = "all-MiniLM-L6-v2"
    dimensionality: int = 384
    pooling: PoolingStrategy = PoolingStrategy.MEAN
    normalize: bool = True
    batch_size: int = 32
    device: str = "auto"
    cache_size: int = 10000
    fallback_dim: int = 384


@dataclass
class ModelInfo:
    """Information about the embedding model.
    
    Attributes:
        name: Model name
        type: Model type
        dimensionality: Output dimension
        vocab_size: Vocabulary size (if applicable)
        parameters: Number of parameters
        description: Model description
    """
    name: str
    type: EmbeddingModelType
    dimensionality: int
    vocab_size: Optional[int] = None
    parameters: Optional[int] = None
    description: str = ""


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.
    
    Attributes:
        embedding: The embedding vector
        model_type: Type of model used
        computation_time: Time to compute (seconds)
        from_cache: Whether result was from cache
        metadata: Additional metadata
    """
    embedding: np.ndarray
    model_type: EmbeddingModelType
    computation_time: float
    from_cache: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEncoder(ABC):
    """Abstract base class for encoders."""
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embeddings."""
        pass
    
    @abstractmethod
    def get_dimensionality(self) -> int:
        """Get output dimensionality."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        pass


class TFIDFEncoder(BaseEncoder):
    """TF-IDF based fallback encoder.
    
    Uses simple TF-IDF vectorization with dimensionality reduction
    when transformer models are unavailable.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize TF-IDF encoder.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.dimensionality = config.fallback_dim
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        self._documents_seen = 0
        self._lock = threading.Lock()
        
        # Initialize random projection matrix for dimensionality reduction
        np.random.seed(42)  # Deterministic
        self._init_projection()
    
    def _init_projection(self):
        """Initialize random projection matrix."""
        # Start with a reasonable vocabulary size estimate
        max_vocab = 10000
        self._projection_matrix = np.random.randn(
            max_vocab, self.dimensionality
        ) / np.sqrt(self.dimensionality)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Basic word tokenization
        text = text.lower()
        # Remove punctuation and split
        tokens = []
        current_token = []
        for char in text:
            if char.isalnum():
                current_token.append(char)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
        if current_token:
            tokens.append(''.join(current_token))
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalize by document length
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}
    
    def _get_token_index(self, token: str) -> int:
        """Get or create index for token."""
        if token not in self._vocabulary:
            self._vocabulary[token] = len(self._vocabulary)
        return self._vocabulary[token]
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using TF-IDF with random projection.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimensionality)
        
        tf = self._compute_tf(tokens)
        
        with self._lock:
            # Update IDF estimates
            self._documents_seen += 1
            for token in set(tokens):
                self._idf[token] = self._idf.get(token, 0) + 1
            
            # Compute TF-IDF vector
            sparse_vec = np.zeros(len(self._projection_matrix))
            for token, tf_val in tf.items():
                idx = self._get_token_index(token)
                if idx < len(sparse_vec):
                    idf = np.log(self._documents_seen / (self._idf.get(token, 1)))
                    sparse_vec[idx] = tf_val * max(idf, 0.1)
        
        # Project to lower dimension
        embedding = sparse_vec[:self._projection_matrix.shape[0]] @ self._projection_matrix
        
        # Normalize if configured
        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings (n_texts, dimensionality)
        """
        return np.array([self.encode(text) for text in texts])
    
    def get_dimensionality(self) -> int:
        """Get output dimensionality."""
        return self.dimensionality
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name="TF-IDF Encoder",
            type=EmbeddingModelType.TFIDF,
            dimensionality=self.dimensionality,
            vocab_size=len(self._vocabulary),
            description="TF-IDF with random projection fallback"
        )


class RandomEncoder(BaseEncoder):
    """Random projection encoder (baseline).
    
    Generates deterministic random embeddings based on text hash.
    Useful as a baseline for comparison.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize random encoder.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.dimensionality = config.fallback_dim
    
    def _text_to_seed(self, text: str) -> int:
        """Convert text to deterministic seed."""
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to random embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Deterministic random embedding
        """
        seed = self._text_to_seed(text)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimensionality).astype(np.float32)
        
        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return np.array([self.encode(text) for text in texts])
    
    def get_dimensionality(self) -> int:
        """Get output dimensionality."""
        return self.dimensionality
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name="Random Encoder",
            type=EmbeddingModelType.RANDOM,
            dimensionality=self.dimensionality,
            description="Deterministic random projection baseline"
        )


class TransformerEncoderImpl(BaseEncoder):
    """Transformer-based encoder using sentence-transformers.
    
    Requires sentence-transformers package to be installed.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize transformer encoder.
        
        Args:
            config: Embedding configuration
            
        Raises:
            ImportError: If sentence-transformers not available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
        
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model."""
        device = self.config.device
        if device == "auto":
            device = None  # Let sentence-transformers decide
        
        self._model = SentenceTransformer(
            self.config.model_name,
            device=device
        )
        # Update dimensionality from model
        self.config.dimensionality = self._model.get_sentence_embedding_dimension()
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using transformer model.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        with self._lock:
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False
            )
        return embedding.astype(np.float32)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        with self._lock:
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False
            )
        return embeddings.astype(np.float32)
    
    def get_dimensionality(self) -> int:
        """Get output dimensionality."""
        return self.config.dimensionality
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name=self.config.model_name,
            type=EmbeddingModelType.TRANSFORMER,
            dimensionality=self.config.dimensionality,
            description=f"Sentence Transformer: {self.config.model_name}"
        )


class EmbeddingCache:
    """LRU cache for embeddings.
    
    Thread-safe cache with configurable size and TTL.
    
    Example:
        >>> import numpy as np
        >>> cache = EmbeddingCache(max_size=1000)
        >>> embedding = np.array([0.1, 0.2, 0.3])
        >>> cache.put("key1", embedding)
        >>> result = cache.get("key1")
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None
    ):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, text: str, model_name: str) -> str:
        """Compute cache key for text and model."""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def get(
        self,
        text: str,
        model_name: str = "default"
    ) -> Optional[np.ndarray]:
        """Get embedding from cache.
        
        Args:
            text: Text that was encoded
            model_name: Name of model used
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._compute_key(text, model_name)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            embedding, timestamp = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds is not None:
                if time.time() - timestamp > self.ttl_seconds:
                    del self._cache[key]
                    self._misses += 1
                    return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return embedding.copy()
    
    def put(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str = "default"
    ):
        """Put embedding in cache.
        
        Args:
            text: Text that was encoded
            embedding: Embedding vector
            model_name: Name of model used
        """
        key = self._compute_key(text, model_name)
        
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (embedding.copy(), time.time())
    
    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], np.ndarray],
        model_name: str = "default"
    ) -> Tuple[np.ndarray, bool]:
        """Get from cache or compute and cache.
        
        Args:
            text: Text to encode
            compute_fn: Function to compute embedding if not cached
            model_name: Name of model
            
        Returns:
            Tuple of (embedding, from_cache)
        """
        cached = self.get(text, model_name)
        if cached is not None:
            return cached, True
        
        embedding = compute_fn(text)
        self.put(text, embedding, model_name)
        return embedding, False
    
    def clear(self):
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }
    
    def precompute_batch(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], np.ndarray],
        model_name: str = "default"
    ) -> int:
        """Precompute and cache embeddings for multiple texts.
        
        Args:
            texts: List of texts to precompute
            compute_fn: Function to compute embeddings in batch
            model_name: Name of model
            
        Returns:
            Number of new embeddings computed
        """
        # Find texts not in cache
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if self.get(text, model_name) is None:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return 0
        
        # Compute in batch
        embeddings = compute_fn(uncached_texts)
        
        # Cache results
        for text, embedding in zip(uncached_texts, embeddings):
            self.put(text, embedding, model_name)
        
        return len(uncached_texts)


class TransformerEncoder:
    """Main transformer encoding interface.
    
    Provides a unified interface for encoding text with automatic
    fallback when transformers are unavailable.
    
    Example:
        >>> encoder = TransformerEncoder()
        >>> embedding = encoder.encode_text("Hello world")
        >>> similarity = encoder.similarity("Hello", "Hi there")
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        use_cache: bool = True
    ):
        """Initialize transformer encoder.
        
        Args:
            config: Embedding configuration
            use_cache: Whether to use embedding cache
        """
        self.config = config or EmbeddingConfig()
        self._encoder: Optional[BaseEncoder] = None
        self._cache: Optional[EmbeddingCache] = None
        self._lock = threading.Lock()
        
        if use_cache:
            self._cache = EmbeddingCache(
                max_size=self.config.cache_size
            )
        
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the appropriate encoder."""
        # Try transformer first
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._encoder = TransformerEncoderImpl(self.config)
                return
            except Exception:
                pass  # Fall through to fallback
        
        # Fall back to TF-IDF
        self._encoder = TFIDFEncoder(self.config)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text to embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        if self._cache is not None:
            model_name = self._encoder.get_model_info().name
            embedding, _ = self._cache.get_or_compute(
                text,
                lambda t: self._encoder.encode(t),
                model_name
            )
            return embedding
        
        return self._encoder.encode(text)
    
    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode multiple texts to embeddings.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress (ignored for now)
            
        Returns:
            Array of embeddings (n_texts, dimensionality)
        """
        if not texts:
            return np.array([]).reshape(0, self.get_dimensionality())
        
        if self._cache is not None:
            model_name = self._encoder.get_model_info().name
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache
            for i, text in enumerate(texts):
                cached = self._cache.get(text, model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Compute uncached
            if uncached_texts:
                new_embeddings = self._encoder.encode_batch(uncached_texts)
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                    self._cache.put(text, emb, model_name)
                    embeddings.append((idx, emb))
            
            # Sort by original index
            embeddings.sort(key=lambda x: x[0])
            return np.array([e[1] for e in embeddings])
        
        return self._encoder.encode_batch(texts)
    
    def encode_tree(self, tree: Any) -> np.ndarray:
        """Encode a sigmalang tree to embedding.
        
        Args:
            tree: Sigmalang semantic tree
            
        Returns:
            Embedding vector
        """
        # Extract text representation from tree
        if hasattr(tree, 'to_text'):
            text = tree.to_text()
        elif hasattr(tree, '__str__'):
            text = str(tree)
        else:
            text = repr(tree)
        
        return self.encode_text(text)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        return self._cosine_similarity(emb1, emb2)
    
    def similarity_batch(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """Compute similarity between query and multiple candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            
        Returns:
            List of similarity scores
        """
        query_emb = self.encode_text(query)
        candidate_embs = self.encode_batch(candidates)
        
        return [
            self._cosine_similarity(query_emb, cand_emb)
            for cand_emb in candidate_embs
        ]
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def get_dimensionality(self) -> int:
        """Get output embedding dimensionality."""
        return self._encoder.get_dimensionality()
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the model being used."""
        return self._encoder.get_model_info()
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self._cache is not None:
            return self._cache.get_stats()
        return None
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()


class HybridEncoder:
    """Combined HD + Transformer encoding.
    
    Creates hybrid embeddings by combining hyperdimensional encodings
    with transformer embeddings, allowing weighted similarity computation.
    
    Example:
        >>> hybrid = HybridEncoder()
        >>> embedding = hybrid.encode_hybrid("Hello world")
        >>> similarity = hybrid.similarity_hybrid("Hello", "Hi there")
    """
    
    def __init__(
        self,
        transformer_config: Optional[EmbeddingConfig] = None,
        hd_encoder: Optional[Any] = None,
        transformer_weight: float = 0.7,
        hd_weight: float = 0.3
    ):
        """Initialize hybrid encoder.
        
        Args:
            transformer_config: Config for transformer encoder
            hd_encoder: Optional HD encoder instance
            transformer_weight: Weight for transformer similarity
            hd_weight: Weight for HD similarity
        """
        self.transformer_encoder = TransformerEncoder(transformer_config)
        self.hd_encoder = hd_encoder
        self._transformer_weight = transformer_weight
        self._hd_weight = hd_weight
        self._lock = threading.Lock()
        
        # Normalize weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = self._transformer_weight + self._hd_weight
        if total > 0:
            self._transformer_weight /= total
            self._hd_weight /= total
    
    def configure_weights(
        self,
        transformer_weight: float,
        hd_weight: float
    ):
        """Configure similarity weights.
        
        Args:
            transformer_weight: Weight for transformer similarity
            hd_weight: Weight for HD similarity
        """
        with self._lock:
            self._transformer_weight = transformer_weight
            self._hd_weight = hd_weight
            self._normalize_weights()
    
    def get_weights(self) -> Tuple[float, float]:
        """Get current weights.
        
        Returns:
            Tuple of (transformer_weight, hd_weight)
        """
        return self._transformer_weight, self._hd_weight
    
    def encode_hybrid(
        self,
        text: str
    ) -> Dict[str, np.ndarray]:
        """Create hybrid embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Dictionary with 'transformer' and optionally 'hd' embeddings
        """
        result = {
            'transformer': self.transformer_encoder.encode_text(text)
        }
        
        if self.hd_encoder is not None:
            try:
                if hasattr(self.hd_encoder, 'encode'):
                    result['hd'] = self.hd_encoder.encode(text)
                elif hasattr(self.hd_encoder, 'encode_text'):
                    result['hd'] = self.hd_encoder.encode_text(text)
            except Exception:
                pass  # HD encoding optional
        
        return result
    
    def similarity_hybrid(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute weighted hybrid similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Weighted similarity score
        """
        emb1 = self.encode_hybrid(text1)
        emb2 = self.encode_hybrid(text2)
        
        # Transformer similarity
        tf_sim = self._cosine_similarity(
            emb1['transformer'],
            emb2['transformer']
        )
        
        # HD similarity (if available)
        hd_sim = 0.0
        if 'hd' in emb1 and 'hd' in emb2:
            hd_sim = self._cosine_similarity(emb1['hd'], emb2['hd'])
        
        # Weighted combination
        with self._lock:
            if 'hd' in emb1 and 'hd' in emb2:
                return (
                    self._transformer_weight * tf_sim +
                    self._hd_weight * hd_sim
                )
            else:
                return tf_sim
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def adaptive_weighting(
        self,
        query: str,
        candidates: List[str],
        relevance_scores: List[float]
    ) -> Tuple[float, float]:
        """Learn optimal weights from relevance feedback.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            relevance_scores: Ground truth relevance scores [0, 1]
            
        Returns:
            Optimized (transformer_weight, hd_weight)
        """
        if len(candidates) != len(relevance_scores):
            raise ValueError("Candidates and scores must have same length")
        
        if not candidates:
            return self.get_weights()
        
        # Compute similarities for different weight combinations
        best_weights = self.get_weights()
        best_correlation = -float('inf')
        
        for tf_w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            hd_w = 1.0 - tf_w
            self.configure_weights(tf_w, hd_w)
            
            # Compute predicted similarities
            predicted = [
                self.similarity_hybrid(query, cand)
                for cand in candidates
            ]
            
            # Compute correlation with ground truth
            correlation = self._correlation(predicted, relevance_scores)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_weights = (tf_w, hd_w)
        
        # Set optimal weights
        self.configure_weights(*best_weights)
        return best_weights
    
    def _correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Compute Pearson correlation."""
        if len(x) < 2:
            return 0.0
        
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)
        
        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sqrt(
            np.sum((x_arr - x_mean) ** 2) *
            np.sum((y_arr - y_mean) ** 2)
        )
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about hybrid encoder."""
        info = {
            'transformer': self.transformer_encoder.get_model_info().__dict__,
            'weights': {
                'transformer': self._transformer_weight,
                'hd': self._hd_weight
            },
            'hd_available': self.hd_encoder is not None
        }
        return info


class TransformerEmbeddings:
    """Unified embedding interface for sigmalang.
    
    High-level interface combining all embedding functionality
    with automatic model selection and caching.
    
    Example:
        >>> embeddings = TransformerEmbeddings()
        >>> vec = embeddings.embed("Hello world")
        >>> similar = embeddings.find_similar("query", ["a", "b", "c"])
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        hd_encoder: Optional[Any] = None,
        use_hybrid: bool = True
    ):
        """Initialize unified embedding interface.
        
        Args:
            config: Embedding configuration
            hd_encoder: Optional HD encoder for hybrid mode
            use_hybrid: Whether to use hybrid encoding
        """
        self.config = config or EmbeddingConfig()
        
        if use_hybrid:
            self._encoder = HybridEncoder(
                transformer_config=self.config,
                hd_encoder=hd_encoder
            )
        else:
            self._encoder = TransformerEncoder(self.config)
        
        self._use_hybrid = use_hybrid
    
    def embed(self, text: str) -> np.ndarray:
        """Embed text to vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self._use_hybrid:
            result = self._encoder.encode_hybrid(text)
            return result['transformer']
        else:
            return self._encoder.encode_text(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of embeddings
        """
        if self._use_hybrid:
            return np.array([
                self._encoder.encode_hybrid(t)['transformer']
                for t in texts
            ])
        else:
            return self._encoder.encode_batch(texts)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        if self._use_hybrid:
            return self._encoder.similarity_hybrid(text1, text2)
        else:
            return self._encoder.similarity(text1, text2)
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Find most similar candidates to query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Maximum results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, similarity) tuples, sorted by similarity
        """
        if not candidates:
            return []
        
        # Compute similarities
        similarities = [
            (cand, self.similarity(query, cand))
            for cand in candidates
        ]
        
        # Filter by threshold
        if threshold is not None:
            similarities = [
                (c, s) for c, s in similarities
                if s >= threshold
            ]
        
        # Sort and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def cluster_by_similarity(
        self,
        texts: List[str],
        threshold: float = 0.7
    ) -> List[List[str]]:
        """Cluster texts by similarity.
        
        Simple greedy clustering - assigns each text to the most
        similar existing cluster or creates a new cluster.
        
        Args:
            texts: List of texts to cluster
            threshold: Similarity threshold for same cluster
            
        Returns:
            List of clusters (each cluster is a list of texts)
        """
        if not texts:
            return []
        
        clusters: List[List[str]] = [[texts[0]]]
        
        for text in texts[1:]:
            best_cluster = -1
            best_similarity = threshold
            
            for i, cluster in enumerate(clusters):
                # Compare to first item in cluster (centroid approximation)
                sim = self.similarity(text, cluster[0])
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = i
            
            if best_cluster >= 0:
                clusters[best_cluster].append(text)
            else:
                clusters.append([text])
        
        return clusters
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._use_hybrid:
            return self._encoder.get_model_info()
        else:
            return self._encoder.get_model_info().__dict__
    
    def is_transformer_available(self) -> bool:
        """Check if transformer models are available."""
        return SENTENCE_TRANSFORMERS_AVAILABLE


# Convenience functions

def create_embeddings(
    model_name: str = "all-MiniLM-L6-v2",
    use_cache: bool = True
) -> TransformerEmbeddings:
    """Create embedding interface with specified model.
    
    Args:
        model_name: Name of transformer model
        use_cache: Whether to use embedding cache
        
    Returns:
        TransformerEmbeddings instance
    """
    config = EmbeddingConfig(model_name=model_name)
    return TransformerEmbeddings(config=config)


def compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts.
    
    Convenience function using default configuration.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score
    """
    encoder = TransformerEncoder()
    return encoder.similarity(text1, text2)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed multiple texts.
    
    Convenience function using default configuration.
    
    Args:
        texts: List of texts
        
    Returns:
        Array of embeddings
    """
    encoder = TransformerEncoder()
    return encoder.encode_batch(texts)
