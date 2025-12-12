"""
Tests for Transformer Embeddings module.

This test suite covers:
- EmbeddingConfig dataclass
- TransformerEncoder and fallback encoders
- HybridEncoder with HD + Transformer
- EmbeddingCache with LRU eviction
- TransformerEmbeddings unified interface
"""

import time
import threading
import hashlib
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np

from core.transformer_embeddings import (
    # Enums
    PoolingStrategy,
    EmbeddingModelType,
    # Dataclasses
    EmbeddingConfig,
    ModelInfo,
    EmbeddingResult,
    # Encoders
    BaseEncoder,
    TFIDFEncoder,
    RandomEncoder,
    TransformerEncoderImpl,
    TransformerEncoder,
    HybridEncoder,
    # Cache
    EmbeddingCache,
    # Main interface
    TransformerEmbeddings,
    # Convenience functions
    create_embeddings,
    compute_similarity,
    embed_texts,
    # Module-level flag
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Create default embedding configuration."""
    return EmbeddingConfig()


@pytest.fixture
def small_config():
    """Create small dimension configuration for fast tests."""
    return EmbeddingConfig(fallback_dim=64, cache_size=100)


@pytest.fixture
def tfidf_encoder(small_config):
    """Create TF-IDF encoder with small dimension."""
    return TFIDFEncoder(small_config)


@pytest.fixture
def random_encoder(small_config):
    """Create random encoder with small dimension."""
    return RandomEncoder(small_config)


@pytest.fixture
def transformer_encoder(small_config):
    """Create transformer encoder (falls back if unavailable)."""
    return TransformerEncoder(config=small_config, use_cache=False)


@pytest.fixture
def cached_encoder(small_config):
    """Create transformer encoder with cache."""
    return TransformerEncoder(config=small_config, use_cache=True)


@pytest.fixture
def embedding_cache():
    """Create embedding cache for testing."""
    return EmbeddingCache(max_size=100)


@pytest.fixture
def hybrid_encoder(small_config):
    """Create hybrid encoder."""
    return HybridEncoder(transformer_config=small_config)


@pytest.fixture
def embeddings_interface(small_config):
    """Create TransformerEmbeddings interface."""
    return TransformerEmbeddings(config=small_config, use_hybrid=False)


# =============================================================================
# Test EmbeddingConfig
# =============================================================================

class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.dimensionality == 384
        assert config.pooling == PoolingStrategy.MEAN
        assert config.normalize is True
        assert config.batch_size == 32
        assert config.device == "auto"
        assert config.cache_size == 10000
        assert config.fallback_dim == 384
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="custom-model",
            dimensionality=768,
            pooling=PoolingStrategy.CLS,
            normalize=False,
            batch_size=64,
            device="cpu"
        )
        
        assert config.model_name == "custom-model"
        assert config.dimensionality == 768
        assert config.pooling == PoolingStrategy.CLS
        assert config.normalize is False
        assert config.batch_size == 64
        assert config.device == "cpu"
    
    def test_pooling_strategies(self):
        """Test all pooling strategies."""
        assert PoolingStrategy.MEAN is not None
        assert PoolingStrategy.CLS is not None
        assert PoolingStrategy.MAX is not None
        assert PoolingStrategy.FIRST_LAST is not None


class TestModelInfo:
    """Tests for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test model info creation."""
        info = ModelInfo(
            name="test-model",
            type=EmbeddingModelType.TRANSFORMER,
            dimensionality=384,
            vocab_size=30000,
            parameters=22000000,
            description="Test model"
        )
        
        assert info.name == "test-model"
        assert info.type == EmbeddingModelType.TRANSFORMER
        assert info.dimensionality == 384
        assert info.vocab_size == 30000
        assert info.parameters == 22000000
        assert info.description == "Test model"
    
    def test_optional_fields(self):
        """Test model info with optional fields."""
        info = ModelInfo(
            name="minimal",
            type=EmbeddingModelType.TFIDF,
            dimensionality=64
        )
        
        assert info.vocab_size is None
        assert info.parameters is None
        assert info.description == ""


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""
    
    def test_embedding_result(self):
        """Test embedding result creation."""
        embedding = np.random.randn(64).astype(np.float32)
        result = EmbeddingResult(
            embedding=embedding,
            model_type=EmbeddingModelType.TRANSFORMER,
            computation_time=0.05,
            from_cache=False
        )
        
        assert np.array_equal(result.embedding, embedding)
        assert result.model_type == EmbeddingModelType.TRANSFORMER
        assert result.computation_time == 0.05
        assert result.from_cache is False
        assert result.metadata == {}


# =============================================================================
# Test TFIDFEncoder
# =============================================================================

class TestTFIDFEncoder:
    """Tests for TF-IDF fallback encoder."""
    
    def test_encode_single_text(self, tfidf_encoder):
        """Test encoding a single text."""
        embedding = tfidf_encoder.encode("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 64
    
    def test_encode_empty_text(self, tfidf_encoder):
        """Test encoding empty text."""
        embedding = tfidf_encoder.encode("")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 64
        assert np.allclose(embedding, 0)
    
    def test_encode_batch(self, tfidf_encoder):
        """Test batch encoding."""
        texts = ["Hello world", "How are you", "Testing batch"]
        embeddings = tfidf_encoder.encode_batch(texts)
        
        assert embeddings.shape == (3, 64)
    
    def test_normalized_output(self, small_config):
        """Test that output is normalized when configured."""
        small_config.normalize = True
        encoder = TFIDFEncoder(small_config)
        
        embedding = encoder.encode("Normalize this text please")
        norm = np.linalg.norm(embedding)
        
        # Should be approximately 1.0 (normalized)
        assert abs(norm - 1.0) < 0.01 or norm == 0
    
    def test_unnormalized_output(self, small_config):
        """Test unnormalized output."""
        small_config.normalize = False
        encoder = TFIDFEncoder(small_config)
        
        embedding = encoder.encode("Do not normalize")
        # Just verify it runs without error
        assert isinstance(embedding, np.ndarray)
    
    def test_vocabulary_growth(self, tfidf_encoder):
        """Test vocabulary grows with new words."""
        initial_vocab = len(tfidf_encoder._vocabulary)
        
        tfidf_encoder.encode("completely unique vocabulary")
        
        assert len(tfidf_encoder._vocabulary) > initial_vocab
    
    def test_tokenization(self, tfidf_encoder):
        """Test tokenization method."""
        tokens = tfidf_encoder._tokenize("Hello, World! How are you?")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens
    
    def test_get_dimensionality(self, tfidf_encoder):
        """Test dimensionality getter."""
        assert tfidf_encoder.get_dimensionality() == 64
    
    def test_get_model_info(self, tfidf_encoder):
        """Test model info getter."""
        info = tfidf_encoder.get_model_info()
        
        assert info.name == "TF-IDF Encoder"
        assert info.type == EmbeddingModelType.TFIDF
        assert info.dimensionality == 64


# =============================================================================
# Test RandomEncoder
# =============================================================================

class TestRandomEncoder:
    """Tests for random projection encoder."""
    
    def test_encode_deterministic(self, random_encoder):
        """Test that encoding is deterministic."""
        text = "Hello world"
        
        embedding1 = random_encoder.encode(text)
        embedding2 = random_encoder.encode(text)
        
        assert np.allclose(embedding1, embedding2)
    
    def test_different_texts_different_embeddings(self, random_encoder):
        """Test different texts produce different embeddings."""
        emb1 = random_encoder.encode("Hello")
        emb2 = random_encoder.encode("World")
        
        assert not np.allclose(emb1, emb2)
    
    def test_encode_batch(self, random_encoder):
        """Test batch encoding."""
        texts = ["A", "B", "C"]
        embeddings = random_encoder.encode_batch(texts)
        
        assert embeddings.shape == (3, 64)
    
    def test_normalized_output(self, small_config):
        """Test normalized output."""
        small_config.normalize = True
        encoder = RandomEncoder(small_config)
        
        embedding = encoder.encode("Test")
        norm = np.linalg.norm(embedding)
        
        assert abs(norm - 1.0) < 0.01
    
    def test_get_model_info(self, random_encoder):
        """Test model info."""
        info = random_encoder.get_model_info()
        
        assert info.type == EmbeddingModelType.RANDOM
        assert info.dimensionality == 64


# =============================================================================
# Test TransformerEncoder
# =============================================================================

class TestTransformerEncoder:
    """Tests for main TransformerEncoder interface."""
    
    def test_initialization(self, transformer_encoder):
        """Test encoder initialization."""
        assert transformer_encoder._encoder is not None
    
    def test_encode_text(self, transformer_encoder):
        """Test text encoding."""
        embedding = transformer_encoder.encode_text("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_encode_batch(self, transformer_encoder):
        """Test batch encoding."""
        texts = ["First", "Second", "Third"]
        embeddings = transformer_encoder.encode_batch(texts)
        
        assert embeddings.shape[0] == 3
    
    def test_encode_empty_batch(self, transformer_encoder):
        """Test empty batch encoding."""
        embeddings = transformer_encoder.encode_batch([])
        
        assert embeddings.shape[0] == 0
    
    def test_similarity(self, transformer_encoder):
        """Test similarity computation."""
        sim = transformer_encoder.similarity("Hello", "Hello")
        
        # Same text should have high similarity
        assert sim > 0.9
    
    def test_similarity_different_texts(self, transformer_encoder):
        """Test similarity for different texts."""
        sim = transformer_encoder.similarity("Hello", "Goodbye")
        
        # Different texts should have lower similarity
        assert -1.0 <= sim <= 1.0
    
    def test_similarity_batch(self, transformer_encoder):
        """Test batch similarity."""
        query = "Hello"
        candidates = ["Hello", "Hi", "Goodbye"]
        
        sims = transformer_encoder.similarity_batch(query, candidates)
        
        assert len(sims) == 3
        assert sims[0] > sims[2]  # "Hello" more similar to "Hello" than "Goodbye"
    
    def test_get_dimensionality(self, transformer_encoder):
        """Test dimensionality getter."""
        dim = transformer_encoder.get_dimensionality()
        assert dim > 0
    
    def test_get_model_info(self, transformer_encoder):
        """Test model info getter."""
        info = transformer_encoder.get_model_info()
        
        assert info.dimensionality > 0
        assert info.name is not None
    
    def test_encode_tree_with_to_text(self, transformer_encoder):
        """Test encoding tree with to_text method."""
        mock_tree = Mock()
        mock_tree.to_text.return_value = "Tree text content"
        
        embedding = transformer_encoder.encode_tree(mock_tree)
        
        assert isinstance(embedding, np.ndarray)
        mock_tree.to_text.assert_called_once()
    
    def test_encode_tree_with_str(self, transformer_encoder):
        """Test encoding tree with __str__."""
        mock_tree = Mock(spec=['__str__'])
        mock_tree.__str__ = Mock(return_value="String representation")
        
        embedding = transformer_encoder.encode_tree(mock_tree)
        
        assert isinstance(embedding, np.ndarray)


# =============================================================================
# Test EmbeddingCache
# =============================================================================

class TestEmbeddingCache:
    """Tests for embedding cache."""
    
    def test_put_and_get(self, embedding_cache):
        """Test basic put and get."""
        embedding = np.random.randn(64).astype(np.float32)
        
        embedding_cache.put("test", embedding)
        result = embedding_cache.get("test")
        
        assert np.allclose(result, embedding)
    
    def test_cache_miss(self, embedding_cache):
        """Test cache miss returns None."""
        result = embedding_cache.get("nonexistent")
        assert result is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = EmbeddingCache(max_size=3)
        
        # Add 3 items
        for i in range(3):
            cache.put(f"key{i}", np.array([i]))
        
        # Add 4th item (should evict first)
        cache.put("key3", np.array([3]))
        
        assert cache.get("key0") is None  # Evicted
        assert cache.get("key3") is not None  # Added
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=0.1)
        
        cache.put("expiring", np.array([1.0]))
        
        # Should exist immediately
        assert cache.get("expiring") is not None
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("expiring") is None
    
    def test_get_or_compute_cached(self, embedding_cache):
        """Test get_or_compute with cached value."""
        embedding = np.array([1.0, 2.0, 3.0])
        embedding_cache.put("key", embedding)
        
        compute_fn = Mock(return_value=np.array([4.0, 5.0, 6.0]))
        result, from_cache = embedding_cache.get_or_compute("key", compute_fn)
        
        assert from_cache is True
        assert np.allclose(result, embedding)
        compute_fn.assert_not_called()
    
    def test_get_or_compute_not_cached(self, embedding_cache):
        """Test get_or_compute without cached value."""
        embedding = np.array([1.0, 2.0, 3.0])
        compute_fn = Mock(return_value=embedding)
        
        result, from_cache = embedding_cache.get_or_compute("new_key", compute_fn)
        
        assert from_cache is False
        assert np.allclose(result, embedding)
        compute_fn.assert_called_once()
    
    def test_clear(self, embedding_cache):
        """Test clearing cache."""
        embedding_cache.put("key1", np.array([1.0]))
        embedding_cache.put("key2", np.array([2.0]))
        
        embedding_cache.clear()
        
        assert embedding_cache.get("key1") is None
        assert embedding_cache.get("key2") is None
    
    def test_get_stats(self, embedding_cache):
        """Test getting cache statistics."""
        embedding_cache.put("key", np.array([1.0]))
        embedding_cache.get("key")  # Hit
        embedding_cache.get("nonexistent")  # Miss
        
        stats = embedding_cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_precompute_batch(self, embedding_cache):
        """Test batch precomputation."""
        def compute_batch(texts):
            return np.array([[len(t)] for t in texts])
        
        # Precompute some
        embedding_cache.put("text1", np.array([5.0]))
        
        count = embedding_cache.precompute_batch(
            ["text1", "text2", "text3"],
            compute_batch
        )
        
        assert count == 2  # text1 was already cached
        assert embedding_cache.get("text2") is not None
        assert embedding_cache.get("text3") is not None
    
    def test_thread_safety(self):
        """Test cache is thread-safe."""
        cache = EmbeddingCache(max_size=1000)
        errors = []
        
        def writer(i):
            try:
                for j in range(100):
                    cache.put(f"key_{i}_{j}", np.array([float(i * j)]))
            except Exception as e:
                errors.append(e)
        
        def reader(i):
            try:
                for j in range(100):
                    cache.get(f"key_{i}_{j}")
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(writer, i))
                futures.append(executor.submit(reader, i))
            
            for f in futures:
                f.result()
        
        assert len(errors) == 0


# =============================================================================
# Test TransformerEncoder with Cache
# =============================================================================

class TestTransformerEncoderWithCache:
    """Tests for TransformerEncoder with caching enabled."""
    
    def test_cache_enabled(self, cached_encoder):
        """Test that cache is enabled."""
        assert cached_encoder._cache is not None
    
    def test_caching_works(self, cached_encoder):
        """Test that caching actually works."""
        text = "Cache this text"
        
        # First call - should compute
        cached_encoder.encode_text(text)
        stats1 = cached_encoder.get_cache_stats()
        
        # Second call - should hit cache
        cached_encoder.encode_text(text)
        stats2 = cached_encoder.get_cache_stats()
        
        assert stats2["hits"] == stats1["hits"] + 1
    
    def test_batch_caching(self, cached_encoder):
        """Test batch encoding uses cache."""
        texts = ["Text1", "Text2", "Text3"]
        
        # First batch
        cached_encoder.encode_batch(texts)
        
        # Second batch (partial overlap)
        cached_encoder.encode_batch(["Text1", "Text4"])
        
        stats = cached_encoder.get_cache_stats()
        assert stats["hits"] >= 1  # At least Text1 should hit
    
    def test_clear_cache(self, cached_encoder):
        """Test clearing cache."""
        cached_encoder.encode_text("Test")
        cached_encoder.clear_cache()
        
        stats = cached_encoder.get_cache_stats()
        assert stats["size"] == 0


# =============================================================================
# Test HybridEncoder
# =============================================================================

class TestHybridEncoder:
    """Tests for hybrid HD + Transformer encoder."""
    
    def test_initialization(self, hybrid_encoder):
        """Test hybrid encoder initialization."""
        assert hybrid_encoder.transformer_encoder is not None
    
    def test_encode_hybrid_without_hd(self, hybrid_encoder):
        """Test hybrid encoding without HD encoder."""
        result = hybrid_encoder.encode_hybrid("Test text")
        
        assert "transformer" in result
        assert "hd" not in result
    
    def test_encode_hybrid_with_hd(self, small_config):
        """Test hybrid encoding with HD encoder."""
        mock_hd = Mock()
        mock_hd.encode.return_value = np.random.randn(64)
        
        hybrid = HybridEncoder(
            transformer_config=small_config,
            hd_encoder=mock_hd
        )
        
        result = hybrid.encode_hybrid("Test text")
        
        assert "transformer" in result
        assert "hd" in result
    
    def test_similarity_hybrid(self, hybrid_encoder):
        """Test hybrid similarity computation."""
        sim = hybrid_encoder.similarity_hybrid("Hello", "Hello")
        
        assert sim > 0.9  # Same text should be very similar
    
    def test_configure_weights(self, hybrid_encoder):
        """Test weight configuration."""
        hybrid_encoder.configure_weights(0.8, 0.2)
        
        tf_w, hd_w = hybrid_encoder.get_weights()
        
        assert tf_w == 0.8
        assert hd_w == 0.2
    
    def test_weights_normalization(self, hybrid_encoder):
        """Test that weights are normalized."""
        hybrid_encoder.configure_weights(2.0, 2.0)
        
        tf_w, hd_w = hybrid_encoder.get_weights()
        
        assert abs(tf_w + hd_w - 1.0) < 0.001
    
    def test_adaptive_weighting(self, hybrid_encoder):
        """Test adaptive weight learning."""
        query = "Hello"
        candidates = ["Hello", "Hi", "Goodbye"]
        relevance = [1.0, 0.8, 0.2]
        
        best_weights = hybrid_encoder.adaptive_weighting(
            query, candidates, relevance
        )
        
        assert len(best_weights) == 2
        assert abs(sum(best_weights) - 1.0) < 0.001
    
    def test_adaptive_weighting_empty(self, hybrid_encoder):
        """Test adaptive weighting with empty input."""
        result = hybrid_encoder.adaptive_weighting("query", [], [])
        
        # Should return current weights
        assert len(result) == 2
    
    def test_adaptive_weighting_mismatch(self, hybrid_encoder):
        """Test adaptive weighting with mismatched lengths."""
        with pytest.raises(ValueError):
            hybrid_encoder.adaptive_weighting(
                "query",
                ["a", "b"],
                [1.0, 0.5, 0.2]  # Wrong length
            )
    
    def test_get_model_info(self, hybrid_encoder):
        """Test getting model info."""
        info = hybrid_encoder.get_model_info()
        
        assert "transformer" in info
        assert "weights" in info
        assert "hd_available" in info


# =============================================================================
# Test TransformerEmbeddings
# =============================================================================

class TestTransformerEmbeddings:
    """Tests for unified TransformerEmbeddings interface."""
    
    def test_initialization(self, embeddings_interface):
        """Test interface initialization."""
        assert embeddings_interface._encoder is not None
    
    def test_embed(self, embeddings_interface):
        """Test single text embedding."""
        embedding = embeddings_interface.embed("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_embed_batch(self, embeddings_interface):
        """Test batch embedding."""
        embeddings = embeddings_interface.embed_batch(["A", "B", "C"])
        
        assert embeddings.shape[0] == 3
    
    def test_similarity(self, embeddings_interface):
        """Test similarity computation."""
        sim = embeddings_interface.similarity("Hello", "Hello")
        
        assert sim > 0.9
    
    def test_find_similar(self, embeddings_interface):
        """Test finding similar texts."""
        query = "Hello"
        candidates = ["Hello", "Hi there", "Goodbye", "Morning"]
        
        results = embeddings_interface.find_similar(query, candidates, top_k=2)
        
        assert len(results) <= 2
        assert results[0][0] == "Hello"  # Most similar should be exact match
    
    def test_find_similar_with_threshold(self, embeddings_interface):
        """Test finding similar with threshold."""
        results = embeddings_interface.find_similar(
            "Hello",
            ["Hello", "Hi", "Completely different"],
            threshold=0.9
        )
        
        # Should filter out low similarity
        for text, score in results:
            assert score >= 0.9
    
    def test_find_similar_empty(self, embeddings_interface):
        """Test finding similar with empty candidates."""
        results = embeddings_interface.find_similar("Hello", [])
        
        assert results == []
    
    def test_cluster_by_similarity(self, embeddings_interface):
        """Test clustering by similarity."""
        texts = ["Hello", "Hello there", "Goodbye", "Bye bye"]
        
        clusters = embeddings_interface.cluster_by_similarity(texts, threshold=0.5)
        
        assert len(clusters) >= 1
        assert sum(len(c) for c in clusters) == len(texts)
    
    def test_cluster_empty(self, embeddings_interface):
        """Test clustering empty list."""
        clusters = embeddings_interface.cluster_by_similarity([])
        
        assert clusters == []
    
    def test_get_model_info(self, embeddings_interface):
        """Test getting model info."""
        info = embeddings_interface.get_model_info()
        
        assert info is not None
    
    def test_is_transformer_available(self, embeddings_interface):
        """Test transformer availability check."""
        available = embeddings_interface.is_transformer_available()
        
        # Should return boolean
        assert isinstance(available, bool)


# =============================================================================
# Test with Hybrid Mode
# =============================================================================

class TestTransformerEmbeddingsHybrid:
    """Tests for TransformerEmbeddings in hybrid mode."""
    
    def test_hybrid_initialization(self, small_config):
        """Test hybrid mode initialization."""
        embeddings = TransformerEmbeddings(
            config=small_config,
            use_hybrid=True
        )
        
        assert embeddings._use_hybrid is True
    
    def test_hybrid_embed(self, small_config):
        """Test hybrid embed."""
        embeddings = TransformerEmbeddings(
            config=small_config,
            use_hybrid=True
        )
        
        result = embeddings.embed("Test")
        
        assert isinstance(result, np.ndarray)
    
    def test_hybrid_similarity(self, small_config):
        """Test hybrid similarity."""
        embeddings = TransformerEmbeddings(
            config=small_config,
            use_hybrid=True
        )
        
        sim = embeddings.similarity("Hello", "Hello")
        
        assert sim > 0.9


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_create_embeddings(self):
        """Test create_embeddings function."""
        embeddings = create_embeddings()
        
        assert isinstance(embeddings, TransformerEmbeddings)
    
    def test_create_embeddings_custom_model(self):
        """Test create_embeddings with custom model."""
        embeddings = create_embeddings(model_name="custom-model")
        
        assert embeddings.config.model_name == "custom-model"
    
    def test_compute_similarity(self):
        """Test compute_similarity function."""
        sim = compute_similarity("Hello", "Hello")
        
        assert sim > 0.9
    
    def test_embed_texts(self):
        """Test embed_texts function."""
        embeddings = embed_texts(["A", "B", "C"])
        
        assert embeddings.shape[0] == 3


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_long_text(self, transformer_encoder):
        """Test encoding very long text."""
        long_text = "word " * 10000
        
        embedding = transformer_encoder.encode_text(long_text)
        
        assert isinstance(embedding, np.ndarray)
    
    def test_special_characters(self, transformer_encoder):
        """Test encoding with special characters."""
        special = "Hello! @#$%^&*() 你好 🎉"
        
        embedding = transformer_encoder.encode_text(special)
        
        assert isinstance(embedding, np.ndarray)
    
    def test_unicode_text(self, transformer_encoder):
        """Test encoding Unicode text."""
        unicode_text = "Привет мир 世界你好"
        
        embedding = transformer_encoder.encode_text(unicode_text)
        
        assert isinstance(embedding, np.ndarray)
    
    def test_whitespace_only(self, transformer_encoder):
        """Test encoding whitespace-only text."""
        embedding = transformer_encoder.encode_text("   \t\n  ")
        
        assert isinstance(embedding, np.ndarray)
    
    def test_single_character(self, transformer_encoder):
        """Test encoding single character."""
        embedding = transformer_encoder.encode_text("a")
        
        assert isinstance(embedding, np.ndarray)
    
    def test_zero_vector_similarity(self, transformer_encoder):
        """Test similarity with zero vectors."""
        # Empty text produces zero vector in some encoders
        sim = transformer_encoder._cosine_similarity(
            np.zeros(64),
            np.array([1.0] * 64)
        )
        
        assert sim == 0.0


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_encoding(self, small_config):
        """Test concurrent encoding is thread-safe."""
        encoder = TransformerEncoder(config=small_config, use_cache=True)
        errors = []
        
        def encode_thread(i):
            try:
                for j in range(50):
                    encoder.encode_text(f"Thread {i} iteration {j}")
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(encode_thread, i) for i in range(5)]
            for f in futures:
                f.result()
        
        assert len(errors) == 0
    
    def test_concurrent_hybrid(self, small_config):
        """Test concurrent hybrid encoding."""
        hybrid = HybridEncoder(transformer_config=small_config)
        errors = []
        
        def encode_thread(i):
            try:
                for j in range(50):
                    hybrid.similarity_hybrid(f"Query {i}", f"Target {j}")
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(encode_thread, i) for i in range(5)]
            for f in futures:
                f.result()
        
        assert len(errors) == 0


# =============================================================================
# Test Module Constants
# =============================================================================

class TestModuleConstants:
    """Tests for module-level constants."""
    
    def test_sentence_transformers_flag(self):
        """Test SENTENCE_TRANSFORMERS_AVAILABLE flag."""
        assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)
    
    def test_embedding_model_types(self):
        """Test all embedding model types exist."""
        assert EmbeddingModelType.TRANSFORMER is not None
        assert EmbeddingModelType.TFIDF is not None
        assert EmbeddingModelType.HD is not None
        assert EmbeddingModelType.RANDOM is not None
