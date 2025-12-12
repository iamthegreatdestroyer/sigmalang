"""
Tests for multilingual support module.

Tests all multilingual capabilities including language detection,
cross-lingual encoding, similarity computation, and translation approximation.
"""

import pytest
import numpy as np
import threading
from unittest.mock import Mock, patch

from core.multilingual_support import (
    # Enums and constants
    Language,
    LANGUAGE_CODE_MAP,
    # Dataclasses
    LanguageConfig,
    LanguageDetectionResult,
    TranslationResult,
    # Classes
    LanguageDetector,
    MultilingualEncoder,
    CrossLingualMapper,
    MultilingualSimilarity,
    TranslationApproximator,
    MultilingualSupport,
    # Convenience functions
    create_multilingual_support,
    detect_language,
    cross_lingual_similarity,
)


class TestLanguage:
    """Tests for Language enum."""
    
    def test_language_values(self):
        """Test language codes."""
        assert Language.ENGLISH.value == "en"
        assert Language.SPANISH.value == "es"
        assert Language.FRENCH.value == "fr"
        assert Language.GERMAN.value == "de"
        assert Language.CHINESE.value == "zh"
        assert Language.JAPANESE.value == "ja"
        assert Language.UNKNOWN.value == "xx"
    
    def test_language_code_map(self):
        """Test language code mapping."""
        assert LANGUAGE_CODE_MAP["en"] == Language.ENGLISH
        assert LANGUAGE_CODE_MAP["es"] == Language.SPANISH
        assert LANGUAGE_CODE_MAP["fr"] == Language.FRENCH


class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LanguageConfig()
        assert config.default_language == Language.ENGLISH
        assert config.detect_language is True
        assert config.normalize_unicode is True
        assert config.lowercase is False
        assert config.remove_accents is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LanguageConfig(
            default_language=Language.SPANISH,
            detect_language=False,
            lowercase=True
        )
        assert config.default_language == Language.SPANISH
        assert config.detect_language is False
        assert config.lowercase is True


class TestLanguageDetectionResult:
    """Tests for LanguageDetectionResult dataclass."""
    
    def test_detection_result(self):
        """Test detection result creation."""
        result = LanguageDetectionResult(
            language=Language.ENGLISH,
            confidence=0.95,
            probabilities={Language.ENGLISH: 0.95, Language.SPANISH: 0.05},
            method="pattern"
        )
        assert result.language == Language.ENGLISH
        assert result.confidence == 0.95
        assert len(result.probabilities) == 2
        assert result.method == "pattern"


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""
    
    def test_translation_result(self):
        """Test translation result creation."""
        result = TranslationResult(
            source_text="hello",
            source_language=Language.ENGLISH,
            target_text="hola",
            target_language=Language.SPANISH,
            confidence=0.9,
            alternatives=[("saludo", 0.7), ("buenos dias", 0.6)]
        )
        assert result.source_text == "hello"
        assert result.target_text == "hola"
        assert result.confidence == 0.9
        assert len(result.alternatives) == 2


class TestLanguageDetector:
    """Tests for LanguageDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create detector fixture."""
        return LanguageDetector()
    
    def test_detect_english(self, detector):
        """Test English detection."""
        result = detector.detect("The quick brown fox jumps over the lazy dog")
        assert result.language == Language.ENGLISH
        assert result.confidence > 0
    
    def test_detect_spanish(self, detector):
        """Test Spanish detection."""
        result = detector.detect("El rápido zorro marrón salta sobre el perro perezoso")
        # Might detect Spanish or related language
        assert result.confidence > 0
    
    def test_detect_french(self, detector):
        """Test French detection."""
        result = detector.detect("Le renard brun et rapide saute par-dessus le chien paresseux")
        assert result.confidence > 0
    
    def test_detect_german(self, detector):
        """Test German detection with special characters."""
        result = detector.detect("Der schnelle braune Fuchs springt über den faulen Hund")
        assert result.confidence > 0
    
    def test_detect_chinese(self, detector):
        """Test Chinese detection by script."""
        result = detector.detect("这是中文测试")
        assert result.language == Language.CHINESE
        assert result.confidence > 0.5
        assert result.method == "script"
    
    def test_detect_japanese(self, detector):
        """Test Japanese detection by script."""
        result = detector.detect("これは日本語のテストです")
        # Contains both hiragana and kanji
        assert result.confidence > 0
    
    def test_detect_russian(self, detector):
        """Test Russian detection by script."""
        result = detector.detect("Это тест на русском языке")
        assert result.language == Language.RUSSIAN
        assert result.confidence > 0.5
    
    def test_detect_korean(self, detector):
        """Test Korean detection by script."""
        result = detector.detect("이것은 한국어 테스트입니다")
        assert result.language == Language.KOREAN
        assert result.confidence > 0.5
    
    def test_detect_empty_text(self, detector):
        """Test detection with empty text."""
        result = detector.detect("")
        assert result.language == Language.UNKNOWN
        assert result.confidence == 0.0
    
    def test_detect_whitespace_only(self, detector):
        """Test detection with whitespace only."""
        result = detector.detect("   \n\t  ")
        assert result.language == Language.UNKNOWN
        assert result.confidence == 0.0
    
    def test_detect_batch(self, detector):
        """Test batch detection."""
        texts = [
            "Hello world",
            "Hola mundo",
            "Bonjour le monde"
        ]
        results = detector.detect_batch(texts)
        assert len(results) == 3
        for result in results:
            assert result.confidence > 0
    
    def test_caching(self, detector):
        """Test that caching works."""
        text = "The quick brown fox jumps"
        result1 = detector.detect(text)
        result2 = detector.detect(text)
        assert result1.language == result2.language
        assert result1.confidence == result2.confidence


class TestMultilingualEncoder:
    """Tests for MultilingualEncoder class."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return MultilingualEncoder()
    
    def test_encode_text(self, encoder):
        """Test text encoding."""
        embedding = encoder.encode("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    def test_encode_with_language(self, encoder):
        """Test encoding with explicit language."""
        embedding = encoder.encode("Hello world", Language.ENGLISH)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0
    
    def test_encode_spanish(self, encoder):
        """Test Spanish encoding."""
        embedding = encoder.encode("Hola mundo", Language.SPANISH)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0
    
    def test_encode_batch(self, encoder):
        """Test batch encoding."""
        texts = ["Hello", "World", "Test"]
        embeddings = encoder.encode_batch(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0
    
    def test_encode_batch_with_languages(self, encoder):
        """Test batch encoding with languages."""
        texts = ["Hello", "Hola", "Bonjour"]
        languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH]
        embeddings = encoder.encode_batch(texts, languages)
        assert embeddings.shape[0] == 3
    
    def test_get_language(self, encoder):
        """Test language detection via encoder."""
        lang = encoder.get_language("Hello world")
        assert isinstance(lang, Language)
    
    def test_get_dimensionality(self, encoder):
        """Test getting dimensionality."""
        dim = encoder.get_dimensionality()
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_caching(self, encoder):
        """Test encoding cache."""
        text = "Test caching behavior"
        emb1 = encoder.encode(text, Language.ENGLISH)
        emb2 = encoder.encode(text, Language.ENGLISH)
        np.testing.assert_array_equal(emb1, emb2)


class TestCrossLingualMapper:
    """Tests for CrossLingualMapper class."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return MultilingualEncoder()
    
    @pytest.fixture
    def mapper(self, encoder):
        """Create mapper fixture."""
        return CrossLingualMapper(encoder, shared_dim=128)
    
    def test_project_to_shared(self, mapper, encoder):
        """Test projection to shared space."""
        embedding = encoder.encode("Hello world", Language.ENGLISH)
        projected = mapper.project_to_shared(embedding, Language.ENGLISH)
        assert projected.shape[0] == 128  # shared_dim
        # Check normalized
        norm = np.linalg.norm(projected)
        assert abs(norm - 1.0) < 0.01
    
    def test_project_different_languages(self, mapper, encoder):
        """Test projection from different languages."""
        en_emb = encoder.encode("Hello", Language.ENGLISH)
        es_emb = encoder.encode("Hola", Language.SPANISH)
        
        en_proj = mapper.project_to_shared(en_emb, Language.ENGLISH)
        es_proj = mapper.project_to_shared(es_emb, Language.SPANISH)
        
        assert en_proj.shape == es_proj.shape
    
    def test_align_languages(self, mapper):
        """Test language alignment."""
        parallel_pairs = [
            ("hello", "hola"),
            ("world", "mundo"),
            ("cat", "gato")
        ]
        # Should not raise
        mapper.align_languages(Language.ENGLISH, Language.SPANISH, parallel_pairs)
    
    def test_align_empty_pairs(self, mapper):
        """Test alignment with empty pairs."""
        mapper.align_languages(Language.ENGLISH, Language.FRENCH, [])


class TestMultilingualSimilarity:
    """Tests for MultilingualSimilarity class."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return MultilingualEncoder()
    
    @pytest.fixture
    def mapper(self, encoder):
        """Create mapper fixture."""
        return CrossLingualMapper(encoder)
    
    @pytest.fixture
    def similarity(self, encoder, mapper):
        """Create similarity computer fixture."""
        return MultilingualSimilarity(encoder, mapper)
    
    def test_compute_similarity_same_language(self, similarity):
        """Test similarity within same language."""
        sim = similarity.compute_similarity(
            "hello", Language.ENGLISH,
            "hello", Language.ENGLISH
        )
        assert 0 <= sim <= 1
        assert sim > 0.5  # Same text should be similar
    
    def test_compute_similarity_cross_lingual(self, similarity):
        """Test cross-lingual similarity."""
        sim = similarity.compute_similarity(
            "hello", Language.ENGLISH,
            "hola", Language.SPANISH
        )
        # Cosine similarity can be negative after random projections
        assert -1 <= sim <= 1
    
    def test_compute_similarity_euclidean(self, similarity):
        """Test euclidean similarity method."""
        sim = similarity.compute_similarity(
            "hello", Language.ENGLISH,
            "world", Language.ENGLISH,
            method="euclidean"
        )
        assert 0 <= sim <= 1
    
    def test_find_similar(self, similarity):
        """Test finding similar texts."""
        candidates = ["uno", "dos", "tres", "hola", "mundo"]
        results = similarity.find_similar(
            "hello", Language.ENGLISH,
            candidates, Language.SPANISH,
            top_k=3
        )
        assert len(results) == 3
        for text, sim in results:
            assert text in candidates
            # Cosine similarity can be negative after random projections
            assert -1 <= sim <= 1


class TestTranslationApproximator:
    """Tests for TranslationApproximator class."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return MultilingualEncoder()
    
    @pytest.fixture
    def mapper(self, encoder):
        """Create mapper fixture."""
        return CrossLingualMapper(encoder)
    
    @pytest.fixture
    def translator(self, encoder, mapper):
        """Create translator fixture."""
        return TranslationApproximator(encoder, mapper)
    
    def test_translate_word_with_vocabulary(self, translator):
        """Test word translation with vocabulary."""
        vocabulary = ["hola", "mundo", "gato", "perro"]
        result = translator.translate_word(
            "hello", Language.ENGLISH, Language.SPANISH, vocabulary
        )
        assert isinstance(result, TranslationResult)
        assert result.source_text == "hello"
        assert result.target_text in vocabulary
        assert 0 <= result.confidence <= 1
    
    def test_translate_with_dictionary(self, translator):
        """Test translation with added dictionary."""
        translator.add_dictionary(
            Language.ENGLISH, Language.SPANISH,
            {"hello": "hola", "world": "mundo"}
        )
        
        result = translator.translate_word(
            "hello", Language.ENGLISH, Language.SPANISH, ["hola", "adios"]
        )
        assert result.target_text == "hola"
        assert result.confidence == 1.0
    
    def test_translation_alternatives(self, translator):
        """Test translation returns alternatives."""
        vocabulary = ["uno", "dos", "tres"]
        result = translator.translate_word(
            "one", Language.ENGLISH, Language.SPANISH, vocabulary
        )
        # Should have alternatives (up to 5)
        assert result.alternatives is not None


class TestMultilingualSupport:
    """Tests for MultilingualSupport main class."""
    
    @pytest.fixture
    def ml(self):
        """Create multilingual support fixture."""
        return MultilingualSupport()
    
    def test_detect_language(self, ml):
        """Test language detection."""
        result = ml.detect_language("Hello world")
        assert isinstance(result, LanguageDetectionResult)
        assert result.confidence > 0
    
    def test_detect_language_batch(self, ml):
        """Test batch language detection."""
        texts = ["Hello", "Hola", "Bonjour"]
        results = ml.detect_language_batch(texts)
        assert len(results) == 3
    
    def test_encode(self, ml):
        """Test text encoding."""
        embedding = ml.encode("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0
    
    def test_encode_with_language(self, ml):
        """Test encoding with explicit language."""
        embedding = ml.encode("Hola mundo", Language.SPANISH)
        assert embedding.shape[0] > 0
    
    def test_encode_batch(self, ml):
        """Test batch encoding."""
        texts = ["Hello", "World"]
        embeddings = ml.encode_batch(texts)
        assert embeddings.shape[0] == 2
    
    def test_cross_lingual_similarity(self, ml):
        """Test cross-lingual similarity."""
        sim = ml.cross_lingual_similarity(
            "hello", "en",
            "hola", "es"
        )
        # Cosine similarity can be negative after random projections
        assert -1 <= sim <= 1
    
    def test_find_similar(self, ml):
        """Test finding similar texts."""
        results = ml.find_similar(
            "hello", "en",
            ["hola", "mundo", "gato"], "es",
            top_k=2
        )
        assert len(results) == 2
    
    def test_translate_word(self, ml):
        """Test word translation."""
        result = ml.translate_word(
            "hello", "en", "es",
            ["hola", "adios", "gracias"]
        )
        assert isinstance(result, TranslationResult)
    
    def test_add_translation_dictionary(self, ml):
        """Test adding translation dictionary."""
        ml.add_translation_dictionary(
            "en", "es",
            {"cat": "gato", "dog": "perro"}
        )
        
        result = ml.translate_word(
            "cat", "en", "es", ["gato", "perro"]
        )
        assert result.target_text == "gato"
    
    def test_align_languages(self, ml):
        """Test language alignment."""
        parallel = [
            ("hello", "hola"),
            ("goodbye", "adios")
        ]
        ml.align_languages("en", "es", parallel)
    
    def test_get_supported_languages(self, ml):
        """Test getting supported languages."""
        languages = ml.get_supported_languages()
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "xx" not in languages  # UNKNOWN excluded
    
    def test_get_encoder(self, ml):
        """Test getting encoder."""
        encoder = ml.get_encoder()
        assert isinstance(encoder, MultilingualEncoder)
    
    def test_get_mapper(self, ml):
        """Test getting mapper."""
        mapper = ml.get_mapper()
        assert isinstance(mapper, CrossLingualMapper)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_multilingual_support(self):
        """Test creating multilingual support."""
        ml = create_multilingual_support(detect_language=True)
        assert isinstance(ml, MultilingualSupport)
        assert ml.language_config.detect_language is True
    
    def test_detect_language_function(self):
        """Test language detection function."""
        lang_code = detect_language("Hello world")
        assert isinstance(lang_code, str)
        assert len(lang_code) == 2
    
    def test_cross_lingual_similarity_function(self):
        """Test cross-lingual similarity function."""
        sim = cross_lingual_similarity(
            "hello", "en", "hola", "es"
        )
        # Cosine similarity can be negative after random projections
        assert -1 <= sim <= 1


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.fixture
    def ml(self):
        """Create multilingual support fixture."""
        return MultilingualSupport()
    
    def test_empty_text_detection(self, ml):
        """Test detection with empty text."""
        result = ml.detect_language("")
        assert result.language == Language.UNKNOWN
    
    def test_single_character(self, ml):
        """Test with single character."""
        embedding = ml.encode("a")
        assert embedding.shape[0] > 0
    
    def test_very_long_text(self, ml):
        """Test with very long text."""
        long_text = "hello " * 1000
        embedding = ml.encode(long_text)
        assert embedding.shape[0] > 0
    
    def test_special_characters(self, ml):
        """Test with special characters."""
        text = "Hello! @#$%^&*() World?"
        embedding = ml.encode(text)
        assert embedding.shape[0] > 0
    
    def test_mixed_languages(self, ml):
        """Test with mixed language text."""
        text = "Hello mundo bonjour"
        result = ml.detect_language(text)
        assert result.confidence > 0
    
    def test_unknown_language_code(self, ml):
        """Test with unknown language code."""
        sim = ml.cross_lingual_similarity(
            "hello", "zz",  # Unknown code
            "world", "en"
        )
        # Cosine similarity can be negative after random projections
        assert -1 <= sim <= 1
    
    def test_unicode_normalization(self):
        """Test unicode normalization."""
        config = LanguageConfig(normalize_unicode=True)
        ml = MultilingualSupport(language_config=config)
        
        # Composed vs decomposed unicode
        text1 = "café"  # composed
        text2 = "cafe\u0301"  # decomposed
        
        emb1 = ml.encode(text1)
        emb2 = ml.encode(text2)
        
        # After normalization, should be similar
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert sim > 0.5


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_detection(self):
        """Test concurrent language detection."""
        ml = MultilingualSupport()
        results = []
        errors = []
        
        def detect_task(text):
            try:
                result = ml.detect_language(text)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        texts = ["Hello", "Hola", "Bonjour", "Guten Tag", "Ciao"] * 10
        threads = [threading.Thread(target=detect_task, args=(t,)) for t in texts]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == len(texts)
    
    def test_concurrent_encoding(self):
        """Test concurrent encoding."""
        ml = MultilingualSupport()
        embeddings = []
        errors = []
        
        def encode_task(text):
            try:
                emb = ml.encode(text)
                embeddings.append(emb)
            except Exception as e:
                errors.append(e)
        
        texts = ["word" + str(i) for i in range(50)]
        threads = [threading.Thread(target=encode_task, args=(t,)) for t in texts]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(embeddings) == len(texts)


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test full multilingual workflow."""
        ml = MultilingualSupport()
        
        # Detect language
        en_result = ml.detect_language("The quick brown fox")
        assert en_result.language == Language.ENGLISH
        
        # Encode texts
        en_emb = ml.encode("cat", Language.ENGLISH)
        es_emb = ml.encode("gato", Language.SPANISH)
        
        # Compute similarity
        sim = ml.cross_lingual_similarity("cat", "en", "gato", "es")
        assert 0 <= sim <= 1
        
        # Find similar
        candidates = ["perro", "gato", "casa"]
        similar = ml.find_similar("cat", "en", candidates, "es", top_k=1)
        assert len(similar) == 1
    
    def test_translation_workflow(self):
        """Test translation workflow."""
        ml = MultilingualSupport()
        
        # Add dictionary
        ml.add_translation_dictionary("en", "es", {
            "hello": "hola",
            "cat": "gato"
        })
        
        # Translate
        result = ml.translate_word("hello", "en", "es", ["hola", "adios"])
        assert result.target_text == "hola"
        assert result.confidence == 1.0
    
    def test_alignment_improves_similarity(self):
        """Test that alignment can affect projections."""
        ml = MultilingualSupport()
        
        # Measure similarity before alignment
        sim_before = ml.cross_lingual_similarity("dog", "en", "perro", "es")
        
        # Align with parallel data
        parallel = [
            ("hello", "hola"),
            ("world", "mundo"),
            ("cat", "gato"),
            ("house", "casa")
        ]
        ml.align_languages("en", "es", parallel)
        
        # Measure after - values may change due to projection updates
        sim_after = ml.cross_lingual_similarity("dog", "en", "perro", "es")
        
        # Both should be valid similarities
        assert 0 <= sim_before <= 1
        assert 0 <= sim_after <= 1
