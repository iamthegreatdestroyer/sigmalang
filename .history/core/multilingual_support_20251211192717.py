"""
Multilingual Support for sigmalang.

This module provides multilingual capabilities for hyperdimensional computing,
enabling cross-language similarity, translation approximation, and language
detection.

Classes:
    Language: Enumeration of supported languages
    LanguageConfig: Configuration for language handling
    LanguageDetector: Detects language of text
    MultilingualEncoder: Encodes text with language awareness
    CrossLingualMapper: Maps between language embedding spaces
    MultilingualSimilarity: Computes similarity across languages
    MultilingualSupport: Main multilingual interface

Example:
    >>> ml = MultilingualSupport()
    >>> sim = ml.cross_lingual_similarity("hello", "en", "hola", "es")
    >>> print(f"Similarity: {sim:.3f}")
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

from .transformer_embeddings import (
    TransformerEncoder,
    EmbeddingConfig,
    EmbeddingCache,
)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    UNKNOWN = "xx"


# Language code to Language mapping
LANGUAGE_CODE_MAP: Dict[str, Language] = {
    lang.value: lang for lang in Language
}


@dataclass
class LanguageConfig:
    """Configuration for multilingual handling.
    
    Attributes:
        default_language: Default language for unlabeled text
        detect_language: Whether to auto-detect language
        normalize_unicode: Whether to normalize unicode
        lowercase: Whether to lowercase text
        remove_accents: Whether to remove accents
    """
    default_language: Language = Language.ENGLISH
    detect_language: bool = True
    normalize_unicode: bool = True
    lowercase: bool = False
    remove_accents: bool = False


@dataclass
class LanguageDetectionResult:
    """Result of language detection.
    
    Attributes:
        language: Detected language
        confidence: Detection confidence [0, 1]
        probabilities: Probability distribution over languages
        method: Detection method used
    """
    language: Language
    confidence: float
    probabilities: Dict[Language, float] = field(default_factory=dict)
    method: str = "character_ngram"


@dataclass 
class TranslationResult:
    """Result of translation approximation.
    
    Attributes:
        source_text: Original text
        source_language: Source language
        target_text: Translated text (approximation)
        target_language: Target language
        confidence: Translation confidence
        alternatives: Alternative translations
    """
    source_text: str
    source_language: Language
    target_text: str
    target_language: Language
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


class LanguageDetector:
    """Detects the language of input text.
    
    Uses character n-gram analysis and language-specific patterns.
    """
    
    # Common character patterns per language
    LANGUAGE_PATTERNS: Dict[Language, Set[str]] = {
        Language.ENGLISH: {"the", "and", "is", "to", "of", "a", "in"},
        Language.SPANISH: {"el", "la", "de", "que", "en", "los", "del", "ñ"},
        Language.FRENCH: {"le", "la", "de", "et", "en", "les", "est", "ç"},
        Language.GERMAN: {"der", "die", "und", "in", "den", "ß", "ä", "ö", "ü"},
        Language.ITALIAN: {"il", "la", "di", "che", "e", "è", "per", "un"},
        Language.PORTUGUESE: {"de", "que", "e", "do", "da", "em", "ã", "ç"},
        Language.DUTCH: {"de", "het", "en", "van", "een", "ij", "oe"},
        Language.RUSSIAN: {"и", "в", "не", "на", "что", "с", "я"},
        Language.CHINESE: set(),  # Detected by Unicode range
        Language.JAPANESE: set(),  # Detected by Unicode range
        Language.KOREAN: set(),    # Detected by Unicode range
        Language.ARABIC: set(),    # Detected by Unicode range
        Language.HINDI: set(),     # Detected by Unicode range
    }
    
    # Unicode ranges for script detection
    SCRIPT_RANGES = {
        Language.CHINESE: [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
        Language.JAPANESE: [(0x3040, 0x309F), (0x30A0, 0x30FF)],
        Language.KOREAN: [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
        Language.ARABIC: [(0x0600, 0x06FF), (0x0750, 0x077F)],
        Language.HINDI: [(0x0900, 0x097F)],
        Language.RUSSIAN: [(0x0400, 0x04FF)],
    }
    
    def __init__(self):
        """Initialize detector."""
        self._cache: Dict[str, LanguageDetectionResult] = {}
        self._lock = threading.Lock()
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text[:500].encode('utf-8', errors='ignore')).hexdigest()[:16]
    
    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult with detected language
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0
            )
        
        cache_key = self._cache_key(text)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Try script-based detection first
        script_result = self._detect_by_script(text)
        if script_result.confidence > 0.7:
            with self._lock:
                self._cache[cache_key] = script_result
            return script_result
        
        # Fall back to pattern-based detection
        pattern_result = self._detect_by_patterns(text)
        
        with self._lock:
            self._cache[cache_key] = pattern_result
        
        return pattern_result
    
    def _detect_by_script(self, text: str) -> LanguageDetectionResult:
        """Detect language by Unicode script.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult
        """
        script_counts: Dict[Language, int] = defaultdict(int)
        total_chars = 0
        
        for char in text:
            code_point = ord(char)
            for lang, ranges in self.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[lang] += 1
                        total_chars += 1
                        break
        
        if total_chars == 0:
            return LanguageDetectionResult(
                language=Language.UNKNOWN,
                confidence=0.0,
                method="script"
            )
        
        # Find dominant script
        best_lang = max(script_counts, key=script_counts.get)
        confidence = script_counts[best_lang] / total_chars
        
        # Build probability distribution
        probs = {
            lang: count / total_chars
            for lang, count in script_counts.items()
        }
        
        return LanguageDetectionResult(
            language=best_lang,
            confidence=confidence,
            probabilities=probs,
            method="script"
        )
    
    def _detect_by_patterns(self, text: str) -> LanguageDetectionResult:
        """Detect language by word patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult
        """
        # Tokenize
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        scores: Dict[Language, float] = {}
        
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if not patterns:
                continue
            
            matches = len(words & patterns)
            scores[lang] = matches / max(len(patterns), 1)
        
        if not scores:
            return LanguageDetectionResult(
                language=Language.ENGLISH,  # Default
                confidence=0.3,
                method="pattern_default"
            )
        
        # Find best match
        best_lang = max(scores, key=scores.get)
        max_score = scores[best_lang]
        
        # Normalize to confidence
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.3
        
        # Build probabilities
        probs = {
            lang: score / total_score if total_score > 0 else 0.0
            for lang, score in scores.items()
        }
        
        return LanguageDetectionResult(
            language=best_lang,
            confidence=min(confidence, 0.9),  # Cap confidence
            probabilities=probs,
            method="pattern"
        )
    
    def detect_batch(self, texts: List[str]) -> List[LanguageDetectionResult]:
        """Detect language for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of detection results
        """
        return [self.detect(text) for text in texts]


class MultilingualEncoder:
    """Encodes text with language awareness.
    
    Uses language-specific preprocessing and encoding strategies.
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        language_config: Optional[LanguageConfig] = None
    ):
        """Initialize encoder.
        
        Args:
            config: Embedding configuration
            language_config: Language configuration
        """
        self.config = config or EmbeddingConfig(fallback_dim=256)
        self.language_config = language_config or LanguageConfig()
        
        self._encoder = TransformerEncoder(config=self.config, use_cache=True)
        self._detector = LanguageDetector()
        self._cache = EmbeddingCache(max_size=1000)
        self._lock = threading.Lock()
    
    def _preprocess(self, text: str, language: Language) -> str:
        """Preprocess text based on language.
        
        Args:
            text: Text to preprocess
            language: Language of text
            
        Returns:
            Preprocessed text
        """
        processed = text
        
        if self.language_config.normalize_unicode:
            import unicodedata
            processed = unicodedata.normalize('NFC', processed)
        
        if self.language_config.lowercase:
            processed = processed.lower()
        
        if self.language_config.remove_accents:
            import unicodedata
            processed = ''.join(
                c for c in unicodedata.normalize('NFD', processed)
                if unicodedata.category(c) != 'Mn'
            )
        
        # Add language marker for better embeddings
        lang_marker = f"[{language.value.upper()}]"
        processed = f"{lang_marker} {processed}"
        
        return processed
    
    def encode(
        self,
        text: str,
        language: Optional[Language] = None
    ) -> np.ndarray:
        """Encode text to embedding.
        
        Args:
            text: Text to encode
            language: Language (detected if not provided)
            
        Returns:
            Embedding vector
        """
        # Detect language if not provided
        if language is None:
            if self.language_config.detect_language:
                detection = self._detector.detect(text)
                language = detection.language
            else:
                language = self.language_config.default_language
        
        # Check cache
        cache_key = f"{language.value}:{text[:100]}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Preprocess and encode
        processed = self._preprocess(text, language)
        embedding = self._encoder.encode_text(processed)
        
        # Cache
        self._cache.put(cache_key, embedding)
        
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        languages: Optional[List[Language]] = None
    ) -> np.ndarray:
        """Encode multiple texts.
        
        Args:
            texts: List of texts
            languages: Languages for each text
            
        Returns:
            Array of embeddings
        """
        if languages is None:
            languages = [None] * len(texts)
        
        embeddings = []
        for text, lang in zip(texts, languages):
            emb = self.encode(text, lang)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def get_language(self, text: str) -> Language:
        """Get detected language for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language
        """
        return self._detector.detect(text).language
    
    def get_dimensionality(self) -> int:
        """Get embedding dimensionality.
        
        Returns:
            Embedding dimension
        """
        return self._encoder.get_dimensionality()


class CrossLingualMapper:
    """Maps between language embedding spaces.
    
    Enables cross-lingual similarity by learning alignments.
    """
    
    def __init__(
        self,
        encoder: MultilingualEncoder,
        shared_dim: int = 256
    ):
        """Initialize mapper.
        
        Args:
            encoder: Multilingual encoder
            shared_dim: Dimension of shared space
        """
        self._encoder = encoder
        self._shared_dim = shared_dim
        self._projections: Dict[Language, np.ndarray] = {}
        self._lock = threading.Lock()
        
        # Initialize random projections
        self._init_projections()
    
    def _init_projections(self):
        """Initialize projection matrices for each language."""
        np.random.seed(42)
        dim = self._encoder.get_dimensionality()
        
        for lang in Language:
            if lang != Language.UNKNOWN:
                # Random orthogonal projection
                matrix = np.random.randn(dim, self._shared_dim)
                # Normalize columns
                norms = np.linalg.norm(matrix, axis=0, keepdims=True)
                norms = np.where(norms > 0, norms, 1)
                self._projections[lang] = (matrix / norms).astype(np.float32)
    
    def project_to_shared(
        self,
        embedding: np.ndarray,
        language: Language
    ) -> np.ndarray:
        """Project embedding to shared space.
        
        Args:
            embedding: Language-specific embedding
            language: Language of embedding
            
        Returns:
            Shared space embedding
        """
        with self._lock:
            if language not in self._projections:
                language = Language.ENGLISH  # Fallback
            
            projection = self._projections[language]
        
        projected = embedding @ projection
        
        # Normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        
        return projected
    
    def align_languages(
        self,
        source_lang: Language,
        target_lang: Language,
        parallel_pairs: List[Tuple[str, str]]
    ):
        """Align two languages using parallel data.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            parallel_pairs: List of (source_text, target_text) pairs
        """
        if not parallel_pairs:
            return
        
        # Encode parallel pairs
        source_embs = []
        target_embs = []
        
        for source_text, target_text in parallel_pairs:
            source_emb = self._encoder.encode(source_text, source_lang)
            target_emb = self._encoder.encode(target_text, target_lang)
            source_embs.append(source_emb)
            target_embs.append(target_emb)
        
        source_matrix = np.array(source_embs)
        target_matrix = np.array(target_embs)
        
        # Project both to shared space
        source_proj = np.array([
            self.project_to_shared(emb, source_lang)
            for emb in source_matrix
        ])
        target_proj = np.array([
            self.project_to_shared(emb, target_lang)
            for emb in target_matrix
        ])
        
        # Simple alignment: adjust target projection to minimize distance
        with self._lock:
            # Compute average direction correction
            diff = np.mean(source_proj - target_proj, axis=0)
            
            # Update target projection slightly
            old_proj = self._projections[target_lang]
            correction = np.outer(np.mean(target_matrix, axis=0), diff)
            self._projections[target_lang] = old_proj + 0.1 * correction


class MultilingualSimilarity:
    """Computes similarity across languages.
    
    Provides various cross-lingual similarity metrics.
    """
    
    def __init__(
        self,
        encoder: MultilingualEncoder,
        mapper: CrossLingualMapper
    ):
        """Initialize similarity computer.
        
        Args:
            encoder: Multilingual encoder
            mapper: Cross-lingual mapper
        """
        self._encoder = encoder
        self._mapper = mapper
    
    def compute_similarity(
        self,
        text1: str,
        lang1: Language,
        text2: str,
        lang2: Language,
        method: str = "cosine"
    ) -> float:
        """Compute cross-lingual similarity.
        
        Args:
            text1: First text
            lang1: Language of first text
            text2: Second text
            lang2: Language of second text
            method: Similarity method
            
        Returns:
            Similarity score
        """
        # Encode
        emb1 = self._encoder.encode(text1, lang1)
        emb2 = self._encoder.encode(text2, lang2)
        
        # Project to shared space
        proj1 = self._mapper.project_to_shared(emb1, lang1)
        proj2 = self._mapper.project_to_shared(emb2, lang2)
        
        if method == "cosine":
            return self._cosine_similarity(proj1, proj2)
        elif method == "euclidean":
            return self._euclidean_similarity(proj1, proj2)
        else:
            return self._cosine_similarity(proj1, proj2)
    
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
    
    def _euclidean_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute euclidean similarity (inverse distance)."""
        distance = np.linalg.norm(vec1 - vec2)
        return 1.0 / (1.0 + distance)
    
    def find_similar(
        self,
        query: str,
        query_lang: Language,
        candidates: List[str],
        candidate_lang: Language,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar texts in another language.
        
        Args:
            query: Query text
            query_lang: Language of query
            candidates: Candidate texts
            candidate_lang: Language of candidates
            top_k: Number of results
            
        Returns:
            List of (text, similarity) tuples
        """
        results = []
        
        for candidate in candidates:
            sim = self.compute_similarity(
                query, query_lang,
                candidate, candidate_lang
            )
            results.append((candidate, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class TranslationApproximator:
    """Approximates translations using embeddings.
    
    Note: This is not a full translation system but provides
    embedding-based translation approximation.
    """
    
    def __init__(
        self,
        encoder: MultilingualEncoder,
        mapper: CrossLingualMapper
    ):
        """Initialize approximator.
        
        Args:
            encoder: Multilingual encoder
            mapper: Cross-lingual mapper
        """
        self._encoder = encoder
        self._mapper = mapper
        self._dictionaries: Dict[Tuple[Language, Language], Dict[str, str]] = {}
        self._lock = threading.Lock()
    
    def add_dictionary(
        self,
        source_lang: Language,
        target_lang: Language,
        translations: Dict[str, str]
    ):
        """Add translation dictionary.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            translations: Word-to-word translations
        """
        with self._lock:
            key = (source_lang, target_lang)
            if key not in self._dictionaries:
                self._dictionaries[key] = {}
            self._dictionaries[key].update(translations)
    
    def translate_word(
        self,
        word: str,
        source_lang: Language,
        target_lang: Language,
        vocabulary: List[str]
    ) -> TranslationResult:
        """Translate single word using embedding similarity.
        
        Args:
            word: Word to translate
            source_lang: Source language
            target_lang: Target language
            vocabulary: Target language vocabulary
            
        Returns:
            TranslationResult
        """
        # Check dictionary first
        dict_key = (source_lang, target_lang)
        with self._lock:
            if dict_key in self._dictionaries:
                if word.lower() in self._dictionaries[dict_key]:
                    translation = self._dictionaries[dict_key][word.lower()]
                    return TranslationResult(
                        source_text=word,
                        source_language=source_lang,
                        target_text=translation,
                        target_language=target_lang,
                        confidence=1.0
                    )
        
        # Embedding-based translation
        source_emb = self._encoder.encode(word, source_lang)
        source_proj = self._mapper.project_to_shared(source_emb, source_lang)
        
        best_word = word
        best_sim = 0.0
        alternatives = []
        
        for target_word in vocabulary:
            target_emb = self._encoder.encode(target_word, target_lang)
            target_proj = self._mapper.project_to_shared(target_emb, target_lang)
            
            sim = self._cosine_similarity(source_proj, target_proj)
            alternatives.append((target_word, sim))
            
            if sim > best_sim:
                best_sim = sim
                best_word = target_word
        
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return TranslationResult(
            source_text=word,
            source_language=source_lang,
            target_text=best_word,
            target_language=target_lang,
            confidence=best_sim,
            alternatives=alternatives[:5]
        )
    
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


class MultilingualSupport:
    """Main multilingual interface.
    
    Unified interface for all multilingual operations.
    
    Example:
        >>> ml = MultilingualSupport()
        >>> lang = ml.detect_language("Hello world")
        >>> sim = ml.cross_lingual_similarity("hello", "en", "hola", "es")
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        language_config: Optional[LanguageConfig] = None
    ):
        """Initialize multilingual support.
        
        Args:
            config: Embedding configuration
            language_config: Language configuration
        """
        self.config = config or EmbeddingConfig(fallback_dim=256)
        self.language_config = language_config or LanguageConfig()
        
        self._encoder = MultilingualEncoder(self.config, self.language_config)
        self._mapper = CrossLingualMapper(self._encoder)
        self._similarity = MultilingualSimilarity(self._encoder, self._mapper)
        self._translator = TranslationApproximator(self._encoder, self._mapper)
        self._detector = LanguageDetector()
        self._lock = threading.Lock()
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageDetectionResult
        """
        return self._detector.detect(text)
    
    def detect_language_batch(
        self,
        texts: List[str]
    ) -> List[LanguageDetectionResult]:
        """Detect languages for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of detection results
        """
        return self._detector.detect_batch(texts)
    
    def encode(
        self,
        text: str,
        language: Optional[Language] = None
    ) -> np.ndarray:
        """Encode text to embedding.
        
        Args:
            text: Text to encode
            language: Language (auto-detected if None)
            
        Returns:
            Embedding vector
        """
        return self._encoder.encode(text, language)
    
    def encode_batch(
        self,
        texts: List[str],
        languages: Optional[List[Language]] = None
    ) -> np.ndarray:
        """Encode multiple texts.
        
        Args:
            texts: List of texts
            languages: Languages (auto-detected if None)
            
        Returns:
            Array of embeddings
        """
        return self._encoder.encode_batch(texts, languages)
    
    def cross_lingual_similarity(
        self,
        text1: str,
        lang1: str,
        text2: str,
        lang2: str
    ) -> float:
        """Compute cross-lingual similarity.
        
        Args:
            text1: First text
            lang1: Language code of first text
            text2: Second text
            lang2: Language code of second text
            
        Returns:
            Similarity score
        """
        language1 = LANGUAGE_CODE_MAP.get(lang1, Language.UNKNOWN)
        language2 = LANGUAGE_CODE_MAP.get(lang2, Language.UNKNOWN)
        
        return self._similarity.compute_similarity(
            text1, language1, text2, language2
        )
    
    def find_similar(
        self,
        query: str,
        query_lang: str,
        candidates: List[str],
        candidate_lang: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar texts across languages.
        
        Args:
            query: Query text
            query_lang: Language code of query
            candidates: Candidate texts
            candidate_lang: Language code of candidates
            top_k: Number of results
            
        Returns:
            List of (text, similarity) tuples
        """
        qlang = LANGUAGE_CODE_MAP.get(query_lang, Language.UNKNOWN)
        clang = LANGUAGE_CODE_MAP.get(candidate_lang, Language.UNKNOWN)
        
        return self._similarity.find_similar(
            query, qlang, candidates, clang, top_k
        )
    
    def translate_word(
        self,
        word: str,
        source_lang: str,
        target_lang: str,
        vocabulary: List[str]
    ) -> TranslationResult:
        """Translate word using embedding similarity.
        
        Args:
            word: Word to translate
            source_lang: Source language code
            target_lang: Target language code
            vocabulary: Target vocabulary
            
        Returns:
            TranslationResult
        """
        slang = LANGUAGE_CODE_MAP.get(source_lang, Language.UNKNOWN)
        tlang = LANGUAGE_CODE_MAP.get(target_lang, Language.UNKNOWN)
        
        return self._translator.translate_word(
            word, slang, tlang, vocabulary
        )
    
    def add_translation_dictionary(
        self,
        source_lang: str,
        target_lang: str,
        translations: Dict[str, str]
    ):
        """Add translation dictionary.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            translations: Word-to-word translations
        """
        slang = LANGUAGE_CODE_MAP.get(source_lang, Language.UNKNOWN)
        tlang = LANGUAGE_CODE_MAP.get(target_lang, Language.UNKNOWN)
        
        self._translator.add_dictionary(slang, tlang, translations)
    
    def align_languages(
        self,
        source_lang: str,
        target_lang: str,
        parallel_texts: List[Tuple[str, str]]
    ):
        """Align language spaces using parallel data.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            parallel_texts: List of (source, target) text pairs
        """
        slang = LANGUAGE_CODE_MAP.get(source_lang, Language.UNKNOWN)
        tlang = LANGUAGE_CODE_MAP.get(target_lang, Language.UNKNOWN)
        
        self._mapper.align_languages(slang, tlang, parallel_texts)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes.
        
        Returns:
            List of language codes
        """
        return [lang.value for lang in Language if lang != Language.UNKNOWN]
    
    def get_encoder(self) -> MultilingualEncoder:
        """Get the multilingual encoder."""
        return self._encoder
    
    def get_mapper(self) -> CrossLingualMapper:
        """Get the cross-lingual mapper."""
        return self._mapper


# Convenience functions

def create_multilingual_support(
    detect_language: bool = True
) -> MultilingualSupport:
    """Create multilingual support with configuration.
    
    Args:
        detect_language: Whether to auto-detect language
        
    Returns:
        MultilingualSupport instance
    """
    config = LanguageConfig(detect_language=detect_language)
    return MultilingualSupport(language_config=config)


def detect_language(text: str) -> str:
    """Detect language of text.
    
    Convenience function.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return result.language.value


def cross_lingual_similarity(
    text1: str,
    lang1: str,
    text2: str,
    lang2: str
) -> float:
    """Compute cross-lingual similarity.
    
    Convenience function.
    
    Args:
        text1: First text
        lang1: Language code of first text
        text2: Second text
        lang2: Language code of second text
        
    Returns:
        Similarity score
    """
    ml = MultilingualSupport()
    return ml.cross_lingual_similarity(text1, lang1, text2, lang2)
