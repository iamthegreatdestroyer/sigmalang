"""
Text Understanding Pipeline for sigmalang.

This module provides text processing and understanding capabilities including
document processing, chunking, summarization, and semantic analysis.

Classes:
    ChunkingStrategy: Enumeration of chunking strategies
    ChunkConfig: Configuration for text chunking
    TextChunk: Represents a chunk of text with metadata
    DocumentMetadata: Metadata about a document
    ProcessedDocument: Processed document with chunks and embeddings
    SummaryResult: Result of text summarization
    TextChunker: Splits text into semantic chunks
    DocumentProcessor: Processes documents for analysis
    TextSummarizer: Generates summaries of text
    SemanticAnalyzer: Analyzes semantic content
    TextUnderstandingPipeline: Main unified interface

Example:
    >>> pipeline = TextUnderstandingPipeline()
    >>> doc = pipeline.process_document("Long document text...")
    >>> summary = pipeline.summarize(doc)
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set, Callable

import numpy as np

from .transformer_embeddings import (
    TransformerEncoder,
    EmbeddingConfig,
    EmbeddingCache,
)


class ChunkingStrategy(Enum):
    """Strategy for splitting text into chunks."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class ChunkConfig:
    """Configuration for text chunking.
    
    Attributes:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size
        respect_sentences: Whether to respect sentence boundaries
        respect_paragraphs: Whether to respect paragraph boundaries
    """
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    chunk_size: int = 500
    overlap: int = 50
    min_chunk_size: int = 100
    respect_sentences: bool = True
    respect_paragraphs: bool = True


@dataclass
class TextChunk:
    """Represents a chunk of text.
    
    Attributes:
        text: The chunk text
        index: Chunk index in document
        start_char: Starting character position
        end_char: Ending character position
        embedding: Optional embedding vector
        metadata: Additional metadata
    """
    text: str
    index: int
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return chunk length."""
        return len(self.text)
    
    def get_hash(self) -> str:
        """Get hash of chunk content."""
        return hashlib.md5(self.text.encode()).hexdigest()[:16]


@dataclass
class DocumentMetadata:
    """Metadata about a document.
    
    Attributes:
        title: Document title
        source: Source of document
        created_at: Creation timestamp
        word_count: Number of words
        char_count: Number of characters
        language: Detected language
        custom: Custom metadata
    """
    title: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    language: str = "en"
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """A processed document with chunks and analysis.
    
    Attributes:
        text: Original document text
        chunks: List of text chunks
        metadata: Document metadata
        embedding: Document-level embedding
        topics: Detected topics
        keywords: Extracted keywords
    """
    text: str
    chunks: List[TextChunk]
    metadata: DocumentMetadata
    embedding: Optional[np.ndarray] = None
    topics: List[str] = field(default_factory=list)
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    
    def get_chunk_count(self) -> int:
        """Get number of chunks."""
        return len(self.chunks)
    
    def get_text_by_chunks(self, indices: List[int]) -> str:
        """Get text from specific chunks."""
        texts = []
        for idx in indices:
            if 0 <= idx < len(self.chunks):
                texts.append(self.chunks[idx].text)
        return " ".join(texts)


@dataclass
class SummaryResult:
    """Result of text summarization.
    
    Attributes:
        summary: Generated summary text
        original_length: Original text length
        summary_length: Summary length
        compression_ratio: Compression ratio
        key_sentences: Key sentences from original
        method: Summarization method used
    """
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_sentences: List[str] = field(default_factory=list)
    method: str = "extractive"


class TextChunker:
    """Splits text into semantic chunks.
    
    Supports multiple chunking strategies optimized for different use cases.
    """
    
    # Sentence boundary patterns
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    PARAGRAPH_PATTERN = re.compile(r'\n\n+')
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        self._lock = threading.Lock()
    
    def chunk(self, text: str) -> List[TextChunk]:
        """Chunk text according to configured strategy.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        strategy_map = {
            ChunkingStrategy.FIXED_SIZE: self._chunk_fixed_size,
            ChunkingStrategy.SENTENCE: self._chunk_by_sentence,
            ChunkingStrategy.PARAGRAPH: self._chunk_by_paragraph,
            ChunkingStrategy.SEMANTIC: self._chunk_semantic,
            ChunkingStrategy.SLIDING_WINDOW: self._chunk_sliding_window,
        }
        
        chunker = strategy_map.get(self.config.strategy, self._chunk_by_sentence)
        return chunker(text)
    
    def _chunk_fixed_size(self, text: str) -> List[TextChunk]:
        """Chunk by fixed character size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        chunks = []
        pos = 0
        index = 0
        
        while pos < len(text):
            end = min(pos + self.config.chunk_size, len(text))
            
            # Try to respect word boundaries
            if end < len(text) and text[end] not in ' \n\t':
                # Find last space
                last_space = text.rfind(' ', pos, end)
                if last_space > pos:
                    end = last_space + 1
            
            chunk_text = text[pos:end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=index,
                    start_char=pos,
                    end_char=end
                ))
                index += 1
            
            pos = end - self.config.overlap if end < len(text) else end
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[TextChunk]:
        """Chunk by sentences, grouping to meet size targets.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        sentences = self._split_sentences(text)
        chunks = []
        current_text = ""
        current_start = 0
        index = 0
        pos = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if len(current_text) + sentence_len > self.config.chunk_size and current_text:
                # Save current chunk
                if len(current_text.strip()) >= self.config.min_chunk_size:
                    chunks.append(TextChunk(
                        text=current_text.strip(),
                        index=index,
                        start_char=current_start,
                        end_char=pos
                    ))
                    index += 1
                
                current_text = sentence
                current_start = pos
            else:
                current_text += " " + sentence if current_text else sentence
            
            pos += sentence_len + 1  # +1 for space
        
        # Add remaining
        if current_text.strip() and len(current_text.strip()) >= self.config.min_chunk_size:
            chunks.append(TextChunk(
                text=current_text.strip(),
                index=index,
                start_char=current_start,
                end_char=pos
            ))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[TextChunk]:
        """Chunk by paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        chunks = []
        index = 0
        pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if len(para) >= self.config.min_chunk_size:
                start = text.find(para, pos)
                chunks.append(TextChunk(
                    text=para,
                    index=index,
                    start_char=start,
                    end_char=start + len(para)
                ))
                index += 1
                pos = start + len(para)
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[TextChunk]:
        """Chunk based on semantic similarity (simplified version).
        
        Uses sentence boundaries with semantic coherence heuristics.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        sentences = self._split_sentences(text)
        chunks = []
        current_sentences = []
        current_start = 0
        index = 0
        pos = 0
        
        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_text = " ".join(current_sentences)
            
            # Check if we should create a new chunk
            should_split = False
            
            if len(current_text) > self.config.chunk_size:
                should_split = True
            elif i < len(sentences) - 1:
                # Heuristic: check for topic shift indicators
                next_sentence = sentences[i + 1]
                if self._is_topic_shift(sentence, next_sentence):
                    should_split = True
            
            if should_split and len(current_text.strip()) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=current_text.strip(),
                    index=index,
                    start_char=current_start,
                    end_char=pos + len(sentence)
                ))
                index += 1
                current_sentences = []
                current_start = pos + len(sentence) + 1
            
            pos += len(sentence) + 1
        
        # Add remaining
        if current_sentences:
            current_text = " ".join(current_sentences)
            if len(current_text.strip()) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=current_text.strip(),
                    index=index,
                    start_char=current_start,
                    end_char=len(text)
                ))
        
        return chunks
    
    def _chunk_sliding_window(self, text: str) -> List[TextChunk]:
        """Chunk using sliding window approach.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks with overlap
        """
        sentences = self._split_sentences(text)
        chunks = []
        index = 0
        
        window_size = max(1, self.config.chunk_size // 100)  # Sentences per window
        step = max(1, window_size - (self.config.overlap // 100))
        
        for i in range(0, len(sentences), step):
            window_sentences = sentences[i:i + window_size]
            chunk_text = " ".join(window_sentences)
            
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    index=index,
                    start_char=0,  # Simplified
                    end_char=len(chunk_text)
                ))
                index += 1
            
            if i + window_size >= len(sentences):
                break
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = self.SENTENCE_PATTERN.split(text)
        # Filter empty
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_topic_shift(self, current: str, next_sentence: str) -> bool:
        """Detect potential topic shift between sentences.
        
        Args:
            current: Current sentence
            next_sentence: Next sentence
            
        Returns:
            True if topic shift detected
        """
        # Heuristics for topic shift
        topic_shift_indicators = [
            "however", "but", "nevertheless", "on the other hand",
            "meanwhile", "in contrast", "conversely", "alternatively",
            "first", "second", "third", "finally", "in conclusion",
            "next", "then", "subsequently", "additionally"
        ]
        
        next_lower = next_sentence.lower()
        for indicator in topic_shift_indicators:
            if next_lower.startswith(indicator):
                return True
        
        # Check for significant vocabulary shift (simplified)
        current_words = set(current.lower().split())
        next_words = set(next_lower.split())
        
        overlap = len(current_words & next_words)
        if overlap < min(len(current_words), len(next_words)) * 0.1:
            return True
        
        return False


class DocumentProcessor:
    """Processes documents for analysis.
    
    Handles document ingestion, metadata extraction, and preparation.
    """
    
    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """Initialize processor.
        
        Args:
            chunk_config: Chunking configuration
            embedding_config: Embedding configuration
        """
        self.chunk_config = chunk_config or ChunkConfig()
        self.embedding_config = embedding_config or EmbeddingConfig(fallback_dim=256)
        
        self._chunker = TextChunker(self.chunk_config)
        self._encoder = TransformerEncoder(config=self.embedding_config)
        self._lock = threading.Lock()
    
    def process(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        compute_embeddings: bool = True
    ) -> ProcessedDocument:
        """Process a document.
        
        Args:
            text: Document text
            metadata: Optional metadata
            compute_embeddings: Whether to compute embeddings
            
        Returns:
            ProcessedDocument
        """
        # Extract metadata
        doc_metadata = self._extract_metadata(text, metadata)
        
        # Chunk the document
        chunks = self._chunker.chunk(text)
        
        # Compute embeddings if requested
        doc_embedding = None
        if compute_embeddings:
            # Compute chunk embeddings
            for chunk in chunks:
                chunk.embedding = self._encoder.encode_text(chunk.text)
            
            # Compute document embedding (average of chunks)
            if chunks:
                chunk_embeddings = [c.embedding for c in chunks if c.embedding is not None]
                if chunk_embeddings:
                    doc_embedding = np.mean(chunk_embeddings, axis=0)
        
        # Extract keywords and topics
        keywords = self._extract_keywords(text)
        topics = self._infer_topics(text, keywords)
        
        return ProcessedDocument(
            text=text,
            chunks=chunks,
            metadata=doc_metadata,
            embedding=doc_embedding,
            topics=topics,
            keywords=keywords
        )
    
    def _extract_metadata(
        self,
        text: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """Extract document metadata.
        
        Args:
            text: Document text
            custom_metadata: Custom metadata to include
            
        Returns:
            DocumentMetadata
        """
        words = text.split()
        
        metadata = DocumentMetadata(
            word_count=len(words),
            char_count=len(text),
            custom=custom_metadata or {}
        )
        
        # Try to extract title (first line if short)
        lines = text.strip().split('\n')
        if lines and len(lines[0]) < 100:
            metadata.title = lines[0].strip()
        
        return metadata
    
    def _extract_keywords(
        self,
        text: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract keywords from text using TF-IDF-like scoring.
        
        Args:
            text: Text to analyze
            top_k: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now', 'this', 'that', 'these',
            'those', 'what', 'which', 'who', 'whom', 'has', 'have', 'had',
            'been', 'being', 'was', 'were', 'is', 'are', 'am'
        }
        
        filtered_words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        total_words = len(filtered_words)
        
        if total_words == 0:
            return []
        
        # Score by frequency with length bonus
        scores = []
        for word, count in word_counts.most_common(top_k * 2):
            tf = count / total_words
            length_bonus = min(len(word) / 10, 0.3)
            score = tf + length_bonus
            scores.append((word, score))
        
        # Normalize and return top-k
        max_score = max(s[1] for s in scores) if scores else 1
        normalized = [(w, s / max_score) for w, s in scores]
        
        return sorted(normalized, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _infer_topics(
        self,
        text: str,
        keywords: List[Tuple[str, float]]
    ) -> List[str]:
        """Infer topics from keywords.
        
        Args:
            text: Document text
            keywords: Extracted keywords
            
        Returns:
            List of topic labels
        """
        # Simple topic inference based on keyword clustering
        topics = []
        
        # Use top keywords as topics
        for keyword, score in keywords[:5]:
            if score > 0.3:
                topics.append(keyword)
        
        return topics


class TextSummarizer:
    """Generates summaries of text.
    
    Supports extractive summarization using sentence importance scoring.
    """
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """Initialize summarizer.
        
        Args:
            embedding_config: Embedding configuration
        """
        self.embedding_config = embedding_config or EmbeddingConfig(fallback_dim=256)
        self._encoder = TransformerEncoder(config=self.embedding_config)
        self._lock = threading.Lock()
    
    def summarize(
        self,
        text: str,
        ratio: float = 0.3,
        max_sentences: int = 5,
        min_sentences: int = 1
    ) -> SummaryResult:
        """Summarize text using extractive summarization.
        
        Args:
            text: Text to summarize
            ratio: Target compression ratio
            max_sentences: Maximum sentences in summary
            min_sentences: Minimum sentences in summary
            
        Returns:
            SummaryResult
        """
        if not text or not text.strip():
            return SummaryResult(
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=1.0
            )
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= min_sentences:
            return SummaryResult(
                summary=text,
                original_length=len(text),
                summary_length=len(text),
                compression_ratio=1.0,
                key_sentences=sentences
            )
        
        # Score sentences
        scores = self._score_sentences(sentences, text)
        
        # Select top sentences
        target_count = min(
            max_sentences,
            max(min_sentences, int(len(sentences) * ratio))
        )
        
        # Sort by score and select top
        scored_sentences = list(zip(sentences, scores, range(len(sentences))))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Get top sentences, then sort by original order
        top_sentences = scored_sentences[:target_count]
        top_sentences.sort(key=lambda x: x[2])
        
        # Build summary
        summary_sentences = [s[0] for s in top_sentences]
        summary = " ".join(summary_sentences)
        
        return SummaryResult(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text) if len(text) > 0 else 1.0,
            key_sentences=summary_sentences,
            method="extractive"
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Pattern for sentence boundaries
        pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentences(
        self,
        sentences: List[str],
        full_text: str
    ) -> List[float]:
        """Score sentences for importance.
        
        Args:
            sentences: List of sentences
            full_text: Full document text
            
        Returns:
            List of scores
        """
        if not sentences:
            return []
        
        # Compute document embedding
        doc_embedding = self._encoder.encode_text(full_text)
        
        # Score each sentence
        scores = []
        for sentence in sentences:
            # Get sentence embedding
            sent_embedding = self._encoder.encode_text(sentence)
            
            # Cosine similarity to document
            sim = self._cosine_similarity(sent_embedding, doc_embedding)
            
            # Position bonus (earlier sentences often more important)
            position_in_doc = full_text.find(sentence) / len(full_text)
            position_bonus = max(0, 0.3 * (1 - position_in_doc))
            
            # Length penalty for very short or very long sentences
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            length_ratio = len(sentence) / avg_len
            length_penalty = 1.0 if 0.5 < length_ratio < 2.0 else 0.8
            
            score = (sim + position_bonus) * length_penalty
            scores.append(score)
        
        return scores
    
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


class SemanticAnalyzer:
    """Analyzes semantic content of text.
    
    Provides semantic similarity, coherence analysis, and more.
    """
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """Initialize analyzer.
        
        Args:
            embedding_config: Embedding configuration
        """
        self.embedding_config = embedding_config or EmbeddingConfig(fallback_dim=256)
        self._encoder = TransformerEncoder(config=self.embedding_config)
        self._cache = EmbeddingCache(max_size=500)
        self._lock = threading.Lock()
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute semantic similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0, 1]
        """
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        return self._cosine_similarity(emb1, emb2)
    
    def compute_coherence(
        self,
        document: ProcessedDocument
    ) -> float:
        """Compute document coherence score.
        
        Measures how semantically connected chunks are.
        
        Args:
            document: Processed document
            
        Returns:
            Coherence score [0, 1]
        """
        if len(document.chunks) < 2:
            return 1.0
        
        # Compute similarity between adjacent chunks
        adjacent_sims = []
        for i in range(len(document.chunks) - 1):
            emb1 = document.chunks[i].embedding
            emb2 = document.chunks[i + 1].embedding
            
            if emb1 is not None and emb2 is not None:
                sim = self._cosine_similarity(emb1, emb2)
                adjacent_sims.append(sim)
        
        if not adjacent_sims:
            return 0.5
        
        return float(np.mean(adjacent_sims))
    
    def find_similar_chunks(
        self,
        query: str,
        document: ProcessedDocument,
        top_k: int = 3
    ) -> List[Tuple[TextChunk, float]]:
        """Find chunks most similar to query.
        
        Args:
            query: Query text
            document: Document to search
            top_k: Number of results
            
        Returns:
            List of (chunk, similarity) tuples
        """
        query_emb = self._get_embedding(query)
        
        results = []
        for chunk in document.chunks:
            if chunk.embedding is not None:
                sim = self._cosine_similarity(query_emb, chunk.embedding)
                results.append((chunk, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        cache_key = hashlib.md5(text[:200].encode()).hexdigest()[:16]
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        embedding = self._encoder.encode_text(text)
        self._cache.put(cache_key, embedding)
        
        return embedding
    
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


class TextUnderstandingPipeline:
    """Main unified interface for text understanding.
    
    Combines document processing, summarization, and semantic analysis.
    
    Example:
        >>> pipeline = TextUnderstandingPipeline()
        >>> doc = pipeline.process_document("Long text...")
        >>> summary = pipeline.summarize(doc)
        >>> chunks = pipeline.find_relevant_chunks(doc, "query")
    """
    
    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """Initialize pipeline.
        
        Args:
            chunk_config: Chunking configuration
            embedding_config: Embedding configuration
        """
        self.chunk_config = chunk_config or ChunkConfig()
        self.embedding_config = embedding_config or EmbeddingConfig(fallback_dim=256)
        
        self._processor = DocumentProcessor(self.chunk_config, self.embedding_config)
        self._summarizer = TextSummarizer(self.embedding_config)
        self._analyzer = SemanticAnalyzer(self.embedding_config)
        self._lock = threading.Lock()
    
    def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        compute_embeddings: bool = True
    ) -> ProcessedDocument:
        """Process a document.
        
        Args:
            text: Document text
            metadata: Optional metadata
            compute_embeddings: Whether to compute embeddings
            
        Returns:
            ProcessedDocument
        """
        return self._processor.process(text, metadata, compute_embeddings)
    
    def summarize(
        self,
        document: ProcessedDocument,
        ratio: float = 0.3,
        max_sentences: int = 5
    ) -> SummaryResult:
        """Summarize a document.
        
        Args:
            document: Processed document
            ratio: Compression ratio
            max_sentences: Maximum summary sentences
            
        Returns:
            SummaryResult
        """
        return self._summarizer.summarize(document.text, ratio, max_sentences)
    
    def summarize_text(
        self,
        text: str,
        ratio: float = 0.3,
        max_sentences: int = 5
    ) -> SummaryResult:
        """Summarize raw text directly.
        
        Args:
            text: Text to summarize
            ratio: Compression ratio
            max_sentences: Maximum summary sentences
            
        Returns:
            SummaryResult
        """
        return self._summarizer.summarize(text, ratio, max_sentences)
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute semantic similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        return self._analyzer.compute_similarity(text1, text2)
    
    def compute_document_similarity(
        self,
        doc1: ProcessedDocument,
        doc2: ProcessedDocument
    ) -> float:
        """Compute similarity between documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score
        """
        if doc1.embedding is None or doc2.embedding is None:
            return self._analyzer.compute_similarity(doc1.text, doc2.text)
        
        return self._analyzer._cosine_similarity(doc1.embedding, doc2.embedding)
    
    def find_relevant_chunks(
        self,
        document: ProcessedDocument,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[TextChunk, float]]:
        """Find chunks most relevant to query.
        
        Args:
            document: Document to search
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, similarity) tuples
        """
        return self._analyzer.find_similar_chunks(query, document, top_k)
    
    def get_document_coherence(
        self,
        document: ProcessedDocument
    ) -> float:
        """Get coherence score for document.
        
        Args:
            document: Document to analyze
            
        Returns:
            Coherence score [0, 1]
        """
        return self._analyzer.compute_coherence(document)
    
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract keywords from text.
        
        Args:
            text: Text to analyze
            top_k: Number of keywords
            
        Returns:
            List of (keyword, score) tuples
        """
        return self._processor._extract_keywords(text, top_k)
    
    def get_chunker(self) -> TextChunker:
        """Get the text chunker."""
        return self._processor._chunker
    
    def get_analyzer(self) -> SemanticAnalyzer:
        """Get the semantic analyzer."""
        return self._analyzer


# Convenience functions

def create_pipeline(
    chunk_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
    chunk_size: int = 500
) -> TextUnderstandingPipeline:
    """Create text understanding pipeline with configuration.
    
    Args:
        chunk_strategy: Chunking strategy to use
        chunk_size: Target chunk size
        
    Returns:
        TextUnderstandingPipeline instance
    """
    config = ChunkConfig(strategy=chunk_strategy, chunk_size=chunk_size)
    return TextUnderstandingPipeline(chunk_config=config)


def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
    chunk_size: int = 500
) -> List[TextChunk]:
    """Chunk text with specified strategy.
    
    Convenience function.
    
    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        
    Returns:
        List of TextChunk objects
    """
    config = ChunkConfig(strategy=strategy, chunk_size=chunk_size)
    chunker = TextChunker(config)
    return chunker.chunk(text)


def summarize_text(
    text: str,
    ratio: float = 0.3,
    max_sentences: int = 5
) -> str:
    """Summarize text.
    
    Convenience function.
    
    Args:
        text: Text to summarize
        ratio: Compression ratio
        max_sentences: Maximum sentences
        
    Returns:
        Summary text
    """
    summarizer = TextSummarizer()
    result = summarizer.summarize(text, ratio, max_sentences)
    return result.summary
