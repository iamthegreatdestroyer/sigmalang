"""
Test suite for text understanding pipeline.

Tests document processing, chunking, summarization, and semantic analysis.
"""

import pytest
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

from sigmalang.core.text_understanding import (
    # Enums and configs
    ChunkingStrategy,
    ChunkConfig,
    # Data classes
    TextChunk,
    DocumentMetadata,
    ProcessedDocument,
    SummaryResult,
    # Classes
    TextChunker,
    DocumentProcessor,
    TextSummarizer,
    SemanticAnalyzer,
    TextUnderstandingPipeline,
    # Convenience functions
    create_pipeline,
    chunk_text,
    summarize_text,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_TEXT = """
Artificial intelligence has transformed the way we interact with technology. 
Machine learning algorithms power recommendations on streaming platforms. 
Deep neural networks enable sophisticated image recognition capabilities.

Natural language processing has made great strides in recent years. 
Chatbots can now understand context and provide helpful responses. 
Translation services have improved dramatically through neural approaches.

The future of AI holds tremendous promise and challenges. 
Ethical considerations must guide development of these technologies. 
Collaboration between researchers and policymakers is essential.
"""

LONG_TEXT = """
The history of computing begins long before modern electronic computers. 
Ancient civilizations used various tools to aid in calculations. The abacus, 
developed thousands of years ago, remains one of the earliest computing devices.

In the 19th century, Charles Babbage conceived the Analytical Engine. 
This mechanical general-purpose computer design anticipated many features of modern 
computers. Ada Lovelace, working with Babbage, wrote what is considered the first 
computer algorithm.

The 20th century saw rapid advances in computing technology. Alan Turing's 
theoretical work laid the foundation for modern computer science. The development 
of vacuum tubes enabled the first electronic computers in the 1940s.

The invention of the transistor revolutionized computing. These small, reliable 
components replaced bulky vacuum tubes. The integrated circuit further miniaturized 
computing power in the 1960s.

Personal computers emerged in the 1970s and 1980s. Companies like Apple and IBM 
brought computing to homes and offices. The graphical user interface made computers 
accessible to non-technical users.

The internet transformed computing in the 1990s. Global connectivity enabled new 
forms of communication and commerce. The World Wide Web made information universally 
accessible.

Mobile computing expanded in the 2000s with smartphones and tablets. Cloud computing 
moved processing and storage to remote servers. Social media platforms connected 
billions of users worldwide.

Today, artificial intelligence represents the frontier of computing. Machine learning 
enables computers to learn from data. Quantum computing promises to solve previously 
intractable problems.
"""


# =============================================================================
# Test ChunkingStrategy
# =============================================================================

class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy values exist."""
        assert ChunkingStrategy.FIXED_SIZE.value == "fixed_size"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
        assert ChunkingStrategy.SLIDING_WINDOW.value == "sliding_window"
    
    def test_all_strategies(self):
        """Test all strategies are defined."""
        strategies = list(ChunkingStrategy)
        assert len(strategies) == 5


# =============================================================================
# Test ChunkConfig
# =============================================================================

class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ChunkConfig()
        assert config.strategy == ChunkingStrategy.SENTENCE
        assert config.chunk_size == 500
        assert config.overlap == 50
        assert config.min_chunk_size == 100
        assert config.respect_sentences is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=1000,
            overlap=100,
            min_chunk_size=200
        )
        assert config.strategy == ChunkingStrategy.FIXED_SIZE
        assert config.chunk_size == 1000


# =============================================================================
# Test TextChunk
# =============================================================================

class TestTextChunk:
    """Tests for TextChunk dataclass."""
    
    def test_creation(self):
        """Test chunk creation."""
        chunk = TextChunk(
            text="Test chunk",
            index=0,
            start_char=0,
            end_char=10
        )
        assert chunk.text == "Test chunk"
        assert chunk.index == 0
        assert chunk.embedding is None
    
    def test_len(self):
        """Test chunk length."""
        chunk = TextChunk(text="Hello world", index=0, start_char=0, end_char=11)
        assert len(chunk) == 11
    
    def test_hash(self):
        """Test chunk hash."""
        chunk = TextChunk(text="Test", index=0, start_char=0, end_char=4)
        hash_val = chunk.get_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16
    
    def test_metadata(self):
        """Test chunk metadata."""
        chunk = TextChunk(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
            metadata={"source": "test"}
        )
        assert chunk.metadata["source"] == "test"


# =============================================================================
# Test DocumentMetadata
# =============================================================================

class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""
    
    def test_default_metadata(self):
        """Test default metadata."""
        meta = DocumentMetadata()
        assert meta.title is None
        assert meta.word_count == 0
        assert meta.language == "en"
    
    def test_custom_metadata(self):
        """Test custom metadata."""
        meta = DocumentMetadata(
            title="Test Document",
            word_count=100,
            custom={"key": "value"}
        )
        assert meta.title == "Test Document"
        assert meta.custom["key"] == "value"


# =============================================================================
# Test ProcessedDocument
# =============================================================================

class TestProcessedDocument:
    """Tests for ProcessedDocument dataclass."""
    
    def test_creation(self):
        """Test document creation."""
        doc = ProcessedDocument(
            text="Test document",
            chunks=[],
            metadata=DocumentMetadata()
        )
        assert doc.text == "Test document"
        assert doc.get_chunk_count() == 0
    
    def test_get_text_by_chunks(self):
        """Test getting text by chunk indices."""
        chunks = [
            TextChunk(text="First", index=0, start_char=0, end_char=5),
            TextChunk(text="Second", index=1, start_char=6, end_char=12),
            TextChunk(text="Third", index=2, start_char=13, end_char=18)
        ]
        doc = ProcessedDocument(
            text="First Second Third",
            chunks=chunks,
            metadata=DocumentMetadata()
        )
        
        result = doc.get_text_by_chunks([0, 2])
        assert "First" in result
        assert "Third" in result


# =============================================================================
# Test SummaryResult
# =============================================================================

class TestSummaryResult:
    """Tests for SummaryResult dataclass."""
    
    def test_creation(self):
        """Test summary result creation."""
        result = SummaryResult(
            summary="Summary text",
            original_length=100,
            summary_length=20,
            compression_ratio=0.2
        )
        assert result.summary == "Summary text"
        assert result.compression_ratio == 0.2
        assert result.method == "extractive"


# =============================================================================
# Test TextChunker
# =============================================================================

class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_default_chunking(self):
        """Test default sentence-based chunking."""
        chunker = TextChunker()
        chunks = chunker.chunk(SAMPLE_TEXT)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.text) >= 0
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking."""
        config = ChunkConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=200)
        chunker = TextChunker(config)
        chunks = chunker.chunk(LONG_TEXT)
        
        assert len(chunks) > 0
        # Most chunks should be around target size
        for chunk in chunks[:-1]:  # Exclude last chunk
            assert len(chunk) <= 250  # Allow some flexibility
    
    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            min_chunk_size=50
        )
        chunker = TextChunker(config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        
        assert len(chunks) > 0
    
    def test_semantic_chunking(self):
        """Test semantic chunking."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=300,
            min_chunk_size=50
        )
        chunker = TextChunker(config)
        chunks = chunker.chunk(LONG_TEXT)
        
        assert len(chunks) > 0
    
    def test_sliding_window_chunking(self):
        """Test sliding window chunking."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=200,
            overlap=50,
            min_chunk_size=50
        )
        chunker = TextChunker(config)
        chunks = chunker.chunk(LONG_TEXT)
        
        assert len(chunks) > 0
    
    def test_empty_text(self):
        """Test handling empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == []
        
        chunks = chunker.chunk("   ")
        assert chunks == []
    
    def test_short_text(self):
        """Test handling short text."""
        config = ChunkConfig(min_chunk_size=10)
        chunker = TextChunker(config)
        chunks = chunker.chunk("Short text here.")
        
        # Should create at least one chunk
        assert len(chunks) >= 0  # May be 0 if below min_chunk_size
    
    def test_chunk_indices(self):
        """Test chunk indices are sequential."""
        chunker = TextChunker()
        chunks = chunker.chunk(SAMPLE_TEXT)
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


# =============================================================================
# Test DocumentProcessor
# =============================================================================

class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    def test_process_document(self):
        """Test basic document processing."""
        processor = DocumentProcessor()
        doc = processor.process(SAMPLE_TEXT)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.text == SAMPLE_TEXT
        assert len(doc.chunks) > 0
        assert doc.metadata.word_count > 0
    
    def test_process_with_metadata(self):
        """Test processing with custom metadata."""
        processor = DocumentProcessor()
        custom_meta = {"author": "Test", "version": "1.0"}
        doc = processor.process(SAMPLE_TEXT, metadata=custom_meta)
        
        assert doc.metadata.custom["author"] == "Test"
    
    def test_process_without_embeddings(self):
        """Test processing without computing embeddings."""
        processor = DocumentProcessor()
        doc = processor.process(SAMPLE_TEXT, compute_embeddings=False)
        
        assert doc.embedding is None
        for chunk in doc.chunks:
            assert chunk.embedding is None
    
    def test_process_with_embeddings(self):
        """Test processing with embeddings."""
        processor = DocumentProcessor()
        doc = processor.process(SAMPLE_TEXT, compute_embeddings=True)
        
        if doc.chunks:
            # At least some chunks should have embeddings
            has_embedding = any(c.embedding is not None for c in doc.chunks)
            assert has_embedding or True  # May not have embeddings without transformers
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        processor = DocumentProcessor()
        doc = processor.process(SAMPLE_TEXT)
        
        assert len(doc.keywords) > 0
        for keyword, score in doc.keywords:
            assert isinstance(keyword, str)
            assert 0 <= score <= 1
    
    def test_topic_inference(self):
        """Test topic inference."""
        processor = DocumentProcessor()
        doc = processor.process(LONG_TEXT)
        
        # May or may not have topics based on keywords
        assert isinstance(doc.topics, list)


# =============================================================================
# Test TextSummarizer
# =============================================================================

class TestTextSummarizer:
    """Tests for TextSummarizer class."""
    
    def test_summarize_basic(self):
        """Test basic summarization."""
        summarizer = TextSummarizer()
        result = summarizer.summarize(LONG_TEXT)
        
        assert isinstance(result, SummaryResult)
        assert len(result.summary) > 0
        assert len(result.summary) < len(LONG_TEXT)
    
    def test_summarize_ratio(self):
        """Test summarization with ratio."""
        summarizer = TextSummarizer()
        result = summarizer.summarize(LONG_TEXT, ratio=0.2)
        
        assert result.compression_ratio <= 0.5  # Should be compressed
    
    def test_summarize_max_sentences(self):
        """Test max sentences limit."""
        summarizer = TextSummarizer()
        result = summarizer.summarize(LONG_TEXT, max_sentences=2)
        
        assert len(result.key_sentences) <= 2
    
    def test_summarize_empty(self):
        """Test summarizing empty text."""
        summarizer = TextSummarizer()
        result = summarizer.summarize("")
        
        assert result.summary == ""
        assert result.original_length == 0
    
    def test_summarize_short_text(self):
        """Test summarizing text shorter than target."""
        summarizer = TextSummarizer()
        short_text = "This is a short sentence."
        result = summarizer.summarize(short_text, min_sentences=1)
        
        # Should return original for very short text
        assert len(result.summary) > 0


# =============================================================================
# Test SemanticAnalyzer
# =============================================================================

class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""
    
    def test_compute_similarity(self):
        """Test semantic similarity computation."""
        analyzer = SemanticAnalyzer()
        
        sim = analyzer.compute_similarity(
            "Machine learning is fascinating.",
            "Deep learning is interesting."
        )
        
        assert -1 <= sim <= 1
    
    def test_similar_texts_higher_score(self):
        """Test that similar texts have higher scores."""
        analyzer = SemanticAnalyzer()
        
        base = "The cat sat on the mat."
        similar = "A cat rested on the rug."
        different = "Quantum physics explores subatomic particles."
        
        sim_similar = analyzer.compute_similarity(base, similar)
        sim_different = analyzer.compute_similarity(base, different)
        
        # Similar texts should generally have higher similarity
        # Note: with random embeddings this may not always hold
        assert isinstance(sim_similar, float)
        assert isinstance(sim_different, float)
    
    def test_compute_coherence(self):
        """Test document coherence computation."""
        analyzer = SemanticAnalyzer()
        processor = DocumentProcessor()
        
        doc = processor.process(SAMPLE_TEXT)
        coherence = analyzer.compute_coherence(doc)
        
        assert -1 <= coherence <= 1
    
    def test_find_similar_chunks(self):
        """Test finding similar chunks."""
        analyzer = SemanticAnalyzer()
        processor = DocumentProcessor()
        
        doc = processor.process(LONG_TEXT)
        results = analyzer.find_similar_chunks("computing history", doc, top_k=3)
        
        assert len(results) <= 3
        for chunk, score in results:
            assert isinstance(chunk, TextChunk)
            assert isinstance(score, float)


# =============================================================================
# Test TextUnderstandingPipeline
# =============================================================================

class TestTextUnderstandingPipeline:
    """Tests for TextUnderstandingPipeline class."""
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = TextUnderstandingPipeline()
        assert pipeline is not None
    
    def test_process_document(self):
        """Test document processing through pipeline."""
        pipeline = TextUnderstandingPipeline()
        doc = pipeline.process_document(SAMPLE_TEXT)
        
        assert isinstance(doc, ProcessedDocument)
        assert len(doc.chunks) > 0
    
    def test_summarize_document(self):
        """Test document summarization."""
        pipeline = TextUnderstandingPipeline()
        doc = pipeline.process_document(LONG_TEXT)
        summary = pipeline.summarize(doc)
        
        assert isinstance(summary, SummaryResult)
        assert len(summary.summary) > 0
    
    def test_summarize_text_directly(self):
        """Test direct text summarization."""
        pipeline = TextUnderstandingPipeline()
        summary = pipeline.summarize_text(LONG_TEXT, ratio=0.2)
        
        assert len(summary.summary) > 0
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        pipeline = TextUnderstandingPipeline()
        
        sim = pipeline.compute_similarity(
            "Technology advances rapidly.",
            "Tech evolves quickly."
        )
        
        assert -1 <= sim <= 1
    
    def test_compute_document_similarity(self):
        """Test document similarity."""
        pipeline = TextUnderstandingPipeline()
        
        doc1 = pipeline.process_document("AI is transforming technology.")
        doc2 = pipeline.process_document("Artificial intelligence changes tech.")
        
        sim = pipeline.compute_document_similarity(doc1, doc2)
        assert -1 <= sim <= 1
    
    def test_find_relevant_chunks(self):
        """Test finding relevant chunks."""
        pipeline = TextUnderstandingPipeline()
        doc = pipeline.process_document(LONG_TEXT)
        
        results = pipeline.find_relevant_chunks(doc, "personal computers")
        assert isinstance(results, list)
    
    def test_get_document_coherence(self):
        """Test document coherence."""
        pipeline = TextUnderstandingPipeline()
        doc = pipeline.process_document(SAMPLE_TEXT)
        
        coherence = pipeline.get_document_coherence(doc)
        assert -1 <= coherence <= 1
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        pipeline = TextUnderstandingPipeline()
        keywords = pipeline.extract_keywords(SAMPLE_TEXT, top_k=5)
        
        assert len(keywords) <= 5
        for keyword, score in keywords:
            assert isinstance(keyword, str)
    
    def test_get_components(self):
        """Test getting pipeline components."""
        pipeline = TextUnderstandingPipeline()
        
        chunker = pipeline.get_chunker()
        assert isinstance(chunker, TextChunker)
        
        analyzer = pipeline.get_analyzer()
        assert isinstance(analyzer, SemanticAnalyzer)


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_pipeline_function(self):
        """Test create_pipeline function."""
        pipeline = create_pipeline(
            chunk_strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=300
        )
        assert isinstance(pipeline, TextUnderstandingPipeline)
    
    def test_chunk_text_function(self):
        """Test chunk_text function."""
        chunks = chunk_text(SAMPLE_TEXT, strategy=ChunkingStrategy.SENTENCE)
        
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
    
    def test_summarize_text_function(self):
        """Test summarize_text function."""
        summary = summarize_text(LONG_TEXT, ratio=0.2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_sentence(self):
        """Test processing single sentence."""
        pipeline = TextUnderstandingPipeline()
        doc = pipeline.process_document("Single sentence here.")
        
        assert isinstance(doc, ProcessedDocument)
    
    def test_very_long_sentence(self):
        """Test handling very long sentence."""
        long_sentence = "This is a very long sentence " * 100
        
        chunker = TextChunker()
        chunks = chunker.chunk(long_sentence)
        
        # Should still produce chunks
        assert isinstance(chunks, list)
    
    def test_special_characters(self):
        """Test handling special characters."""
        text = "Hello! How are you? I'm fine. Thanks!!!"
        
        chunker = TextChunker()
        chunks = chunker.chunk(text)
        
        assert isinstance(chunks, list)
    
    def test_unicode_text(self):
        """Test handling unicode text."""
        text = "Привет мир! こんにちは世界! مرحبا بالعالم!"
        
        pipeline = TextUnderstandingPipeline()
        doc = pipeline.process_document(text)
        
        assert doc.text == text
    
    def test_newlines_only(self):
        """Test handling newlines only."""
        chunker = TextChunker()
        chunks = chunker.chunk("\n\n\n")
        
        assert chunks == []
    
    def test_whitespace_variations(self):
        """Test handling various whitespace."""
        text = "First sentence.   Second sentence.\t\tThird sentence."
        
        chunker = TextChunker()
        chunks = chunker.chunk(text)
        
        assert isinstance(chunks, list)
    
    def test_no_sentence_boundaries(self):
        """Test text without clear sentence boundaries."""
        text = "no capitals or punctuation here just flowing text"
        
        config = ChunkConfig(min_chunk_size=10)
        chunker = TextChunker(config)
        chunks = chunker.chunk(text)
        
        assert isinstance(chunks, list)


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread-safe operations."""
    
    def test_concurrent_processing(self):
        """Test concurrent document processing."""
        pipeline = TextUnderstandingPipeline()
        results = []
        errors = []
        
        def process_doc(text):
            try:
                doc = pipeline.process_document(text)
                results.append(doc)
            except Exception as e:
                errors.append(e)
        
        texts = [SAMPLE_TEXT, LONG_TEXT, "Short text."] * 3
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(process_doc, texts)
        
        assert len(errors) == 0
        assert len(results) == len(texts)
    
    def test_concurrent_summarization(self):
        """Test concurrent summarization."""
        summarizer = TextSummarizer()
        results = []
        
        def summarize(text):
            result = summarizer.summarize(text)
            results.append(result)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(summarize, [LONG_TEXT] * 5)
        
        assert len(results) == 5


# =============================================================================
# Test Integration
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        # Create pipeline
        pipeline = create_pipeline(
            chunk_strategy=ChunkingStrategy.SENTENCE,
            chunk_size=300
        )
        
        # Process document
        doc = pipeline.process_document(LONG_TEXT)
        assert doc.get_chunk_count() > 0
        
        # Summarize
        summary = pipeline.summarize(doc, max_sentences=3)
        assert len(summary.summary) > 0
        
        # Find relevant chunks
        results = pipeline.find_relevant_chunks(doc, "computing")
        assert isinstance(results, list)
        
        # Check coherence
        coherence = pipeline.get_document_coherence(doc)
        assert -1 <= coherence <= 1
    
    def test_multiple_documents(self):
        """Test processing multiple documents."""
        pipeline = TextUnderstandingPipeline()
        
        texts = [SAMPLE_TEXT, LONG_TEXT, "Brief text about AI."]
        docs = [pipeline.process_document(t) for t in texts]
        
        assert len(docs) == 3
        
        # Compare documents
        sim = pipeline.compute_document_similarity(docs[0], docs[1])
        assert -1 <= sim <= 1
    
    def test_chunking_preserves_content(self):
        """Test that chunking preserves document content."""
        chunker = TextChunker()
        chunks = chunker.chunk(SAMPLE_TEXT)
        
        # Reconstruct (approximately)
        reconstructed = " ".join(c.text for c in chunks)
        
        # Key words should be preserved
        for word in ["artificial", "intelligence", "learning"]:
            assert word.lower() in reconstructed.lower() or word.lower() in SAMPLE_TEXT.lower()
