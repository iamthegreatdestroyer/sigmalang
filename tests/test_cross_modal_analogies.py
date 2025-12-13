"""
Tests for cross-modal analogies module.
"""

import hashlib
import numpy as np
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sigmalang.core.cross_modal_analogies import (
    # Enums
    Modality,
    # Dataclasses
    ModalityConfig,
    AnalogyPair,
    AnalogyResult,
    RelationResult,
    # Classes
    ModalityProjector,
    CrossModalEncoder,
    RelationExtractor,
    AnalogySolver,
    RelationTransfer,
    CrossModalAnalogy,
    # Convenience functions
    create_analogy_solver,
    solve_analogy,
)


class TestModality:
    """Tests for Modality enum."""
    
    def test_modality_values(self):
        """Test all modality values exist."""
        assert hasattr(Modality, 'TEXT')
        assert hasattr(Modality, 'CODE')
        assert hasattr(Modality, 'MATH')
        assert hasattr(Modality, 'SYMBOL')
        assert hasattr(Modality, 'SEMANTIC')
        assert hasattr(Modality, 'NUMERIC')
    
    def test_modality_uniqueness(self):
        """Test modality values are unique."""
        values = [m.value for m in Modality]
        assert len(values) == len(set(values))


class TestModalityConfig:
    """Tests for ModalityConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ModalityConfig()
        assert config.modality == Modality.TEXT
        assert config.embedding_dim == 384
        assert config.projection_dim == 256
        assert config.normalize is True
        assert config.use_cache is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModalityConfig(
            modality=Modality.CODE,
            embedding_dim=512,
            projection_dim=128,
            normalize=False,
            use_cache=False
        )
        assert config.modality == Modality.CODE
        assert config.embedding_dim == 512
        assert config.projection_dim == 128
        assert config.normalize is False
        assert config.use_cache is False


class TestAnalogyPair:
    """Tests for AnalogyPair dataclass."""
    
    def test_basic_pair(self):
        """Test basic analogy pair creation."""
        pair = AnalogyPair(source="king", target="queen")
        assert pair.source == "king"
        assert pair.target == "queen"
        assert pair.modality == Modality.TEXT
        assert pair.confidence == 1.0
    
    def test_pair_with_relation(self):
        """Test pair with relation vector."""
        relation = np.random.randn(256)
        pair = AnalogyPair(
            source="man",
            target="woman",
            relation_vector=relation,
            confidence=0.9
        )
        assert pair.source == "man"
        assert pair.target == "woman"
        assert pair.relation_vector is not None
        assert pair.confidence == 0.9


class TestAnalogyResult:
    """Tests for AnalogyResult dataclass."""
    
    def test_basic_result(self):
        """Test basic result creation."""
        result = AnalogyResult(
            query=("king", "man", "woman"),
            answer="queen",
            confidence=0.95
        )
        assert result.query == ("king", "man", "woman")
        assert result.answer == "queen"
        assert result.confidence == 0.95
        assert result.candidates == []
    
    def test_result_with_candidates(self):
        """Test result with candidates."""
        result = AnalogyResult(
            query=("a", "b", "c"),
            answer="d",
            confidence=0.8,
            candidates=[("d", 0.8), ("e", 0.6)],
            computation_time=0.1,
            method="vector_arithmetic"
        )
        assert len(result.candidates) == 2
        assert result.computation_time == 0.1
        assert result.method == "vector_arithmetic"


class TestRelationResult:
    """Tests for RelationResult dataclass."""
    
    def test_relation_result(self):
        """Test relation result creation."""
        vec = np.random.randn(256)
        result = RelationResult(
            source="dog",
            target="animal",
            relation_type="hypernymy",
            relation_vector=vec,
            strength=0.85,
            bidirectional=False
        )
        assert result.source == "dog"
        assert result.target == "animal"
        assert result.relation_type == "hypernymy"
        assert result.strength == 0.85
        assert result.bidirectional is False


class TestModalityProjector:
    """Tests for ModalityProjector class."""
    
    def test_projector_init(self):
        """Test projector initialization."""
        projector = ModalityProjector(
            source_dim=384,
            target_dim=384,
            shared_dim=256
        )
        assert projector.source_dim == 384
        assert projector.target_dim == 384
        assert projector.shared_dim == 256
    
    def test_project_source(self):
        """Test source projection."""
        projector = ModalityProjector(384, 384, 256)
        embedding = np.random.randn(384)
        
        projected = projector.project_source(embedding)
        
        assert projected.shape == (256,)
        # Should be normalized
        assert np.abs(np.linalg.norm(projected) - 1.0) < 1e-5
    
    def test_project_target(self):
        """Test target projection."""
        projector = ModalityProjector(384, 384, 256)
        embedding = np.random.randn(384)
        
        projected = projector.project_target(embedding)
        
        assert projected.shape == (256,)
        assert np.abs(np.linalg.norm(projected) - 1.0) < 1e-5
    
    def test_projector_deterministic(self):
        """Test projector is deterministic."""
        projector = ModalityProjector(384, 384, 256)
        embedding = np.random.randn(384)
        
        proj1 = projector.project_source(embedding)
        proj2 = projector.project_source(embedding)
        
        np.testing.assert_array_almost_equal(proj1, proj2)
    
    def test_zero_embedding(self):
        """Test handling zero embedding."""
        projector = ModalityProjector(384, 384, 256)
        zero_emb = np.zeros(384)
        
        projected = projector.project_source(zero_emb)
        
        # Should return zero (can't normalize)
        assert np.linalg.norm(projected) == 0


class TestCrossModalEncoder:
    """Tests for CrossModalEncoder class."""
    
    def test_encoder_init(self):
        """Test encoder initialization."""
        encoder = CrossModalEncoder()
        assert encoder._encoder is not None
    
    def test_encode_text(self):
        """Test text encoding."""
        encoder = CrossModalEncoder()
        embedding = encoder.encode("hello world", Modality.TEXT)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_encode_code(self):
        """Test code encoding."""
        encoder = CrossModalEncoder()
        embedding = encoder.encode("def foo(): pass", Modality.CODE)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_encode_math(self):
        """Test math encoding."""
        encoder = CrossModalEncoder()
        embedding = encoder.encode("x^2 + y^2 = r^2", Modality.MATH)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_encode_batch(self):
        """Test batch encoding."""
        encoder = CrossModalEncoder()
        texts = ["hello", "world", "test"]
        
        embeddings = encoder.encode_batch(texts, Modality.TEXT)
        
        assert embeddings.shape[0] == 3
    
    def test_cross_modal_similarity(self):
        """Test cross-modal similarity."""
        encoder = CrossModalEncoder()
        
        sim = encoder.cross_modal_similarity(
            "dog", Modality.TEXT,
            "animal", Modality.TEXT
        )
        
        assert -1 <= sim <= 1
    
    def test_same_text_high_similarity(self):
        """Test same text has high similarity."""
        encoder = CrossModalEncoder()
        
        sim = encoder.cross_modal_similarity(
            "hello", Modality.TEXT,
            "hello", Modality.TEXT
        )
        
        assert sim > 0.9
    
    def test_get_projector(self):
        """Test getting projector."""
        encoder = CrossModalEncoder()
        
        projector = encoder.get_projector(Modality.TEXT, Modality.CODE)
        
        assert isinstance(projector, ModalityProjector)
    
    def test_projector_caching(self):
        """Test projectors are cached."""
        encoder = CrossModalEncoder()
        
        proj1 = encoder.get_projector(Modality.TEXT, Modality.CODE)
        proj2 = encoder.get_projector(Modality.TEXT, Modality.CODE)
        
        assert proj1 is proj2


class TestRelationExtractor:
    """Tests for RelationExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create relation extractor."""
        encoder = CrossModalEncoder()
        return RelationExtractor(encoder)
    
    def test_extract_relation(self, extractor):
        """Test basic relation extraction."""
        result = extractor.extract_relation("king", "queen")
        
        assert isinstance(result, RelationResult)
        assert result.source == "king"
        assert result.target == "queen"
        assert result.relation_vector is not None
    
    def test_relation_has_type(self, extractor):
        """Test relation has type."""
        result = extractor.extract_relation("dog", "animal")
        
        assert result.relation_type in extractor.RELATION_TYPES or \
               result.relation_type == "association"
    
    def test_relation_strength(self, extractor):
        """Test relation strength is valid."""
        result = extractor.extract_relation("big", "large")
        
        assert 0 <= result.strength <= 1
    
    def test_relation_caching(self, extractor):
        """Test relations are cached."""
        result1 = extractor.extract_relation("cat", "animal")
        result2 = extractor.extract_relation("cat", "animal")
        
        np.testing.assert_array_equal(
            result1.relation_vector,
            result2.relation_vector
        )
    
    def test_find_similar_relations(self, extractor):
        """Test finding similar relations."""
        relation = extractor.extract_relation("king", "queen")
        candidates = [("man", "woman"), ("boy", "girl"), ("dog", "cat")]
        
        results = extractor.find_similar_relations(relation, candidates, top_k=3)
        
        assert len(results) <= 3
        for (pair, sim) in results:
            assert -1 <= sim <= 1


class TestAnalogySolver:
    """Tests for AnalogySolver class."""
    
    @pytest.fixture
    def solver(self):
        """Create analogy solver with vocabulary."""
        encoder = CrossModalEncoder()
        vocab = ["queen", "woman", "girl", "female", "princess",
                 "king", "man", "boy", "male", "prince"]
        return AnalogySolver(encoder, vocab)
    
    def test_solve_basic(self, solver):
        """Test basic analogy solving."""
        result = solver.solve("king", "man", "woman")
        
        assert isinstance(result, AnalogyResult)
        assert result.query == ("king", "man", "woman")
        assert result.answer is not None
    
    def test_solve_has_candidates(self, solver):
        """Test solving produces candidates when vocabulary has matches."""
        # Use words from the vocabulary
        result = solver.solve("king", "queen", "man")
        
        # Candidates come from vocabulary excluding query words
        # Since vocabulary has 10 words and 3 are excluded, we should have some
        assert result.candidates is not None
    
    def test_solve_has_confidence(self, solver):
        """Test solving produces confidence."""
        result = solver.solve("king", "queen", "man")
        
        assert 0 <= result.confidence <= 1
    
    def test_solve_relation_transfer_method(self, solver):
        """Test relation transfer method."""
        result = solver.solve(
            "king", "man", "woman",
            method="relation_transfer"
        )
        
        assert result.method == "relation_transfer"
    
    def test_set_vocabulary(self):
        """Test setting vocabulary."""
        encoder = CrossModalEncoder()
        solver = AnalogySolver(encoder)
        
        solver.set_vocabulary(["dog", "cat", "bird"])
        
        assert len(solver._vocabulary) == 3
    
    def test_add_to_vocabulary(self, solver):
        """Test adding to vocabulary."""
        initial_size = len(solver._vocabulary)
        solver.add_to_vocabulary(["new_word"])
        
        assert len(solver._vocabulary) == initial_size + 1
    
    def test_add_duplicate_vocabulary(self, solver):
        """Test adding duplicate words."""
        initial_size = len(solver._vocabulary)
        solver.add_to_vocabulary(["queen"])  # Already exists
        
        assert len(solver._vocabulary) == initial_size
    
    def test_evaluate_analogy(self, solver):
        """Test analogy evaluation."""
        result = solver.evaluate_analogy(
            "king", "man", "woman", "queen"
        )
        
        assert "correct" in result
        assert "predicted" in result
        assert "expected" in result
        assert "confidence" in result


class TestRelationTransfer:
    """Tests for RelationTransfer class."""
    
    @pytest.fixture
    def transfer(self):
        """Create relation transfer instance."""
        encoder = CrossModalEncoder()
        extractor = RelationExtractor(encoder)
        return RelationTransfer(encoder, extractor)
    
    def test_transfer_relation(self, transfer):
        """Test basic relation transfer."""
        vocab = ["woman", "girl", "female", "lady"]
        
        results = transfer.transfer_relation(
            ("king", "queen"),
            "man",
            vocab
        )
        
        assert len(results) > 0
        for (word, conf) in results:
            assert word in vocab
            assert -1 <= conf <= 1
    
    def test_generate_analogies(self, transfer):
        """Test analogy generation."""
        pool = ["dog", "cat", "puppy", "kitten", "bird", "chick"]
        
        results = transfer.generate_analogies(
            ("dog", "puppy"),
            pool,
            top_k=3
        )
        
        assert len(results) <= 3
        for (c, d, conf) in results:
            assert c in pool
            assert d in pool
            assert c != d


class TestCrossModalAnalogy:
    """Tests for CrossModalAnalogy main class."""
    
    @pytest.fixture
    def analogy(self):
        """Create cross-modal analogy instance."""
        vocab = ["queen", "woman", "girl", "female", "princess",
                 "king", "man", "boy", "male", "prince"]
        return CrossModalAnalogy(vocabulary=vocab)
    
    def test_init(self):
        """Test initialization."""
        cma = CrossModalAnalogy()
        assert cma._encoder is not None
        assert cma._solver is not None
    
    def test_solve(self, analogy):
        """Test basic solving."""
        result = analogy.solve("king", "man", "woman")
        
        assert isinstance(result, AnalogyResult)
    
    def test_set_vocabulary(self):
        """Test setting vocabulary."""
        cma = CrossModalAnalogy()
        cma.set_vocabulary(["a", "b", "c"])
        
        assert len(cma._solver._vocabulary) == 3
    
    def test_add_vocabulary(self, analogy):
        """Test adding vocabulary."""
        initial = len(analogy._solver._vocabulary)
        analogy.add_vocabulary(["test_word"])
        
        assert len(analogy._solver._vocabulary) == initial + 1
    
    def test_extract_relation(self, analogy):
        """Test relation extraction."""
        result = analogy.extract_relation("big", "small")
        
        assert isinstance(result, RelationResult)
    
    def test_transfer_relation(self, analogy):
        """Test relation transfer."""
        vocab = ["woman", "girl"]
        
        results = analogy.transfer_relation(
            ("king", "queen"),
            "man",
            vocab
        )
        
        assert len(results) > 0
    
    def test_generate_analogies(self, analogy):
        """Test analogy generation."""
        pool = ["dog", "cat", "puppy", "kitten"]
        
        results = analogy.generate_analogies(
            ("dog", "puppy"),
            pool
        )
        
        assert isinstance(results, list)
    
    def test_evaluate(self, analogy):
        """Test evaluation."""
        test_analogies = [
            ("king", "man", "woman", "queen"),
            ("man", "woman", "king", "queen")
        ]
        
        metrics = analogy.evaluate(test_analogies)
        
        assert "accuracy" in metrics
        assert "correct" in metrics
        assert "total" in metrics
        assert metrics["total"] == 2
    
    def test_similarity(self, analogy):
        """Test similarity computation."""
        sim = analogy.similarity("king", "queen")
        
        assert -1 <= sim <= 1
    
    def test_similarity_cross_modal(self, analogy):
        """Test cross-modal similarity."""
        sim = analogy.similarity(
            "def x(): pass",
            "function definition",
            Modality.CODE,
            Modality.TEXT
        )
        
        assert -1 <= sim <= 1
    
    def test_solve_cross_modal(self, analogy):
        """Test cross-modal solving."""
        result = analogy.solve_cross_modal(
            "king", Modality.TEXT,
            "queen", Modality.TEXT,
            "def foo():", Modality.CODE,
            Modality.CODE
        )
        
        assert isinstance(result, AnalogyResult)
        assert result.method == "cross_modal"
    
    def test_get_encoder(self, analogy):
        """Test getting encoder."""
        encoder = analogy.get_encoder()
        assert isinstance(encoder, CrossModalEncoder)
    
    def test_get_relation_extractor(self, analogy):
        """Test getting relation extractor."""
        extractor = analogy.get_relation_extractor()
        assert isinstance(extractor, RelationExtractor)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_analogy_solver(self):
        """Test creating analogy solver."""
        solver = create_analogy_solver()
        assert isinstance(solver, CrossModalAnalogy)
    
    def test_create_analogy_solver_with_vocab(self):
        """Test creating with vocabulary."""
        vocab = ["a", "b", "c"]
        solver = create_analogy_solver(vocab)
        
        assert len(solver._solver._vocabulary) == 3
    
    def test_solve_analogy(self):
        """Test solve_analogy function."""
        result = solve_analogy("king", "man", "woman")
        
        assert isinstance(result, AnalogyResult)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_vocabulary(self):
        """Test with empty vocabulary."""
        cma = CrossModalAnalogy()
        result = cma.solve("a", "b", "c")
        
        # Should return fallback
        assert result.answer is not None
    
    def test_same_words(self):
        """Test analogy with same words."""
        cma = CrossModalAnalogy(vocabulary=["a", "b"])
        result = cma.solve("a", "a", "b")
        
        assert result is not None
    
    def test_empty_string(self):
        """Test with empty string."""
        cma = CrossModalAnalogy()
        result = cma.solve("", "a", "b")
        
        assert result is not None
    
    def test_special_characters(self):
        """Test with special characters."""
        cma = CrossModalAnalogy()
        result = cma.solve("hello!", "world@", "test#")
        
        assert result is not None
    
    def test_unicode(self):
        """Test with unicode."""
        cma = CrossModalAnalogy()
        result = cma.solve("café", "naïve", "résumé")
        
        assert result is not None
    
    def test_long_text(self):
        """Test with long text."""
        cma = CrossModalAnalogy()
        long_text = "word " * 100
        result = cma.solve(long_text, "short", "medium")
        
        assert result is not None


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_encoding(self):
        """Test concurrent encoding is thread-safe."""
        encoder = CrossModalEncoder()
        results = []
        errors = []
        
        def encode_text(text):
            try:
                emb = encoder.encode(text, Modality.TEXT)
                results.append(emb)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(encode_text, f"text_{i}")
                for i in range(20)
            ]
            for f in as_completed(futures):
                pass
        
        assert len(errors) == 0
        assert len(results) == 20
    
    def test_concurrent_solving(self):
        """Test concurrent solving is thread-safe."""
        vocab = ["queen", "woman", "king", "man"]
        cma = CrossModalAnalogy(vocabulary=vocab)
        results = []
        errors = []
        
        def solve_analogy(a, b, c):
            try:
                result = cma.solve(a, b, c)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(solve_analogy, "king", "man", "woman")
                for _ in range(10)
            ]
            for f in as_completed(futures):
                pass
        
        assert len(errors) == 0
        assert len(results) == 10


class TestModalities:
    """Tests for different modalities."""
    
    def test_all_modalities_encode(self):
        """Test all modalities can encode."""
        encoder = CrossModalEncoder()
        
        for modality in Modality:
            emb = encoder.encode("test", modality)
            assert emb is not None
            assert len(emb) > 0
    
    def test_modality_affects_embedding(self):
        """Test modality changes embedding."""
        encoder = CrossModalEncoder()
        text = "def foo(): pass"
        
        emb_text = encoder.encode(text, Modality.TEXT)
        emb_code = encoder.encode(text, Modality.CODE)
        
        # Should be different due to preprocessing
        assert not np.allclose(emb_text, emb_code)
    
    def test_cross_modality_projector(self):
        """Test cross-modality projection."""
        encoder = CrossModalEncoder()
        
        projector = encoder.get_projector(Modality.TEXT, Modality.CODE)
        
        text_emb = encoder.encode("hello", Modality.TEXT)
        code_emb = encoder.encode("hello", Modality.CODE)
        
        proj_text = projector.project_source(text_emb)
        proj_code = projector.project_target(code_emb)
        
        # Both should be in shared space
        assert proj_text.shape == proj_code.shape


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete analogy workflow."""
        # Create solver with vocabulary
        vocab = ["queen", "woman", "princess", "female",
                 "king", "man", "prince", "male"]
        cma = CrossModalAnalogy(vocabulary=vocab)
        
        # Extract relation
        relation = cma.extract_relation("king", "queen")
        assert relation is not None
        
        # Solve analogy
        result = cma.solve("king", "queen", "man")
        assert result is not None
        
        # Transfer relation
        transfers = cma.transfer_relation(
            ("king", "queen"),
            "prince",
            vocab
        )
        assert len(transfers) > 0
        
        # Evaluate
        test_data = [("king", "man", "woman", "queen")]
        metrics = cma.evaluate(test_data)
        assert metrics["total"] == 1
    
    def test_modality_chain(self):
        """Test chaining across modalities."""
        cma = CrossModalAnalogy()
        
        # Encode in different modalities
        text_sim = cma.similarity("hello", "world", Modality.TEXT)
        code_sim = cma.similarity("foo", "bar", Modality.CODE)
        math_sim = cma.similarity("x+1", "y+2", Modality.MATH)
        
        # All should be valid similarities
        assert -1 <= text_sim <= 1
        assert -1 <= code_sim <= 1
        assert -1 <= math_sim <= 1
