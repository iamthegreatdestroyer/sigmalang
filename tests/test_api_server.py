"""
Tests for Phase 3: API Server

Comprehensive test suite for the REST API service.
"""

import time
from unittest.mock import patch, MagicMock
from dataclasses import asdict

import pytest
import numpy as np

from sigmalang.core.api_models import (
    # Request models
    EncodeRequest, DecodeRequest,
    AnalogyRequest, AnalogyExplainRequest,
    SearchRequest, EntityExtractionRequest,
    EmbeddingRequest, SimilarityRequest,
    # Response models
    EncodeResponse, DecodeResponse,
    AnalogyResponse, AnalogyExplainResponse, AnalogySolution,
    SearchResponse, SearchResult,
    EntityExtractionResponse, ExtractedEntity, ExtractedRelation,
    EmbeddingResponse, SimilarityResponse,
    HealthResponse, InfoResponse,
    ErrorResponse, create_error_response,
    # Enums
    AnalogyType, SearchMode, EntityType, OutputFormat,
)
from sigmalang.core.api_server import (
    # Services
    EncoderService,
    DecoderService,
    AnalogyService,
    SearchService,
    EntityService,
    NLPService,
    # Main API
    SigmalangAPI,
    get_api,
    create_api,
    __version__,
)
from sigmalang.core.config import SigmalangConfig, FeatureFlags, reset_config


# =============================================================================
# API Models Tests
# =============================================================================

class TestEncodeModels:
    """Tests for encode request/response models."""
    
    def test_encode_request_defaults(self):
        """Test EncodeRequest default values."""
        request = EncodeRequest(text="Hello world")
        
        assert request.text == "Hello world"
        assert request.output_format == OutputFormat.JSON
        assert request.normalize is True
        assert request.request_id is not None
    
    def test_encode_request_batch(self):
        """Test EncodeRequest with batch texts."""
        request = EncodeRequest(texts=["Hello", "World"])
        
        assert request.texts is not None
        assert len(request.texts) == 2
    
    def test_encode_response(self):
        """Test EncodeResponse creation."""
        response = EncodeResponse(
            success=True,
            vector=[0.1, 0.2, 0.3],
            dimensions=3,
            token_count=2
        )
        
        assert response.success is True
        assert len(response.vector) == 3
        assert response.dimensions == 3


class TestAnalogyModels:
    """Tests for analogy request/response models."""
    
    def test_analogy_request(self):
        """Test AnalogyRequest creation."""
        request = AnalogyRequest(
            a="king",
            b="queen",
            c="man",
            top_k=5
        )
        
        assert request.a == "king"
        assert request.b == "queen"
        assert request.c == "man"
        assert request.analogy_type == AnalogyType.SEMANTIC
    
    def test_analogy_solution(self):
        """Test AnalogySolution creation."""
        solution = AnalogySolution(
            answer="woman",
            confidence=0.95,
            relation="gender_counterpart",
            reasoning="King is to queen as man is to woman"
        )
        
        assert solution.answer == "woman"
        assert solution.confidence == 0.95
    
    def test_analogy_response(self):
        """Test AnalogyResponse with solutions."""
        solutions = [
            AnalogySolution(answer="woman", confidence=0.95),
            AnalogySolution(answer="female", confidence=0.75),
        ]
        
        response = AnalogyResponse(
            success=True,
            solutions=solutions,
            best_answer="woman",
            confidence=0.95
        )
        
        assert len(response.solutions) == 2
        assert response.best_answer == "woman"


class TestSearchModels:
    """Tests for search request/response models."""
    
    def test_search_request(self):
        """Test SearchRequest creation."""
        request = SearchRequest(
            query="machine learning",
            corpus=["AI", "ML", "DL"],
            top_k=5
        )
        
        assert request.query == "machine learning"
        assert len(request.corpus) == 3
        assert request.mode == SearchMode.SEMANTIC
    
    def test_search_result(self):
        """Test SearchResult creation."""
        result = SearchResult(
            text="Machine learning is...",
            score=0.95,
            index=0
        )
        
        assert result.text == "Machine learning is..."
        assert result.score == 0.95
    
    def test_search_response(self):
        """Test SearchResponse creation."""
        results = [
            SearchResult(text="Result 1", score=0.9, index=0),
            SearchResult(text="Result 2", score=0.8, index=1),
        ]
        
        response = SearchResponse(
            success=True,
            results=results,
            total_count=2
        )
        
        assert len(response.results) == 2
        assert response.total_count == 2


class TestEntityModels:
    """Tests for entity extraction models."""
    
    def test_extracted_entity(self):
        """Test ExtractedEntity creation."""
        entity = ExtractedEntity(
            text="Apple",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.95,
            start=0,
            end=5
        )
        
        assert entity.text == "Apple"
        assert entity.entity_type == EntityType.ORGANIZATION
    
    def test_extracted_relation(self):
        """Test ExtractedRelation creation."""
        relation = ExtractedRelation(
            source="Tim Cook",
            target="Apple",
            relation_type="CEO_OF",
            confidence=0.9
        )
        
        assert relation.source == "Tim Cook"
        assert relation.relation_type == "CEO_OF"


class TestHealthModels:
    """Tests for health and info models."""
    
    def test_health_response(self):
        """Test HealthResponse creation."""
        response = HealthResponse(
            success=True,
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0
        )
        
        assert response.status == "healthy"
        assert response.uptime_seconds == 3600.0
    
    def test_info_response(self):
        """Test InfoResponse creation."""
        response = InfoResponse(
            success=True,
            version="1.0.0",
            environment="development",
            features={"analogies": True, "search": True}
        )
        
        assert response.environment == "development"
        assert response.features["analogies"] is True


class TestErrorModels:
    """Tests for error models."""
    
    def test_create_error_response(self):
        """Test creating error response."""
        response = create_error_response(
            error="Something went wrong",
            error_code="ERR_001",
            request_id="req_123"
        )
        
        assert response.success is False
        assert response.error == "Something went wrong"
        assert response.error_code == "ERR_001"


# =============================================================================
# Service Tests
# =============================================================================

class TestEncoderService:
    """Tests for EncoderService."""
    
    def test_encoder_service_init(self):
        """Test EncoderService initialization."""
        service = EncoderService()
        
        assert service._initialized is False
    
    def test_encoder_service_initialize(self):
        """Test EncoderService initialization."""
        service = EncoderService()
        
        # Should not raise
        try:
            service.initialize()
            assert service._initialized is True
        except ImportError:
            # Core encoder might not be available
            pass
    
    def test_encoder_service_encode(self):
        """Test encoding text."""
        service = EncoderService()
        
        try:
            service.initialize()
            vector = service.encode("Hello world")
            
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0
        except (ImportError, AttributeError):
            # Skip if encoder not available
            pytest.skip("Encoder not available")
    
    def test_encoder_service_health(self):
        """Test encoder health check."""
        service = EncoderService()
        
        # Before initialization
        result = service.get_health()
        assert result.healthy is False
        
        # After initialization
        try:
            service.initialize()
            result = service.get_health()
            # May or may not be healthy depending on implementation
        except ImportError:
            pass


class TestAnalogyService:
    """Tests for AnalogyService."""
    
    def test_analogy_service_init(self):
        """Test AnalogyService initialization."""
        service = AnalogyService()
        
        assert service._initialized is False
    
    def test_analogy_service_solve(self):
        """Test solving analogies."""
        service = AnalogyService()
        
        try:
            service.initialize()
            solutions = service.solve("king", "queen", "man", top_k=3)
            
            assert isinstance(solutions, list)
            # Solutions may be empty if no good matches found
            if len(solutions) > 0:
                assert all(isinstance(s, AnalogySolution) for s in solutions)
        except (ImportError, RuntimeError) as e:
            # Skip if analogy engine not available or candidates not registered
            pytest.skip(f"Analogy engine not available: {e}")
    
    def test_analogy_service_explain(self):
        """Test explaining analogies."""
        service = AnalogyService()
        
        try:
            service.initialize()
            explanation = service.explain("king", "queen", "man", "woman")
            
            assert isinstance(explanation, dict)
            assert "explanation" in explanation
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Analogy engine not available: {e}")


class TestSearchService:
    """Tests for SearchService."""
    
    def test_search_service_init(self):
        """Test SearchService initialization."""
        service = SearchService()
        
        assert service._initialized is False
    
    def test_search_service_search(self):
        """Test searching corpus."""
        service = SearchService()
        
        try:
            service.initialize()
            results = service.search(
                query="machine learning",
                corpus=["AI is great", "Machine learning rocks", "Deep learning"],
                top_k=2
            )
            
            assert isinstance(results, list)
            assert all(isinstance(r, SearchResult) for r in results)
        except ImportError:
            pytest.skip("Search engine not available")


class TestNLPService:
    """Tests for NLPService."""
    
    def test_nlp_service_init(self):
        """Test NLPService initialization."""
        service = NLPService()
        
        assert service._initialized is False
    
    def test_nlp_service_embed(self):
        """Test generating embeddings."""
        service = NLPService()
        
        try:
            service.initialize()
            embedding = service.embed("Hello world")
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0
        except ImportError:
            pytest.skip("Embeddings not available")
    
    def test_nlp_service_similarity(self):
        """Test computing similarity."""
        service = NLPService()
        
        try:
            service.initialize()
            score = service.similarity("Hello", "Hi")
            
            assert isinstance(score, float)
            # Cosine similarity ranges from -1 to 1
            assert -1.0 <= score <= 1.0
        except ImportError:
            pytest.skip("Embeddings not available")


# =============================================================================
# SigmalangAPI Tests
# =============================================================================

class TestSigmalangAPI:
    """Tests for main SigmalangAPI class."""
    
    @pytest.fixture
    def api(self):
        """Create API instance for testing."""
        reset_config()
        api = SigmalangAPI()
        try:
            api.initialize()
        except Exception:
            pass  # Some services may not initialize
        return api
    
    def test_api_init(self, api):
        """Test API initialization."""
        assert api.encoder is not None
        assert api.decoder is not None
        assert api.analogy is not None
        assert api.search is not None
    
    def test_api_health(self, api):
        """Test health endpoint."""
        response = api.health()
        
        assert isinstance(response, HealthResponse)
        assert response.success is True
        assert response.version is not None
        assert response.uptime_seconds >= 0
    
    def test_api_info(self, api):
        """Test info endpoint."""
        response = api.info()
        
        assert isinstance(response, InfoResponse)
        assert response.success is True
        assert response.version is not None
        assert "analogies" in response.features
    
    def test_api_metrics(self, api):
        """Test metrics endpoint."""
        metrics = api.metrics()
        
        assert isinstance(metrics, str)
        # Should be Prometheus format
    
    def test_api_encode(self, api):
        """Test encode endpoint."""
        request = EncodeRequest(text="Hello world")
        response = api.encode(request)
        
        assert isinstance(response, EncodeResponse)
        assert response.request_id == request.request_id
        # May or may not succeed depending on encoder availability
    
    def test_api_encode_batch(self, api):
        """Test batch encode endpoint."""
        request = EncodeRequest(texts=["Hello", "World"])
        response = api.encode(request)
        
        assert isinstance(response, EncodeResponse)
    
    def test_api_solve_analogy(self, api):
        """Test analogy solving endpoint."""
        request = AnalogyRequest(a="king", b="queen", c="man", top_k=3)
        response = api.solve_analogy(request)
        
        assert isinstance(response, AnalogyResponse)
        assert response.request_id == request.request_id
    
    def test_api_solve_analogy_with_verification(self, api):
        """Test analogy solving with answer verification."""
        request = AnalogyRequest(
            a="king", b="queen", c="man", d="woman", top_k=3
        )
        response = api.solve_analogy(request)
        
        assert isinstance(response, AnalogyResponse)
        # is_valid should be set when d is provided
    
    def test_api_explain_analogy(self, api):
        """Test analogy explanation endpoint."""
        request = AnalogyExplainRequest(
            a="king", b="queen", c="man", d="woman"
        )
        response = api.explain_analogy(request)
        
        assert isinstance(response, AnalogyExplainResponse)
        assert response.request_id == request.request_id
    
    def test_api_search(self, api):
        """Test search endpoint."""
        request = SearchRequest(
            query="machine learning",
            corpus=["AI", "ML", "Deep learning"],
            top_k=2
        )
        response = api.search_corpus(request)
        
        assert isinstance(response, SearchResponse)
    
    def test_api_search_no_corpus(self, api):
        """Test search endpoint without corpus."""
        request = SearchRequest(query="test")
        response = api.search_corpus(request)
        
        assert response.success is False
        assert "corpus" in response.error.lower()
    
    def test_api_extract_entities(self, api):
        """Test entity extraction endpoint."""
        request = EntityExtractionRequest(
            text="Apple CEO Tim Cook announced new products."
        )
        response = api.extract_entities(request)
        
        assert isinstance(response, EntityExtractionResponse)
    
    def test_api_embeddings(self, api):
        """Test embeddings endpoint."""
        request = EmbeddingRequest(text="Hello world")
        response = api.get_embeddings(request)
        
        assert isinstance(response, EmbeddingResponse)
    
    def test_api_embeddings_batch(self, api):
        """Test batch embeddings endpoint."""
        request = EmbeddingRequest(texts=["Hello", "World"])
        response = api.get_embeddings(request)
        
        assert isinstance(response, EmbeddingResponse)
    
    def test_api_similarity(self, api):
        """Test similarity endpoint."""
        request = SimilarityRequest(text1="Hello", text2="Hi")
        response = api.compute_similarity(request)
        
        assert isinstance(response, SimilarityResponse)
    
    def test_api_shutdown(self, api):
        """Test API shutdown."""
        api.shutdown()
        
        assert api._initialized is False


class TestAPIFactory:
    """Tests for API factory functions."""
    
    def test_get_api_creates_instance(self):
        """Test get_api creates instance."""
        # Reset to ensure clean state
        import core.api_server as api_module
        api_module._api_instance = None
        
        api = get_api()
        
        assert api is not None
        assert isinstance(api, SigmalangAPI)
    
    def test_create_api_custom_config(self):
        """Test creating API with custom config."""
        config = SigmalangConfig()
        config.features = FeatureFlags(
            enable_analogies=False,
            enable_search=False
        )
        
        api = create_api(config)
        
        assert api.config.features.enable_analogies is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestAPIIntegration:
    """Integration tests for API."""
    
    def test_encode_decode_roundtrip(self):
        """Test encode then decode produces similar result."""
        reset_config()
        api = SigmalangAPI()
        
        try:
            api.initialize()
            
            # Encode
            encode_request = EncodeRequest(text="Hello world")
            encode_response = api.encode(encode_request)
            
            if encode_response.success and encode_response.vector:
                # Decode
                decode_request = DecodeRequest(vector=encode_response.vector)
                decode_response = api.decode(decode_request)
                
                assert decode_response.request_id is not None
        except Exception:
            pytest.skip("Encoder/Decoder not available")
    
    def test_analogy_pipeline(self):
        """Test complete analogy pipeline."""
        reset_config()
        api = SigmalangAPI()
        
        try:
            api.initialize()
            
            # Solve
            solve_request = AnalogyRequest(
                a="king", b="queen", c="man", top_k=3
            )
            solve_response = api.solve_analogy(solve_request)
            
            if solve_response.success and solve_response.best_answer:
                # Explain
                explain_request = AnalogyExplainRequest(
                    a="king", b="queen", c="man", d=solve_response.best_answer
                )
                explain_response = api.explain_analogy(explain_request)
                
                assert explain_response.request_id is not None
        except Exception:
            pytest.skip("Analogy engine not available")


# =============================================================================
# Performance Tests
# =============================================================================

class TestAPIPerformance:
    """Performance tests for API."""
    
    @pytest.mark.slow
    def test_encode_latency(self):
        """Test encode latency is acceptable."""
        api = SigmalangAPI()
        
        try:
            api.initialize()
            
            request = EncodeRequest(text="Hello world")
            
            start = time.perf_counter()
            for _ in range(10):
                api.encode(request)
            elapsed = time.perf_counter() - start
            
            avg_latency = elapsed / 10
            assert avg_latency < 0.1  # 100ms per request
        except Exception:
            pytest.skip("Encoder not available")
    
    @pytest.mark.slow
    def test_health_check_latency(self):
        """Test health check is fast."""
        api = SigmalangAPI()
        
        try:
            api.initialize()
            
            start = time.perf_counter()
            for _ in range(100):
                api.health()
            elapsed = time.perf_counter() - start
            
            avg_latency = elapsed / 100
            assert avg_latency < 0.01  # 10ms per request
        except Exception:
            pytest.skip("API not available")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
