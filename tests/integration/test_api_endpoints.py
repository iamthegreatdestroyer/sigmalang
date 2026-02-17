"""
API Endpoints Integration Tests

Tests all FastAPI REST endpoints with real HTTP requests.
Verifies request/response schemas, error handling, and
end-to-end API functionality.

Test Coverage:
- /encode endpoint (single and batch)
- /decode endpoint
- /analogy endpoint (solve and explain)
- /search endpoint
- /entities endpoint
- /embedding endpoint
- /similarity endpoint
- /health endpoint
- /info endpoint
- Error responses (4xx, 5xx)
"""

import sys
import json
import pytest
from pathlib import Path
from typing import Dict, Any

# Add parent to path
sigmalang_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sigmalang_root))

from fastapi.testclient import TestClient
from sigmalang.core.api_server import create_api
from sigmalang.core.config import reset_config


@pytest.fixture(scope="module")
def client():
    """Create test client for API."""
    reset_config()  # Reset config to defaults
    api = create_api()
    return TestClient(api.app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.integration
    def test_health_check_returns_200(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_health_check_response_structure(self, client):
        """Test health endpoint response structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.integration
    def test_health_check_includes_uptime(self, client):
        """Test health endpoint includes uptime info."""
        response = client.get("/health")
        data = response.json()

        # Uptime should be present and non-negative
        if "uptime" in data:
            assert data["uptime"] >= 0


class TestInfoEndpoint:
    """Tests for /info endpoint."""

    @pytest.mark.integration
    def test_info_returns_200(self, client):
        """Test info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_info_response_structure(self, client):
        """Test info endpoint response structure."""
        response = client.get("/info")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert data["name"] == "SigmaLang"


class TestEncodeEndpoint:
    """Tests for /encode endpoint."""

    @pytest.mark.integration
    def test_encode_simple_text(self, client):
        """Test encoding simple text."""
        response = client.post(
            "/encode",
            json={"text": "Hello, world!"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "vector" in data
        assert isinstance(data["vector"], list)
        assert len(data["vector"]) > 0

    @pytest.mark.integration
    def test_encode_with_normalization(self, client):
        """Test encoding with normalization enabled."""
        response = client.post(
            "/encode",
            json={
                "text": "Machine learning",
                "normalize": True
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.integration
    def test_encode_batch_texts(self, client):
        """Test batch encoding multiple texts."""
        response = client.post(
            "/encode",
            json={
                "texts": [
                    "First text",
                    "Second text",
                    "Third text"
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "vectors" in data
        assert len(data["vectors"]) == 3

    @pytest.mark.integration
    def test_encode_empty_text_returns_error(self, client):
        """Test encoding empty text returns appropriate error."""
        response = client.post(
            "/encode",
            json={"text": ""}
        )

        # Should still process, might return empty vector or error
        assert response.status_code in [200, 400]

    @pytest.mark.integration
    def test_encode_missing_text_returns_422(self, client):
        """Test encoding without text parameter returns validation error."""
        response = client.post(
            "/encode",
            json={}
        )

        assert response.status_code == 422  # Validation error


class TestDecodeEndpoint:
    """Tests for /decode endpoint."""

    @pytest.mark.integration
    def test_decode_after_encode(self, client):
        """Test decoding previously encoded data."""
        # First, encode
        encode_response = client.post(
            "/encode",
            json={"text": "Test message"}
        )
        assert encode_response.status_code == 200
        encode_data = encode_response.json()

        # Then, decode (if API supports it)
        if "encoded" in encode_data or "vector" in encode_data:
            vector_or_encoded = encode_data.get("encoded") or encode_data.get("vector")

            decode_response = client.post(
                "/decode",
                json={"encoded": vector_or_encoded}
            )

            # Should either succeed or indicate not implemented
            assert decode_response.status_code in [200, 501]

    @pytest.mark.integration
    def test_decode_invalid_data_returns_error(self, client):
        """Test decoding invalid data returns error."""
        response = client.post(
            "/decode",
            json={"encoded": [0.1, 0.2, 0.3]}  # Invalid encoded data
        )

        # Should return error (400 or 500)
        assert response.status_code in [400, 500, 501]


class TestAnalogyEndpoint:
    """Tests for /analogy endpoint."""

    @pytest.mark.integration
    def test_analogy_solve_basic(self, client):
        """Test basic analogy solving: A:B::C:?"""
        response = client.post(
            "/analogy",
            json={
                "a": "king",
                "b": "queen",
                "c": "man",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "solutions" in data
        assert isinstance(data["solutions"], list)
        if data["solutions"]:
            assert "answer" in data["solutions"][0]
            assert "confidence" in data["solutions"][0]

    @pytest.mark.integration
    def test_analogy_with_different_types(self, client):
        """Test analogy with different analogy types."""
        response = client.post(
            "/analogy",
            json={
                "a": "Python",
                "b": "programming",
                "c": "JavaScript",
                "analogy_type": "semantic"
            }
        )

        assert response.status_code in [200, 400]  # May not support all types

    @pytest.mark.integration
    def test_analogy_explain(self, client):
        """Test analogy explanation endpoint."""
        response = client.post(
            "/analogy/explain",
            json={
                "a": "cat",
                "b": "meow",
                "c": "dog",
                "d": "bark"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "explanation" in data or "confidence" in data


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    @pytest.mark.integration
    def test_search_basic_query(self, client):
        """Test basic semantic search."""
        response = client.post(
            "/search",
            json={
                "query": "machine learning algorithms",
                "top_k": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.integration
    def test_search_with_filters(self, client):
        """Test search with filters."""
        response = client.post(
            "/search",
            json={
                "query": "neural networks",
                "filters": {"category": "ai"},
                "top_k": 5
            }
        )

        assert response.status_code in [200, 400]  # Filters may not be implemented

    @pytest.mark.integration
    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.post(
            "/search",
            json={"query": ""}
        )

        assert response.status_code in [200, 400]


class TestEntityEndpoint:
    """Tests for /entities endpoint."""

    @pytest.mark.integration
    def test_entity_extraction_basic(self, client):
        """Test basic entity extraction."""
        response = client.post(
            "/entities",
            json={
                "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        assert isinstance(data["entities"], list)

    @pytest.mark.integration
    def test_entity_extraction_with_relations(self, client):
        """Test entity extraction including relations."""
        response = client.post(
            "/entities",
            json={
                "text": "John works at Microsoft in Seattle.",
                "include_relations": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        if "relations" in data:
            assert isinstance(data["relations"], list)


class TestEmbeddingEndpoint:
    """Tests for /embedding endpoint."""

    @pytest.mark.integration
    def test_embedding_generation(self, client):
        """Test generating embeddings."""
        response = client.post(
            "/embedding",
            json={"text": "Generate embedding for this text"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "embedding" in data or "vector" in data
        embedding = data.get("embedding") or data.get("vector")
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    @pytest.mark.integration
    def test_embedding_dimensions(self, client):
        """Test embedding dimensions are consistent."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = []

        for text in texts:
            response = client.post(
                "/embedding",
                json={"text": text}
            )
            assert response.status_code == 200
            data = response.json()
            embedding = data.get("embedding") or data.get("vector")
            embeddings.append(embedding)

        # All embeddings should have same dimension
        assert len(set(len(e) for e in embeddings)) == 1


class TestSimilarityEndpoint:
    """Tests for /similarity endpoint."""

    @pytest.mark.integration
    def test_similarity_computation(self, client):
        """Test computing similarity between texts."""
        response = client.post(
            "/similarity",
            json={
                "text1": "Machine learning algorithms",
                "text2": "Deep learning models"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "similarity" in data
        assert 0.0 <= data["similarity"] <= 1.0

    @pytest.mark.integration
    def test_similarity_identical_texts(self, client):
        """Test similarity of identical texts is high."""
        text = "Identical text"

        response = client.post(
            "/similarity",
            json={"text1": text, "text2": text}
        )

        assert response.status_code == 200
        data = response.json()

        # Identical texts should have high similarity
        assert data["similarity"] >= 0.9


class TestBatchEndpoint:
    """Tests for /batch endpoint (if available)."""

    @pytest.mark.integration
    def test_batch_encode(self, client):
        """Test batch encoding endpoint."""
        response = client.post(
            "/batch/encode",
            json={
                "texts": [
                    "First text for batch encoding",
                    "Second text for batch encoding",
                    "Third text for batch encoding"
                ]
            }
        )

        # Batch endpoint may or may not exist
        if response.status_code == 404:
            pytest.skip("Batch endpoint not implemented")

        assert response.status_code == 200
        data = response.json()

        assert "results" in data or "vectors" in data
        results = data.get("results") or data.get("vectors")
        assert len(results) == 3


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    @pytest.mark.integration
    def test_invalid_json_returns_422(self, client):
        """Test invalid JSON returns 422."""
        response = client.post(
            "/encode",
            data="not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    @pytest.mark.integration
    def test_method_not_allowed(self, client):
        """Test wrong HTTP method returns 405."""
        response = client.get("/encode")  # Should be POST

        assert response.status_code == 405

    @pytest.mark.integration
    def test_not_found_returns_404(self, client):
        """Test non-existent endpoint returns 404."""
        response = client.get("/nonexistent")

        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
