"""
Comprehensive API tests for OpenClassifier endpoints.
Tests all API functionality with proper mocking and error scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import json

from open_classifier.main import app
from open_classifier.api.models import ClassificationRequest, BatchClassificationRequest


class TestClassificationAPI:
    """Test suite for classification API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_classification_request(self):
        """Sample classification request data."""
        return {
            "text": "This product is absolutely amazing and exceeded expectations!",
            "labels": ["positive", "negative", "neutral"],
            "use_ensemble": True,
            "return_probabilities": True,
            "confidence_threshold": 0.7
        }

    @pytest.fixture
    def mock_classification_response(self):
        """Mock classification response."""
        return {
            "class": "positive",
            "confidence": 0.94,
            "probabilities": {
                "positive": 0.94,
                "negative": 0.03,
                "neutral": 0.03
            },
            "explanation": "High confidence positive sentiment with strong emotional indicators",
            "metadata": {
                "model_used": "ensemble",
                "processing_time": 0.234,
                "ensemble_agreement": True
            }
        }

    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    def test_classify_endpoint_success(self, mock_classify, client, sample_classification_request, mock_classification_response):
        """Test successful classification."""
        # Arrange
        mock_classify.return_value = mock_classification_response

        # Act
        response = client.post("/api/v1/classify", json=sample_classification_request)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["classification"]["class"] == "positive"
        assert data["classification"]["confidence"] == 0.94
        assert "positive" in data["classification"]["probabilities"]
        assert data["metadata"]["model_used"] == "ensemble"

    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    def test_classify_endpoint_validation_error(self, mock_classify, client):
        """Test classification with invalid input."""
        # Arrange
        invalid_request = {
            "text": "",  # Empty text
            "labels": [],  # Empty labels
        }

        # Act
        response = client.post("/api/v1/classify", json=invalid_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    def test_classify_endpoint_text_too_long(self, mock_classify, client):
        """Test classification with text exceeding length limit."""
        # Arrange
        long_text_request = {
            "text": "x" * 100000,  # Very long text
            "labels": ["positive", "negative"]
        }

        # Act
        response = client.post("/api/v1/classify", json=long_text_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "too long" in str(data["detail"]).lower()

    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    def test_classify_endpoint_service_error(self, mock_classify, client, sample_classification_request):
        """Test classification service error handling."""
        # Arrange
        mock_classify.side_effect = Exception("Model service unavailable")

        # Act
        response = client.post("/api/v1/classify", json=sample_classification_request)

        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data["detail"].lower()

    @patch('open_classifier.services.classifier_service.ClassifierService.batch_classify')
    def test_batch_classify_endpoint_success(self, mock_batch_classify, client):
        """Test successful batch classification."""
        # Arrange
        batch_request = {
            "texts": [
                "This is great!",
                "This is terrible!",
                "This is okay."
            ],
            "labels": ["positive", "negative", "neutral"],
            "use_ensemble": True
        }
        
        mock_batch_classify.return_value = [
            {"class": "positive", "confidence": 0.9, "text": "This is great!"},
            {"class": "negative", "confidence": 0.85, "text": "This is terrible!"},
            {"class": "neutral", "confidence": 0.7, "text": "This is okay."}
        ]

        # Act
        response = client.post("/api/v1/classify/batch", json=batch_request)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 3
        assert data["results"][0]["class"] == "positive"
        assert data["results"][1]["class"] == "negative"
        assert data["results"][2]["class"] == "neutral"

    @patch('open_classifier.services.classifier_service.ClassifierService.batch_classify')
    def test_batch_classify_endpoint_empty_batch(self, mock_batch_classify, client):
        """Test batch classification with empty batch."""
        # Arrange
        empty_batch_request = {
            "texts": [],
            "labels": ["positive", "negative"]
        }

        # Act
        response = client.post("/api/v1/classify/batch", json=empty_batch_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('open_classifier.services.classifier_service.ClassifierService.batch_classify')
    def test_batch_classify_endpoint_exceeds_limit(self, mock_batch_classify, client):
        """Test batch classification exceeding batch size limit."""
        # Arrange
        large_batch_request = {
            "texts": ["text"] * 1000,  # Exceeds limit
            "labels": ["positive", "negative"]
        }

        # Act
        response = client.post("/api/v1/classify/batch", json=large_batch_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "batch size" in str(data["detail"]).lower()


class TestSimilarityAPI:
    """Test suite for similarity search endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_similarity_request(self):
        """Sample similarity request data."""
        return {
            "query": "machine learning algorithms",
            "candidates": [
                "neural networks and deep learning",
                "cooking recipes and ingredients",
                "artificial intelligence research"
            ],
            "top_k": 2
        }

    @patch('open_classifier.models.embeddings.EmbeddingModel.compute_similarity')
    def test_similarity_endpoint_success(self, mock_similarity, client, sample_similarity_request):
        """Test successful similarity computation."""
        # Arrange
        mock_similarity.return_value = [
            {"text": "neural networks and deep learning", "similarity": 0.92},
            {"text": "artificial intelligence research", "similarity": 0.78}
        ]

        # Act
        response = client.post("/api/v1/similarity", json=sample_similarity_request)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["similarity"] > data["results"][1]["similarity"]

    @patch('open_classifier.models.embeddings.EmbeddingModel.compute_similarity')
    def test_similarity_endpoint_empty_query(self, mock_similarity, client):
        """Test similarity with empty query."""
        # Arrange
        invalid_request = {
            "query": "",
            "candidates": ["text1", "text2"]
        }

        # Act
        response = client.post("/api/v1/similarity", json=invalid_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('open_classifier.models.embeddings.EmbeddingModel.compute_similarity')
    def test_similarity_endpoint_no_candidates(self, mock_similarity, client):
        """Test similarity with no candidates."""
        # Arrange
        invalid_request = {
            "query": "test query",
            "candidates": []
        }

        # Act
        response = client.post("/api/v1/similarity", json=invalid_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestEmbeddingAPI:
    """Test suite for embedding generation endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch('open_classifier.models.embeddings.EmbeddingModel.generate_embeddings')
    def test_embeddings_endpoint_success(self, mock_embeddings, client):
        """Test successful embedding generation."""
        # Arrange
        request_data = {
            "texts": ["Hello world", "How are you?"],
            "model": "text-embedding-3-large"
        }
        
        mock_embeddings.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4] * 768,  # 3072 dimensions
                [0.5, 0.6, 0.7, 0.8] * 768
            ],
            "model": "text-embedding-3-large",
            "dimensions": 3072
        }

        # Act
        response = client.post("/api/v1/embeddings", json=request_data)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["embeddings"]) == 2
        assert data["dimensions"] == 3072
        assert data["model"] == "text-embedding-3-large"

    @patch('open_classifier.models.embeddings.EmbeddingModel.generate_embeddings')
    def test_embeddings_endpoint_empty_texts(self, mock_embeddings, client):
        """Test embedding generation with empty texts."""
        # Arrange
        invalid_request = {
            "texts": [],
            "model": "text-embedding-3-large"
        }

        # Act
        response = client.post("/api/v1/embeddings", json=invalid_request)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSystemAPI:
    """Test suite for system and monitoring endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_system_info_endpoint(self, client):
        """Test system information endpoint."""
        # Act
        response = client.get("/api/v1/system/info")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "version" in data
        assert "environment" in data
        assert "uptime" in data

    def test_models_endpoint(self, client):
        """Test available models endpoint."""
        # Act
        response = client.get("/api/v1/models")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        # Act
        response = client.get("/metrics")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        assert "text/plain" in response.headers["content-type"]

    def test_openapi_docs(self, client):
        """Test OpenAPI documentation."""
        # Act
        response = client.get("/docs")

        # Assert
        assert response.status_code == status.HTTP_200_OK

    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema."""
        # Act
        response = client.get("/openapi.json")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "openapi" in data
        assert "info" in data


class TestAuthenticationAndSecurity:
    """Test suite for authentication and security features."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def authenticated_headers(self):
        """Headers with valid API key."""
        return {"X-API-Key": "test-api-key"}

    @pytest.fixture
    def sample_request(self):
        """Sample classification request."""
        return {
            "text": "Test text",
            "labels": ["positive", "negative"]
        }

    @patch('open_classifier.core.config.settings.API_KEY_REQUIRED', True)
    def test_api_key_required(self, client, sample_request):
        """Test that API key is required when configured."""
        # Act
        response = client.post("/api/v1/classify", json=sample_request)

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch('open_classifier.core.config.settings.API_KEY_REQUIRED', True)
    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    def test_valid_api_key_accepted(self, mock_classify, client, sample_request, authenticated_headers):
        """Test that valid API key is accepted."""
        # Arrange
        mock_classify.return_value = {"class": "positive", "confidence": 0.8}

        # Act
        response = client.post("/api/v1/classify", json=sample_request, headers=authenticated_headers)

        # Assert
        assert response.status_code == status.HTTP_200_OK

    def test_invalid_api_key_rejected(self, client, sample_request):
        """Test that invalid API key is rejected."""
        # Arrange
        invalid_headers = {"X-API-Key": "invalid-key"}

        # Act
        response = client.post("/api/v1/classify", json=sample_request, headers=invalid_headers)

        # Assert
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @patch('open_classifier.core.middleware.rate_limiter')
    def test_rate_limiting(self, mock_rate_limiter, client, sample_request):
        """Test rate limiting functionality."""
        # Arrange
        mock_rate_limiter.side_effect = Exception("Rate limit exceeded")

        # Act
        response = client.post("/api/v1/classify", json=sample_request)

        # Assert
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        # Act
        response = client.options("/api/v1/classify")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in response.headers

    def test_request_size_limit(self, client):
        """Test request size limitation."""
        # Arrange
        large_request = {
            "text": "x" * 10000000,  # Very large text
            "labels": ["positive", "negative"]
        }

        # Act
        response = client.post("/api/v1/classify", json=large_request)

        # Assert
        assert response.status_code in [status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, status.HTTP_422_UNPROCESSABLE_ENTITY]


class TestWebSocketAPI:
    """Test suite for WebSocket functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    def test_websocket_classification(self, mock_classify, client):
        """Test WebSocket classification."""
        # Arrange
        mock_classify.return_value = {"class": "positive", "confidence": 0.9}

        # Act & Assert
        with client.websocket_connect("/ws/classify") as websocket:
            # Send classification request
            websocket.send_json({
                "text": "This is great!",
                "labels": ["positive", "negative"]
            })
            
            # Receive response
            data = websocket.receive_json()
            assert data["class"] == "positive"
            assert data["confidence"] == 0.9

    def test_websocket_invalid_message(self, client):
        """Test WebSocket with invalid message format."""
        # Act & Assert
        with client.websocket_connect("/ws/classify") as websocket:
            # Send invalid message
            websocket.send_text("invalid json")
            
            # Should receive error response
            data = websocket.receive_json()
            assert "error" in data


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test suite for async endpoint behavior."""

    @patch('open_classifier.services.classifier_service.ClassifierService.classify')
    async def test_concurrent_requests(self, mock_classify):
        """Test handling of concurrent requests."""
        # Arrange
        mock_classify.return_value = {"class": "positive", "confidence": 0.8}
        
        async with AsyncMock() as client:
            # Simulate multiple concurrent requests
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    client.post("/api/v1/classify", json={
                        "text": f"Test text {i}",
                        "labels": ["positive", "negative"]
                    })
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks)
            
            # Assert all requests completed successfully
            assert len(responses) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 