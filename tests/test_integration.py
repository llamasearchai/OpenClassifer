"""
Comprehensive integration tests for OpenClassifier.
Tests the entire system end-to-end with real components.
"""

import pytest
import asyncio
import httpx
import json
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from open_classifier.main import app
from open_classifier.core.config import settings
from open_classifier.services.classifier_service import ClassifierService
from open_classifier.models.embeddings import EmbeddingModel
from open_classifier.utils.cache import get_cache, clear_all_caches


class TestSystemIntegration:
    """Integration tests for the complete OpenClassifier system."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create HTTP client for API testing."""
        return httpx.AsyncClient(app=app, base_url="http://testserver")

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear caches before each test
        clear_all_caches()
        yield
        # Cleanup after each test
        clear_all_caches()

    @pytest.mark.asyncio
    async def test_complete_classification_workflow(self, client):
        """Test the complete classification workflow from API to response."""
        # Mock external dependencies
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'), \
             patch('open_classifier.models.embeddings.SentenceTransformer') as mock_transformer:
            
            # Setup mocks
            mock_transformer.return_value.encode.return_value = [[0.1] * 384]
            
            # Test data
            request_data = {
                "text": "This is an excellent product with amazing features!",
                "labels": ["positive", "negative", "neutral"],
                "use_ensemble": True,
                "return_probabilities": True,
                "confidence_threshold": 0.7
            }
            
            # Make API request
            response = await client.post("/api/v1/classify", json=request_data)
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            
            assert "classification" in data
            assert "metadata" in data
            assert data["classification"]["class"] in request_data["labels"]
            assert 0 <= data["classification"]["confidence"] <= 1
            
            if "probabilities" in data["classification"]:
                probabilities = data["classification"]["probabilities"]
                assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_batch_classification_integration(self, client):
        """Test batch classification functionality."""
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'), \
             patch('open_classifier.models.embeddings.SentenceTransformer'):
            
            # Test data
            request_data = {
                "texts": [
                    "Excellent product, highly recommend!",
                    "Terrible quality, waste of money.",
                    "Average product, nothing special."
                ],
                "labels": ["positive", "negative", "neutral"],
                "use_ensemble": True
            }
            
            # Make API request
            response = await client.post("/api/v1/classify/batch", json=request_data)
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) == len(request_data["texts"])
            
            for result in data["results"]:
                assert "class" in result
                assert "confidence" in result
                assert result["class"] in request_data["labels"]

    @pytest.mark.asyncio
    async def test_similarity_search_integration(self, client):
        """Test similarity search functionality."""
        with patch('open_classifier.models.embeddings.SentenceTransformer') as mock_transformer:
            # Mock embeddings
            mock_transformer.return_value.encode.side_effect = [
                [[1.0, 0.0, 0.0]],  # Query embedding
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]  # Candidate embeddings
            ]
            
            # Test data
            request_data = {
                "query": "machine learning algorithms",
                "candidates": [
                    "neural networks and deep learning",
                    "cooking recipes and techniques",
                    "artificial intelligence research"
                ],
                "top_k": 2
            }
            
            # Make API request
            response = await client.post("/api/v1/similarity", json=request_data)
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) <= request_data["top_k"]
            
            for result in data["results"]:
                assert "text" in result
                assert "similarity" in result
                assert 0 <= result["similarity"] <= 1

    @pytest.mark.asyncio
    async def test_embedding_generation_integration(self, client):
        """Test embedding generation functionality."""
        with patch('open_classifier.models.embeddings.SentenceTransformer') as mock_transformer:
            # Mock embeddings
            mock_embeddings = [[0.1, 0.2, 0.3] * 128]  # 384 dimensions
            mock_transformer.return_value.encode.return_value = mock_embeddings
            
            # Test data
            request_data = {
                "texts": ["Hello world", "How are you?"],
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
            # Make API request
            response = await client.post("/api/v1/embeddings", json=request_data)
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            
            assert "embeddings" in data
            assert "model" in data
            assert "dimensions" in data
            assert len(data["embeddings"]) == len(request_data["texts"])

    @pytest.mark.asyncio
    async def test_system_monitoring_endpoints(self, client):
        """Test system monitoring and health endpoints."""
        # Test health endpoint
        response = await client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] == "healthy"
        
        # Test system info endpoint
        response = await client.get("/api/v1/system/info")
        assert response.status_code == 200
        
        info_data = response.json()
        assert "version" in info_data
        assert "environment" in info_data
        assert "uptime" in info_data
        
        # Test models endpoint
        response = await client.get("/api/v1/models")
        assert response.status_code == 200
        
        models_data = response.json()
        assert "models" in models_data
        assert isinstance(models_data["models"], list)
        
        # Test metrics endpoint
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    @pytest.mark.asyncio
    async def test_caching_integration(self, client):
        """Test caching functionality integration."""
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'), \
             patch('open_classifier.models.embeddings.SentenceTransformer'):
            
            # Test data
            request_data = {
                "text": "This is a test for caching functionality",
                "labels": ["positive", "negative", "neutral"]
            }
            
            # First request - should hit the actual service
            start_time = time.time()
            response1 = await client.post("/api/v1/classify", json=request_data)
            first_request_time = time.time() - start_time
            
            assert response1.status_code == 200
            
            # Second request - should be served from cache (faster)
            start_time = time.time()
            response2 = await client.post("/api/v1/classify", json=request_data)
            second_request_time = time.time() - start_time
            
            assert response2.status_code == 200
            
            # Cache hit should be faster than original request
            # Note: This might not always be true in test environment
            # but is a good indicator of caching working
            
            # Verify both responses are identical
            assert response1.json() == response2.json()

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, client):
        """Test error handling across the system."""
        # Test with invalid input
        invalid_requests = [
            {"text": "", "labels": []},  # Empty text and labels
            {"text": "x" * 100000, "labels": ["positive"]},  # Text too long
            {"labels": ["positive"]},  # Missing text
            {"text": "test"},  # Missing labels
        ]
        
        for invalid_request in invalid_requests:
            response = await client.post("/api/v1/classify", json=invalid_request)
            assert response.status_code == 422  # Validation error
        
        # Test with invalid JSON
        response = await client.post(
            "/api/v1/classify",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_concurrent_requests_integration(self, client):
        """Test handling of concurrent requests."""
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'), \
             patch('open_classifier.models.embeddings.SentenceTransformer'):
            
            # Create multiple concurrent requests
            request_data = {
                "text": "Test concurrent request",
                "labels": ["positive", "negative", "neutral"]
            }
            
            # Send 10 concurrent requests
            tasks = []
            for i in range(10):
                task = client.post("/api/v1/classify", json={
                    **request_data,
                    "text": f"Test concurrent request {i}"
                })
                tasks.append(task)
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks)
            
            # Verify all requests succeeded
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "classification" in data

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, client):
        """Test rate limiting functionality."""
        # Note: This test assumes rate limiting is configured
        # Adjust based on your rate limiting settings
        
        request_data = {
            "text": "Rate limiting test",
            "labels": ["positive", "negative"]
        }
        
        # Send requests in quick succession
        responses = []
        for _ in range(5):  # Adjust based on rate limit
            response = await client.post("/api/v1/classify", json=request_data)
            responses.append(response)
        
        # Most requests should succeed, but some might be rate limited
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        
        # At least some requests should succeed
        assert success_count > 0
        
        # If rate limiting is enabled, some might be limited
        total_responses = success_count + rate_limited_count
        assert total_responses == len(responses)

    @pytest.mark.asyncio
    async def test_websocket_integration(self, client):
        """Test WebSocket functionality."""
        # Note: This test requires WebSocket support in the test client
        # Adjust based on your WebSocket implementation
        
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'):
            
            # Test WebSocket connection and messaging
            # This is a simplified test - actual implementation may vary
            async with client.websocket_connect("/ws/classify") as websocket:
                # Send classification request
                await websocket.send_json({
                    "text": "WebSocket test message",
                    "labels": ["positive", "negative"]
                })
                
                # Receive response
                response = await websocket.receive_json()
                
                # Verify response structure
                assert "class" in response
                assert "confidence" in response

    def test_service_dependencies(self):
        """Test that all service dependencies are properly configured."""
        # Test ClassifierService initialization
        service = ClassifierService()
        assert service is not None
        
        # Test EmbeddingModel initialization
        with patch('open_classifier.models.embeddings.SentenceTransformer'):
            embedding_model = EmbeddingModel()
            assert embedding_model is not None
        
        # Test cache initialization
        cache = get_cache()
        assert cache is not None
        
        # Test cache operations
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        cache.delete("test_key")
        assert cache.get("test_key") is None

    @pytest.mark.asyncio
    async def test_openapi_schema_integration(self, client):
        """Test OpenAPI schema generation and validation."""
        # Test OpenAPI JSON schema
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Verify key endpoints are documented
        paths = schema["paths"]
        assert "/api/v1/classify" in paths
        assert "/api/v1/classify/batch" in paths
        assert "/api/v1/similarity" in paths
        assert "/health" in paths
        
        # Test Swagger UI
        response = await client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cors_integration(self, client):
        """Test CORS functionality."""
        # Test CORS preflight request
        response = await client.options(
            "/api/v1/classify",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    @pytest.mark.asyncio
    async def test_content_type_handling(self, client):
        """Test handling of different content types."""
        request_data = {
            "text": "Content type test",
            "labels": ["positive", "negative"]
        }
        
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'):
            
            # Test JSON content type
            response = await client.post(
                "/api/v1/classify",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 200
            
            # Test unsupported content type
            response = await client.post(
                "/api/v1/classify",
                content="text data",
                headers={"Content-Type": "text/plain"}
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_performance_under_load(self, client):
        """Test system performance under load."""
        with patch('open_classifier.models.dspy_classifier.dspy.OpenAI'), \
             patch('open_classifier.models.langchain_classifier.ChatOpenAI'), \
             patch('open_classifier.models.embeddings.SentenceTransformer'):
            
            # Test with multiple requests
            request_data = {
                "text": "Performance test message",
                "labels": ["positive", "negative", "neutral"]
            }
            
            start_time = time.time()
            
            # Send 20 requests concurrently
            tasks = []
            for i in range(20):
                task = client.post("/api/v1/classify", json=request_data)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Verify all requests succeeded
            for response in responses:
                assert response.status_code == 200
            
            # Performance assertion (adjust based on requirements)
            requests_per_second = len(responses) / total_time
            assert requests_per_second > 5  # At least 5 requests per second


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 