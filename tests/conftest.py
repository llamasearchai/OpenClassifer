"""
Pytest configuration and shared fixtures for Open Classifier tests.
"""

import asyncio
import os
import tempfile
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import httpx
from fastapi.testclient import TestClient

# Set test environment variables
os.environ.update({
    "OPENAI_API_KEY": "test-key",
    "DEBUG": "true",
    "LOG_LEVEL": "DEBUG",
    "CACHE_TTL": "1",
    "RATE_LIMIT_REQUESTS": "1000",
    "RATE_LIMIT_PERIOD": "60",
    "MODEL_CACHE_DIR": tempfile.mkdtemp(),
    "DSPY_CACHE_DIR": tempfile.mkdtemp(),
    "LANGCHAIN_CACHE_DIR": tempfile.mkdtemp(),
})

from open_classifier.main import app
from open_classifier.core.config import settings
from open_classifier.services.classifier_service import ClassifierService
from open_classifier.models.dspy_classifier import DSPyClassifier
from open_classifier.models.langchain_classifier import LangChainClassifier
from open_classifier.models.agent import ClassificationAgent
from open_classifier.models.embeddings import EmbeddingModel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "positive",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


@pytest.fixture
def mock_embedding_response():
    """Mock embedding API response."""
    return {
        "data": [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim vector
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I love this product! It's amazing.",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic experience!",
        "Could be better, but not bad."
    ]


@pytest.fixture
def sample_labels():
    """Sample classification labels."""
    return ["positive", "negative", "neutral"]


@pytest.fixture
def classification_request():
    """Sample classification request."""
    return {
        "text": "I love this product!",
        "labels": ["positive", "negative", "neutral"],
        "mode": "dspy_only"
    }


@pytest.fixture
def batch_classification_request():
    """Sample batch classification request."""
    return {
        "texts": [
            "I love this product!",
            "This is terrible.",
            "It's okay."
        ],
        "labels": ["positive", "negative", "neutral"],
        "mode": "ensemble"
    }


@pytest.fixture
def mock_dspy_classifier():
    """Mock DSPy classifier."""
    mock = Mock(spec=DSPyClassifier)
    mock.classify = AsyncMock(return_value={
        "label": "positive",
        "confidence": 0.95,
        "reasoning": "The text expresses positive sentiment",
        "metadata": {"model": "dspy", "tokens_used": 15}
    })
    mock.classify_batch = AsyncMock(return_value=[
        {
            "text": "I love this product!",
            "label": "positive",
            "confidence": 0.95,
            "reasoning": "Positive sentiment",
            "metadata": {"model": "dspy", "tokens_used": 15}
        }
    ])
    return mock


@pytest.fixture
def mock_langchain_classifier():
    """Mock LangChain classifier."""
    mock = Mock(spec=LangChainClassifier)
    mock.classify = AsyncMock(return_value={
        "label": "positive",
        "confidence": 0.92,
        "reasoning": "The text shows positive sentiment",
        "metadata": {"model": "langchain", "tokens_used": 18}
    })
    mock.classify_batch = AsyncMock(return_value=[
        {
            "text": "I love this product!",
            "label": "positive",
            "confidence": 0.92,
            "reasoning": "Positive sentiment",
            "metadata": {"model": "langchain", "tokens_used": 18}
        }
    ])
    return mock


@pytest.fixture
def mock_agent():
    """Mock classification agent."""
    mock = Mock(spec=ClassificationAgent)
    mock.classify = AsyncMock(return_value={
        "label": "positive",
        "confidence": 0.98,
        "reasoning": "Agent analysis shows strong positive sentiment",
        "metadata": {
            "model": "agent",
            "tokens_used": 25,
            "tools_used": ["classification", "similarity"],
            "iterations": 2
        }
    })
    return mock


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    mock = Mock(spec=EmbeddingModel)
    mock.get_embeddings = AsyncMock(return_value=[
        [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim vector
    ])
    mock.find_similar = AsyncMock(return_value=[
        {
            "text": "Similar text",
            "similarity": 0.85,
            "index": 0
        }
    ])
    mock.cluster_texts = AsyncMock(return_value={
        "clusters": [
            {
                "cluster_id": 0,
                "texts": ["I love this product!"],
                "centroid": [0.1, 0.2, 0.3, 0.4, 0.5] * 100,
                "size": 1
            }
        ],
        "metadata": {
            "n_clusters": 1,
            "silhouette_score": 0.8
        }
    })
    return mock


@pytest.fixture
def mock_classifier_service(
    mock_dspy_classifier,
    mock_langchain_classifier,
    mock_agent,
    mock_embedding_model
):
    """Mock classifier service with all components."""
    mock = Mock(spec=ClassifierService)
    mock.dspy_classifier = mock_dspy_classifier
    mock.langchain_classifier = mock_langchain_classifier
    mock.agent = mock_agent
    mock.embedding_model = mock_embedding_model
    mock.current_mode = "dspy_only"
    
    # Mock service methods
    mock.classify = AsyncMock(return_value={
        "label": "positive",
        "confidence": 0.95,
        "reasoning": "Positive sentiment detected",
        "metadata": {"model": "dspy", "tokens_used": 15}
    })
    
    mock.classify_batch = AsyncMock(return_value=[
        {
            "text": "I love this product!",
            "label": "positive",
            "confidence": 0.95,
            "reasoning": "Positive sentiment",
            "metadata": {"model": "dspy", "tokens_used": 15}
        }
    ])
    
    mock.get_embeddings = AsyncMock(return_value=[
        [0.1, 0.2, 0.3, 0.4, 0.5] * 100
    ])
    
    mock.find_similar = AsyncMock(return_value=[
        {
            "text": "Similar text",
            "similarity": 0.85,
            "index": 0
        }
    ])
    
    mock.cluster_texts = AsyncMock(return_value={
        "clusters": [
            {
                "cluster_id": 0,
                "texts": ["I love this product!"],
                "centroid": [0.1, 0.2, 0.3, 0.4, 0.5] * 100,
                "size": 1
            }
        ],
        "metadata": {
            "n_clusters": 1,
            "silhouette_score": 0.8
        }
    })
    
    mock.get_metrics = Mock(return_value={
        "total_requests": 100,
        "successful_requests": 95,
        "failed_requests": 5,
        "average_response_time": 0.5,
        "cache_hit_rate": 0.8,
        "current_mode": "dspy_only"
    })
    
    mock.set_mode = Mock()
    mock.clear_cache = Mock()
    
    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch("openai.AsyncOpenAI") as mock_client:
        # Mock chat completions
        mock_instance = mock_client.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(
                        message=Mock(content="positive"),
                        finish_reason="stop"
                    )
                ],
                usage=Mock(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15
                )
            )
        )
        
        # Mock embeddings
        mock_instance.embeddings.create = AsyncMock(
            return_value=Mock(
                data=[
                    Mock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 100)
                ],
                usage=Mock(
                    prompt_tokens=5,
                    total_tokens=5
                )
            )
        )
        
        yield mock_instance


@pytest.fixture
def performance_data():
    """Sample performance data for testing."""
    return {
        "response_times": [0.1, 0.2, 0.15, 0.3, 0.25],
        "memory_usage": [100, 120, 110, 130, 115],
        "cpu_usage": [10, 15, 12, 18, 14],
        "cache_hits": 80,
        "cache_misses": 20,
        "total_requests": 100
    }


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test."""
    # This would reset any global metrics if they exist
    yield
    # Cleanup after test


@pytest.fixture
def api_headers():
    """Standard API headers for testing."""
    return {
        "Authorization": "Bearer demo-key",
        "Content-Type": "application/json"
    }


@pytest.fixture
def invalid_api_headers():
    """Invalid API headers for testing authentication."""
    return {
        "Authorization": "Bearer invalid-key",
        "Content-Type": "application/json"
    }


# Pytest markers for different test types
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.benchmark = pytest.mark.benchmark 